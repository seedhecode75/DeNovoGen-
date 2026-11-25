import torch
from torch import nn
from torch.distributions import Categorical

from .chem import BOS, EOS, PAD, UNK


class Generator(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg["embedding"], padding_idx=PAD)
        self.lstm = nn.LSTM(
            cfg["embedding"], cfg["hidden"], num_layers=2, batch_first=True
        )
        self.output = nn.Linear(cfg["hidden"], vocab_size)

    def forward(self, tokens, state=None):
        states, state = self.lstm(self.embedding(tokens), state)
        return self.output(states), state


def action_logits(logits, temperature=1.0, top_p=1.0):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    logits = logits.float().clone() / temperature
    logits[..., [BOS, PAD, UNK]] = -torch.inf
    if top_p < 1:
        values, indices = logits.sort(descending=True)
        probs = values.softmax(-1)
        values = values.masked_fill(probs.cumsum(-1) - probs >= top_p, -torch.inf)
        logits = logits.scatter(-1, indices, values)
    return logits


@torch.inference_mode()
def sample(model, batch, steps, device, gen, temperature=1.0, top_p=1.0):
    # Keep a dense batch for efficient LSTM kernels; finished rows emit only PAD.
    prefix = torch.full((batch, 1), BOS, device=device, dtype=torch.long)
    seq = [prefix[:, 0]]
    done = torch.zeros(batch, device=device, dtype=torch.bool)
    state = None
    for _ in range(steps):
        logits, state = model(prefix, state)
        logits = action_logits(logits[:, 0], temperature, top_p)
        token = torch.multinomial(logits.softmax(-1), 1, generator=gen).squeeze(1)
        token = token.masked_fill(done, PAD)
        seq.append(token)
        done |= token.eq(EOS)
        if done.all():
            break
        prefix = token[:, None]
    return torch.stack(seq, 1)


def decode(seqs, vocab):
    result = []
    for row in seqs.tolist():
        tokens = []
        eos = False
        for i in row[1:]:
            if i == EOS:
                eos = True
                break
            if i in (PAD, BOS, UNK):
                raise ValueError("Unexpected special token in active sequence")
            tokens.append(vocab[i])
        result.append(("".join(tokens), eos))
    return result


def trajectory(model, seqs):
    logits, _ = model(seqs[:, :-1])
    dist = Categorical(logits=action_logits(logits))
    target = seqs[:, 1:]
    active = target.ne(PAD)
    actions = target.masked_fill(~active, EOS)
    logs = dist.log_prob(actions).masked_fill(~active, 0.0)
    entropy = dist.entropy().masked_select(active).mean()
    return logs.sum(1), entropy


def reinforce_loss(log_probs, returns):
    if len(returns) > 1:
        baseline = (returns.sum() - returns) / (len(returns) - 1)
    else:
        baseline = torch.zeros_like(returns)
    return -((returns - baseline).detach() * log_probs).mean()
