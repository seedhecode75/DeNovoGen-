import math
import time
from collections import defaultdict

import torch
from torch.nn import functional as F

from .chem import PAD, UNK, category
from .common import (
    checkpoint,
    environment,
    read,
    restore,
    rng_state,
    save_json,
    seed_all,
    set_rng,
)
from .data import Sequences, batches
from .model import Generator


@torch.inference_mode()
def teacher_metrics(model, dataset, cfg, device, vocab):
    model.eval()
    supports = torch.zeros(len(vocab), dtype=torch.long, device=device)
    correct = torch.zeros_like(supports)
    loss = top = all_correct = positions = 0
    for seq in batches(dataset, cfg, 0, False):
        seq = seq.to(device)
        out, _ = model(seq[:, :-1])
        y = seq[:, 1:]
        mask = y.ne(PAD)
        loss += F.cross_entropy(
            out.transpose(1, 2), y, ignore_index=PAD, reduction="sum"
        ).item()
        pred = out.argmax(-1)
        supports += torch.bincount(y[mask], minlength=len(vocab))
        hit = mask & pred.eq(y)
        correct += torch.bincount(y[hit], minlength=len(vocab))
        top += int(
            (
                (out.topk(min(5, len(vocab)), dim=-1).indices == y[..., None]).any(-1)
                & mask
            ).sum()
        )
        all_correct += int(pred.eq(y).sum())
        positions += y.numel()
    sums = defaultdict(lambda: [0, 0])
    for t, n, k in zip(vocab, supports.tolist(), correct.tolist()):
        vals = sums[category(t)]
        vals[0] += k
        vals[1] += n
    n = int(supports.sum())
    return dict(
        nll=loss / n,
        ppl=math.exp(min(700, loss / n)),
        accuracy=int(correct.sum()) / n,
        top5=top / n,
        unmasked_accuracy=all_correct / positions,
        targets=n,
        oov_fraction=int(supports[UNK]) / n,
        by_category={k: dict(correct=a, total=b) for k, (a, b) in sums.items()},
    )


def train(folder, variant, cfg, device):
    data = folder / "data"
    manifest = read(data / "manifest.json")
    vocab = manifest["vocab"]
    dest = folder / "pretrain" / variant
    dest.mkdir(parents=True, exist_ok=True)
    last = dest / "last.pt"
    seed_all(cfg["pretrain_seed"])
    model = Generator(len(vocab), cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["pretrain_lr"], weight_decay=cfg["weight_decay"]
    )
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    canonical = Sequences(data, "train")
    alternate = Sequences(data, "train", variant)
    trainset = canonical
    valid = Sequences(data, "val")
    history = []
    if last.exists():
        s = restore(last)
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        scaler.load_state_dict(s["scaler"])
        set_rng(s["rng"])
        history = s["history"]
        # A crash between last and best writes is recoverable.
        if history[-1]["val"]["nll"] == min(r["val"]["nll"] for r in history) and (
            not (dest / "best.pt").exists()
            or restore(dest / "best.pt")["epoch"] != len(history)
        ):
            checkpoint(dest / "best.pt", dict(model=s["model"], epoch=len(history)))
    for epoch in range(len(history) + 1, cfg["epochs"] + 1):
        start = time.perf_counter()
        trainset = alternate if epoch % 2 else canonical
        model.train()
        total_loss = 0.0
        tokens = 0
        optimizer.zero_grad(set_to_none=True)
        accum_tokens = accum_batches = 0

        def step(n):
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for seq in batches(trainset, cfg, cfg["pretrain_seed"] + epoch, True):
            seq = seq.to(device)
            y = seq[:, 1:]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(seq[:, :-1])
                loss = F.cross_entropy(
                    logits.transpose(1, 2), y, ignore_index=PAD, reduction="sum"
                )
            scaler.scale(loss).backward()
            n = int(y.ne(PAD).sum())
            total_loss += loss.item()
            tokens += n
            accum_tokens += n
            accum_batches += 1
            if accum_batches == cfg["accum_batches"]:
                step(accum_tokens)
                accum_batches = accum_tokens = 0
        if accum_tokens:
            step(accum_tokens)
        metrics = teacher_metrics(model, valid, cfg, device, vocab)
        history.append(
            dict(
                epoch=epoch,
                view=variant if epoch % 2 else "canonical",
                train_nll=total_loss / tokens,
                targets=tokens,
                seen_unique=len(trainset),
                elapsed_s=time.perf_counter() - start,
                val=metrics,
            )
        )
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            rng=rng_state(),
            history=history,
        )
        checkpoint(last, state)
        if metrics["nll"] <= min(r["val"]["nll"] for r in history):
            checkpoint(dest / "best.pt", dict(model=model.state_dict(), epoch=epoch))
        save_json(dest / "history.json", history)
        print(
            f"{variant} epoch {epoch}/{cfg['epochs']}: val NLL {metrics['nll']:.4f}, masked accuracy {metrics['accuracy']:.3f}",
            flush=True,
        )
    save_json(dest / "history.json", history)
    save_json(
        dest / "training_scope.json",
        dict(
            unique_seen=len(trainset),
            target_exposures=sum(r["targets"] for r in history),
            env=environment(device),
            data_id=manifest["request_id"],
        ),
    )
