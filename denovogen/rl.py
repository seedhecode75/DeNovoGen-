from collections import Counter, deque

import numpy as np
import torch

from .chem import inspect, query, score
from .common import (
    checkpoint,
    file_hash,
    read,
    restore,
    rng_state,
    save_json,
    seed_all,
    set_rng,
)
from .evaluate import quick_eval
from .model import Generator, decode, reinforce_loss, sample, trajectory


def load_model(path, vocab, cfg, device):
    model = Generator(len(vocab), cfg).to(device)
    model.load_state_dict(restore(path)["model"])
    return model.eval()


def valid_rank(m):
    # Common selection rule. Only independent validation draws enter ranking.
    return (
        m["validity"] or 0.0,
        m["qualified_scaffolds_per_1k"] or 0.0,
        m["novel_unique_yield"] or 0.0,
    )


def choose(folder, cfg, device, ref):
    vocab = ref.manifest["vocab"]
    results = {}
    for variant in ("canonical", "randomized"):
        root = folder / "pretrain" / variant
        model = load_model(root / "best.pt", vocab, cfg, device)
        ev = quick_eval(model, vocab, ref, cfg, device, cfg["validation_seed"])
        best_nll = min(v["val"]["nll"] for v in read(root / "history.json"))
        results[variant] = dict(
            generation=ev, val_nll=best_nll, sha256=file_hash(root / "best.pt")
        )
    name = max(
        results,
        key=lambda v: (valid_rank(results[v]["generation"]), -results[v]["val_nll"]),
    )
    save_json(folder / "selection.json", dict(selected=name, variants=results))
    return name


def train_arm(folder, cfg, device, ref, arm, seed):
    vocab = ref.manifest["vocab"]
    choice = read(folder / "selection.json")
    prior_path = folder / "pretrain" / choice["selected"] / "best.pt"
    prior_id = file_hash(prior_path)
    if prior_id != choice["variants"][choice["selected"]]["sha256"]:
        raise ValueError(
            "Selected prior fingerprint changed. Re-run selection in a new workspace."
        )
    dest = folder / "rl" / f"R{arm}" / f"seed_{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    prior = load_model(prior_path, vocab, cfg, device).requires_grad_(False)
    policy = load_model(prior_path, vocab, cfg, device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["rl_lr"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["updates"], eta_min=cfg["rl_lr"] * 0.2
    )
    history = []
    evaluations = []
    history_scaffolds = deque(maxlen=cfg["memory_size"])
    last = dest / "last.pt"
    goal = query(cfg["fragment"])
    if last.exists():
        s = restore(last)
        if s["prior_sha256"] != prior_id:
            raise ValueError("RL restart would use a different prior")
        policy.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        schedule.load_state_dict(s["scheduler"])
        set_rng(s["rng"])
        if s["generator_device"] != str(device):
            raise ValueError("Resume this RL job on the original device type")
        gen.set_state(s["gen"])
        history, evaluations = s["history"], s["evaluations"]
        history_scaffolds.extend(s["memory"])
    for update in range(len(history) + 1, cfg["updates"] + 1):
        sampled = sample(policy, cfg["rl_batch"], cfg["max_steps"], device, gen)
        # Inference tensors must be cloned before use in a differentiable replay.
        seq = sampled.clone()
        decoded = decode(seq.cpu(), vocab)
        rows = [inspect(sm, ref, goal) for sm, _ in decoded]
        counts = Counter(history_scaffolds) if arm == 6 else Counter()
        rewards = torch.tensor(
            [score(row, arm, counts, cfg) for row in rows], device=device
        )
        logp, entropy = trajectory(policy, seq)
        with torch.no_grad():
            logprior, _ = trajectory(prior, seq)
            kl = logp.detach() - logprior
            returns = rewards - cfg["kl_beta"] * kl
        loss = reinforce_loss(logp, returns)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        opt.step()
        schedule.step()
        if arm == 6:
            history_scaffolds.extend(r["scaffold"] for r in rows if r["valid"])
        valid = [r for r in rows if r["valid"]]
        qs = [r["qed"] for r in valid if r["qed"] is not None]
        history.append(
            dict(
                update=update,
                reward_mean=rewards.mean().item(),
                reward_sd=rewards.std(unbiased=False).item(),
                loss=loss.item(),
                validity=len(valid) / len(rows),
                train_novelty_among_valid=float(
                    np.mean([r["train_novel"] for r in valid])
                )
                if valid
                else None,
                fragment_among_valid=float(np.mean([r["fragment"] for r in valid]))
                if valid
                else None,
                mean_qed=float(np.mean(qs)) if qs else None,
                entropy_per_token=entropy.item(),
                sequence_kl_estimate=kl.mean().item(),
                gradient_norm=float(norm),
                lr=opt.param_groups[0]["lr"],
                cap_rate=sum(not eos for _, eos in decoded) / len(rows),
            )
        )
        if update % cfg["eval_every"] == 0 or update == cfg["updates"]:
            ev = quick_eval(
                policy, vocab, ref, cfg, device, cfg["validation_seed"] + update
            )
            evaluations.append(dict(update=update, metrics=ev))
            state = dict(
                model=policy.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=schedule.state_dict(),
                rng=rng_state(),
                prior_sha256=prior_id,
                history=history,
                evaluations=evaluations,
                memory=list(history_scaffolds),
                gen=gen.get_state(),
                generator_device=str(device),
            )
            checkpoint(last, state)
            save_json(dest / "history.json", history)
            save_json(dest / "validation.json", evaluations)
            print(
                f"R{arm} seed {seed} update {update}: independent validation validity {ev['validity']:.3f}",
                flush=True,
            )
    save_json(dest / "history.json", history)
    save_json(dest / "validation.json", evaluations)
    final = dest / "final.pt"
    if not final.exists():
        checkpoint(
            final,
            dict(
                model=policy.state_dict(),
                prior_sha256=prior_id,
                arm=arm,
                seed=seed,
                updates=len(history),
            ),
        )
