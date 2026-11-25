import argparse
from pathlib import Path

import torch
from filelock import FileLock
from rdkit import RDLogger

from .cli import checkpoint_for
from .common import device_for, file_hash, read
from .data import Reference
from .evaluate import export_run
from .rl import load_model


def main():
    p = argparse.ArgumentParser(
        description="Generate and score another batch from an existing run"
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--policy",
        choices=["prior", "canonical", "randomized", *[f"R{i}" for i in range(1, 7)]],
        default="prior",
    )
    p.add_argument("--training-seed", type=int)
    p.add_argument("--sample-seed", type=int, default=6001)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if min(a.count, a.batch) < 1:
        p.error("Count and batch must be positive.")
    cfg = read(a.run / "run.json")["settings"]
    cfg.update(temperature=a.temperature, top_p=a.top_p)
    d = device_for(a.device)
    torch.set_num_threads(cfg["cpu_threads"])
    RDLogger.DisableLog("rdApp.error")
    ref = Reference(a.run / "data")
    try:
        seed = cfg["seeds"][0] if a.training_seed is None else a.training_seed
        path = checkpoint_for(a.run, a.policy, seed)
        model = load_model(path, ref.manifest["vocab"], cfg, d)
        a.out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(a.out / ".export.lock"), timeout=0):
            result = export_run(
                model,
                ref.manifest["vocab"],
                ref,
                cfg,
                d,
                a.out,
                a.sample_seed,
                a.count,
                a.batch,
                file_hash(path),
            )
        print(
            f"{result['valid']} valid, {result['unique_valid']} distinct from {result['attempts']} attempts. See {a.out}."
        )
    finally:
        ref.close()


if __name__ == "__main__":
    main()
