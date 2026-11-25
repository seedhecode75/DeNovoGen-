import argparse
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from rdkit import RDLogger

from .common import device_for, environment, file_hash, read, save_json
from .data import Reference, Sequences, prepare
from .evaluate import export_run
from .pretrain import teacher_metrics, train
from .report import make_report
from .rl import choose, load_model, train_arm, valid_rank


def checkpoint_for(folder, policy, seed):
    if policy == "prior":
        policy = read(folder / "selection.json")["selected"]
    if policy in ("canonical", "randomized"):
        return folder / "pretrain" / policy / "best.pt"
    return folder / "rl" / policy / f"seed_{seed}" / "final.pt"


def final_evaluation(folder, cfg, device, ref):
    from .quality import compare_reference

    vocab = ref.manifest["vocab"]
    for p in ["canonical", "randomized", "prior", *[f"R{a}" for a in range(1, 7)]]:
        for seed in cfg["seeds"]:
            path = checkpoint_for(folder, p, seed)
            model = load_model(path, vocab, cfg, device)
            dest = folder / "final" / p / f"seed_{seed}"
            export_run(
                model,
                vocab,
                ref,
                cfg,
                device,
                dest,
                cfg["test_seed"] + seed,
                cfg["eval_samples"],
                cfg["sample_batch"],
                file_hash(path),
            )
            if p in ("canonical", "randomized") and seed == cfg["seeds"][0]:
                save_json(
                    dest.parent / "teacher_test.json",
                    teacher_metrics(
                        model, Sequences(folder / "data", "test"), cfg, device, vocab
                    ),
                )
    compare_reference(folder, cfg, ref)


def benchmark(folder, cfg, device, ref):
    # Select one policy on independent validation, before interpreting test scores.
    ranks = {}
    for a in range(1, 7):
        vals = [
            read(folder / "rl" / f"R{a}" / f"seed_{s}" / "validation.json")[-1][
                "metrics"
            ]
            for s in cfg["seeds"]
        ]
        ranks[f"R{a}"] = np.mean([valid_rank(v) for v in vals], axis=0).tolist()
    p = max(ranks, key=ranks.get)
    seed = cfg["seeds"][0]
    save_json(
        folder / "benchmark_policy.json",
        dict(policy=p, seed=seed, validation_ranks=ranks),
    )
    path = checkpoint_for(folder, p, seed)
    model = load_model(path, ref.manifest["vocab"], cfg, device)
    for n in cfg["workloads"]:
        modes = {"batched": cfg["sample_batch"]}
        if n <= cfg["serial_max"]:
            modes["serial"] = 1
        for name, batch in modes.items():
            export_run(
                model,
                ref.manifest["vocab"],
                ref,
                cfg,
                device,
                folder / "bench" / str(n) / name,
                cfg["benchmark_seed"],
                n,
                batch,
                file_hash(path),
            )


def main():
    parser = argparse.ArgumentParser(description="DeNovoGen")
    steps = ["prepare", "pretrain", "select", "rl", "evaluate", "benchmark", "report"]
    parser.add_argument("stage", choices=["all", *steps])
    parser.add_argument("--config", type=Path, default=Path("configs/small.json"))
    parser.add_argument(
        "--source", type=Path, help="Directory of SMILES-first .smi files or one file"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--arm", type=int, choices=range(1, 7), help="Single arm for the rl stage"
    )
    parser.add_argument(
        "--seed", type=int, help="Single configured seed for the rl stage"
    )
    parser.add_argument(
        "--variant",
        choices=["canonical", "randomized"],
        help="Single variant for the pretrain stage",
    )
    args = parser.parse_args()
    cfg = read(args.config)
    if (
        cfg["split"] not in ("random", "scaffold")
        or cfg["rl_batch"] < 2
        or cfg["pairs"] < 2
        or cfg["crowd_cap"] < 1
    ):
        parser.error("Invalid split, RL batch, diversity pair budget or crowd capacity")
    if args.seed is not None and args.seed not in cfg["seeds"]:
        parser.error("--seed must appear in config.seeds")
    if (args.seed is not None or args.arm is not None) and args.stage != "rl":
        parser.error("--arm and --seed apply to rl only")
    if args.variant and args.stage != "pretrain":
        parser.error("--variant applies to pretrain only")
    if args.stage in ("prepare", "all") and args.source is None:
        parser.error("--source is required for preparation")
    folder = args.run
    folder.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    torch.set_num_threads(cfg["cpu_threads"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    signature = dict(
        settings=cfg,
        code={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    )
    with FileLock(str(folder / ".run.lock"), timeout=0):
        if (folder / "run.json").exists() and read(folder / "run.json") != signature:
            raise ValueError(
                "Settings or implementation changed. Select a new run directory."
            )
        save_json(folder / "run.json", signature)
        save_json(folder / "environment.json", environment(device))
        for stage in steps if args.stage == "all" else [args.stage]:
            print(f"[{stage}] {folder}", flush=True)
            if stage == "prepare":
                prepare(args.source, folder / "data", cfg)
            elif stage == "pretrain":
                for v in (
                    [args.variant] if args.variant else ["canonical", "randomized"]
                ):
                    train(folder, v, cfg, device)
            elif stage == "report":
                make_report(folder, cfg)
            else:
                ref = Reference(folder / "data")
                try:
                    if stage == "select":
                        choose(folder, cfg, device, ref)
                    elif stage == "rl":
                        for a in [args.arm] if args.arm else range(1, 7):
                            for s in (
                                [args.seed] if args.seed is not None else cfg["seeds"]
                            ):
                                train_arm(folder, cfg, device, ref, a, s)
                    elif stage == "evaluate":
                        final_evaluation(folder, cfg, device, ref)
                    elif stage == "benchmark":
                        benchmark(folder, cfg, device, ref)
                finally:
                    ref.close()


if __name__ == "__main__":
    main()
