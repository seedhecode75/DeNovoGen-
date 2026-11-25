import csv
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .common import read

METRICS = (
    "validity",
    "uniqueness",
    "novelty_train",
    "novelty_source",
    "novel_unique_yield",
    "mean_qed",
    "diversity",
    "fragment_rate",
    "qualified_scaffolds_per_1k",
    "cap_rate",
)


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def interval(values):
    if not values:
        return (None,) * 4
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, None, None, None
    sd = float(np.std(values, ddof=1))
    width = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return mean, sd, mean - width, mean + width


def plots(folder, out, groups):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for variant in ["canonical", "randomized"]:
        hist = read(folder / "pretrain" / variant / "history.json")
        for ax, key, label in [
            (axes[0, 0], "nll", "Validation NLL"),
            (axes[0, 1], "accuracy", "Masked validation accuracy"),
        ]:
            ax.plot(
                [r["epoch"] for r in hist],
                [r["val"][key] for r in hist],
                marker=".",
                label=variant,
            )
            ax.set(xlabel="Epoch", ylabel=label)
    arms = [f"R{a}" for a in range(1, 7)]
    for ax, key in [(axes[1, 0], "validity"), (axes[1, 1], "mean_qed")]:
        for i, k in enumerate(["prior", *arms]):
            vals = [r[key] for r in groups[k].values() if r[key] is not None]
            if vals:
                ax.scatter([i] * len(vals), vals, s=16)
        ax.set_xticks(range(7), ["prior", *arms])
        ax.set(ylabel=key)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    signals = ["reward_mean", "reward_sd", "entropy_per_token", "sequence_kl_estimate"]
    for arm in arms:
        paths = sorted((folder / "rl" / arm).glob("seed_*/history.json"))
        histories = [read(p) for p in paths]
        for ax, key in zip(axes.flat, signals):
            avg = np.mean([[r[key] for r in h] for h in histories], axis=0)
            ax.plot(range(1, len(avg) + 1), avg, alpha=0.2)
            window = min(10, len(avg))
            ax.plot(
                range(window, len(avg) + 1),
                np.convolve(avg, np.ones(window) / window, "valid"),
                label=arm,
            )
            ax.set_title(key)
        for ax, key in zip(
            list(axes.flat)[4:],
            ["validity", "novelty_train", "mean_qed", "diversity", "fragment_rate"],
        ):
            vals = [read(p.parent / "validation.json") for p in paths]
            for i, v in enumerate(vals):
                ax.plot(
                    [r["update"] for r in v],
                    [r["metrics"][key] for r in v],
                    label=arm if i == 0 else None,
                    alpha=0.65,
                )
            ax.set_title("Independent validation: " + key, fontsize=9)
    axes[0, 0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rl_diagnostics.png", dpi=145)
    plt.close(fig)


def make_report(folder, cfg):
    from rdkit import Chem
    from rdkit.Chem import Draw

    out = folder / "report"
    out.mkdir(exist_ok=True)
    groups = defaultdict(dict)
    rows = []
    for path in sorted((folder / "final").glob("*/seed_*/metrics.json")):
        policy = path.parent.parent.name
        seed = int(path.parent.name.split("_")[1])
        result = read(path)
        groups[policy][seed] = result
        rows.append(dict(policy=policy, seed=seed, **{k: result[k] for k in METRICS}))
    policies = ["canonical", "randomized", "prior", *[f"R{a}" for a in range(1, 7)]]
    if any(set(groups[p]) != set(cfg["seeds"]) for p in policies):
        raise ValueError(
            "Complete final evaluation for every policy and seed before reporting."
        )
    csv_file(out / "per_seed.csv", rows)
    stats = []
    for p in policies:
        for key in METRICS:
            valid = [r[key] for r in groups[p].values() if r[key] is not None]
            mean, sd, lo, hi = interval(valid)
            paired = [
                (r[key], groups["prior"][s][key])
                for s, r in groups[p].items()
                if r[key] is not None and groups["prior"][s][key] is not None
            ]
            diff, paired_sd, dl, dh = interval([x - y for x, y in paired])
            base = float(np.mean([y for _, y in paired])) if paired else None
            pct = key in (
                "validity",
                "uniqueness",
                "novelty_train",
                "novelty_source",
                "novel_unique_yield",
                "fragment_rate",
                "cap_rate",
            )
            stats.append(
                dict(
                    policy=p,
                    metric=key,
                    n=len(valid),
                    mean=mean,
                    sd=sd,
                    ci95_low=lo,
                    ci95_high=hi,
                    paired_n=len(paired),
                    difference_from_prior=diff,
                    paired_sd=paired_sd,
                    paired_ci95_low=dl,
                    paired_ci95_high=dh,
                    percentage_point_change=100 * diff
                    if diff is not None and pct
                    else None,
                    relative_change_pct=100 * diff / base if base else None,
                )
            )
    csv_file(out / "comparisons.csv", stats)
    scaling = []
    for path in sorted((folder / "bench").glob("*/*/metrics.json")):
        m = read(path)
        scaling.append(
            dict(workload=m["attempts"], mode=path.parent.name, **m["benchmark"])
        )
    csv_file(out / "scaling.csv", scaling)
    plots(folder, out, groups)
    chosen = (
        read(folder / "benchmark_policy.json")["policy"]
        if (folder / "benchmark_policy.json").exists()
        else "prior"
    )
    imgs = []
    seen = set()
    for path in sorted(
        (folder / "final" / chosen / f"seed_{cfg['seeds'][0]}").glob("chunk_*.csv")
    ):
        with path.open() as f:
            for row in csv.DictReader(f):
                k = row["identity"]
                if k and k not in seen:
                    imgs.append(Chem.MolFromSmiles(k))
                    seen.add(k)
                if len(imgs) == 12:
                    break
        if len(imgs) == 12:
            break
    if imgs:
        Draw.MolsToGridImage(imgs, molsPerRow=4, subImgSize=(230, 180)).save(
            out / "first_valid_structures.png"
        )
    selected = read(folder / "selection.json")["selected"]
    lines = [
        "# DeNovoGen experiment results",
        "",
        f"Pretrained variant chosen on validation: {selected}.",
        f"Final samples per policy and seed: {cfg['eval_samples']}. Training seeds: {cfg['seeds']}.",
        "",
        "| Policy | Validity | Canonical uniqueness | Train novelty | Source novelty | Mean QED | Diversity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in policies:
        vals = []
        for k in (
            "validity",
            "uniqueness",
            "novelty_train",
            "novelty_source",
            "mean_qed",
            "diversity",
        ):
            values = [r[k] for r in groups[p].values() if r[k] is not None]
            vals.append(f"{np.mean(values):.4f}" if values else "unavailable")
        lines.append("| " + " | ".join([p, *vals]) + " |")
    lines += [
        "",
        "Rates are fractions. Each mean uses the defined seed-level values. Check comparisons.csv for denominators, intervals and paired differences.",
        "The prior repeats the selected pretrained variant under common evaluation seeds. RL seeds share one pretrained checkpoint.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
