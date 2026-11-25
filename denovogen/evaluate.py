import csv
import threading
import time

import numpy as np
import psutil
import torch

from .chem import diversity, inspect, query
from .common import digest, environment, read, save_json
from .model import decode, sample


def summarize(rows, seed, cfg):
    n = len(rows)
    valid = [r for r in rows if r["valid"]]
    unique = {r["identity"]: r for r in valid}
    u = len(unique)

    def fraction(a, b):
        return a / b if b else None

    q = [r["qed"] for r in valid if r["qed"] is not None]
    novel = sum(r["train_novel"] for r in unique.values())
    full = bool(u) and all(r["source_novel"] is not None for r in unique.values())
    source_n = sum(r["source_novel"] for r in unique.values()) if full else None
    qualified = {
        r["scaffold"]
        for r in unique.values()
        if r["qed"] is not None and r["qed"] >= cfg["qed_floor"] and r["scaffold"]
    }
    div, pairs, se = diversity([r["identity"] for r in valid], seed, cfg["pairs"])
    return dict(
        attempts=n,
        valid=len(valid),
        unique_valid=u,
        novel_unique_train=novel,
        novel_unique_source=source_n,
        validity=fraction(len(valid), n),
        uniqueness=fraction(u, len(valid)),
        raw_uniqueness=fraction(len({r["smiles"] for r in valid}), len(valid)),
        novelty_train=fraction(novel, u),
        novelty_source=fraction(source_n, u) if full else None,
        novel_unique_yield=fraction(novel, n),
        mean_qed=float(np.mean(q)) if q else None,
        qed_failures=len(valid) - len(q),
        qed_quantiles=np.quantile(q, [0.1, 0.5, 0.9]).tolist() if q else [],
        fragment_rate=fraction(sum(r["fragment"] for r in valid), len(valid)),
        fragment_yield=fraction(sum(r["fragment"] for r in valid), n),
        cap_rate=fraction(sum(not r["eos"] for r in rows), n),
        diversity=div,
        diversity_pairs=pairs,
        diversity_mc_se=se,
        scaffolds=len({r["scaffold"] for r in valid if r["scaffold"]}),
        acyclic_fraction=fraction(sum(not r["scaffold"] for r in valid), len(valid)),
        qualified_scaffolds_per_1k=fraction(1000 * len(qualified), n),
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def score_batch(
    model, vocab, reference, cfg, device, gen, n, goal, *, extended=False, full=False
):
    seq = sample(
        model, n, cfg["max_steps"], device, gen, cfg["temperature"], cfg["top_p"]
    )
    rows = []
    for smi, eos in decode(seq.cpu(), vocab):
        row = inspect(smi, reference, goal, extended=extended, full=full)
        row["eos"] = eos
        rows.append(row)
    return rows


def quick_eval(model, vocab, reference, cfg, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    goal = query(cfg["fragment"])
    rows = []
    while len(rows) < cfg["val_samples"]:
        count = min(cfg["sample_batch"], cfg["val_samples"] - len(rows))
        rows.extend(score_batch(model, vocab, reference, cfg, device, gen, count, goal))
    return summarize(rows, seed, cfg)


def export_run(model, vocab, reference, cfg, device, dest, seed, n, batch, model_id):
    dest.mkdir(parents=True, exist_ok=True)
    signature = dict(
        model=model_id,
        data=reference.manifest["request_id"],
        cfg=digest(cfg),
        attempts=n,
        batch=batch,
        seed=seed,
        env=environment(device),
    )
    if (dest / "request.json").exists() and read(dest / "request.json") != signature:
        raise ValueError(f"Incompatible export at {dest}. Use a new directory.")
    if (dest / "metrics.json").exists():
        return read(dest / "metrics.json")
    save_json(dest / "request.json", signature)
    model.eval()
    goal = query(cfg["fragment"])
    warm = torch.Generator(device=device).manual_seed(seed)
    sample(model, min(n, batch), min(8, cfg["max_steps"]), device, warm)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    proc = psutil.Process()

    def rss():
        return proc.memory_info().rss

    rows = []
    index_fields = (
        "identity",
        "smiles",
        "valid",
        "eos",
        "train_novel",
        "source_novel",
        "source_id",
        "scaffold",
        "fragment",
        "qed",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable",
        "fsp3",
        "ro5_violations",
    )
    chunk_size = cfg["chunk_size"]
    timers = []
    # Each atomic chunk owns its output IDs and a deterministic RNG stream.
    for start in range(0, n, chunk_size):
        path = dest / f"chunk_{start:09d}.json"
        if path.exists():
            saved = read(path)
            rows.extend(saved["rows"])
            timers.append(saved["timing"])
            continue
        chunk_start = time.perf_counter()
        current = []
        gen = torch.Generator(device=device).manual_seed(seed + start)
        stats = dict(
            generation_s=0.0, chemistry_s=0.0, batch_latency_s=[], rss_peak=rss()
        )
        stop = threading.Event()

        def watch():
            while not stop.wait(0.05):
                stats["rss_peak"] = max(stats["rss_peak"], rss())

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            while len(current) < min(chunk_size, n - start):
                size = min(batch, min(chunk_size, n - start) - len(current))
                tick = time.perf_counter()
                seq = sample(
                    model,
                    size,
                    cfg["max_steps"],
                    device,
                    gen,
                    cfg["temperature"],
                    cfg["top_p"],
                )
                decoded = decode(seq.cpu(), vocab)
                sync(device)
                delta = time.perf_counter() - tick
                stats["generation_s"] += delta
                stats["batch_latency_s"].append(delta)
                tick = time.perf_counter()
                for smi, eos in decoded:
                    row = inspect(smi, reference, goal, extended=True, full=True)
                    row.update(eos=eos, attempt_id=start + len(current))
                    current.append(row)
                stats["chemistry_s"] += time.perf_counter() - tick
        finally:
            stop.set()
            thread.join()
        stats["rss_peak"] = max(stats["rss_peak"], rss())
        stats["gpu_peak"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        # Time a streamed rows write as part of the end-to-end export.
        csv_path = path.with_suffix(".csv")
        tmp = csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["attempt_id", *index_fields])
            writer.writeheader()
            writer.writerows(current)
        tmp.replace(csv_path)
        stats["active_s"] = time.perf_counter() - chunk_start
        tick = time.perf_counter()
        save_json(path, dict(rows=current, timing=stats))
        # Retain measured metadata-write overhead for resumed benchmarks.
        commit_s = time.perf_counter() - tick
        save_json(path.with_suffix(".time"), commit_s)
        rows.extend(current)
        timers.append(stats)
    finish = time.perf_counter()
    metrics = summarize(rows, seed, cfg)
    fields = [
        "qed",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "fsp3",
        "rotatable",
        "ro5_violations",
    ]
    metrics["properties"] = {}
    for k in fields:
        values = [r[k] for r in rows if r.get(k) is not None]
        if values:
            metrics["properties"][k] = dict(
                mean=float(np.mean(values)),
                p10_p50_p90=np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
            )
    latencies = [v for t in timers for v in t["batch_latency_s"]]
    overhead = sum(read(p) for p in dest.glob("chunk_*.time"))
    wall = (
        sum(t["active_s"] for t in timers) + overhead + (time.perf_counter() - finish)
    )
    metrics["benchmark"] = dict(
        active_end_to_end_s=wall,
        attempts_per_s=n / wall,
        valid_unique_per_s=metrics["unique_valid"] / wall,
        generation_s=sum(t["generation_s"] for t in timers),
        chemistry_s=sum(t["chemistry_s"] for t in timers),
        batch_latency_p50_p95_s=np.quantile(latencies, [0.5, 0.95]).tolist(),
        gpu_peak_allocated_bytes=max(t["gpu_peak"] for t in timers),
        peak_generation_rss_bytes=max(t["rss_peak"] for t in timers),
        rss_after_aggregation_bytes=rss(),
        batch=batch,
        generation_attempts_per_s=n / max(1e-9, sum(t["generation_s"] for t in timers)),
    )
    save_json(dest / "metrics.json", metrics)
    return metrics
