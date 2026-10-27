#!/usr/bin/env python
"""Phase-1 benchmark: build HNSW, measure recall/latency/memory, compare hnswlib.

Examples
--------
    python scripts/run_phase1.py --dataset siftsmall
    python scripts/run_phase1.py --dataset sift100k --M 16 --ef-construction 100 \\
        --ef 16 32 64 128 --k 10 --compare-hnswlib
    python scripts/run_phase1.py --config configs/experiments/e1_sift100k.yaml

The script writes ``result.json``, ``summary.csv`` and plots under ``--out``.
It never treats the shipped SIFT ground truth as valid for a truncated base:
when the base is a prefix or the live set changed, exact ground truth is
computed (and cached) for that snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import yaml  # noqa: E402

from churnvec import HNSWIndex  # noqa: E402
from churnvec.bench.datasets import (  # noqa: E402
    load_dataset,
    load_or_compute_groundtruth,
)
from churnvec.bench.metrics import (  # noqa: E402
    RSSMonitor,
    latency_stats,
    recall_at_k,
    write_json,
)

DEFAULT_EFS = [16, 32, 64, 128]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML experiment config")
    parser.add_argument(
        "--dataset", default="siftsmall", choices=["siftsmall", "sift10k", "sift100k", "sift1m"]
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef", type=int, nargs="+", default=DEFAULT_EFS)
    parser.add_argument("--gate-ef", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--queries", type=int, default=None, help="limit query count")
    parser.add_argument("--base-limit", type=int, default=None, help="prefix length (recomputes GT)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=["heuristic", "simple"], default="heuristic")
    parser.add_argument(
        "--keep-pruned",
        dest="keep_pruned",
        action="store_true",
        help="fill rejected candidates (paper option; fragments some SIFT builds)",
    )
    parser.set_defaults(keep_pruned=False)
    parser.add_argument("--metric", choices=["l2", "cosine", "ip"], default="l2")
    parser.add_argument("--compare-hnswlib", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.set_defaults(plots=True)
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--gt-cache", type=Path, default=None)
    parser.add_argument(
        "--progress-every", type=int, default=10_000, help="build progress interval; 0 disables"
    )
    parser.add_argument(
        "--parity-tolerance-pp",
        type=float,
        default=2.0,
        help="allowed recall gap versus hnswlib, in percentage points",
    )
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        config = yaml.safe_load(known.config.read_text()) or {}
        if not isinstance(config, dict):
            raise ValueError(f"config must be a mapping: {known.config}")
        parser.set_defaults(**config)
    return parser.parse_args(argv)


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    short = run("rev-parse", "--short", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": short,
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
    }


def environment(args: argparse.Namespace) -> dict:
    try:
        from importlib.metadata import version

        hnswlib_version = version("hnswlib")
    except Exception:
        hnswlib_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hnswlib": hnswlib_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "git": git_info(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def resolve_groundtruth(dataset, args: argparse.Namespace, nq: int) -> tuple[np.ndarray, dict]:
    """Use shipped GT only when the benchmark base is exactly the shipped base."""
    k = max(int(args.k), int(args.gt_k))
    shipped = dataset.groundtruth
    shipped_valid = (
        shipped is not None
        and dataset.name in {"siftsmall", "sift1m"}
        and dataset.n_base == (10_000 if dataset.name == "siftsmall" else 1_000_000)
        and args.base_limit is None
    )
    if shipped_valid:
        gt = np.asarray(shipped, dtype=np.int64)[:nq, :k]
        return gt, {"source": "shipped", "path": None, "shape": list(gt.shape)}

    if args.gt_cache is not None:
        cache_path = args.gt_cache
    else:
        cache_path = (
            ROOT
            / "data"
            / "processed"
            / f"gt_{args.dataset}_{dataset.n_base}_q{nq}_k{k}.npy"
        )
    gt = load_or_compute_groundtruth(
        dataset.base, dataset.queries[:nq], cache_path, k=k, metric=args.metric
    )
    return np.asarray(gt, dtype=np.int64), {
        "source": "computed",
        "path": str(cache_path),
        "shape": list(gt.shape),
    }


def quantiles(samples_ns: np.ndarray) -> dict:
    return latency_stats(samples_ns)


def evaluate_custom(
    index: HNSWIndex,
    queries: np.ndarray,
    gt: np.ndarray,
    efs: list[int],
    k: int,
    warmup: int = 32,
) -> list[dict]:
    nq = queries.shape[0]
    k_eff = min(k, index.size)
    rows: list[dict] = []
    for ef in efs:
        ef = max(int(ef), k_eff)
        for query in queries[: min(warmup, nq)]:
            index.search_ids(query, k=k_eff, ef=ef)
        samples = np.empty(nq, dtype=np.int64)
        returned = np.empty((nq, k_eff), dtype=np.int64)
        for i, query in enumerate(queries):
            start = time.perf_counter_ns()
            ids, _ = index.search_ids(query, k=k_eff, ef=ef)
            samples[i] = time.perf_counter_ns() - start
            returned[i] = ids
        stats = quantiles(samples)
        recall = recall_at_k(returned, gt[:, :k_eff], k=k_eff)
        rows.append(
            {
                "engine": "churnvec",
                "ef": int(ef),
                "recall": recall,
                "k": int(k_eff),
                **stats,
            }
        )
        print(
            f"  churnvec ef={ef:>4} recall@{k_eff}={recall:.4f} "
            f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
            f"qps={stats['qps']:.1f}"
        )
    return rows


def evaluate_hnswlib(
    base: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    efs: list[int],
    k: int,
    M: int,
    ef_construction: int,
    seed: int,
    out_dir: Path,
    metric: str = "l2",
    warmup: int = 32,
) -> tuple[list[dict], dict]:
    import hnswlib

    n = base.shape[0]
    k_eff = min(k, n)
    with RSSMonitor() as monitor:
        index = hnswlib.Index(space=metric, dim=base.shape[1])
        index.init_index(
            max_elements=n,
            ef_construction=ef_construction,
            M=M,
            random_seed=seed,
            allow_replace_deleted=False,
        )
        start = time.perf_counter()
        index.add_items(base, np.arange(n, dtype=np.int64), num_threads=1)
        build_seconds = time.perf_counter() - start
    index.set_num_threads(1)
    print(
        f"  hnswlib built in {build_seconds:.2f}s "
        f"(peak RSS {monitor.peak_mb:.0f} MB)"
    )

    index_path = out_dir / "hnswlib_index.bin"
    index.save_index(str(index_path))
    index_bytes = index_path.stat().st_size

    nq = queries.shape[0]
    rows: list[dict] = []
    for ef in efs:
        ef = max(int(ef), k_eff)
        index.set_ef(ef)
        for query in queries[: min(warmup, nq)]:
            index.knn_query(query.reshape(1, -1), k=k_eff, num_threads=1)
        samples = np.empty(nq, dtype=np.int64)
        returned = np.empty((nq, k_eff), dtype=np.int64)
        for i, query in enumerate(queries):
            start_ns = time.perf_counter_ns()
            labels, _ = index.knn_query(query.reshape(1, -1), k=k_eff, num_threads=1)
            samples[i] = time.perf_counter_ns() - start_ns
            returned[i] = labels[0].astype(np.int64)
        stats = quantiles(samples)
        recall = recall_at_k(returned, gt[:, :k_eff], k=k_eff)
        rows.append(
            {
                "engine": "hnswlib",
                "ef": int(ef),
                "recall": recall,
                "k": int(k_eff),
                **stats,
            }
        )
        print(
            f"  hnswlib  ef={ef:>4} recall@{k_eff}={recall:.4f} "
            f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
            f"qps={stats['qps']:.1f}"
        )
    return rows, {
        "build_seconds": build_seconds,
        "rss_start_mb": monitor.start_mb,
        "rss_peak_mb": monitor.peak_mb,
        "rss_delta_mb": monitor.peak_mb - monitor.start_mb,
        "index_file_bytes": index_bytes,
        "index_file_mb": index_bytes / (1024 * 1024),
    }


def make_plots(rows: list[dict], out_dir: Path, dataset: str, k: int) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  plotting skipped: {exc}")
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    engines = sorted({row["engine"] for row in rows})
    for metric, ylabel, name in [
        ("recall", f"Recall@{k}", "recall_vs_ef"),
        ("p50_ms", "p50 latency (ms)", "latency_vs_ef"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for engine in engines:
            subset = sorted(
                (row for row in rows if row["engine"] == engine), key=lambda r: r["ef"]
            )
            if not subset:
                continue
            ax.plot(
                [row["ef"] for row in subset],
                [row[metric] for row in subset],
                marker="o",
                label=engine,
            )
        ax.set_xlabel("efSearch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset} ({k}-NN)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def projection(custom_build_seconds: float, n_base: int, p50_ms: float) -> dict:
    inserts_per_second = n_base / max(custom_build_seconds, 1e-9)
    projected_hours = 1_000_000 / max(inserts_per_second, 1e-9) / 3600.0
    triggers = []
    if projected_hours > 3.0:
        triggers.append("projected_1m_build_over_3h")
    if p50_ms > 10.0:
        triggers.append("p50_over_10ms")
    return {
        "inserts_per_second": inserts_per_second,
        "projected_1m_build_hours": projected_hours,
        "measured_p50_ms": p50_ms,
        "native_core_triggered": bool(triggers),
        "triggers": triggers,
        "note": (
            "1M values are linear projections from this run; query latency at 1M "
            "is likely higher because the graph has more layers/nodes."
        ),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "engine",
        "ef",
        "k",
        "recall",
        "count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "qps",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_name = "siftsmall" if args.dataset == "sift10k" else args.dataset
    out_dir = args.out or (
        ROOT / "results" / "phase1" / f"{dataset_name}_M{args.M}_efC{args.ef_construction}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase1] dataset={dataset_name} out={out_dir}")
    print(
        f"[phase1] M={args.M} efConstruction={args.ef_construction} "
        f"efSearch={args.ef} k={args.k} metric={args.metric}"
    )

    dataset = load_dataset(
        dataset_name,
        root=args.data_root,
        base_limit=args.base_limit,
        query_limit=args.queries,
        include_learn=False,
    )
    nq = dataset.n_queries
    print(
        f"[phase1] loaded base={dataset.n_base:,}x{dataset.dim} "
        f"queries={nq:,} base_mb={dataset.base.nbytes / 1024**2:.1f}"
    )
    gt, gt_info = resolve_groundtruth(dataset, args, nq)
    print(f"[phase1] ground truth: {gt_info['source']} shape={gt_info['shape']}")

    efs = sorted({max(int(ef), args.k) for ef in args.ef} | {max(args.gate_ef, args.k)})
    print("[phase1] building churnvec HNSW...")
    with RSSMonitor() as monitor:
        # Construct inside the monitor: capacity is preallocated, so this is
        # part of the index's memory footprint.
        index = HNSWIndex(
            dim=dataset.dim,
            M=args.M,
            ef_construction=args.ef_construction,
            metric=args.metric,
            seed=args.seed,
            capacity=max(dataset.n_base, 1),
            selection=args.selection,
            keep_pruned=args.keep_pruned,
        )
        build_start = time.perf_counter()
        index.add_items(
            dataset.base,
            dataset.base_ids,
            verbose=True,
            progress_every=args.progress_every,
        )
        build_seconds = time.perf_counter() - build_start
    print(
        f"[phase1] built in {build_seconds:.2f}s "
        f"(peak RSS {monitor.peak_mb:.0f} MB, start {monitor.start_mb:.0f} MB)"
    )
    build_stats = index.stats()
    print(
        f"[phase1] graph: max_level={build_stats['max_level']} "
        f"unreachable={build_stats['unreachable']} "
        f"memory={build_stats['memory_bytes'] / 1024**2:.1f} MB "
        f"repair_links={build_stats['repair_links']}"
    )
    if build_stats["unreachable"]:
        raise SystemExit("graph is not connected after build; refusing to benchmark")

    custom_rows = evaluate_custom(index, dataset.queries, gt, efs, args.k)

    hnswlib_rows: list[dict] = []
    hnswlib_build: dict | None = None
    if args.compare_hnswlib:
        print("[phase1] building hnswlib...")
        hnswlib_rows, hnswlib_build = evaluate_hnswlib(
            dataset.base,
            dataset.queries,
            gt,
            efs,
            args.k,
            M=args.M,
            ef_construction=args.ef_construction,
            seed=args.seed,
            out_dir=out_dir,
            metric=args.metric,
        )

    all_rows = custom_rows + hnswlib_rows
    write_summary_csv(out_dir / "summary.csv", all_rows)
    plots = make_plots(all_rows, out_dir, dataset_name, args.k) if args.plots else []

    def row_at(rows: list[dict], ef: int) -> dict | None:
        return next((row for row in rows if row["ef"] == max(ef, args.k)), None)

    gate_custom = row_at(custom_rows, args.gate_ef)
    gate_lib = row_at(hnswlib_rows, args.gate_ef)
    recall_pass = bool(gate_custom and gate_custom["recall"] >= args.recall_target)
    parity_pass = None
    delta_pp = None
    if gate_custom and gate_lib:
        delta_pp = (gate_lib["recall"] - gate_custom["recall"]) * 100.0
        parity_pass = bool(delta_pp <= args.parity_tolerance_pp)
    week5_gate = bool(recall_pass and (parity_pass is not False))

    custom_index_info: dict = {
        "memory_bytes": build_stats["memory_bytes"],
        "memory_mb": build_stats["memory_bytes"] / 1024**2,
        "stats": build_stats,
    }
    if args.save_index:
        path = index.save(out_dir / "churnvec_index")
        custom_index_info["saved_path"] = str(path)
        custom_index_info["saved_bytes"] = path.stat().st_size

    projected = projection(
        build_seconds, dataset.n_base, gate_custom["p50_ms"] if gate_custom else float("nan")
    )
    result = {
        "run": environment(args),
        "dataset": dataset.summary(),
        "groundtruth": gt_info,
        "params": {
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": efs,
            "gate_ef": args.gate_ef,
            "k": args.k,
            "metric": args.metric,
            "selection": args.selection,
            "keep_pruned": args.keep_pruned,
            "seed": args.seed,
        },
        "build": {
            "seconds": build_seconds,
            "rss_start_mb": monitor.start_mb,
            "rss_peak_mb": monitor.peak_mb,
            "rss_delta_mb": monitor.peak_mb - monitor.start_mb,
            "n_base": dataset.n_base,
            **custom_index_info,
        },
        "churnvec": custom_rows,
        "hnswlib": hnswlib_rows,
        "hnswlib_build": hnswlib_build,
        "gate": {
            "ef": args.gate_ef,
            "target_recall": args.recall_target,
            "custom_recall": gate_custom["recall"] if gate_custom else None,
            "hnswlib_recall": gate_lib["recall"] if gate_lib else None,
            "recall_pass": recall_pass,
            "delta_pp": delta_pp,
            "parity_tolerance_pp": args.parity_tolerance_pp,
            "parity_pass": parity_pass,
            "week5_gate_passed": week5_gate,
        },
        "projection": projected,
        "plots": plots,
    }
    write_json(out_dir / "result.json", result)

    print("\n[phase1] summary")
    print(f"{'engine':<10} {'ef':>4} {'recall':>8} {'p50 ms':>9} {'p95 ms':>9} {'qps':>9}")
    for row in all_rows:
        print(
            f"{row['engine']:<10} {row['ef']:>4} {row['recall']:>8.4f} "
            f"{row['p50_ms']:>9.3f} {row['p95_ms']:>9.3f} {row['qps']:>9.1f}"
        )
    print(
        f"\n[phase1] Week-5 gate at ef={args.gate_ef}: "
        f"custom recall={gate_custom['recall']:.4f} (target {args.recall_target:.2f}); "
        f"hnswlib={gate_lib['recall']:.4f}; delta={delta_pp:+.2f}pp; "
        f"passed={week5_gate}"
        if gate_custom and gate_lib
        else f"\n[phase1] Week-5 gate at ef={args.gate_ef}: custom recall="
        f"{gate_custom['recall']:.4f}; passed={week5_gate}"
    )
    print(
        f"[phase1] projected 1M build: {projected['projected_1m_build_hours']:.2f}h; "
        f"native-core triggers: {projected['triggers'] or 'none'}"
    )
    print(f"[phase1] results: {out_dir / 'result.json'}")
    return 0 if week5_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
