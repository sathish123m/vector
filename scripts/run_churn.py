#!/usr/bin/env python
"""Phase-2 churn benchmark: deletes, updates, inserts, snapshots, compaction.

The experiment starts from a static SIFT prefix, seals it as one HNSW segment,
then applies a deterministic churn schedule.  After every batch it recomputes
exact ground truth for the *current live set* and measures Recall@k, latency,
tombstone pressure, and snapshot stability.  A final compaction merges all
live vectors into one segment and the same measurements are repeated.

Examples
--------
    python scripts/run_churn.py --config configs/experiments/e2_churn_sift10k.yaml
    python scripts/run_churn.py --dataset sift100k --queries 2000 \\
        --flush-threshold 8192 --compact-at-end

Ground truth is never reused from the shipped SIFT files: deletes, updates,
and inserts change the live set, so exact top-k is recomputed from
``snapshot().live_arrays()`` for each measurement.
"""

from __future__ import annotations

import argparse
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

from churnvec import ChurnIndex, brute_force_topk  # noqa: E402
from churnvec.bench.datasets import load_dataset  # noqa: E402
from churnvec.bench.metrics import (  # noqa: E402
    current_rss_mb,
    latency_stats,
    recall_at_k,
    write_json,
)

DEFAULT_SCHEDULE = [
    {"name": "warm", "delete_fraction": 0.05, "update_fraction": 0.05, "insert_fraction": 0.02},
    {"name": "steady", "delete_fraction": 0.05, "update_fraction": 0.05, "insert_fraction": 0.02},
    {"name": "heavy", "delete_fraction": 0.10, "update_fraction": 0.10, "insert_fraction": 0.04},
    {"name": "stress", "delete_fraction": 0.20, "update_fraction": 0.20, "insert_fraction": 0.08},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML experiment config")
    parser.add_argument(
        "--dataset", default="sift10k", choices=["siftsmall", "sift10k", "sift100k", "sift1m"]
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "phase2" / "e2_churn")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["l2", "cosine", "ip"], default="l2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries", type=int, default=None, help="queries used for recall/latency")
    parser.add_argument(
        "--gt-queries",
        type=int,
        default=None,
        help="queries used for exact ground truth; defaults to --queries",
    )
    parser.add_argument(
        "--stability-queries",
        type=int,
        default=16,
        help="queries used to verify old snapshots after a batch",
    )
    parser.add_argument("--flush-threshold", type=int, default=8192)
    parser.add_argument(
        "--gt-chunk",
        type=int,
        default=256,
        help="query chunk for exact ground truth; lower values reduce peak memory",
    )
    parser.add_argument("--base-limit", type=int, default=None, help="prefix of the SIFT base")
    parser.add_argument("--insert-id-start", type=int, default=1_000_000)
    parser.add_argument("--schedule", type=list, default=None, help="YAML list of batch dicts")
    parser.add_argument("--compact-at-end", dest="compact_at_end", action="store_true")
    parser.set_defaults(compact_at_end=True)
    parser.add_argument("--no-compact-at-end", dest="compact_at_end", action="store_false")
    parser.add_argument("--compare-hnswlib", action="store_true")
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--plots", dest="plots", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.set_defaults(plots=True)
    parser.add_argument("--progress-every", type=int, default=20_000)
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
    args = parser.parse_args(argv)
    if args.schedule is None:
        args.schedule = DEFAULT_SCHEDULE
    if args.gt_queries is None:
        args.gt_queries = args.queries
    return args


def git_info() -> dict:
    def run(*parts: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), *parts], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
    }


def environment(args: argparse.Namespace) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "git": git_info(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def measure(
    index: ChurnIndex,
    queries: np.ndarray,
    live_ids: np.ndarray,
    live_vectors: np.ndarray,
    efs: list[int],
    k: int,
    metric: str,
    warmup: int = 16,
    gt_chunk: int | None = 256,
) -> tuple[dict, dict, np.ndarray]:
    """Recompute exact GT against the live set and evaluate every ef."""
    k_eff = min(int(k), int(live_ids.size))
    if k_eff == 0:
        return {}, {"n_live": 0, "n_queries": 0, "gt_seconds": 0.0}, np.empty((0, 0), dtype=np.int64)

    gt_start = time.perf_counter()
    truth_idx, _ = brute_force_topk(
        live_vectors, queries, k=k_eff, metric=metric, chunk_size=gt_chunk
    )
    gt_ids = live_ids[truth_idx]
    gt_seconds = time.perf_counter() - gt_start

    rows: dict[str, dict] = {}
    for ef in sorted({max(int(e), k_eff) for e in efs}):
        snapshot = index.snapshot()
        for query in queries[: min(warmup, queries.shape[0])]:
            snapshot.search_ids(query, k=k_eff, ef=ef)
        samples = np.empty(queries.shape[0], dtype=np.int64)
        returned = np.empty((queries.shape[0], k_eff), dtype=np.int64)
        for i, query in enumerate(queries):
            start = time.perf_counter_ns()
            ids, _ = snapshot.search_ids(query, k=k_eff, ef=ef)
            samples[i] = time.perf_counter_ns() - start
            returned[i] = ids
        stats = latency_stats(samples)
        recall = recall_at_k(returned, gt_ids, k=k_eff)
        rows[str(ef)] = {
            "ef": int(ef),
            "k": int(k_eff),
            "recall": float(recall),
            **stats,
        }
        print(
            f"    ef={ef:>4} recall@{k_eff}={recall:.4f} "
            f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
            f"qps={stats['qps']:.1f}"
        )
    gt_info = {
        "n_live": int(live_ids.size),
        "n_queries": int(queries.shape[0]),
        "gt_k": int(k_eff),
        "gt_seconds": float(gt_seconds),
        "method": "exact brute force recomputed on the live set",
    }
    return rows, gt_info, gt_ids


def apply_batch(
    index: ChurnIndex,
    batch: dict,
    n_initial: int,
    live_ids: np.ndarray,
    learn: np.ndarray,
    rng: np.random.Generator,
    update_pool_offset: int,
    next_insert_id: int,
) -> tuple[dict, int, int, dict]:
    """Apply one deterministic churn batch.  Returns counts, pools, and ops."""
    requested_delete = int(round(float(batch.get("delete_fraction", 0.0)) * n_initial))
    requested_update = int(round(float(batch.get("update_fraction", 0.0)) * n_initial))
    requested_insert = int(round(float(batch.get("insert_fraction", 0.0)) * n_initial))

    candidates = live_ids.copy()
    rng.shuffle(candidates)
    n_delete = min(requested_delete, max(0, candidates.size - 1))
    delete_ids = candidates[:n_delete]
    remaining = candidates[n_delete:]
    n_update = min(requested_update, remaining.size)
    update_ids = remaining[:n_update]

    update_vectors = np.stack(
        [learn[(update_pool_offset + offset) % learn.shape[0]] for offset in range(n_update)]
    ) if n_update else np.empty((0, learn.shape[1]), dtype=np.float32)
    inserted_ids = np.arange(next_insert_id, next_insert_id + requested_insert, dtype=np.int64)
    insert_vectors = np.stack(
        [
            learn[(update_pool_offset + n_update + i) % learn.shape[0]]
            for i in range(requested_insert)
        ]
    ) if requested_insert else np.empty((0, learn.shape[1]), dtype=np.float32)

    start = time.perf_counter()
    deleted = index.delete_many(delete_ids.tolist())
    updated = 0
    for external_id, vector in zip(update_ids.tolist(), update_vectors):
        updated += int(index.upsert(external_id, vector))
    for external_id, vector in zip(inserted_ids.tolist(), insert_vectors):
        index.add(int(external_id), vector)
    elapsed = time.perf_counter() - start

    operations = {
        "delete_ids": delete_ids.astype(np.int64, copy=True),
        "update_ids": update_ids.astype(np.int64, copy=True),
        "update_vectors": update_vectors.astype(np.float32, copy=True),
        "insert_ids": inserted_ids.astype(np.int64, copy=True),
        "insert_vectors": insert_vectors.astype(np.float32, copy=True),
    }
    info = {
        "requested": {
            "delete": requested_delete,
            "update": requested_update,
            "insert": requested_insert,
        },
        "applied": {
            "delete": int(deleted),
            "update": int(updated),
            "insert": int(requested_insert),
        },
        "mutations": int(deleted + updated + requested_insert),
        "seconds": float(elapsed),
        "mutations_per_second": float((deleted + updated + requested_insert) / max(elapsed, 1e-9)),
        "batch_name": batch.get("name", "batch"),
    }
    return info, update_pool_offset + n_update + requested_insert, next_insert_id + requested_insert, operations


def hnswlib_run(
    base: np.ndarray,
    operations: list[dict],
    queries: np.ndarray,
    schedule: list[dict],
    args: argparse.Namespace,
    gt_live: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Replay the exact ChurnVec operation lists against hnswlib."""
    import hnswlib

    n_initial = base.shape[0]
    total_insert = sum(int(ops["insert_ids"].size) for ops in operations)
    total_update = sum(int(ops["update_ids"].size) for ops in operations)
    index = hnswlib.Index(space=args.metric, dim=base.shape[1])
    max_elements = n_initial + total_insert + total_update + 1024
    index.init_index(
        max_elements=max_elements,
        ef_construction=args.ef_construction,
        M=args.M,
        random_seed=args.seed,
        allow_replace_deleted=False,
    )
    index.set_num_threads(1)
    build_start = time.perf_counter()
    index.add_items(base, np.arange(n_initial, dtype=np.int64))
    build_seconds = time.perf_counter() - build_start

    # hnswlib's replace-deleted path reuses labels, which makes an external-id
    # mapping ambiguous.  Instead model an update the same way ChurnVec does:
    # tombstone the old label and append a fresh label for the same external
    # id.  Results are mapped back to external ids before scoring.
    external_by_label = np.full(max_elements, -1, dtype=np.int64)
    external_by_label[:n_initial] = np.arange(n_initial, dtype=np.int64)
    live_label_by_external = {external_id: external_id for external_id in range(n_initial)}
    next_label = n_initial

    batches = []
    for batch_number, (batch, ops, truth_data) in enumerate(zip(schedule, operations, gt_live)):
        start = time.perf_counter()
        deleted = 0
        for external_id in ops["delete_ids"].tolist():
            label = live_label_by_external.pop(int(external_id), None)
            if label is None:
                continue
            index.mark_deleted(int(label))
            deleted += 1

        updated = 0
        for external_id, vector in zip(ops["update_ids"].tolist(), ops["update_vectors"]):
            old_label = live_label_by_external.pop(int(external_id), None)
            if old_label is not None:
                index.mark_deleted(int(old_label))
            label = next_label
            next_label += 1
            external_by_label[label] = int(external_id)
            live_label_by_external[int(external_id)] = label
            index.add_items(vector.reshape(1, -1), np.asarray([label], dtype=np.int64))
            updated += 1

        for external_id, vector in zip(ops["insert_ids"].tolist(), ops["insert_vectors"]):
            label = next_label
            next_label += 1
            external_by_label[label] = int(external_id)
            live_label_by_external[int(external_id)] = label
            index.add_items(vector.reshape(1, -1), np.asarray([label], dtype=np.int64))
        elapsed = time.perf_counter() - start

        truth_ids = truth_data[2]
        rows = {}
        for ef in args.ef:
            ef = max(int(ef), args.k)
            index.set_ef(ef)
            samples = np.empty(queries.shape[0], dtype=np.int64)
            returned = np.empty((queries.shape[0], args.k), dtype=np.int64)
            for i, query in enumerate(queries):
                start_ns = time.perf_counter_ns()
                labels, _ = index.knn_query(query.reshape(1, -1), k=args.k, num_threads=1)
                samples[i] = time.perf_counter_ns() - start_ns
                returned[i] = external_by_label[labels[0].astype(np.int64)]
            stats = latency_stats(samples)
            rows[str(ef)] = {
                "ef": int(ef),
                "k": int(args.k),
                "recall": float(recall_at_k(returned, truth_ids, k=args.k)),
                **stats,
            }
        batches.append(
            {
                "batch": batch_number,
                "name": batch.get("name", "batch"),
                "applied": {
                    "delete": int(deleted),
                    "update": int(updated),
                    "insert": int(ops["insert_ids"].size),
                },
                "seconds": float(elapsed),
                "evaluation": rows,
            }
        )
        print(
            f"    hnswlib batch {batch_number} ({batch.get('name', 'batch')}) "
            f"applied in {elapsed:.2f}s"
        )

    index_path = ROOT / "results" / "phase2" / "e2_churn" / "hnswlib_churn.bin"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(index_path))
    return {
        "build_seconds": float(build_seconds),
        "batches": batches,
        "index_file_bytes": int(index_path.stat().st_size),
        "note": (
            "hnswlib keeps marked-deleted slots and appends fresh labels for updates; "
            "it has no segment compaction."
        ),
    }


def make_plots(result: dict, out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  plotting skipped: {exc}")
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    batches = result["batches"]
    labels = [f"{b['batch']}\n{b.get('batch_name', b.get('name', 'batch'))}" for b in batches]
    primary_ef = str(max(int(ef) for ef in result["config"]["ef"]))
    if result.get("final") is not None:
        labels.append("compacted")
        points = batches + [result["final"]]
    else:
        points = batches
    paths = []
    for metric, ylabel, filename in [
        ("recall", f"Recall@{result['config']['k']}", "recall_vs_churn.png"),
        ("p50_ms", "p50 latency (ms)", "latency_vs_churn.png"),
    ]:
        values = [point["evaluation"][primary_ef][metric] for point in points]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(range(len(labels)), values, marker="o", color="#0D9488")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"ChurnVec {result['dataset']} — efSearch={primary_ef}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.text(0.995, 0.005, "sathish123m", ha="right", va="bottom", fontsize=6, alpha=0.28)
        path = plots_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_name = "siftsmall" if args.dataset == "sift10k" else args.dataset
    out_dir = args.out or ROOT / "results" / "phase2" / f"e2_churn_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[churn] dataset={dataset_name} out={out_dir} M={args.M} "
        f"efC={args.ef_construction} ef={args.ef} k={args.k}"
    )

    dataset = load_dataset(
        dataset_name,
        root=args.data_root,
        base_limit=args.base_limit,
        query_limit=args.queries,
        include_learn=True,
    )
    base = dataset.base
    queries = dataset.queries
    learn = dataset.learn if dataset.learn is not None else base
    n_initial = base.shape[0]
    print(
        f"[churn] base={n_initial:,}x{dataset.dim} queries={queries.shape[0]:,} "
        f"learn={learn.shape[0]:,}"
    )

    index = ChurnIndex(
        dim=dataset.dim,
        M=args.M,
        ef_construction=args.ef_construction,
        metric=args.metric,
        seed=args.seed,
        flush_threshold=max(n_initial, 1),
    )
    rss_start = current_rss_mb()
    build_start = time.perf_counter()
    index.add_many(base, np.arange(n_initial, dtype=np.int64))
    index.flush()
    build_seconds = time.perf_counter() - build_start
    rss_after_build = current_rss_mb()
    index.flush_threshold = args.flush_threshold
    print(
        f"[churn] initial build {build_seconds:.2f}s "
        f"segments={index.stats()['segment_count']} live={len(index):,}"
    )

    rng = np.random.default_rng(args.seed)
    live_ids, _ = index.snapshot().live_arrays()
    update_pool_offset = 0
    next_insert_id = int(args.insert_id_start)
    batches = []
    gt_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    operation_plan: list[dict] = []
    compact_eval = None
    stability_checks = 0
    stability_failures = 0

    for batch_number, batch in enumerate(args.schedule):
        snapshot_before = index.snapshot()
        stability_ids = min(int(args.stability_queries), queries.shape[0])
        before_results = [
            snapshot_before.search_ids(query, k=args.k, ef=max(args.ef))
            for query in queries[:stability_ids]
        ]
        before_live = snapshot_before.live_count
        before_generation = snapshot_before.generation
        before_dead_fraction = index.dead_fraction

        live_ids, _ = index.snapshot().live_arrays()
        info, update_pool_offset, next_insert_id, operations = apply_batch(
            index,
            batch,
            n_initial,
            live_ids,
            learn,
            rng,
            update_pool_offset,
            next_insert_id,
        )
        if args.compare_hnswlib:
            operation_plan.append(operations)

        # Old snapshot must be byte-for-byte stable after mutations and flushes.
        after_results = [
            snapshot_before.search_ids(query, k=args.k, ef=max(args.ef))
            for query in queries[:stability_ids]
        ]
        stable = snapshot_before.live_count == before_live
        for (old_ids, old_distances), (new_ids, new_distances) in zip(before_results, after_results):
            stable = stable and np.array_equal(old_ids, new_ids)
            stable = stable and np.allclose(old_distances, new_distances, rtol=0, atol=0)
        stability_checks += 1
        stability_failures += int(not stable)
        info["snapshot_stability"] = {
            "checked": True,
            "queries": int(stability_ids),
            "live_count_before": int(before_live),
            "generation_before": int(before_generation),
            "passed": bool(stable),
        }

        live_ids, live_vectors = index.snapshot().live_arrays()
        table, gt_info, gt_ids = measure(
            index,
            queries,
            live_ids,
            live_vectors,
            args.ef,
            args.k,
            args.metric,
            gt_chunk=args.gt_chunk,
        )
        gt_sets.append((live_ids, live_vectors, gt_ids))
        stats = index.stats()
        info.update(
            {
                "batch": batch_number,
                "index": stats,
                "dead_fraction_before": float(before_dead_fraction),
                "evaluation": table,
                "groundtruth": gt_info,
                "rss_mb": current_rss_mb(),
            }
        )
        batches.append(info)
        print(
            f"  batch {batch_number} ({info['batch_name']}): "
            f"live={stats['live_count']:,} segments={stats['segment_count']} "
            f"phys={stats['physical_count']:,} dead={stats['dead_fraction']:.3f} "
            f"mutations={info['mutations']} ({info['seconds']:.2f}s)"
        )

    if args.compact_at_end:
        print("[churn] compacting all live vectors into one segment...")
        compact_start = time.perf_counter()
        compacted = index.compact()
        compact_seconds = time.perf_counter() - compact_start
        live_ids, live_vectors = index.snapshot().live_arrays()
        table, gt_info, _ = measure(
            index,
            queries,
            live_ids,
            live_vectors,
            args.ef,
            args.k,
            args.metric,
            gt_chunk=args.gt_chunk,
        )
        compact_eval = {
            "vectors": int(compacted),
            "seconds": float(compact_seconds),
            "index": index.stats(),
            "evaluation": table,
            "groundtruth": gt_info,
            "rss_mb": current_rss_mb(),
        }
        print(
            f"  compacted {compacted:,} vectors in {compact_seconds:.2f}s; "
            f"segments={index.stats()['segment_count']} dead={index.dead_fraction:.3f}"
        )

    baseline = None
    if args.compare_hnswlib:
        print("[churn] running hnswlib baseline with matching churn operations...")
        baseline = hnswlib_run(base, operation_plan, queries, args.schedule, args, gt_sets)

    result = {
        "experiment": "e2_churn",
        "dataset": dataset_name,
        "config": {
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef": args.ef,
            "k": args.k,
            "metric": args.metric,
            "flush_threshold": args.flush_threshold,
            "gt_chunk": args.gt_chunk,
            "schedule": args.schedule,
            "queries": int(queries.shape[0]),
            "gt_queries": int(args.gt_queries or queries.shape[0]),
            "seed": args.seed,
            "base_limit": args.base_limit,
        },
        "environment": environment(args),
        "dataset_info": dataset.summary(),
        "build": {
            "seconds": float(build_seconds),
            "vectors": int(n_initial),
            "rss_start_mb": float(rss_start),
            "rss_after_build_mb": float(rss_after_build),
            "rss_delta_mb": float(rss_after_build - rss_start),
        },
        "batches": batches,
        "final": compact_eval,
        "snapshot_stability": {
            "checks": int(stability_checks),
            "failures": int(stability_failures),
            "passed": bool(stability_failures == 0),
        },
        "hnswlib": baseline,
    }
    if args.save_index:
        print("[churn] saving final index...")
        result["saved_index"] = str(index.save(out_dir / "churnvec_index"))
    if args.plots:
        result["plots"] = make_plots(result, out_dir)
    write_json(out_dir / "result.json", result)
    print(f"[churn] wrote {out_dir / 'result.json'}")
    if not result["snapshot_stability"]["passed"]:
        print("[churn] SNAPSHOT STABILITY FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
