# ChurnVec

A from-scratch HNSW vector index built as the **Week 1–5 milestone** of a
larger project on real-time vector indexing, safe deletion, tiered storage,
and a learned compaction scheduler.

**Status:** Week-5 gate passed on static SIFT100K. Deletion, concurrency,
tiering, and compaction are not implemented yet.

## Week-5 result

SIFT100K, 10,000 queries, exact recomputed ground truth, `M=16`,
`efConstruction=100`, `efSearch=64`:

| Engine | Recall@10 | p50 | Build |
|---|---:|---:|---:|
| ChurnVec (Python/NumPy) | **0.9846** | 0.978 ms | 241.82 s |
| hnswlib 0.8.0 | 0.9851 | 0.155 ms | 20.57 s |

- Recall gap: **0.052 pp**, well inside the 2 pp gate.
- Graph: 100,000 / 100,000 reachable, 0 repair links.
- Index memory: 63.8 MB (internal estimate); process peak 167.8 MB.
- 1M build projection: 0.67 h (linear; not a 1M run).
- Native-core trigger: not fired (projected build < 3 h, p50 < 10 ms).
- hnswlib is ~6.3× faster per query and ~11.8× faster to build; this is
  recall parity, not speed parity.

SIFT10K smoke test: 0.9970 recall for both engines at `ef=64`.

## What exists

- `.fvecs` / `.ivecs` readers and SIFT dataset adapters.
- Exact brute-force top-k baseline with chunked distance computation.
- From-scratch HNSW: level sampling, greedy descent, ef-bounded search,
  neighbour-selection heuristic, link pruning, persistence, invariants.
- In-degree tracking and a bulk-build connectivity repair for degenerate
  duplicate sets.
- 70 tests: unit, Hypothesis property tests, and SIFT10K integration.
- Benchmark harness: Recall@k, latency percentiles, build time, RSS, exact
  GT caching, hnswlib comparison, JSON/CSV, plots, Week-5 gate.
- Reproducible experiment configs for SIFT10K and SIFT100K.

## Quick start

```bash
make setup      # uv sync + dev dependencies
make data       # official SIFT archives, MD5-verified
make test       # 70 tests (dataset test skips if data is absent)
make e0         # SIFT10K smoke benchmark
make e1         # SIFT100K Week-5 gate
```

Or run the harness directly:

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e1_sift100k.yaml
```

The scripts prepend `src` themselves; `pytest` uses `pythonpath = ["src"]`
from `pyproject.toml`. Interactive `import churnvec` outside the make targets
needs `PYTHONPATH=src` because this macOS environment marks the editable
install’s `.pth` file hidden (see `BUILD_REPORT.md` §11.2).

## Layout

```text
src/churnvec/core/      distance, graph, HNSW, brute force, invariants
src/churnvec/bench/     dataset readers, exact search helpers, metrics
scripts/                data download, phase-1 benchmark
tests/                  unit, property, integration
configs/experiments/    e0_siftsmall, e1_sift100k
docs/                   claims, dataset notes, ADRs
reports/phase1/         committed result evidence (JSON/CSV/plots/logs)
results/                generated run outputs (gitignored)
```

## Week-5 gate definition

`SIFT100K`, all 10K queries, Recall@10 ≥ 0.90 at `efSearch=64`, within
roughly two points of hnswlib at matched parameters. If the 1M build is
projected above three hours or query p50 exceeds 10 ms, the native-core
decision is recorded.

## Documentation

- [Build report](BUILD_REPORT.md) — every step, command, problem, fix, and
  result, including the failed pruning experiment.
- [Claims](docs/claims.md) — allowed and forbidden claims.
- [Dataset notes](docs/dataset.md) — SIFT files, checksums, GT semantics.
- [ADRs](docs/adr/) — Python-core, pruning, and exact-GT decisions.
- [Result evidence](reports/phase1/) — committed benchmark artifacts.
- [Evidence index](reports/README.md) — what each artifact is, configurations,
  and provenance caveats.

## Not implemented

Deletes, updates, churn workloads, concurrent snapshots, hot/warm/cold
tiering, compaction, the ML scheduler, an API/service, a full SIFT1M build,
and text/GIST benchmarks. Do not read the Week-5 result as a claim about any
of those.
