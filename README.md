# ChurnVec

A from-scratch HNSW vector index with **deletion, updates, churn workloads,
snapshot-isolated readers, and compaction**, built as the **Week 1–14
milestone (50%)** of a 28-week project on real-time vector indexing.

**Status:** 14 / 28 weeks complete (50%).

- Weeks 1–5: static HNSW core, SIFT10K/SIFT100K benchmarks, Week-5 gate.
- Weeks 6–14: immutable segments + tombstones, updates, append-only insert
  buffer, atomic snapshots, concurrent readers, explicit compaction, and
  directory persistence.
- Not implemented: hot/warm/cold tiering, the learned compaction scheduler,
  an API/service, and a full SIFT1M run.

This repository contains the source code, tests, experiment configs, and
runner scripts needed to reproduce the results. The full written report,
submission package, benchmark evidence archive, and interactive demo are
maintained separately.

## Measured results

### Static SIFT100K (Week-5 gate)

10,000 queries, exact recomputed ground truth, `M=16`, `efConstruction=100`,
`efSearch=64`:

| Engine | Recall@10 | p50 | Build |
|---|---:|---:|---:|
| ChurnVec (Python/NumPy) | **0.9846** | 0.978 ms | 241.82 s |
| hnswlib 0.8.0 | 0.9851 | 0.155 ms | 20.57 s |

- Recall gap: **0.052 pp**, well inside the 2 pp gate.
- Graph: 100,000 / 100,000 reachable, 0 repair links.
- hnswlib was ~6.3× faster per query and ~11.8× faster to build; this is
  recall parity, not speed parity.

### SIFT10K churn workload (Phase 2)

10,000 base vectors, 10,000 queries, exact ground truth **recomputed on the
current live set after every batch**:

| Batch | Live | Segments | Dead fraction | Recall@10 | p50 |
|---|---:|---:|---:|---:|---:|
| warm | 9,700 | 1 | 0.093 | 0.9977 | 2.882 ms |
| steady | 9,400 | 1 | 0.175 | 0.9979 | 3.134 ms |
| heavy | 8,800 | 2 | 0.292 | 0.9982 | 3.600 ms |
| stress | 7,600 | 3 | 0.495 | 0.9993 | 9.696 ms |
| compacted | 7,600 | 1 | 0.000 | 0.9973 | **0.864 ms** |

- Cumulative churn: 40% deletes, 40% updates, 16% inserts.
- Snapshot stability: **4 / 4** batches returned byte-identical old-snapshot
  results after later mutations and flushes.
- Compaction: 7,600 live vectors rebuilt in **17.56 s**, 3 segments → 1,
  p50 **9.696 ms → 0.864 ms** (11.2× faster).

## What is implemented

- `.fvecs` / `.ivecs` readers and SIFT dataset adapters.
- Exact brute-force top-k baseline with chunked distance computation.
- From-scratch HNSW: level sampling, greedy descent, ef-bounded search,
  neighbour-selection heuristic, link pruning, persistence, invariants.
- Churn layer: `ChurnIndex`, immutable `Segment`s, tombstone deletes,
  `upsert`, append-only insert buffer, `flush`, `compact`, `save`/`load`.
- Snapshot isolation: writers publish an immutable `Snapshot`; readers search
  without locks. HNSW scratch state is thread-local.
- Exact ground-truth recomputation on every mutated live set.
- 100 tests: unit, Hypothesis property tests, SIFT10K integration, and
  concurrent reader/writer stress.
- Benchmark harness: Recall@k, latency percentiles, build time, RSS, exact
  GT caching, hnswlib replay comparison, JSON/CSV, plots.
- Reproducible experiment configs for E0, E1, and E2 (churn).

## Quick start

```bash
make setup      # uv sync + dev dependencies
make data       # official SIFT archives, MD5-verified
make test       # 100 tests (dataset tests skip if data is absent)
make e0         # SIFT10K smoke benchmark
make e1         # SIFT100K Week-5 gate
make e2         # SIFT10K churn workload + compaction
make e2-100k    # SIFT100K churn config (needs a host with free memory)
make e3         # SIFT1M static benchmark (needs a 24 GB host)
```

Or run the harness directly:

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e1_sift100k.yaml
.venv/bin/python scripts/run_churn.py \
  --config configs/experiments/e2_churn_sift10k.yaml
```

The scripts prepend `src` themselves; `pytest` uses `pythonpath = ["src"]`
from `pyproject.toml`. Interactive `import churnvec` outside the make targets
needs `PYTHONPATH=src` because this macOS environment marks the editable
install’s `.pth` file hidden.

### Cloud setup

The heavy SIFT100K churn and SIFT1M runs need more RAM than the reference
laptop provides. [`docs/cloud.md`](docs/cloud.md) contains a one-command
Oracle Cloud Always Free bootstrap for a 24 GB ARM VM.

### Churn API

```python
import numpy as np
from churnvec import ChurnIndex

index = ChurnIndex(dim=128, M=16, ef_construction=100, flush_threshold=2048)

index.add_many(vectors, ids)      # buffered inserts
index.delete(42)                  # O(segments) tombstone
index.upsert(7, new_vector)       # tombstone old + buffer new

snapshot = index.snapshot()       # capture once, then search lock-free
ids, distances = snapshot.search_ids(query, k=10, ef=64)

index.flush()                     # seal buffer into a new segment
index.compact()                   # merge all live vectors into one segment
index.save("path/to/index")       # manifest + segments + buffer
restored = ChurnIndex.load("path/to/index")
```

## Repository layout

```text
src/churnvec/core/      distance, graph, HNSW, brute force, segments, churn
src/churnvec/bench/     dataset readers, exact search helpers, metrics
scripts/                data download, phase-1 and churn benchmarks
tests/                  unit, property, integration, concurrency
configs/experiments/    e0_siftsmall, e1_sift100k, e2_churn_*
docs/adr/               architecture decision records
docs/claims.md          allowed and forbidden claims
docs/dataset.md         SIFT files, checksums, ground-truth semantics
data/                   local dataset cache (gitignored)
results/                generated run outputs (gitignored)
```

## Documentation

- [Claims](docs/claims.md) — allowed and forbidden claims.
- [Dataset notes](docs/dataset.md) — SIFT files, checksums, GT semantics.
- [ADRs](docs/adr/) — Python core, pruning, exact GT, segments/tombstones,
  snapshots, compaction/persistence.

## Data

The SIFT archives (~740 MB extracted) are **not committed** because GitHub
rejects files larger than 100 MB. `make data` downloads the official TEXMEX
archives over FTP, verifies their MD5 checksums, and extracts them under
`data/raw/`. Ground truth for prefixes and mutated live sets is recomputed
and cached under `data/processed/`.

## Not implemented

Hot/warm/cold tiering, the learned/ML compaction scheduler, an API or
service, crash-consistent journaling, finer-grained writer concurrency, a
full SIFT1M churn run, and text/GIST benchmarks. Do not read the results
above as claims about any of those.

---

<sub>Crafted and signed by <a href="https://github.com/sathish123m">Sathish</a> · 2026</sub>
