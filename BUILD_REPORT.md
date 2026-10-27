# ChurnVec — Week 1–5 Build Report

**Milestone:** Weeks 1–5 of the ChurnVec project  
**Report date:** 2026-10-27  
**Repository:** `/Users/msk/Documents/vector`  
**Status:** Week 5 gate passed for the static SIFT100K benchmark.  
**Scope deliberately stops before deletion, concurrency, tiering, and the
learned compaction scheduler.**

---

## 0. Executive summary

I built a from-scratch HNSW vector index in Python/NumPy, with SIFT dataset
loaders, an exact brute-force baseline, a measurement harness, a 70-test
pytest/Hypothesis suite, and a reproducible SIFT10K → SIFT100K benchmark path.

The Week 5 gate asked for **Recall@10 ≥ 0.90 on SIFT100K at `efSearch=64`**,
within roughly two percentage points of hnswlib at matched parameters.

| Gate item | Result |
|---|---:|
| SIFT100K base | 100,000 × 128d |
| Queries | 10,000 |
| `M` | 16 |
| `efConstruction` | 100 |
| `efSearch` | 64 |
| ChurnVec Recall@10 | **0.9846** |
| hnswlib Recall@10 | **0.9851** |
| Gap vs hnswlib | **0.052 pp worse** (well inside 2 pp) |
| Gate | **PASS** |
| Build time | 241.82 s (4.03 min) |
| Query p50 / p95 | 0.978 ms / 1.142 ms |
| Index memory (theoretical) | 63.8 MB |
| Graph reachability | 100,000 / 100,000; 0 repair links |
| Projected 1M build | 0.67 h by linear projection |
| Native-core trigger | **Not triggered** (build < 3 h, p50 < 10 ms) |

The Python/NumPy core is sufficient for the Week 5 static gate. hnswlib is
still **~6.3× faster per query** and **~11.8× faster to build** on this
machine, so the native-core question remains open for later throughput,
churn, and concurrency milestones. No 1M index was built; the 1M numbers are
projections and are labelled as such.

Two non-obvious issues were found and fixed during the build:

1. The optional HNSW “keep pruned connections” fill fragmented the SIFT
   layer-0 graph at scale. Disabling it (matching hnswlib’s default pruning)
   removed 99,673 emergency repair links on the 100K build and cut build time.
2. Exact duplicate vectors can exhaust fixed-capacity links and leave nodes
   unreachable. The implementation now tracks in-degrees and has an explicit
   bulk-build repair path for degenerate duplicate sets.

Both are documented with evidence in §11.

---

## 1. Scope and constraints

### In scope for Weeks 1–5

- Repository, environment, and reproducible data acquisition.
- TEXMEX `.fvecs` / `.ivecs` readers and SIFT adapters.
- Exact brute-force top-k baseline.
- From-scratch HNSW: insertion, multi-layer search, pruning, persistence,
  invariant checking.
- Unit, property-based, and integration tests.
- Recall/latency/build-time/memory harness with hnswlib comparison.
- SIFT10K smoke run and SIFT100K Week-5 validation gate.
- Honest build documentation and claims boundaries.

### Explicitly out of scope for this milestone

- Deletes, updates, tombstones, or segment compaction.
- Concurrent reads/writes, snapshots, or crash consistency.
- Hot/warm/cold tiering or storage offload.
- The ML-based compaction scheduler.
- HTTP/gRPC API, service deployment, or production operations.
- A full SIFT1M build or churn workloads.
- Text-embedding benchmarks or full GIST1M.

These are later milestones. Nothing in this report should be read as claiming
they are implemented.

---

## 2. Environment and hardware

| Item | Value |
|---|---|
| Machine | Apple M1, arm64 |
| macOS | 15.1 |
| RAM | 8 GB |
| CPUs | 8 |
| Free disk at start | 8.2 GB |
| Python | 3.12.14 (`.venv`) |
| Package manager | uv 0.12.10 |
| NumPy | 2.5.4 |
| pandas | 3.1.0 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| hnswlib | 0.8.0 |
| pytest | 9.1.1 |
| Hypothesis | 6.170.12 |
| psutil | 7.2.2 |
| PyYAML | 6.0.3 |

The machine has only 8 GB of RAM and a nearly full disk. That is enough for
SIFT100K, but it is a real constraint for a future SIFT1M build and for
materialising exact 10K × 1M ground truth.

---

## 3. Repository layout

```text
churnvec/
├── BUILD_REPORT.md                 # this report
├── README.md
├── Makefile
├── pyproject.toml
├── configs/
│   └── experiments/
│       ├── e0_siftsmall.yaml
│       └── e1_sift100k.yaml
├── data/                           # gitignored
│   ├── raw/sift/                   # SIFT1M files
│   ├── raw/siftsmall/              # SIFT10K files
│   └── processed/                  # exact ground-truth caches
├── docs/
│   ├── claims.md
│   ├── dataset.md
│   └── adr/
├── reports/
│   └── phase1/                     # committed evidence copies
├── results/                        # gitignored run outputs
├── logs/                           # gitignored raw logs
├── scripts/
│   ├── download_data.sh
│   ├── make_manifest.py
│   └── run_phase1.py
├── src/churnvec/
│   ├── core/
│   │   ├── bruteforce.py
│   │   ├── distance.py
│   │   ├── exceptions.py
│   │   ├── graph.py
│   │   └── hnsw.py
│   └── bench/
│       ├── datasets.py
│       └── metrics.py
└── tests/
    ├── unit/
    ├── property/
    └── integration/
```

The design keeps the index core independent of the benchmark harness. The
harness can load, build, search, and compare without touching core code.

---

## 4. Data acquisition and verification

### Source

Official IRISA/CNRS TEXMEX corpus:

```text
ftp://ftp.irisa.fr/local/texmex/corpus/
```

The project uses the official SIFT archives, not a mirror. The site’s HTTPS
certificate is broken; the FTP source and the official checksums are the
authority used here.

### Files and checksums

| Archive | Official MD5 | Verified locally |
|---|---|---|
| `sift.tar.gz` | `b23d1b3b2ee8469d819b61ca900ef0ed` | yes |
| `siftsmall.tar.gz` | `0b8324a7a82d7f2663d7dcbd57642df7` | yes |

### Extracted data

| Set | Base | Queries | Learn | Ground truth |
|---|---:|---:|---:|---:|
| SIFT10K (`siftsmall`) | 10,000 × 128d | 100 × 128d | 25,000 × 128d | 100 × 100 IDs |
| SIFT1M (`sift`) | 1,000,000 × 128d | 10,000 × 128d | 100,000 × 128d | 10,000 × 100 IDs |

SIFT1M extracted size is ~550 MB; the base file alone is 516,000,000 bytes.
The SIFT10K query file has only **100 queries**; an early assumption that it
had 10,000 was wrong and is recorded in §11.

### Ground-truth policy

The shipped SIFT ground truth is valid only for the exact original base set,
with squared L2 distance and 0-based IDs. It is **not** valid for:

- a 100K prefix of SIFT1M,
- a base after inserts or deletes,
- a different metric.

The harness therefore:

- uses shipped GT only when the base is exactly the shipped base;
- recomputes exact top-100 IDs for SIFT100K with chunked brute force;
- caches the result as `data/processed/gt_sift100k_100000_q10000_k100.npy`
  (3.8 MB, 10,000 × 100 `int32`);
- recomputes GT for any truncated/prefix benchmark;
- rejects an unreachable build before benchmarking.

Exact 10K × 1M materialised as a float32 matrix would be 40 GB, so the
brute-force baseline chunks queries and keeps only top-k. This is the same
policy future churn checkpoints will use.

---

## 5. Implementation

### 5.1 Distance kernels (`core/distance.py`)

- Squared L2 scalar and batch kernels.
- Norm-precomputed L2 batch kernel for the hot path:
  `||x||² − 2x·q + ||q||²`, clamped at zero.
- Cosine and inner-product variants (inner product returned as negative so
  “smaller is closer” holds everywhere).
- Row normalisation helper.
- Float32 storage throughout; no square roots in the L2 hot path.

### 5.2 Graph storage (`core/graph.py`)

- Dense layer-0 adjacency matrix: `(capacity, 2M)` `int32`.
- Sparse dictionaries for upper layers, because only ~1/M of nodes reach
  layer 1 and fewer reach higher layers.
- Dynamic capacity growth that preserves vectors, norms, IDs, levels,
  adjacency, and in-degree counts.
- External `int64` IDs mapped to compact internal slots.
- Incremental in-degree bookkeeping for layer 0 and upper layers.
- `InvariantReport` covering:
  - entry point validity and maximum level;
  - per-node level and degree bounds;
  - out-of-range links, self-loops, duplicate neighbours;
  - ID-map consistency and duplicate IDs;
  - in-degree bookkeeping consistency;
  - layer-0 reachability from the entry point.

### 5.3 HNSW (`core/hnsw.py`)

Implemented from the Malkov–Yashunin algorithm:

1. **Level sampling:** `floor(−ln(U) / ln(M))`, clamped at 32.
2. **Greedy descent:** from the entry point through upper layers.
3. **`ef`-bounded search:** best-first candidate heap + max-heap of results,
   with a visited-token array to avoid re-visiting nodes.
4. **Neighbour selection:** diversity heuristic; a candidate is kept only if
   it is not closer to an already-selected neighbour than to the query.
5. **Reciprocal linking and pruning:** each selected neighbour gets a back
   link; a full list is pruned with the same heuristic.
6. **Persistence:** `.npz` with metadata, vectors, IDs, levels, layer-0
   adjacency, upper-layer adjacency, and RNG state. Load rebuilds in-degree
   bookkeeping.
7. **Connectivity repair:** a bulk-build-only fallback for degenerate data
   such as large sets of exactly equal vectors. It threads a deterministic
   chain through the graph only when reachability is actually broken. Healthy
   SIFT builds report 0 repaired links.
8. **Diagnostics:** size, max level, layer counts, degrees, reachability,
   memory estimate, distance-call count, visited-node count, repair count.

The default is `keep_pruned=False`, i.e. hnswlib-style pruning without
filling rejected candidates. See §11.5 for why.

### 5.4 Exact baseline (`core/bruteforce.py`)

- Chunked query processing to bound peak memory.
- BLAS matrix multiply and `argpartition` for top-k.
- Sorted output, stable top-k selection, metric-aware distances.
- Used for correctness tests and exact ground-truth generation.

### 5.5 Dataset adapters (`bench/datasets.py`)

- `.fvecs` / `.ivecs` readers and writers.
- SIFT10K, SIFT100K (prefix), and SIFT1M adapters.
- Lazy exact-GT cache with shape validation.
- Explicit refusal to reuse 1M GT for a prefix.
- Dataset summaries for result JSON.

### 5.6 Measurement harness (`bench/metrics.py` and `scripts/run_phase1.py`)

- Recall@k against exact IDs.
- Per-query latency samples with mean, p50, p95, p99, min, max, QPS.
- Background RSS sampler with start/peak/delta.
- JSON and CSV output.
- Recall-vs-ef and latency-vs-ef plots.
- hnswlib build/search comparison at matched `M`, `efConstruction`, and
  `efSearch`, with one thread for both engines.
- Gate evaluation and 1M projection.
- Run metadata: timestamp, command, Python/NumPy/hnswlib versions, host,
  git commit and dirty status.

---

## 6. Tests

Command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Result on the final code:

```text
70 passed in 22.89s
```

Breakdown:

| Test file | Tests | What is covered |
|---|---:|---|
| `unit/test_distance.py` | 13 | L2/cosine/IP batch vs scalar, norm kernel, zero vectors, validation |
| `unit/test_datasets.py` | 9 | `.fvecs`/`.ivecs` round-trip, limits, SIFT adapters, GT cache |
| `unit/test_bruteforce.py` | 10 | Exact top-k vs naive, chunking, cosine/IP, input validation |
| `unit/test_graph.py` | 6 | Capacity growth, duplicate IDs, neighbour validation, in-degree/invariants |
| `unit/test_hnsw.py` | 23 | Exact search, IDs, duplicate handling, persistence, metrics, selection variants, repair |
| `unit/test_metrics.py` | 5 | Recall, latency, RSS, JSON serialisation |
| `property/test_hnsw_properties.py` | 2 | Random small graphs; connectivity; exact distances at full-width `ef` |
| `integration/test_siftsmall.py` | 2 | SIFT10K recall smoke; shipped-GT distance agreement |

Notable properties:

- With `ef = n`, search must return the exact k smallest distances (ties may
  reorder equal-distance IDs).
- Every sequential insert leaves the graph valid and reachable for random
  data.
- 100 identical vectors trigger the connectivity repair and end fully
  reachable.
- Save/load round-trips search results and still accepts new inserts.
- Duplicate IDs and dimension mismatches are rejected without mutating the
  index.

The SIFT10K integration test is skipped automatically when data is absent.
`make test-fast` excludes slow/dataset tests.

---

## 7. Benchmarks

### 7.1 E0 — SIFT10K smoke test

Configuration: `M=16`, `efConstruction=100`, `keep_pruned=False`, seed 42,
100 queries, 10,000 base vectors. Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e0_siftsmall.yaml
```

| efSearch | ChurnVec Recall@10 | ChurnVec p50 (ms) | hnswlib Recall@10 | hnswlib p50 (ms) |
|---:|---:|---:|---:|---:|
| 16 | 0.9590 | 0.295 | 0.9620 | 0.035 |
| 32 | 0.9900 | 0.561 | 0.9910 | 0.042 |
| 64 | **0.9970** | 0.874 | **0.9970** | 0.071 |
| 128 | 1.0000 | 1.620 | 1.0000 | 0.112 |

| Build metric | ChurnVec | hnswlib |
|---|---:|---:|
| Build time | 21.41 s | 1.11 s |
| Build RSS start → peak | 40.1 → 46.6 MB | 33.3 → 43.3 MB |
| Index size | 6.4 MB theoretical | 6.30 MB file |

SIFT10K is small and easy; its value is harness validation, not a performance
claim.

### 7.2 E1 — SIFT100K Week-5 gate

Configuration: same `M`, `efConstruction`, `keep_pruned`, and seed; 100,000
base vectors; all 10,000 queries; exact recomputed GT. Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e1_sift100k.yaml
```

| efSearch | ChurnVec Recall@10 | ChurnVec p50 (ms) | ChurnVec p95 (ms) | hnswlib Recall@10 | hnswlib p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8637 | 0.334 | 0.442 | 0.8633 | 0.057 |
| 32 | 0.9467 | 0.554 | 0.676 | 0.9468 | 0.094 |
| **64** | **0.9846** | **0.978** | **1.142** | **0.9851** | **0.155** |
| 128 | 0.9962 | 1.828 | 2.252 | 0.9965 | 0.263 |
| 256 | 0.9987 | 3.485 | 4.371 | 0.9990 | 0.451 |

| Build metric | ChurnVec | hnswlib |
|---|---:|---:|
| Build time | 241.82 s | 20.57 s |
| Insertion rate | 413.5 vectors/s | 4,862 vectors/s |
| Build RSS start → peak | 150.2 → 167.8 MB | 96.6 → 124.3 MB |
| Index size | 63.8 MB theoretical | 62.99 MB file |
| Max level | 4 | not exposed |
| Reachability | 100,000 / 100,000 | not exposed |
| Repair links | 0 | n/a |

The gate result is a **recall-parity** result, not a speed result. ChurnVec is
~6.3× slower per query and ~11.8× slower to build than hnswlib on this
machine. That is expected for pure Python/NumPy and is the main input to the
native-core decision.

### 7.3 1M projection

Using the measured 100K insertion rate and p50:

| Projection | Value |
|---|---:|
| Inserts/s | 413.5 |
| Linear 1M build | 0.672 h (40.3 min) |
| Measured p50 at ef=64 | 0.978 ms |
| Build trigger (>3 h) | no |
| Latency trigger (>10 ms) | no |

This is a **linear projection from 100K**, not a 1M run. HNSW insertion cost
grows with graph depth and Python overhead, so a real 1M build may take
longer. The projection is a gate input, not a benchmark result.

---

## 8. Week-5 gate decision

| Criterion | Required | Measured | Verdict |
|---|---:|---:|---|
| SIFT100K Recall@10 at ef=64 | ≥ 0.90 | 0.9846 | pass |
| Gap to hnswlib | ≤ ~2 pp | 0.052 pp worse | pass |
| 1M build projection | < 3 h for Python core | 0.67 h | pass |
| Query p50 | < 10 ms | 0.978 ms | pass |

**Decision: the Python/NumPy core is sufficient for the Week 5 static
SIFT100K gate. Do not start a native core solely for this gate.**

Caveats that keep this decision honest:

- The 1M projection is not a 1M measurement.
- hnswlib’s C++ implementation is ~6.3× faster at query time and ~11.8×
  faster at build time.
- The gate measures static recall, not churn, deletion, concurrency, or
  tiering throughput.
- A native core remains a reasonable later choice if the churn/concurrency
  milestones show Python cannot meet their latency or throughput targets.
  Profile those workloads before committing to a rewrite.

---

## 9. What was built, in chronological order

This is the actual build sequence, including the dead ends.

1. Initialised the repository and directory layout on `main`.
2. Created `pyproject.toml`, `.python-version`, `.gitignore`, `Makefile`,
   `README.md`.
3. Attempted `uv sync --extra dev`; PyPI requests timed out repeatedly.
4. Installed dependencies with `UV_CONCURRENT_DOWNLOADS=1` and a longer
   timeout; verified imports.
5. Wrote `scripts/download_data.sh` with official MD5 verification.
6. Downloaded and extracted `siftsmall.tar.gz` and `sift.tar.gz`; verified
   both checksums.
7. Implemented distance kernels, `GraphStore`, HNSW, brute force, dataset
   readers, and metrics.
8. Hit the imported-package failure caused by macOS’s hidden flag on the
   editable `.pth`; diagnosed with `python -v`, then moved to `PYTHONPATH=src`
   and pytest’s `pythonpath` setting.
9. Ran a synthetic 500-vector HNSW sanity check: recall 1.0 at `ef=50`,
   all 500 nodes reachable.
10. Wrote and ran unit/property/integration tests; fixed test-helper and
    NumPy reader bugs.
11. Added duplicate-vector connectivity handling after a Hypothesis test
    found an unreachable node in an all-identical set.
12. Implemented the benchmark harness and both experiment configs.
13. Ran SIFT10K with `keep_pruned=True`: 42.3 s build, recall 1.000 at
    `ef=64`.
14. Ran SIFT100K with `keep_pruned=True`: recall 0.9881, but 99,673 repair
    links and a fragmented layer-0 graph.
15. Isolated the fragmentation on 15K/20K builds; found two large components
    and one cross-edge.
16. Tested `keep_pruned=False` and simple selection; `keep_pruned=False`
    stayed connected (0 unreachable) and built ~2× faster, with 0.9952 recall
    on a 20K exact-GT check.
17. Changed the default to `keep_pruned=False`, reran SIFT10K and SIFT100K.
18. Reran both benchmarks after fixing RSS measurement so index preallocation
    is inside the memory monitor.
19. Ran the full 70-test suite on the final code.
20. Wrote this report, claims, dataset notes, ADRs, and committed evidence.

---

## 10. Commands used

### Environment

```bash
uv sync --extra dev
# PyPI timeouts; retried with:
UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=1 uv pip install \
  --python .venv/bin/python numpy pyyaml psutil pytest hypothesis
UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=1 uv pip install \
  --python .venv/bin/python pandas matplotlib pydantic scikit-learn tqdm hnswlib
uv sync --extra dev
```

### Data

```bash
bash scripts/download_data.sh
md5 -q data/raw/sift.tar.gz          # b23d1b3b2ee8469d819b61ca900ef0ed
md5 -q data/raw/siftsmall.tar.gz     # 0b8324a7a82d7f2663d7dcbd57642df7
```

### Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest -q -m "not slow and not dataset"
```

### Benchmarks

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e0_siftsmall.yaml

PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --config configs/experiments/e1_sift100k.yaml
```

### Make targets

```bash
make setup
make data
make test
make e0
make e1
```

`make e0` / `make e1` export `PYTHONPATH` internally. The Python scripts also
insert `src` on their own path, so direct invocation works without the
environment variable. `pytest` gets `pythonpath = ["src"]` from
`pyproject.toml`.

---

## 11. Problems, diagnoses, and fixes

### 11.1 PyPI timeouts during `uv sync`

**Symptom:** `uv sync --extra dev` failed after ~382 s with repeated
`https://pypi.org/simple/...` timeouts.

**Diagnosis:** Concurrent package downloads were unreliable on this network;
individual HTTPS requests worked when tested with `curl`.

**Fix:** Set `UV_CONCURRENT_DOWNLOADS=1` and a longer `UV_HTTP_TIMEOUT`,
install in two smaller groups, then rerun `uv sync`. The environment was
verified with import/version checks.

**Lesson:** Keep the lockfile and record the exact workaround; a bare
`uv sync` may still be slow on this network.

### 11.2 Editable install skipped by macOS hidden flag

**Symptom:** `import churnvec` failed even though `uv sync` reported an
editable install. `python -v` printed:

```text
Skipping hidden .pth file: .../__editable__.churnvec-0.1.0.pth
```

`ls -lO` showed the macOS `hidden` flag on both `.pth` files.

**Diagnosis:** Python 3.12.14’s `site.py` skips `.pth` files carrying the
macOS hidden flag. Clearing the flag worked, but a background process
re-applied it within seconds, so the editable install was not reliable.

**Fix:** Stop depending on the `.pth`:

- `[tool.pytest.ini_options] pythonpath = ["src"]` for tests.
- `scripts/run_phase1.py` prepends `src` to `sys.path`.
- `Makefile` exports `PYTHONPATH=$(CURDIR)/src`.
- README documents `export PYTHONPATH="$PWD/src"` for interactive use.

This is an environment quirk, not a packaging failure. It is recorded so a
future contributor does not lose time on it.

### 11.3 NumPy 2.5 rejects `np.fromfile(count=None)`

**Symptom:** `.fvecs`/`.ivecs` round-trip tests failed with
`TypeError: 'NoneType' object cannot be interpreted as an integer`.

**Diagnosis:** The readers passed `count=None` when no limit was requested;
NumPy 2.5 no longer accepts that.

**Fix:** Omit the `count` keyword when `limit is None`; keep the bounded path
for prefixes. Added round-trip tests.

### 11.4 Test helper swapped IDs and distances

**Symptom:** Several HNSW recall tests failed with recall ≈ 0.04, while a
standalone reproduction returned 1.0.

**Diagnosis:** The test helper wrote `_, truth = brute_force_topk(...)`, but
the function returns `(ids, distances)`, so `truth` contained distances.

**Fix:** Changed to `truth, _ = brute_force_topk(...)`. This was a test bug,
not an index bug. The standalone reproduction and the corrected tests both
returned exact results.

### 11.5 `keep_pruned=True` fragmented the SIFT layer-0 graph

**Symptom:** The first SIFT100K run passed recall (0.9881) but reported
**99,673 repair links**. The graph was not naturally connected; the emergency
repair chain had rebuilt it.

**Diagnosis:** On a 15K SIFT build, 7,950 of 15,000 nodes were unreachable
from the entry point. BFS found two large components with only one directed
cross-edge. Both components had high internal degree, so this was not random
node loss. The optional “keep pruned connections” fill was retaining links
that were farther from the source than an already-selected neighbour;
these low-quality links occupied capacity and displaced cross-cluster
bridges during pruning. The effect appeared only once the graph saturated
(after ~10K nodes).

**Experiment (20K SIFT, exact 20K GT):**

| Policy | Build | Unreachable | Repair links | Recall@10, ef=64 |
|---|---:|---:|---:|---:|
| `keep_pruned=True` | 94.2 s | 10,813 | 0 (pre-repair) | 0.9970 |
| `keep_pruned=False` | 46.0 s | 0 | 0 | 0.9952 |

The fragmented variant happened to retain recall because SIFT contains
near-duplicate descriptors in both components, so a reachable twin could
satisfy many queries. That is not a graph property to rely on.

**Fix:** Default `keep_pruned=False`, matching hnswlib’s pruning behaviour.
Rerun SIFT100K:

- `unreachable = 0`
- `repair_links = 0`
- build 241.82 s (vs 435.19 s)
- recall 0.9846 (vs 0.9881)
- p50 0.978 ms (vs 1.018 ms)

The 0.35 pp recall change is far inside the 2 pp gate and buys a naturally
connected graph plus a much faster build.

**Caveat:** `keepPrunedConnections` is an option in the HNSW paper, not a
universally bad idea. It was bad for this data, this insertion order, and
this implementation. The default was chosen from measurements, not from the
claim that the paper is wrong.

### 11.6 Exact duplicate vectors and reachability

**Symptom:** A Hypothesis case with all-zero vectors left one node
unreachable. With 100 identical vectors, fixed-capacity pruning could rotate
out every incoming link for some nodes.

**Diagnosis:** All distances are zero, so tie-breaking favours low slot IDs.
Reciprocal links to new nodes are often rejected when a neighbour list is
full. A node can keep outgoing links but have no incoming link, making it
unreachable from the entry point.

**Fix, in three layers:**

1. Track in-degrees incrementally in `GraphStore`.
2. When adding a link, if the heuristic rejects the new node, leave the
   existing list unchanged; if a link is added, preserve old nodes when
   possible. Try several selected sources before forcing a link.
3. After a bulk `add_items`, run `repair_connectivity()` only if reachability
   is actually broken. It threads a deterministic chain through unreachable
   nodes and reports the number of repair links. Healthy SIFT builds report
   zero.

**Evidence:** A 100-vector all-identical build now ends with 100/100
reachable and a positive repair count; the property test and
`test_large_duplicate_set_repairs_connectivity` cover it. SIFT100K remains at
zero repairs.

**Limitation:** The repair chain is a structural fallback, not a
quality-preserving HNSW construction for adversarial duplicate
distributions. It is documented rather than hidden.

### 11.7 RSS measurement included preallocated index

**Symptom:** The first result JSON reported `rss_delta_mb = 0` for a build
that obviously allocated a 63.8 MB index.

**Diagnosis:** `HNSWIndex` preallocates capacity in its constructor, and the
constructor ran before the RSS monitor started.

**Fix:** Construct the index (and hnswlib’s index) inside the monitor; report
start, peak, and delta with the index allocation included. The theoretical
index size (`memory_bytes`) is reported separately.

### 11.8 SIFT10K has 100 queries, not 10,000

**Symptom:** The first E0 run loaded `queries=100`, which looked like a bug.

**Diagnosis:** The official `siftsmall_query.fvecs` contains 100 queries
(51,600 bytes = 100 × (4 + 128×4)). SIFT1M has 10,000 queries.

**Fix:** None needed; E0 is documented as a 100-query smoke test. The gate
uses SIFT100K/SIFT1M’s 10,000 queries.

### 11.9 Shipped SIFT ground truth is not valid for prefixes

**Symptom:** A naive SIFT100K run could reuse the 1M GT and report wrong
recall.

**Diagnosis:** The official GT is for the exact 1M base; IDs and ranks change
for a 100K prefix.

**Fix:** The harness refuses to reuse shipped GT when `n_base` differs, and
computes exact GT with `brute_force_topk` for the actual snapshot. The 100K
cache is 10,000 × 100 IDs (3.8 MB). The same policy is the basis for future
churn checkpoints.

### 11.10 `hnswlib.__version__` is absent

**Symptom:** `environment()` reported `hnswlib: null` because the module has
no `__version__` attribute.

**Fix:** Use `importlib.metadata.version("hnswlib")`. The benchmark results
in this report were produced before that cosmetic fix; the installed version
was 0.8.0 and this is recorded in §2.

---

## 12. Decisions

### D1 — Python/NumPy core for Weeks 1–5

**Decision:** Use a pure Python/NumPy HNSW implementation through the Week 5
gate; do not start a Rust/native core for this milestone.

**Evidence:** SIFT100K recall 0.9846 at ef=64, p50 0.978 ms, 241.82 s build,
projected 1M build 0.67 h. Neither native-core trigger (3 h build, 10 ms
p50) fired.

**Consequence:** Keep the native-core option open for later churn,
concurrency, and throughput work. hnswlib’s ~6.3× query advantage shows the
performance headroom a native core could recover, but a rewrite now would
not improve the Week 5 gate result.

### D2 — Static SIFT100K as the Week-5 gate

**Decision:** Use SIFT100K, all 10K queries, exact recomputed GT, `M=16`,
`efConstruction=100`, `efSearch=64`, Recall@10.

**Evidence:** This is large enough to expose graph-saturation issues (the
15K–20K fragmentation only appeared above 10K) and small enough to run
repeatedly on an 8 GB laptop.

**Consequence:** SIFT1M remains the eventual primary benchmark, but it was
not built in this milestone.

### D3 — hnswlib-style pruning (`keep_pruned=False`)

**Decision:** Default to diversity pruning without filling rejected
candidates.

**Evidence:** §11.5: 0 unreachable / 0 repairs / 241.82 s build on SIFT100K,
versus 99,673 repairs / 435.19 s with the fill enabled.

**Consequence:** The option remains available for experiments, but it is not
the default.

### D4 — Exact GT per snapshot

**Decision:** Never reuse shipped GT for a different base or after churn.

**Evidence:** Official GT is 1M-specific; SIFT100K needs its own exact GT.

**Consequence:** Future churn experiments will recompute exact GT at
checkpoints (chunked, cached) rather than comparing against stale IDs.

### D5 — Connectivity repair is a fallback, not a feature claim

**Decision:** Keep the repair path for degenerate duplicate sets, report its
link count, and do not present it as solving graph connectivity in general.

**Evidence:** SIFT100K needs zero repairs; the all-identical case needs many.

**Consequence:** The build report always exposes `repair_links`, and the
claims doc forbids “guaranteed connectivity” language.

---

## 13. Claims discipline

### Allowed claims

- “We implemented a from-scratch HNSW index in Python/NumPy for the Week 1–5
  milestone.”
- “On SIFT100K with 10K queries, the index reached Recall@10 = 0.9846 at
  `efSearch=64`, within 0.052 pp of hnswlib at matched parameters.”
- “On this Apple M1 / 8 GB machine, the 100K build took 241.82 s and query
  p50 was 0.978 ms.”
- “We found and fixed a pruning-policy-induced graph fragmentation on SIFT.”
- “The Week-5 gate passed, and the native-core trigger did not fire.”

### Forbidden claims

- “HNSW cannot delete vectors.” (Deletion is not implemented here; that is
  not a statement about HNSW.)
- “We solved deletion / concurrency / tiering / compaction.”
- “First vector database with deletes.”
- “Production-ready.”
- “Faster than hnswlib.” (We have recall parity, but hnswlib is much faster.)
- “We built and validated SIFT1M.” (Only a projection exists.)
- “The repair path guarantees connectivity for arbitrary data.”
- “The shipped SIFT ground truth is valid after inserts/deletes.”
- “The ML compaction scheduler works.” (It is not implemented.)

See `docs/claims.md` for the short version.

---

## 14. Limitations and open questions

1. **No 1M run.** The 0.67 h projection is linear; a real build may be
   slower, and memory/disk may be tight.
2. **Python speed.** hnswlib is ~6.3× faster at query time; this matters for
   a production service even though the Week-5 gate passes.
3. **No churn.** No deletes, updates, tombstones, segment compaction, or
   live-set GT recomputation beyond the static prefix.
4. **No concurrency.** No snapshot registry, lock-free reads, or writer
   coordination.
5. **No tiering.** No hot/warm/cold storage or memory mapping.
6. **Duplicate repair.** The chain fallback is structural; its recall on
   adversarial duplicate distributions is not characterized.
7. **Exact GT cost.** Future 10K × 1M recomputation must be chunked; a naive
   float32 matrix would be 40 GB.
8. **Ties.** Official GT is IDs only; equal-distance ties can reorder IDs and
   slightly perturb recall. The integration test compares distances where
   possible.
9. **RSS vs index size.** RSS includes Python, NumPy, the base vectors, and
   the GT cache; the 63.8 MB figure is the index’s own estimate.
10. **Environment quirk.** Interactive `import churnvec` needs
    `PYTHONPATH=src` because of the macOS `.pth` flag; make targets and
    scripts handle it.

Open decisions for later: exact deadline, whether the canonical architecture
is hot/cold or hot/warm/cold, whether a larger experiment machine is
available, and whether churn/concurrency requirements justify a native core.

---

## 15. Reproducing the result

```bash
# 1. Environment
make setup

# 2. Data (official IRISA TEXMEX, MD5-verified)
make data

# 3. Tests
make test

# 4. SIFT10K smoke
make e0

# 5. SIFT100K Week-5 gate
make e1
```

Artifacts:

```text
results/phase1/e0_siftsmall/result.json
results/phase1/e0_siftsmall/summary.csv
results/phase1/e0_siftsmall/plots/
results/phase1/e1_sift100k/result.json
results/phase1/e1_sift100k/summary.csv
results/phase1/e1_sift100k/plots/
data/processed/gt_sift100k_100000_q10000_k100.npy
logs/e0_siftsmall.log
logs/e1_sift100k.log
```

Committed evidence copies live under `reports/phase1/` (JSON, CSV, plots,
logs). See `reports/README.md` for the evidence map, configurations, and
provenance caveats. Large generated indexes and raw data are gitignored.

To reproduce the exact run manually:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py \
  --dataset sift100k --M 16 --ef-construction 100
```

`keep_pruned=False` is the default. The opt-in `--keep-pruned` flag is kept
only for experiments; it is not used by the Week-5 gate.

---

## 16. Next steps (Week 6+)

1. **SIFT1M static build** on a machine with more disk/RAM, or a measured
   partial build to validate the 0.67 h projection.
2. **Deletion/update layer** using immutable segments or tombstones, with
   exact GT recomputation at checkpoints.
3. **Concurrency** via an atomic snapshot registry; compare against
   per-node/global-lock baselines.
4. **Hot/cold tiering** and the hot/warm/cold ablation.
5. **Churn workloads** (inserts/deletes/updates), recall/latency/memory under
   churn, and compaction cost.
6. **Native core decision** after profiling churn and concurrency; hnswlib’s
   speed gap is the benchmark to beat, not a reason to rewrite blindly.
7. **CI** for the fast test suite; dataset tests remain opt-in.

---

## Appendix A — Key result fields

`result.json` contains:

- `run`: timestamp, command, Python/NumPy/hnswlib versions, host, git state,
  arguments.
- `dataset`: base/query counts, dimension, GT shape, byte sizes.
- `groundtruth`: source (`shipped` or `computed`), cache path, shape.
- `params`: M, efConstruction, efSearch list, gate ef, k, metric, selection,
  keep_pruned, seed.
- `build`: seconds, RSS start/peak/delta, index memory, graph stats.
- `churnvec`: per-ef recall, latency quantiles, QPS.
- `hnswlib`: per-ef recall, latency quantiles, QPS.
- `hnswlib_build`: build time, RSS, index file size.
- `gate`: target, measured recalls, delta, parity, pass/fail.
- `projection`: insertion rate, 1M build projection, p50, triggers.
- `plots`: generated plot paths.

## Appendix B — Evidence map

| Claim | Evidence |
|---|---|
| SIFT100K recall 0.9846 | `reports/phase1/e1_sift100k/result.json`, `summary.csv`, log |
| hnswlib 0.9851 | same result JSON / log |
| Build 241.82 s, p50 0.978 ms | same |
| Zero repair links | `build.stats.repair_links` and log |
| 70 tests pass | pytest output; `reports/phase1/test_run.txt` |
| SIFT10K 0.997 | `reports/phase1/e0_siftsmall/result.json` |
| MD5 verification | `scripts/download_data.sh`, §4 |
| keep_pruned fragmentation | §11.5 table and archived logs |
| Duplicate repair | `tests/unit/test_hnsw.py`, §11.6 |

## Appendix C — Notable implementation files

| File | Responsibility |
|---|---|
| `src/churnvec/core/distance.py` | L2/cosine/IP kernels, norm fast path |
| `src/churnvec/core/graph.py` | array-backed graph, in-degrees, invariants |
| `src/churnvec/core/hnsw.py` | insertion, search, pruning, persistence, repair |
| `src/churnvec/core/bruteforce.py` | chunked exact top-k, recall |
| `src/churnvec/bench/datasets.py` | TEXMEX readers, SIFT adapters, GT cache |
| `src/churnvec/bench/metrics.py` | recall, latency, RSS, JSON |
| `scripts/run_phase1.py` | benchmark harness, gate, plots |
| `scripts/download_data.sh` | MD5-verified SIFT download |
| `tests/` | unit, property, integration coverage |

---

*End of Week 1–5 build report.*
