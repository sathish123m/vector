# Benchmark evidence — phase 1 (Weeks 1–5)

This directory holds the committed evidence for the Week-5 static benchmark.
The raw run outputs under `results/` and `logs/` are gitignored; the files
here are the durable copies referenced by `BUILD_REPORT.md`.

## Contents

| Path | What it is |
|---|---|
| `e0_siftsmall/result.json` | Full E0 result metadata, numbers, gate, projection |
| `e0_siftsmall/summary.csv` | Per-ef recall/latency table for both engines |
| `e0_siftsmall/run.log` | Console transcript of the E0 run |
| `e0_siftsmall/plots/` | Recall-vs-ef and p50-latency-vs-ef plots |
| `e1_sift100k/result.json` | Full E1 result metadata, numbers, gate, projection |
| `e1_sift100k/summary.csv` | Per-ef recall/latency table for both engines |
| `e1_sift100k/run.log` | Console transcript of the E1 gate run |
| `e1_sift100k/plots/` | Recall-vs-ef and p50-latency-vs-ef plots |
| `test_run.txt` | Final full pytest transcript (70 passed) |
| `archive/` | Logs from the superseded `keep_pruned=True` runs |

## Configurations

Both runs used `M=16`, `efConstruction=100`, `keep_pruned=False`, seed 42,
metric `l2`, and `k=10`.

- **E0** — SIFT10K (`siftsmall`), 10,000 base, 100 queries, shipped ground
  truth. Smoke test only.
- **E1** — SIFT100K (prefix of SIFT1M), 100,000 base, 10,000 queries, exact
  ground truth recomputed for that prefix and cached as
  `data/processed/gt_sift100k_100000_q10000_k100.npy`. This is the Week-5
  gate.

Commands are in `BUILD_REPORT.md` §10 and in the `Makefile` (`make e0`,
`make e1`).

## Headline numbers (E1)

| Engine | Recall@10, ef=64 | p50 | Build |
|---|---:|---:|---:|
| ChurnVec | 0.9846 | 0.978 ms | 241.82 s |
| hnswlib 0.8.0 | 0.9851 | 0.155 ms | 20.57 s |

Gate: recall ≥ 0.90 and within 2 pp of hnswlib → pass (0.052 pp gap).
Graph: 100,000 / 100,000 reachable, 0 repair links.

## Provenance caveats

These are properties of when the evidence was captured, not errors in the
numbers. They are kept visible rather than patched into the JSON:

- `run.hnswlib` is `null` in both archived `result.json` files. The capture
  code read `hnswlib.__version__`, which does not exist in the 0.8.0 wheel.
  The installed version at run time was **0.8.0** (`importlib.metadata`);
  the helper now uses `importlib.metadata.version("hnswlib")`. See
  `BUILD_REPORT.md` §11.10.
- `run.git.commit` is `null` and `run.git.dirty` is `true` because the
  repository had no commits yet when the benchmarks ran. The evidence was
  produced from the working tree that this milestone commits; the exact code
  is the code in this commit.
- Timestamps, machine, Python/NumPy versions, and the full argument list are
  recorded in each `result.json` under `run`.
- `archive/` contains the earlier `keep_pruned=True` runs: recall 0.9881 but
  99,673 repair links and 435.19 s build. They are retained to document the
  pruning experiment described in `BUILD_REPORT.md` §11.5, not as the gate
  result.

## Reproducing

```bash
make setup
make data
make test
make e1
```

`make e1` rewrites `results/phase1/e1_sift100k/`. To refresh this evidence
directory, copy `result.json`, `summary.csv`, `run.log`, and `plots/` from
the new run. Expect small latency differences between runs; recall should be
stable, since construction is seeded and single-threaded.
