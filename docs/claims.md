# Claims policy

This file is the short version of the claims boundaries for the Week 1–5
milestone. When in doubt, use the weaker claim.

## Allowed

- “We implemented a from-scratch Python/NumPy HNSW index for the Week 1–5
  milestone.”
- “On the static SIFT100K benchmark (10,000 queries), ChurnVec reached
  Recall@10 = 0.9846 at `efSearch=64`, within 0.052 percentage points of
  hnswlib at matched `M=16`, `efConstruction=100`.”
- “On this Apple M1 / 8 GB machine, the 100K build took 241.82 s, query p50
  was 0.978 ms, and the index used 63.8 MB by its internal estimate.”
- “The Week-5 recall and native-core gates passed, with a 0.67 h linear
  projection for a 1M build.”
- “We found a pruning-policy-induced layer-0 fragmentation on SIFT and fixed
  it by disabling the optional keep-pruned fill.”
- “Exact duplicate vectors are supported with an explicit bulk-build
  connectivity repair path.”

## Forbidden

- “HNSW cannot delete.” Deletion is not implemented in this milestone; that
  says nothing about what HNSW can do.
- “We solved deletion, concurrency, tiering, or compaction.”
- “First vector database with deletes.”
- “Production-ready.”
- “Faster than hnswlib.” Recall parity is not speed parity; hnswlib was
  ~6.3× faster per query and ~11.8× faster to build here.
- “We built SIFT1M.” Only SIFT10K and SIFT100K were built; 1M is a
  projection.
- “Guaranteed graph connectivity for arbitrary data.”
- “The shipped SIFT ground truth is valid after inserts/deletes.”
- “The ML compaction scheduler works.”
- “The hot/warm/cold architecture is implemented.” Only the Week 1–5 static
  core exists.

## Wording for results

Use:

> On a static SIFT100K snapshot with exact recomputed ground truth, the
> Python/NumPy HNSW implementation matched hnswlib’s Recall@10 within
> 0.052 pp at matched parameters, while being substantially slower.

Do not shorten that to “we beat hnswlib” or “we match hnswlib” without the
metric and the speed caveat.

## Scope reminder

Implemented: data loaders, brute force, HNSW insert/search/save/load,
invariants, tests, benchmark harness, SIFT10K/SIFT100K results.

Not implemented: deletes, updates, churn workloads, concurrency, snapshots,
tiering, compaction, scheduler, API/service, SIFT1M run, text/GIST
benchmarks.
