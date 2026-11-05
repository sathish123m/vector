# Claims policy

This file is the short version of the claims boundaries for the **Week 1–14
milestone (50% of the 28-week plan)**. When in doubt, use the weaker claim.

## Allowed

Static core (Weeks 1–5):

- “We implemented a from-scratch Python/NumPy HNSW index.”
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

Churn layer (Weeks 6–14):

- “Deletes and updates are implemented with immutable sealed segments and
  tombstone sets; the underlying HNSW graph is never mutated by a delete.”
- “On a 10,000-vector SIFT churn workload with 10,000 queries and exact
  ground truth recomputed after every batch, Recall@10 stayed at or above
  0.9973 at `efSearch=64` through 40% deletes, 40% updates, and 16% inserts.”
- “Snapshot isolation is implemented: a reader that captures a snapshot keeps
  seeing a consistent pre-mutation view; 4/4 snapshot-stability checks passed
  byte-for-byte after later deletes, updates, inserts, and flushes.”
- “Concurrent readers share sealed segments safely; HNSW visited state is
  thread-local. Tests cover parallel readers, reader/writer stress, and
  serialised writers.”
- “Compaction is implemented and measured: 7,600 live vectors rebuilt in
  17.56 s, reducing 3 segments to 1, dead fraction from 0.495 to 0, and p50
  from 9.696 ms to 0.864 ms.”
- “The index can be saved to and loaded from a versioned directory format and
  remains writable after loading.”
- “The test suite is 100 tests and passes.”

## Forbidden

- “HNSW cannot delete.” Deletion is not part of the graph algorithm; it is
  implemented above it in this project.
- “First vector database with deletes.”
- “Production-ready.”
- “Faster than hnswlib.” Recall parity is not speed parity; hnswlib was
  ~6.3× faster per query on static SIFT100K and 35×–72× faster per query in
  the 10K churn run.
- “We built SIFT1M.” Only SIFT10K and SIFT100K were built; 1M is a
  projection. The churn experiment uses a 10K base.
- “SIFT100K churn was measured.” The churn result is SIFT10K base with 10K
  queries; the SIFT100K config exists but was not completed on the reference
  host under memory pressure.
- “Guaranteed graph connectivity for arbitrary data.”
- “The shipped SIFT ground truth is valid after inserts/deletes.” It is not;
  every mutated measurement recomputes exact GT on the live set.
- “The ML compaction scheduler works.” Compaction is explicit; the learned
  scheduler is not implemented.
- “The hot/warm/cold architecture is implemented.”
- “Concurrent throughput scales linearly.” Tests prove safety and
  determinism, not parallel speedup.

## Wording for results

Use:

> On a static SIFT100K snapshot with exact recomputed ground truth, the
> Python/NumPy HNSW implementation matched hnswlib’s Recall@10 within
> 0.052 pp at matched parameters, while being substantially slower.
>
> On a 10K SIFT churn workload with exact live-set ground truth, Recall@10
> stayed at or above 0.997 through 40% deletes and 40% updates; compaction
> reduced tombstone pressure from 49.5% to 0 and p50 latency by 11.2×.

Do not shorten that to “we beat hnswlib” or “we match hnswlib” without the
metric and the speed caveat.

## Scope reminder

Implemented: data loaders, brute force, HNSW insert/search/save/load,
invariants, tests, benchmark harness, SIFT10K/SIFT100K results, deletion,
updates, tombstone segments, append-only buffer, snapshots, concurrent
readers, flush, compaction, persistence, and a 10K churn experiment.

Not implemented: hot/warm/cold tiering, learned compaction scheduler,
API/service, crash-consistent journaling, finer-grained writer concurrency,
SIFT1M run, SIFT100K churn run, text/GIST benchmarks.
