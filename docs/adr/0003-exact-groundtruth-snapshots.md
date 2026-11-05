# ADR 0003 — Recompute exact ground truth per snapshot

- Status: Accepted
- Date: 2026-10-27

## Context

The official SIFT ground truth contains 0-based IDs for the exact original
base, using squared L2. It is not valid for:

- a prefix such as SIFT100K;
- a base after inserts or deletes;
- a different metric.

Future churn experiments will mutate the live set, so stale IDs would
silently corrupt recall measurements.

## Decision

The harness uses shipped GT only when the benchmark base is exactly the
shipped base. For prefixes and mutated snapshots it computes exact top-k over
the actual live base, caches the IDs, and records the source in the result
JSON.

## Implementation

- `load_dataset` returns shipped GT only for the unmodified set.
- `resolve_groundtruth` in `scripts/run_phase1.py` checks base size and
  recomputes when needed.
- `load_or_compute_groundtruth` validates cache shape and recomputes on a
  mismatch.
- Exact search is chunked: a full 10K × 1M float32 matrix would be 40 GB.
- The SIFT100K cache is `(10000, 100)` int32, 3.8 MB.

## Consequences

- Recall numbers are tied to a specific live-set snapshot.
- Churn checkpoints must pay the exact-GT cost or reuse a valid cache.
- The shipped SIFT1M GT remains usable for an unmodified 1M build, but not
  for a prefix or after churn.
- The result JSON must record `groundtruth.source` and the cache path; this
  is part of the evidence, not an optional detail.
