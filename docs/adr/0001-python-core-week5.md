# ADR 0001 — Use a Python/NumPy core through the Week-5 gate

- Status: Accepted
- Date: 2026-10-27

## Context

The project plan requires a from-scratch HNSW implementation and a Week-5
gate on SIFT100K: Recall@10 ≥ 0.90 at `efSearch=64`, within roughly two
points of hnswlib. The plan also defines a native-core trigger: start a
native core if the 1M build is projected above three hours or query p50
exceeds 10 ms.

A pure Python/NumPy implementation is the fastest way to get a correct,
inspectable HNSW and a reusable experimental harness. The risk is that it is
too slow for the eventual 1M/churn/concurrency work.

## Decision

Implement the Week 1–5 index in Python/NumPy and defer the native-core
decision until after the static gate.

## Evidence

SIFT100K, 100K × 128d, 10K queries, `M=16`, `efConstruction=100`,
`efSearch=64`, `keep_pruned=False`:

- Recall@10: 0.9846 (hnswlib 0.9851).
- Build: 241.82 s (413.5 inserts/s).
- Query p50/p95: 0.978 / 1.142 ms.
- Theoretical index memory: 63.8 MB.
- Linear 1M build projection: 0.672 h (40.3 min).
- Native-core triggers: none.

hnswlib was ~6.3× faster per query and ~11.8× faster to build on the same
machine, so Python is not performance-parity with C++.

## Consequences

- The Week-5 gate passes without a rewrite.
- Keep the core small and algorithm-focused so a later native port can use
  it as the reference implementation and test oracle.
- Revisit the decision after a real 1M build and after churn/concurrency
  experiments. Profile those workloads before committing to Rust/C++.
- Do not describe the Python core as production-ready.
