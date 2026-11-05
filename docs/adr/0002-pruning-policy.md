# ADR 0002 — Default to hnswlib-style pruning (`keep_pruned=False`)

- Status: Accepted
- Date: 2026-10-27

## Context

HNSW’s neighbour-selection heuristic can either keep only the diverse
selected candidates or fill remaining slots with rejected candidates (the
paper’s optional `keepPrunedConnections`). This implementation initially
defaulted to filling.

On SIFT100K, the fill caused the layer-0 graph to fragment at scale:

- 15K SIFT build: 7,950 / 15,000 nodes unreachable; two large components and
  one directed cross-edge.
- 100K SIFT build: 99,673 emergency repair links after the bulk build.
- 20K SIFT: `keep_pruned=True` built in 94.2 s with 10,813 unreachable;
  `keep_pruned=False` built in 46.0 s with 0 unreachable.

The fragmented variant still showed high recall on SIFT because the data
contains near-duplicate descriptors in both components, so a reachable twin
could satisfy many queries. That is not a property to rely on.

## Decision

Default to `keep_pruned=False`, matching hnswlib’s pruning behaviour. Keep
the option available for experiments, but do not use it for the Week-5 gate.

## Evidence

Final SIFT100K run with `keep_pruned=False`:

- 0 unreachable, 0 repair links.
- Recall@10 0.9846 at ef=64.
- Build 241.82 s.
- p50 0.978 ms.

The earlier `keep_pruned=True` run had recall 0.9881 but required 99,673
repair links and built in 435.19 s.

## Consequences

- A naturally connected graph on the benchmark data.
- Faster construction and lower memory churn.
- The optional fill remains a configurable experiment, not a default.
- This decision is data- and implementation-specific; it is not a claim that
  the paper’s option is universally wrong.
