# ADR 0004 — Immutable sealed segments with tombstones

- Status: Accepted
- Date: 2026-11-04

## Context

HNSW has no cheap delete. Removing a node from a multi-layer proximity graph
requires repairing every incoming edge; doing that per delete is both slow and
likely to fragment the graph. Updates have the same problem, and a churn
workload can issue thousands of mutations per second.

The index also has to keep old readers correct: a query that started before a
delete must not observe a half-mutated graph.

## Decision

The mutable index is a sequence of **immutable sealed segments** plus one
append-only raw buffer.

- A sealed segment wraps an `HNSWIndex` that is never modified after it is
  built.
- A delete records the external id in a `frozenset` tombstone set attached to
  a new `Segment` view. The HNSW index object is shared with the previous view.
- An update is a tombstone of the old occurrence plus an append of the new
  vector in the buffer. Re-adding a deleted id is allowed.
- The raw buffer stores ids and vectors in preallocated NumPy arrays. A
  `BufferView` freezes `size` and the set of dead sequence numbers, so appends
  never change an existing view.

`Segment.with_dead` only records ids that are actually present, which keeps
`live_count = size - len(dead)` O(1) on the search path.

## Consequences

- Deletes are O(number of segments), not O(graph degree).
- Search over-fetches around tombstones and filters them before returning.
- Physical storage grows with churn until compaction; `dead_fraction` makes
  the cost visible.
- Old snapshots retain their old tombstone sets because frozensets are
  immutable and segment views are value objects.
- Compaction is the only operation that rebuilds a graph; it is explicit and
  measurable (ADR 0006).
