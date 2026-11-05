# ADR 0006 — Explicit compaction and directory persistence

- Status: Accepted
- Date: 2026-11-04

## Context

Tombstones and append-only segments make deletes and updates cheap, but
physical storage and per-query work grow with churn. A workload that deletes
40% of its vectors should not keep searching 40% dead nodes forever. The
index also needs to survive a process restart without losing tombstones or the
pending buffer.

## Decision

Compaction is explicit and observable:

- `flush()` seals the live buffer into a new HNSW segment.
- `compact()` flushes, collects every live vector from every segment and the
  buffer, builds one fresh HNSW segment, and atomically publishes a snapshot
  containing only that segment.
- `maybe_compact(dead_fraction, min_vectors)` is an opt-in policy hook. The
  default benchmark calls `compact()` at a controlled checkpoint so the cost
  is measured rather than hidden inside a query.
- `dead_fraction = (physical - live) / physical` reports tombstone pressure.

Persistence writes a directory:

```text
manifest.json       format, parameters, generation, live_count, tombstones
segment-0000.npz    HNSW payload (reuses the Week-5 format)
...
buffer.npz          pending ids, vectors, and dead sequence numbers
```

The manifest is written last. Loading reconstructs segment views, the buffer
index, and the current snapshot; a loaded index remains writable.

## Consequences

- Compaction costs one full HNSW build over the live set. It is I/O- and
  CPU-heavy and must be scheduled, not hidden.
- Old snapshots remain valid across compaction because the new segment is a
  new object and old segments are not mutated.
- The directory format is versioned (`churnvec-churn-v1`). There is no
  crash-consistent journal yet; a torn directory should be treated as invalid
  rather than silently repaired.
- The ML scheduler from the original plan can later choose compaction
  checkpoints using `dead_fraction`, segment count, and query latency.
