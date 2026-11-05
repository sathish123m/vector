# ADR 0005 — Atomic snapshots and thread-local search state

- Status: Accepted
- Date: 2026-11-04

## Context

Real-time indexing has two reader requirements that conflict with a naive
mutable graph:

1. A query must see one consistent live set even while a writer deletes or
   updates vectors.
2. Readers must not block on writers, and writers must not corrupt an
   in-flight search.

The Week 1–5 HNSW used mutable per-index visited arrays, so two concurrent
searches on the same index would race on the visited tokens.

## Decision

`ChurnIndex` publishes a new immutable `Snapshot` after every mutation:

- `Snapshot` contains `(segments, buffer_view, generation, live_count)`.
- Writers hold one `threading.RLock`, build the next segment tuple and buffer
  view, then assign `self._snapshot` in one atomic reference store.
- Readers call `index.snapshot()` once and search that object without a lock.
- `HNSWIndex` stores its visited-token array in `threading.local()`. Search is
  read-only, so concurrent readers share a sealed index safely; only the
  scratch array is per-thread.

The buffer view is safe because the writer never rewrites slots below the
view's frozen `size`, and flushes allocate fresh arrays instead of reusing the
old ones.

## Consequences

- A reader that captured an old snapshot keeps seeing pre-mutation data until
  it asks for a new snapshot.
- Snapshot publication is O(1) in graph size; only the tombstone frozenset and
  the small buffer view are new.
- `_distance_calls` and `_visited_nodes` are diagnostic counters and are not
  atomic under concurrent reads. Search correctness never depends on them.
- Old snapshots keep old segment/buffer objects alive. Callers that retain
  snapshots indefinitely also retain the associated memory.
- The writer lock is coarse. It serialises mutations and segment builds, but
  not reads. Finer-grained per-segment writer locks are later work.
