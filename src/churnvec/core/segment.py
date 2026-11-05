"""Immutable segments, append-only buffer views, and reader snapshots.

ChurnVec separates two kinds of storage:

* a **sealed segment** wraps an :class:`~churnvec.core.hnsw.HNSWIndex` that is
  never mutated after it is built.  Deletes are recorded as tombstones in a
  frozenset, so publishing a new snapshot is cheap and old snapshots keep
  their old tombstone sets.
* an **append-only buffer** holds recently inserted vectors as a flat NumPy
  matrix plus a parallel id array.  A view freezes ``size`` and the set of
  dead sequence numbers.  Appending never rewrites earlier slots, so a view
  stays valid while writers keep appending.

A :class:`Snapshot` binds one tuple of segments to one buffer view.  Readers
capture a snapshot once and can then run lock-free searches while writers
build the next snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .distance import Metric, distance_batch
from .hnsw import HNSWIndex


def _validate_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    vec = np.ascontiguousarray(vector, dtype=np.float32)
    if vec.shape != (dim,):
        raise ValueError(f"expected vector shape ({dim},), got {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError("vector contains NaN or infinity")
    return vec


@dataclass(frozen=True)
class Segment:
    """A sealed HNSW index plus the external ids tombstoned in this view."""

    index: HNSWIndex
    dead: frozenset[int] = frozenset()

    @property
    def size(self) -> int:
        """Number of physical vectors in the segment, including dead ones."""
        return self.index.size

    @property
    def dead_count(self) -> int:
        # ``with_dead`` only records ids that exist in this segment, so the
        # tombstone set is exact and this stays O(1) on the search path.
        return len(self.dead)

    @property
    def live_count(self) -> int:
        return self.size - len(self.dead)

    def contains(self, external_id: int) -> bool:
        if external_id in self.dead:
            return False
        return int(external_id) in self.index.store.id_to_slot

    def with_dead(self, external_ids: set[int] | frozenset[int]) -> "Segment":
        """Return a new view with extra tombstones, sharing the HNSW index."""
        extra = {int(i) for i in external_ids if int(i) in self.index.store.id_to_slot}
        if not extra:
            return self
        return Segment(index=self.index, dead=self.dead | frozenset(extra))

    def search_ids(self, query: np.ndarray, k: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k live ids in this segment, over-fetching around tombstones."""
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.ascontiguousarray(query, dtype=np.float32)
        if self.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if not self.dead:
            fetch = min(self.size, max(int(k), int(ef)))
            return self.index.search_ids(query, k=fetch, ef=max(int(ef), fetch))

        fetch = min(self.size, max(int(k), int(ef)))
        while True:
            ids, distances = self.index.search_ids(query, k=fetch, ef=max(int(ef), fetch))
            if not self.dead or ids.size == 0:
                return ids[:k], distances[:k]
            keep = np.fromiter(
                (int(external_id) not in self.dead for external_id in ids.tolist()),
                dtype=bool,
                count=ids.size,
            )
            ids = ids[keep]
            distances = distances[keep]
            if ids.size >= k:
                return ids[:k], distances[:k]
            if fetch >= self.size:
                return ids, distances
            fetch = min(self.size, max(fetch * 2, k))

    def iter_live(self) -> Iterator[tuple[int, np.ndarray]]:
        store = self.index.store
        ids = store.ids[: self.index.size]
        vectors = store.vectors[: self.index.size]
        for slot in range(self.index.size):
            external_id = int(ids[slot])
            if external_id in self.dead:
                continue
            yield external_id, vectors[slot]


class BufferView:
    """A frozen prefix of the append-only raw insert buffer."""

    __slots__ = ("ids", "vectors", "size", "dead_seq", "metric", "_live_count")

    def __init__(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        size: int,
        dead_seq: frozenset[int],
        metric: Metric,
        live_count: int | None = None,
    ) -> None:
        self.ids = ids
        self.vectors = vectors
        self.size = int(size)
        self.dead_seq = dead_seq
        self.metric = metric
        if live_count is None:
            seen: set[int] = set()
            for seq in range(self.size - 1, -1, -1):
                if seq in self.dead_seq:
                    continue
                seen.add(int(self.ids[seq]))
            live_count = len(seen)
        self._live_count = int(live_count)

    @property
    def live_count(self) -> int:
        return self._live_count

    def live_records(self) -> list[tuple[int, int]]:
        """``(external_id, sequence)`` for the newest live record of each id."""
        seen: set[int] = set()
        records: list[tuple[int, int]] = []
        for seq in range(self.size - 1, -1, -1):
            if seq in self.dead_seq:
                continue
            external_id = int(self.ids[seq])
            if external_id in seen:
                continue
            seen.add(external_id)
            records.append((external_id, seq))
        records.reverse()
        return records

    def contains(self, external_id: int) -> bool:
        external_id = int(external_id)
        for seq in range(self.size - 1, -1, -1):
            if seq in self.dead_seq:
                continue
            if int(self.ids[seq]) == external_id:
                return True
        return False

    def search_ids(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        records = self.live_records()
        if not records:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        seqs = np.fromiter((seq for _, seq in records), dtype=np.int64, count=len(records))
        ext_ids = np.fromiter((external_id for external_id, _ in records), dtype=np.int64, count=len(records))
        distances = distance_batch(self.vectors[seqs], query, self.metric)
        take = min(k, distances.size)
        if take < distances.size:
            order = np.argpartition(distances, take - 1)[:take]
        else:
            order = np.arange(distances.size)
        order = order[np.argsort(distances[order], kind="stable")]
        return ext_ids[order], distances[order].astype(np.float32, copy=False)

    def iter_live(self) -> Iterator[tuple[int, np.ndarray]]:
        for external_id, seq in self.live_records():
            yield external_id, self.vectors[seq]


@dataclass(frozen=True)
class Snapshot:
    """A consistent, immutable view of segments plus the mutable buffer prefix."""

    dim: int
    metric: Metric
    segments: tuple[Segment, ...]
    buffer: BufferView
    generation: int
    live_count: int

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def contains(self, external_id: int) -> bool:
        external_id = int(external_id)
        if self.buffer.contains(external_id):
            return True
        return any(segment.contains(external_id) for segment in self.segments)

    def iter_live(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield every live vector once; newer sources win on duplicate ids."""
        seen: set[int] = set()
        for segment in self.segments:
            for external_id, vector in segment.iter_live():
                if external_id in seen:
                    continue
                seen.add(external_id)
                yield external_id, vector
        for external_id, vector in self.buffer.iter_live():
            if external_id in seen:
                continue
            seen.add(external_id)
            yield external_id, vector

    def live_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        records = list(self.iter_live())
        if not records:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.dim), dtype=np.float32),
            )
        ids = np.fromiter((i for i, _ in records), dtype=np.int64, count=len(records))
        vectors = np.stack([v for _, v in records]).astype(np.float32, copy=False)
        return ids, vectors

    def search_ids(self, query: np.ndarray, k: int = 10, ef: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Merge segment and buffer candidates into one ranked result."""
        if k <= 0:
            raise ValueError("k must be positive")
        q = _validate_vector(query, self.dim)
        parts: list[tuple[np.ndarray, np.ndarray]] = []

        per_source = max(int(k), int(ef))
        for segment in self.segments:
            if segment.live_count == 0:
                continue
            ids, distances = segment.search_ids(
                q, k=min(segment.live_count, per_source), ef=ef
            )
            if ids.size:
                parts.append((ids, distances))
        if self.buffer.live_count:
            ids, distances = self.buffer.search_ids(q, k=min(self.buffer.live_count, per_source))
            if ids.size:
                parts.append((ids, distances))

        if not parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        ids = np.concatenate([part[0] for part in parts])
        distances = np.concatenate([part[1] for part in parts])
        order = np.argsort(distances, kind="stable")

        kept_ids: list[int] = []
        kept_dists: list[float] = []
        seen: set[int] = set()
        for position in order.tolist():
            external_id = int(ids[position])
            if external_id in seen:
                continue
            seen.add(external_id)
            kept_ids.append(external_id)
            kept_dists.append(float(distances[position]))
            if len(kept_ids) >= k:
                break
        return (
            np.asarray(kept_ids, dtype=np.int64),
            np.asarray(kept_dists, dtype=np.float32),
        )


__all__ = ["BufferView", "Segment", "Snapshot"]
