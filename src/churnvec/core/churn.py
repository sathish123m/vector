"""Churn-aware HNSW index: deletes, updates, snapshots, flush, and compaction.

The index is a sequence of immutable sealed segments plus one append-only
buffer of raw vectors.  A single atomic snapshot reference gives readers a
consistent view without locking:

    snapshot = index.snapshot()
    ids, distances = snapshot.search_ids(query, k=10, ef=64)

Writers serialise on ``_lock``.  Each mutation publishes a new immutable
:class:`~churnvec.core.segment.Snapshot`; readers that grabbed the previous
snapshot keep seeing the pre-mutation data until they ask for a new one.
Deletes are tombstones, so they are O(number of segments) and never rebuild
a graph.  ``compact()`` merges all live vectors into one fresh segment.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .distance import Metric
from .exceptions import DimensionMismatchError, DuplicateIDError
from .hnsw import HNSWIndex, Neighbor
from .segment import BufferView, Segment, Snapshot


class ChurnIndex:
    """A mutable vector index with snapshot-isolated readers."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 100,
        metric: Metric = "l2",
        seed: int = 42,
        selection: str = "heuristic",
        keep_pruned: bool = False,
        max_level_cap: int = 32,
        flush_threshold: int = 2048,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if flush_threshold <= 0:
            raise ValueError("flush_threshold must be positive")
        self.dim = int(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.metric: Metric = metric
        self.seed = int(seed)
        self.selection = selection
        self.keep_pruned = bool(keep_pruned)
        self.max_level_cap = int(max_level_cap)
        self.flush_threshold = int(flush_threshold)

        self._lock = threading.RLock()
        self._buffer_ids = np.empty(64, dtype=np.int64)
        self._buffer_vectors = np.empty((64, self.dim), dtype=np.float32)
        self._buffer_size = 0
        self._buffer_dead: set[int] = set()
        self._buffer_index: dict[int, list[int]] = {}
        self._buffer_live = 0
        self._next_id = 0
        self._generation = 0
        self._snapshot = Snapshot(
            dim=self.dim,
            metric=self.metric,
            segments=(),
            buffer=BufferView(
                ids=self._buffer_ids,
                vectors=self._buffer_vectors,
                size=0,
                dead_seq=frozenset(),
                metric=self.metric,
            ),
            generation=0,
            live_count=0,
        )

    # ------------------------------------------------------------------
    # Snapshot access
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        """Return the current immutable snapshot.  This is the reader entry point."""
        return self._snapshot

    def __len__(self) -> int:
        return self._snapshot.live_count

    def __contains__(self, external_id: int) -> bool:
        return self._snapshot.contains(int(external_id))

    @property
    def generation(self) -> int:
        return self._snapshot.generation

    # ------------------------------------------------------------------
    # Internal publication
    # ------------------------------------------------------------------
    def _buffer_view(self) -> BufferView:
        return BufferView(
            ids=self._buffer_ids,
            vectors=self._buffer_vectors,
            size=self._buffer_size,
            dead_seq=frozenset(self._buffer_dead),
            metric=self.metric,
            live_count=self._buffer_live,
        )

    def _publish(
        self,
        segments: tuple[Segment, ...],
        live_count: int,
        buffer: BufferView | None = None,
    ) -> None:
        self._generation += 1
        self._snapshot = Snapshot(
            dim=self.dim,
            metric=self.metric,
            segments=segments,
            buffer=buffer if buffer is not None else self._buffer_view(),
            generation=self._generation,
            live_count=int(live_count),
        )

    def _ensure_buffer_capacity(self, extra: int = 1) -> None:
        needed = self._buffer_size + extra
        capacity = self._buffer_ids.size
        if needed <= capacity:
            return
        new_capacity = max(64, capacity)
        while new_capacity < needed:
            new_capacity *= 2
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_vectors = np.empty((new_capacity, self.dim), dtype=np.float32)
        new_ids[: self._buffer_size] = self._buffer_ids[: self._buffer_size]
        new_vectors[: self._buffer_size] = self._buffer_vectors[: self._buffer_size]
        self._buffer_ids = new_ids
        self._buffer_vectors = new_vectors

    def _append_buffer(self, external_id: int, vector: np.ndarray) -> None:
        self._ensure_buffer_capacity(1)
        seq = self._buffer_size
        self._buffer_ids[seq] = int(external_id)
        self._buffer_vectors[seq] = vector
        self._buffer_size += 1
        self._buffer_index.setdefault(int(external_id), []).append(seq)
        self._next_id = max(self._next_id, int(external_id) + 1)

    def _tombstone(self, external_id: int) -> tuple[tuple[Segment, ...], bool, bool]:
        """Tombstone one id everywhere.

        Returns ``(new_segments, found, buffer_had_live)``.
        """
        external_id = int(external_id)
        found = False
        new_segments: list[Segment] = []
        for segment in self._snapshot.segments:
            if segment.contains(external_id):
                found = True
                new_segments.append(segment.with_dead({external_id}))
            else:
                new_segments.append(segment)

        buffer_had_live = False
        sequences = self._buffer_index.pop(external_id, [])
        for seq in sequences:
            if seq in self._buffer_dead:
                continue
            self._buffer_dead.add(seq)
            buffer_had_live = True
            found = True
        return tuple(new_segments), found, buffer_had_live

    def _validate(self, vector: np.ndarray) -> np.ndarray:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatchError(f"expected vector shape ({self.dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector contains NaN or infinity")
        return vec

    def _build_segment(self, ids: np.ndarray, vectors: np.ndarray, generation: int) -> Segment:
        index = HNSWIndex(
            dim=self.dim,
            M=self.M,
            ef_construction=self.ef_construction,
            metric=self.metric,
            seed=self.seed + generation,
            capacity=max(int(ids.size), 1),
            selection=self.selection,
            keep_pruned=self.keep_pruned,
            max_level_cap=self.max_level_cap,
        )
        index.add_items(vectors, ids)
        return Segment(index=index)

    def _reset_buffer(self) -> None:
        self._buffer_ids = np.empty(64, dtype=np.int64)
        self._buffer_vectors = np.empty((64, self.dim), dtype=np.float32)
        self._buffer_size = 0
        self._buffer_dead = set()
        self._buffer_index = {}
        self._buffer_live = 0

    def _buffer_live_count(self) -> int:
        return self._buffer_live

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add(self, external_id: int, vector: np.ndarray) -> None:
        """Insert a new id.  Raises if the id is already live."""
        vec = self._validate(vector)
        with self._lock:
            if self._snapshot.contains(int(external_id)):
                raise DuplicateIDError(f"duplicate external id: {external_id}")
            self._append_buffer(int(external_id), vec)
            self._buffer_live += 1
            self._publish(self._snapshot.segments, self._snapshot.live_count + 1)
            if self._buffer_live >= self.flush_threshold:
                self._flush_locked()

    def add_many(
        self,
        vectors: np.ndarray,
        ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise DimensionMismatchError(f"expected (n, {self.dim}) vectors, got {data.shape}")
        if ids is None:
            id_array = np.arange(self._next_id, self._next_id + data.shape[0], dtype=np.int64)
        else:
            id_array = np.asarray(ids, dtype=np.int64)
            if id_array.shape != (data.shape[0],):
                raise ValueError("ids length must match the number of vectors")
        for external_id, vector in zip(id_array, data):
            self.add(int(external_id), vector)

    def upsert(self, external_id: int, vector: np.ndarray) -> bool:
        """Insert or replace an id.  Returns ``True`` when an old value existed."""
        vec = self._validate(vector)
        external_id = int(external_id)
        with self._lock:
            segments, found, buffer_had_live = self._tombstone(external_id)
            self._append_buffer(external_id, vec)
            if not buffer_had_live:
                self._buffer_live += 1
            live_count = self._snapshot.live_count + (0 if found else 1)
            self._publish(segments, live_count)
            if self._buffer_live >= self.flush_threshold:
                self._flush_locked()
            return found

    def delete(self, external_id: int) -> bool:
        """Tombstone an id.  Returns ``False`` when it was not live."""
        external_id = int(external_id)
        with self._lock:
            segments, found, buffer_had_live = self._tombstone(external_id)
            if not found:
                return False
            if buffer_had_live:
                self._buffer_live -= 1
            self._publish(segments, self._snapshot.live_count - 1)
            return True

    def delete_many(self, external_ids: Iterable[int]) -> int:
        removed = 0
        with self._lock:
            for external_id in external_ids:
                if self.delete(int(external_id)):
                    removed += 1
        return removed

    # ------------------------------------------------------------------
    # Flush and compaction
    # ------------------------------------------------------------------
    def flush(self) -> int:
        """Seal the live buffer into a new HNSW segment.  Returns vectors sealed."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        records = self._buffer_view().live_records()
        if not records:
            if self._buffer_size or self._buffer_dead:
                self._reset_buffer()
                self._publish(self._snapshot.segments, self._snapshot.live_count)
            return 0
        ids = np.fromiter((i for i, _ in records), dtype=np.int64, count=len(records))
        vectors = np.stack([self._buffer_vectors[seq] for _, seq in records]).astype(np.float32)
        segment = self._build_segment(ids, vectors, self._generation + 1)
        segments = self._snapshot.segments + (segment,)
        self._reset_buffer()
        self._publish(segments, self._snapshot.live_count)
        return int(ids.size)

    def compact(self) -> int:
        """Merge every live vector into one fresh segment.  Returns vector count."""
        with self._lock:
            if self._buffer_size:
                self._flush_locked()
            snapshot = self._snapshot
            ids, vectors = snapshot.live_arrays()
            if ids.size == 0:
                self._reset_buffer()
                self._publish((), 0)
                return 0
            segment = self._build_segment(ids, vectors, self._generation + 1)
            self._reset_buffer()
            self._publish((segment,), int(ids.size))
            return int(ids.size)

    @property
    def dead_fraction(self) -> float:
        snapshot = self._snapshot
        physical = sum(segment.size for segment in snapshot.segments) + snapshot.buffer.size
        if physical == 0:
            return 0.0
        live = snapshot.live_count
        return max(0.0, (physical - live) / physical)

    def maybe_compact(self, dead_fraction: float = 0.3, min_vectors: int = 1024) -> bool:
        """Compact when tombstones exceed a fraction of physical storage."""
        with self._lock:
            if self._snapshot.live_count < min_vectors:
                return False
            if self.dead_fraction < dead_fraction:
                return False
            self.compact()
            return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_ids(self, query: np.ndarray, k: int = 10, ef: int = 64) -> tuple[np.ndarray, np.ndarray]:
        snapshot = self._snapshot
        return snapshot.search_ids(query, k=k, ef=ef)

    def search(self, query: np.ndarray, k: int = 10, ef: int = 64) -> list[Neighbor]:
        ids, distances = self.search_ids(query, k=k, ef=ef)
        return [
            Neighbor(id=int(external_id), distance=float(distance))
            for external_id, distance in zip(ids, distances)
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        snapshot = self._snapshot
        segments = []
        for segment in snapshot.segments:
            segments.append(
                {
                    "size": segment.size,
                    "dead": segment.dead_count,
                    "live": segment.live_count,
                    "memory_bytes": segment.index.store.memory_bytes(),
                    "max_level": segment.index.max_level,
                }
            )
        buffer_bytes = int(self._buffer_ids.nbytes + self._buffer_vectors.nbytes)
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "generation": snapshot.generation,
            "live_count": snapshot.live_count,
            "physical_count": sum(item["size"] for item in segments) + snapshot.buffer.size,
            "segment_count": len(segments),
            "segments": segments,
            "buffer_size": snapshot.buffer.size,
            "buffer_live": snapshot.buffer.live_count,
            "buffer_dead": len(snapshot.buffer.dead_seq),
            "dead_fraction": self.dead_fraction,
            "memory_bytes": sum(item["memory_bytes"] for item in segments) + buffer_bytes,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the current snapshot to a directory and return it."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        snapshot = self._snapshot
        segment_entries = []
        for index, segment in enumerate(snapshot.segments):
            name = f"segment-{index:04d}.npz"
            segment.index.save(path / name)
            segment_entries.append(
                {
                    "file": name,
                    "dead": sorted(int(i) for i in segment.dead),
                    "size": segment.size,
                }
            )
        buffer_name = "buffer.npz"
        np.savez(
            path / buffer_name,
            ids=self._buffer_ids[: self._buffer_size],
            vectors=self._buffer_vectors[: self._buffer_size],
            dead=np.asarray(sorted(self._buffer_dead), dtype=np.int64),
        )
        manifest = {
            "format": "churnvec-churn-v1",
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "seed": self.seed,
            "selection": self.selection,
            "keep_pruned": self.keep_pruned,
            "max_level_cap": self.max_level_cap,
            "flush_threshold": self.flush_threshold,
            "generation": snapshot.generation,
            "live_count": snapshot.live_count,
            "next_id": self._next_id,
            "segments": segment_entries,
            "buffer": {
                "file": buffer_name,
                "size": self._buffer_size,
                "dead": sorted(int(i) for i in self._buffer_dead),
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ChurnIndex":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("format") != "churnvec-churn-v1":
            raise ValueError(f"unsupported churnvec format: {manifest.get('format')}")
        index = cls(
            dim=int(manifest["dim"]),
            M=int(manifest["M"]),
            ef_construction=int(manifest["ef_construction"]),
            metric=manifest["metric"],
            seed=int(manifest["seed"]),
            selection=manifest.get("selection", "heuristic"),
            keep_pruned=bool(manifest.get("keep_pruned", False)),
            max_level_cap=int(manifest.get("max_level_cap", 32)),
            flush_threshold=int(manifest.get("flush_threshold", 2048)),
        )
        segments = []
        for entry in manifest["segments"]:
            segment_index = HNSWIndex.load(path / entry["file"])
            segments.append(
                Segment(index=segment_index, dead=frozenset(int(i) for i in entry.get("dead", [])))
            )

        buffer_entry = manifest["buffer"]
        with np.load(path / buffer_entry["file"], allow_pickle=False) as data:
            ids = data["ids"].astype(np.int64)
            vectors = data["vectors"].astype(np.float32)
            dead = frozenset(int(i) for i in data["dead"].tolist())
        size = int(buffer_entry["size"])
        capacity = max(int(ids.size), 64)
        index._buffer_ids = np.empty(capacity, dtype=np.int64)
        index._buffer_vectors = np.empty((capacity, index.dim), dtype=np.float32)
        index._buffer_ids[:size] = ids[:size]
        index._buffer_vectors[:size] = vectors[:size]
        index._buffer_size = size
        index._buffer_dead = set(dead)
        index._buffer_live = BufferView(
            ids=index._buffer_ids,
            vectors=index._buffer_vectors,
            size=size,
            dead_seq=frozenset(dead),
            metric=index.metric,
        ).live_count
        index._buffer_index = {}
        for seq in range(size):
            if seq in dead:
                continue
            index._buffer_index.setdefault(int(index._buffer_ids[seq]), []).append(seq)
        index._next_id = int(manifest.get("next_id", 0))
        index._generation = int(manifest.get("generation", 0))
        index._snapshot = Snapshot(
            dim=index.dim,
            metric=index.metric,
            segments=tuple(segments),
            buffer=index._buffer_view(),
            generation=index._generation,
            live_count=int(manifest["live_count"]),
        )
        return index


__all__ = ["ChurnIndex"]
