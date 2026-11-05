"""Segment, buffer-view, and snapshot isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from churnvec import HNSWIndex
from churnvec.core.segment import BufferView, Segment, Snapshot


def _segment(vectors: np.ndarray, ids: np.ndarray | None = None) -> Segment:
    n, dim = vectors.shape
    index = HNSWIndex(dim=dim, M=8, ef_construction=40, capacity=n, seed=3)
    index.add_items(vectors, ids if ids is not None else np.arange(n))
    return Segment(index=index)


def test_segment_search_excludes_tombstones(rng):
    vectors = rng.normal(size=(60, 6)).astype(np.float32)
    segment = _segment(vectors)
    ids, distances = segment.search_ids(vectors[7], k=5, ef=64)
    assert ids[0] == 7
    assert distances[0] == pytest.approx(0.0, abs=1e-5)

    dead = segment.with_dead({7, 8})
    assert dead.live_count == 58
    assert dead.size == 60
    assert not dead.contains(7)
    assert dead.contains(9)
    ids, _ = dead.search_ids(vectors[7], k=5, ef=64)
    assert 7 not in ids and 8 not in ids
    assert ids.size == 5


def test_with_dead_shares_index_and_keeps_old_view_alive(rng):
    vectors = rng.normal(size=(40, 4)).astype(np.float32)
    segment = _segment(vectors)
    dead = segment.with_dead({1})
    assert dead.index is segment.index
    assert segment.contains(1)
    assert not dead.contains(1)
    # An unrelated tombstone id is ignored rather than inflating dead_count.
    assert segment.with_dead({9999}) is segment


def test_buffer_view_freezes_size_when_writer_appends(rng):
    dim = 4
    ids = np.empty(16, dtype=np.int64)
    vectors = np.empty((16, dim), dtype=np.float32)
    ids[0], ids[1] = 11, 12
    vectors[0], vectors[1] = rng.normal(size=dim), rng.normal(size=dim)
    view = BufferView(ids=ids, vectors=vectors, size=2, dead_seq=frozenset(), metric="l2")

    # Simulate a writer appending after the view was published.
    ids[2] = 13
    vectors[2] = rng.normal(size=dim)
    assert view.size == 2
    assert view.live_count == 2
    assert [i for i, _ in view.iter_live()] == [11, 12]


def test_buffer_view_latest_record_wins_and_dead_sequences_are_skipped(rng):
    dim = 4
    ids = np.array([5, 6, 5, 7, 8], dtype=np.int64)
    vectors = rng.normal(size=(5, dim)).astype(np.float32)
    view = BufferView(ids=ids, vectors=vectors, size=5, dead_seq=frozenset({1}), metric="l2")
    assert view.live_count == 3
    assert not view.contains(6)
    assert view.contains(5)
    records = view.live_records()
    assert [external_id for external_id, _ in records] == [5, 7, 8]
    # Sequence 2 is the newest live copy of id 5.
    assert records[0][1] == 2

    ids_hit, _ = view.search_ids(vectors[2], k=3)
    assert ids_hit[0] == 5
    assert set(ids_hit.tolist()) == {5, 7, 8}


def test_snapshot_merges_segments_and_buffer_and_deduplicates(rng):
    dim = 5
    seg_vectors = rng.normal(size=(30, dim)).astype(np.float32)
    segment = _segment(seg_vectors)
    ids = np.array([100, 101, 101], dtype=np.int64)
    vectors = rng.normal(size=(3, dim)).astype(np.float32)
    buffer = BufferView(ids=ids, vectors=vectors, size=3, dead_seq=frozenset(), metric="l2")
    snapshot = Snapshot(
        dim=dim,
        metric="l2",
        segments=(segment,),
        buffer=buffer,
        generation=3,
        live_count=32,
    )
    assert snapshot.contains(100)
    assert snapshot.contains(0)
    assert not snapshot.contains(999)
    live_ids, live_vectors = snapshot.live_arrays()
    assert live_ids.size == 32
    assert live_vectors.shape == (32, dim)
    assert len(set(live_ids.tolist())) == 32

    ids_hit, _ = snapshot.search_ids(vectors[0], k=5, ef=64)
    assert ids_hit[0] == 100
    assert ids_hit.size == 5


def test_snapshot_empty_is_safe(rng):
    buffer = BufferView(
        ids=np.empty(0, dtype=np.int64),
        vectors=np.empty((0, 3), dtype=np.float32),
        size=0,
        dead_seq=frozenset(),
        metric="l2",
    )
    snapshot = Snapshot(dim=3, metric="l2", segments=(), buffer=buffer, generation=0, live_count=0)
    ids, distances = snapshot.search_ids(rng.normal(size=3), k=5, ef=10)
    assert ids.size == 0 and distances.size == 0
    assert snapshot.live_arrays()[0].size == 0
    assert not snapshot.contains(1)
