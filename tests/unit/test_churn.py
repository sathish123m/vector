"""ChurnIndex unit tests: deletes, updates, flushing, compaction, persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churnvec import ChurnIndex, brute_force_topk
from churnvec.bench.metrics import recall_at_k
from churnvec.core.exceptions import DimensionMismatchError, DuplicateIDError


def _index(dim: int = 6, flush_threshold: int = 64) -> ChurnIndex:
    return ChurnIndex(dim=dim, M=8, ef_construction=60, flush_threshold=flush_threshold, seed=11)


def _live_arrays(index: ChurnIndex) -> tuple[np.ndarray, np.ndarray]:
    return index.snapshot().live_arrays()


def _assert_respects_live_set(index: ChurnIndex, ids: np.ndarray) -> None:
    live = {int(i) for i in _live_arrays(index)[0].tolist()}
    assert set(int(i) for i in ids.tolist()).issubset(live)


def test_add_search_and_membership(rng):
    base = rng.normal(size=(80, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(80))
    assert len(index) == 80
    assert 17 in index and 999 not in index
    ids, distances = index.search_ids(base[17], k=5, ef=64)
    assert ids[0] == 17
    assert distances[0] == pytest.approx(0.0, abs=1e-5)
    _assert_respects_live_set(index, ids)


def test_duplicate_add_raises_and_upsert_replaces(rng):
    base = rng.normal(size=(50, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(50))
    with pytest.raises(DuplicateIDError):
        index.add(3, base[3])

    replacement = np.full(6, 100.0, dtype=np.float32)
    assert index.upsert(3, replacement) is True
    assert len(index) == 50
    ids, _ = index.search_ids(replacement, k=3, ef=64)
    assert ids[0] == 3
    ids, _ = index.search_ids(base[3], k=5, ef=64)
    assert 3 not in ids
    _assert_respects_live_set(index, ids)


def test_upsert_new_id_reports_insert(rng):
    index = _index()
    index.add(1, rng.normal(size=6).astype(np.float32))
    assert index.upsert(2, rng.normal(size=6).astype(np.float32)) is False
    assert len(index) == 2


def test_delete_hides_vector_and_returns_false_when_missing(rng):
    base = rng.normal(size=(70, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(70))
    assert index.delete(4) is True
    assert index.delete(4) is False
    assert index.delete(999) is False
    assert len(index) == 69
    assert 4 not in index
    ids, _ = index.search_ids(base[4], k=10, ef=128)
    assert 4 not in ids
    _assert_respects_live_set(index, ids)


def test_re_add_after_delete_is_live_again(rng):
    base = rng.normal(size=(30, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(30))
    index.delete(8)
    index.add(8, base[8])
    assert 8 in index
    ids, _ = index.search_ids(base[8], k=3, ef=64)
    assert ids[0] == 8


def test_delete_many_counts_only_live_ids(rng):
    base = rng.normal(size=(40, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(40))
    assert index.delete_many([1, 2, 2, 500]) == 2
    assert len(index) == 38


def test_buffer_hits_flush_threshold(rng):
    base = rng.normal(size=(100, 6)).astype(np.float32)
    index = _index(flush_threshold=16)
    index.add_many(base, np.arange(100))
    snapshot = index.snapshot()
    assert snapshot.segment_count == 6
    assert snapshot.buffer.size == 4
    assert len(index) == 100


def test_flush_seals_buffer_and_preserves_search(rng):
    base = rng.normal(size=(40, 6)).astype(np.float32)
    index = _index(flush_threshold=1000)
    index.add_many(base, np.arange(40))
    assert index.snapshot().segment_count == 0
    assert index.flush() == 40
    assert index.snapshot().segment_count == 1
    assert index.snapshot().buffer.size == 0
    assert len(index) == 40
    ids, _ = index.search_ids(base[9], k=5, ef=64)
    assert ids[0] == 9


def test_compaction_merges_segments_and_removes_tombstones(rng):
    base = rng.normal(size=(90, 6)).astype(np.float32)
    index = _index(flush_threshold=20)
    index.add_many(base, np.arange(90))
    for external_id in (1, 5, 9, 20, 44, 70):
        index.delete(external_id)
    before = len(index)
    assert index.stats()["segment_count"] >= 4
    assert index.stats()["dead_fraction"] > 0
    assert index.compact() == before
    stats = index.stats()
    assert stats["segment_count"] == 1
    assert stats["dead_fraction"] == 0
    assert len(index) == before
    ids, _ = index.search_ids(base[4], k=10, ef=128)
    assert 5 not in ids
    _assert_respects_live_set(index, ids)


def test_snapshot_isolated_from_later_delete_and_compaction(rng):
    base = rng.normal(size=(60, 6)).astype(np.float32)
    index = _index(flush_threshold=1000)
    index.add_many(base, np.arange(60))
    snapshot = index.snapshot()
    old_ids, _ = snapshot.search_ids(base[5], k=10, ef=128)
    assert 5 in old_ids

    index.delete(5)
    assert not index.snapshot().contains(5)
    new_ids, _ = index.snapshot().search_ids(base[5], k=10, ef=128)
    assert 5 not in new_ids

    # The snapshot captured before the mutation still sees pre-delete data.
    again, _ = snapshot.search_ids(base[5], k=10, ef=128)
    assert 5 in again
    index.compact()
    again, _ = snapshot.search_ids(base[5], k=10, ef=128)
    assert 5 in again


def test_buffer_view_snapshot_stays_visible_after_flush(rng):
    base = rng.normal(size=(20, 6)).astype(np.float32)
    index = _index(flush_threshold=1000)
    index.add_many(base, np.arange(20))
    snapshot = index.snapshot()
    assert snapshot.buffer.size == 20
    ids, _ = snapshot.search_ids(base[2], k=5, ef=64)
    assert ids[0] == 2

    index.flush()
    assert index.snapshot().buffer.size == 0
    assert snapshot.buffer.size == 20
    old_ids, _ = snapshot.search_ids(base[2], k=5, ef=64)
    assert old_ids[0] == 2


def test_len_and_live_arrays_track_updates(rng):
    base = rng.normal(size=(25, 6)).astype(np.float32)
    index = _index()
    index.add_many(base, np.arange(25))
    index.delete(0)
    index.upsert(1, rng.normal(size=6).astype(np.float32))
    index.add(100, rng.normal(size=6).astype(np.float32))
    ids, vectors = _live_arrays(index)
    assert ids.size == 25
    assert vectors.shape == (25, 6)
    assert 0 not in ids.tolist()
    assert 100 in ids.tolist()
    assert len(index) == 25


def test_exact_nearest_after_churn_on_clustered_data(clustered_data):
    points, queries = clustered_data
    index = ChurnIndex(dim=4, M=8, ef_construction=60, flush_threshold=25, seed=5)
    index.add_many(points, np.arange(points.shape[0]))
    index.delete_many(range(0, 100, 7))
    for external_id in (1, 2, 3):
        index.upsert(external_id, queries[0] + np.float32(external_id * 1e-3))
    index.compact()

    live_ids, live_vectors = _live_arrays(index)
    truth, _ = brute_force_topk(live_vectors, queries, k=5)
    truth_ids = live_ids[truth]
    returned = np.stack([index.search_ids(q, k=5, ef=200)[0] for q in queries])
    assert recall_at_k(returned, truth_ids, k=5) >= 0.95


def test_dimension_and_finite_validation(rng):
    index = _index()
    with pytest.raises(DimensionMismatchError):
        index.add(1, np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        index.add(1, np.full(6, np.nan, dtype=np.float32))
    index.add(1, rng.normal(size=6).astype(np.float32))
    with pytest.raises(ValueError):
        index.search_ids(np.full(6, np.inf, dtype=np.float32))
    with pytest.raises(ValueError):
        index.search_ids(np.zeros(6, dtype=np.float32), k=0)


def test_empty_index_returns_empty_results(rng):
    index = _index()
    ids, distances = index.search_ids(rng.normal(size=6), k=5)
    assert ids.size == 0 and distances.size == 0
    assert index.flush() == 0
    assert index.compact() == 0
    assert len(index) == 0


def test_maybe_compact_respects_thresholds(rng):
    base = rng.normal(size=(60, 6)).astype(np.float32)
    index = _index(flush_threshold=10)
    index.add_many(base, np.arange(60))
    index.delete_many(range(0, 40))
    assert index.maybe_compact(dead_fraction=0.9, min_vectors=1000) is False
    assert index.maybe_compact(dead_fraction=0.3, min_vectors=10) is True
    assert index.stats()["segment_count"] == 1
    assert index.stats()["dead_fraction"] == 0


def test_save_load_roundtrip_with_segments_buffer_and_tombstones(tmp_path: Path, rng):
    base = rng.normal(size=(70, 6)).astype(np.float32)
    index = ChurnIndex(dim=6, M=8, ef_construction=60, flush_threshold=20, seed=13)
    index.add_many(base, np.arange(70))
    index.delete_many([2, 7, 30])
    index.upsert(4, rng.normal(size=6).astype(np.float32))
    # Leave a pending buffer as well as sealed segments.
    index.add_many(rng.normal(size=(5, 6)).astype(np.float32), np.arange(200, 205))

    saved = index.save(tmp_path / "snapshot")
    loaded = ChurnIndex.load(saved)

    assert len(loaded) == len(index)
    assert loaded.stats()["segment_count"] == index.stats()["segment_count"]
    assert loaded.stats()["buffer_size"] == index.stats()["buffer_size"]
    assert loaded.stats()["dead_fraction"] == index.stats()["dead_fraction"]
    for external_id in (2, 7, 30):
        assert external_id not in loaded
    assert 4 in loaded and 200 in loaded

    for query in base[:10]:
        expected, expected_distances = index.search_ids(query, k=5, ef=64)
        actual, actual_distances = loaded.search_ids(query, k=5, ef=64)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual_distances, expected_distances, rtol=1e-6, atol=1e-6)

    # The restored index remains writable.
    loaded.add(999, rng.normal(size=6).astype(np.float32))
    assert 999 in loaded


def test_stats_reports_physical_and_live_counts(rng):
    base = rng.normal(size=(30, 6)).astype(np.float32)
    index = _index(flush_threshold=10)
    index.add_many(base, np.arange(30))
    index.delete_many([0, 1, 2])
    stats = index.stats()
    assert stats["live_count"] == 27
    assert stats["physical_count"] == 30
    assert stats["segment_count"] == 3
    assert 0 < stats["dead_fraction"] < 1
    assert stats["memory_bytes"] > 0
