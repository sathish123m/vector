"""HNSW unit tests: correctness, persistence, and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from churnvec import HNSWIndex, brute_force_topk
from churnvec.bench.metrics import recall_at_k
from churnvec.core.exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyIndexError,
)


def _recall(index: HNSWIndex, base, queries, k=10, ef=200) -> float:
    truth, _ = brute_force_topk(base, queries, k=k)
    ids = np.stack([index.search_ids(q, k=k, ef=ef)[0] for q in queries])
    return recall_at_k(ids, truth, k=k)


def test_empty_index_search_raises():
    index = HNSWIndex(dim=4)
    with pytest.raises(EmptyIndexError):
        index.search(np.zeros(4, dtype=np.float32))


def test_single_vector_and_self_search(rng):
    index = HNSWIndex(dim=8, M=4, ef_construction=20, capacity=4)
    vec = rng.normal(size=8).astype(np.float32)
    index.add_item(42, vec)
    ids, dists = index.search_ids(vec, k=3, ef=10)
    assert ids.tolist() == [42]
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert index.check_invariants().ok


def test_exact_search_with_large_ef(clustered_data):
    base, queries = clustered_data
    index = HNSWIndex(dim=4, M=8, ef_construction=60, seed=3, capacity=len(base))
    index.add_items(base, np.arange(len(base)))
    assert _recall(index, base, queries, k=5, ef=len(base)) == 1.0
    for query in queries:
        nearest = index.search(query, k=1, ef=len(base))[0]
        exact_id = int(np.argmin(np.sum((base - query) ** 2, axis=1)))
        assert nearest.id == exact_id


def test_default_ids_and_size(rng):
    data = rng.normal(size=(40, 6)).astype(np.float32)
    index = HNSWIndex(dim=6, M=4, ef_construction=30, capacity=10)
    index.add_items(data)
    assert len(index) == 40
    assert index.size == 40
    ids = {neighbor.id for neighbor in index.search(data[0], k=5, ef=30)}
    assert 0 in ids
    report = index.check_invariants()
    assert report.ok
    assert report.unreachable == 0
    assert report.layer_counts[0] == 40


def test_add_items_with_explicit_ids(rng):
    data = rng.normal(size=(20, 5)).astype(np.float32)
    index = HNSWIndex(dim=5, M=4, ef_construction=20)
    index.add_items(data, ids=np.arange(100, 120))
    assert index.search_ids(data[0], k=1, ef=20)[0].tolist() == [100]


def test_duplicate_id_rejected_without_mutation(rng):
    index = HNSWIndex(dim=4, M=4, ef_construction=20)
    index.add_item(7, rng.normal(size=4).astype(np.float32))
    before = index.size
    with pytest.raises(DuplicateIDError):
        index.add_item(7, rng.normal(size=4).astype(np.float32))
    assert index.size == before


def test_dimension_mismatch(rng):
    index = HNSWIndex(dim=4)
    with pytest.raises(DimensionMismatchError):
        index.add_item(1, rng.normal(size=5).astype(np.float32))
    index.add_item(1, rng.normal(size=4).astype(np.float32))
    with pytest.raises(DimensionMismatchError):
        index.search(rng.normal(size=5).astype(np.float32))
    with pytest.raises(DimensionMismatchError):
        index.add_items(rng.normal(size=(3, 5)).astype(np.float32))


def test_non_finite_values_rejected(rng):
    index = HNSWIndex(dim=4)
    bad = rng.normal(size=4).astype(np.float32)
    bad[2] = np.nan
    with pytest.raises(ValueError):
        index.add_item(1, bad)
    index.add_item(1, rng.normal(size=4).astype(np.float32))
    with pytest.raises(ValueError):
        index.search(bad)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        HNSWIndex(dim=0)
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, M=1)
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, ef_construction=0)
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, metric="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HNSWIndex(dim=4, selection="bad")  # type: ignore[arg-type]


def test_k_and_ef_validation(rng):
    index = HNSWIndex(dim=4)
    index.add_items(rng.normal(size=(5, 4)).astype(np.float32))
    with pytest.raises(ValueError):
        index.search(rng.normal(size=4).astype(np.float32), k=0)
    ids, _ = index.search_ids(rng.normal(size=4).astype(np.float32), k=100, ef=1)
    assert ids.size <= 5


@pytest.mark.parametrize("selection", ["heuristic", "simple"])
@pytest.mark.parametrize("keep_pruned", [True, False])
def test_selection_variants_remain_connected(selection, keep_pruned, clustered_data):
    base, queries = clustered_data
    index = HNSWIndex(
        dim=4,
        M=8,
        ef_construction=60,
        selection=selection,
        keep_pruned=keep_pruned,
        capacity=len(base),
    )
    index.add_items(base)
    report = index.check_invariants()
    assert report.ok
    assert report.unreachable == 0
    assert _recall(index, base, queries, k=5, ef=len(base)) == 1.0


def test_save_load_roundtrip(tmp_path, rng):
    data = rng.normal(size=(120, 12)).astype(np.float32)
    queries = rng.normal(size=(5, 12)).astype(np.float32)
    index = HNSWIndex(dim=12, M=8, ef_construction=50, seed=9, capacity=16)
    index.add_items(data, np.arange(1000, 1120))
    before = np.stack([index.search_ids(q, k=10, ef=40)[0] for q in queries])

    path = index.save(tmp_path / "index")
    assert path.exists()
    assert path.suffix == ".npz"

    loaded = HNSWIndex.load(path)
    assert loaded.size == index.size
    assert loaded.dim == 12
    assert loaded.M == 8
    after = np.stack([loaded.search_ids(q, k=10, ef=40)[0] for q in queries])
    np.testing.assert_array_equal(before, after)
    assert loaded.check_invariants().ok

    # The loaded index must still accept new vectors.
    extra = rng.normal(size=12).astype(np.float32)
    loaded.add_item(9999, extra)
    assert loaded.size == index.size + 1
    assert loaded.search_ids(extra, k=1, ef=40)[0].tolist() == [9999]
    assert loaded.check_invariants().ok


def test_save_load_empty(tmp_path):
    index = HNSWIndex(dim=3)
    loaded = HNSWIndex.load(index.save(tmp_path / "empty"))
    assert loaded.size == 0
    assert loaded.entry_point == -1


def test_duplicate_vectors_are_supported(rng):
    base = np.repeat(rng.normal(size=(1, 8)).astype(np.float32), 30, axis=0)
    index = HNSWIndex(dim=8, M=4, ef_construction=20, seed=1)
    index.add_items(base, np.arange(30))
    report = index.check_invariants()
    assert report.ok
    ids, dists = index.search_ids(base[0], k=5, ef=30)
    assert len(ids) == 5
    assert np.all(dists == pytest.approx(0.0, abs=1e-5))


def test_large_duplicate_set_repairs_connectivity():
    # Fixed-capacity pruning cannot keep 100 identical vectors mutually
    # reachable; the bulk-build repair path must restore a connected graph.
    base = np.zeros((100, 6), dtype=np.float32)
    index = HNSWIndex(dim=6, M=4, ef_construction=20, seed=7, capacity=100)
    index.add_items(base, np.arange(100))
    report = index.check_invariants()
    assert report.ok
    assert report.unreachable == 0
    assert index.stats()["repair_links"] > 0
    ids, _ = index.search_ids(base[0], k=5, ef=100)
    assert len(ids) == 5


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_metrics_match_bruteforce(metric, rng):
    base = rng.normal(size=(80, 10)).astype(np.float32)
    query = rng.normal(size=10).astype(np.float32)
    index = HNSWIndex(dim=10, M=8, ef_construction=40, metric=metric, seed=2)
    index.add_items(base)
    ids, _ = index.search_ids(query, k=5, ef=len(base))
    truth, _ = brute_force_topk(base, np.stack([query]), k=5, metric=metric)
    np.testing.assert_array_equal(ids, truth[0])


def test_invariants_report_contents(clustered_data):
    base, _ = clustered_data
    index = HNSWIndex(dim=4, M=4, ef_construction=30, capacity=4)
    index.add_items(base)
    report = index.check_invariants()
    assert report.n_nodes == len(base)
    assert report.max_degree0 <= 2 * 4
    assert report.max_degree_upper <= 4
    assert 0 <= report.entry_point < len(base)
    assert report.layer_counts[0] == len(base)
    for layer in report.layer_counts:
        assert report.layer_counts[layer] >= report.layer_counts.get(layer + 1, 0)


def test_stats_keys(clustered_data):
    base, _ = clustered_data
    index = HNSWIndex(dim=4, M=4, ef_construction=30)
    index.add_items(base)
    stats = index.stats()
    for key in ("size", "dim", "M", "max_level", "memory_bytes", "distance_calls"):
        assert key in stats
    assert stats["size"] == len(base)
