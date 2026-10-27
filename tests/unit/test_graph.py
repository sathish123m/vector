"""GraphStore invariants and capacity growth."""

from __future__ import annotations

import numpy as np
import pytest

from churnvec.core.exceptions import DuplicateIDError, InvariantError
from churnvec.core.graph import GraphStore


def test_capacity_growth_preserves_nodes(rng):
    store = GraphStore(dim=4, M=2, capacity=2)
    vectors = rng.normal(size=(9, 4)).astype(np.float32)
    for i, vec in enumerate(vectors):
        store.add_node(i, vec, 0)
    assert store.size == 9
    assert store.capacity >= 9
    np.testing.assert_array_equal(store.vectors[:9], vectors)
    assert store.id_to_slot == {i: i for i in range(9)}
    assert store.validate().ok


def test_duplicate_id_rejected(rng):
    store = GraphStore(dim=4, M=2)
    vec = rng.normal(size=4).astype(np.float32)
    store.add_node(1, vec, 0)
    with pytest.raises(DuplicateIDError):
        store.add_node(1, vec, 0)


def test_set_neighbors_validation(rng):
    store = GraphStore(dim=4, M=2, capacity=5)
    for i in range(4):
        store.add_node(i, rng.normal(size=4).astype(np.float32), 0)
    store.set_neighbors(0, 0, [1, 2, 3])
    assert store.get_degree(0, 0) == 3
    with pytest.raises(ValueError):
        store.set_neighbors(0, 0, [0])
    with pytest.raises(ValueError):
        store.set_neighbors(0, 0, [1, 1])
    with pytest.raises(ValueError):
        store.set_neighbors(0, 0, [1, 2, 3, 4, 5])
    with pytest.raises(IndexError):
        store.set_neighbors(0, 0, [99])
    store.set_neighbors(0, 0, [3, 1])
    assert store.get_neighbors(0, 0).tolist() == [3, 1]


def test_upper_layers_are_created(rng):
    store = GraphStore(dim=4, M=2, capacity=8)
    store.add_node(0, rng.normal(size=4).astype(np.float32), 3)
    for i in range(1, 5):
        store.add_node(i, rng.normal(size=4).astype(np.float32), 0)
    store.set_neighbors(0, 1, [1])
    assert store.get_neighbors(0, 1).tolist() == [1]
    assert store.get_degree(0, 1) == 1
    report = store.validate()
    assert report.ok
    assert report.layer_counts == {0: 5, 1: 1, 2: 1, 3: 1}


def test_self_loop_is_detected(rng):
    store = GraphStore(dim=4, M=2, capacity=4)
    for i in range(3):
        store.add_node(i, rng.normal(size=4).astype(np.float32), 0)
    store.neighbors0[0, 0] = 0
    store.degree0[0] = 1
    report = store.validate()
    assert not report.ok
    with pytest.raises(InvariantError):
        report.assert_ok()


def test_empty_store_report():
    store = GraphStore(dim=4, M=2)
    report = store.validate()
    assert report.ok
    assert report.n_nodes == 0
    assert report.entry_point == -1
    assert report.layer_counts == {}
