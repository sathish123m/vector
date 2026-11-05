"""Property-based tests for HNSW on small random graphs."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from churnvec import HNSWIndex, brute_force_topk

float32 = st.floats(
    min_value=-20.0,
    max_value=20.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def dataset(draw):
    n = draw(st.integers(min_value=5, max_value=30))
    dim = draw(st.integers(min_value=2, max_value=6))
    rows = draw(st.lists(st.lists(float32, min_size=dim, max_size=dim), min_size=n, max_size=n))
    query = [draw(float32) for _ in range(dim)]
    return np.asarray(rows, dtype=np.float32), np.asarray([query], dtype=np.float32)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(data=dataset())
def test_graph_is_connected_and_exact_with_full_ef(data):
    base, queries = data
    n, dim = base.shape
    index = HNSWIndex(dim=dim, M=4, ef_construction=25, seed=n, capacity=4)
    index.add_items(base, np.arange(n))

    report = index.check_invariants()
    assert report.ok
    assert report.unreachable == 0
    assert report.n_nodes == n
    assert report.max_degree0 <= 8

    k = min(5, n)
    ids, dists = index.search_ids(queries[0], k=k, ef=n + 5)
    assert ids.size == k
    _, truth_dists = brute_force_topk(base, queries, k=k)
    # Full-width ef explores every reachable node, so the returned distances
    # must be the exact k smallest (ties may reorder equal-distance IDs).
    assert float(np.max(dists)) <= float(np.max(truth_dists[0])) + 1e-3


@settings(max_examples=20, deadline=None)
@given(n=st.integers(min_value=2, max_value=25), dim=st.integers(min_value=2, max_value=5))
def test_every_node_is_reachable_after_sequential_inserts(n, dim):
    rng = np.random.default_rng(n * 100 + dim)
    index = HNSWIndex(dim=dim, M=4, ef_construction=20, seed=1, capacity=1)
    for i in range(n):
        index.add_item(i, rng.normal(size=dim).astype(np.float32))
        assert index.size == i + 1
        assert index.check_invariants().ok
    report = index.check_invariants()
    assert report.unreachable == 0
    assert report.layer_counts[0] == n
