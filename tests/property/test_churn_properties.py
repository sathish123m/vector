"""Property-based model tests for churn operations."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from churnvec import ChurnIndex

DIM = 4


@st.composite
def operation_sequences(draw):
    operations = []
    for _ in range(draw(st.integers(min_value=1, max_value=35))):
        kind = draw(st.sampled_from(["add", "upsert", "delete", "flush", "compact", "delete", "upsert"]))
        external_id = draw(st.integers(min_value=0, max_value=14))
        vector = [draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)) for _ in range(DIM)]
        operations.append((kind, external_id, vector))
    return operations


@settings(max_examples=20, deadline=None)
@given(operation_sequences())
def test_random_churn_matches_python_model(operations):
    index = ChurnIndex(dim=DIM, M=4, ef_construction=20, flush_threshold=7, seed=3)
    model: dict[int, np.ndarray] = {}

    for kind, external_id, vector in operations:
        vec = np.asarray(vector, dtype=np.float32)
        if kind == "add":
            if external_id in model:
                continue
            index.add(external_id, vec)
            model[external_id] = vec
        elif kind == "upsert":
            index.upsert(external_id, vec)
            model[external_id] = vec
        elif kind == "delete":
            removed = index.delete(external_id)
            assert removed == (external_id in model)
            model.pop(external_id, None)
        elif kind == "flush":
            index.flush()
        else:
            index.compact()

        assert len(index) == len(model)
        snapshot = index.snapshot()
        live_ids, live_vectors = snapshot.live_arrays()
        assert set(live_ids.tolist()) == set(model)
        assert live_vectors.shape == (len(model), DIM)
        for row, external_id in zip(live_vectors, live_ids.tolist()):
            np.testing.assert_allclose(row, model[external_id], rtol=0, atol=0)

        for external_id in range(15):
            assert snapshot.contains(external_id) == (external_id in model)

        query = np.zeros(DIM, dtype=np.float32)
        returned, _ = snapshot.search_ids(query, k=5, ef=32)
        assert set(returned.tolist()).issubset(model)
        assert returned.size == min(5, len(model))


@settings(max_examples=15, deadline=None)
@given(operation_sequences())
def test_previous_snapshot_never_changes(operations):
    index = ChurnIndex(dim=DIM, M=4, ef_construction=20, flush_threshold=6, seed=8)
    model: dict[int, np.ndarray] = {}
    snapshots = []

    for kind, external_id, vector in operations:
        vec = np.asarray(vector, dtype=np.float32)
        if kind == "add" and external_id not in model:
            index.add(external_id, vec)
            model[external_id] = vec
        elif kind == "upsert":
            index.upsert(external_id, vec)
            model[external_id] = vec
        elif kind == "delete":
            index.delete(external_id)
            model.pop(external_id, None)
        elif kind == "flush":
            index.flush()
        elif kind == "compact":
            index.compact()
        snapshots.append((index.snapshot(), set(model), dict(model)))

    for snapshot, expected_ids, expected_vectors in snapshots:
        live_ids, live_vectors = snapshot.live_arrays()
        assert set(live_ids.tolist()) == expected_ids
        for row, external_id in zip(live_vectors, live_ids.tolist()):
            np.testing.assert_allclose(row, expected_vectors[external_id], rtol=0, atol=0)
