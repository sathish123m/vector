"""Exact brute-force baseline tests."""

from __future__ import annotations

import numpy as np
import pytest

from churnvec.core.bruteforce import brute_force_topk, exact_recall_at_k


def naive_topk(base, queries, k):
    dists = np.sum((queries[:, None, :] - base[None, :, :]) ** 2, axis=2)
    order = np.argsort(dists, axis=1)[:, :k]
    return order, np.take_along_axis(dists, order, axis=1)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, None])
def test_topk_matches_naive(chunk_size, rng):
    base = rng.normal(size=(25, 6)).astype(np.float32)
    queries = rng.normal(size=(4, 6)).astype(np.float32)
    ids, dists = brute_force_topk(base, queries, k=5, chunk_size=chunk_size)
    expected_ids, expected_dists = naive_topk(base, queries, 5)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-4, atol=1e-4)


def test_topk_is_sorted(rng):
    base = rng.normal(size=(100, 8)).astype(np.float32)
    queries = rng.normal(size=(10, 8)).astype(np.float32)
    _, dists = brute_force_topk(base, queries, k=10)
    assert np.all(np.diff(dists, axis=1) >= 0)


def test_k_larger_than_base_is_clamped(rng):
    base = rng.normal(size=(7, 4)).astype(np.float32)
    queries = rng.normal(size=(2, 4)).astype(np.float32)
    ids, dists = brute_force_topk(base, queries, k=50)
    assert ids.shape == (2, 7)
    assert dists.shape == (2, 7)


def test_cosine_and_ip_metrics(rng):
    base = rng.normal(size=(30, 5)).astype(np.float32)
    queries = rng.normal(size=(3, 5)).astype(np.float32)
    ids, _ = brute_force_topk(base, queries, k=5, metric="cosine")
    assert ids.shape == (3, 5)
    ip_ids, _ = brute_force_topk(base, queries, k=5, metric="ip")
    assert ip_ids.shape == (3, 5)


def test_validation_errors(rng):
    base = rng.normal(size=(10, 4)).astype(np.float32)
    queries = rng.normal(size=(2, 5)).astype(np.float32)
    with pytest.raises(ValueError):
        brute_force_topk(base, queries, k=3)
    with pytest.raises(ValueError):
        brute_force_topk(base, base, k=0)
    with pytest.raises(ValueError):
        brute_force_topk(base, base, k=3, metric="bad")  # type: ignore[arg-type]


def test_exact_recall_at_k():
    returned = np.array([[1, 2, 3], [4, 5, 6]])
    truth = np.array([[1, 9, 3], [7, 8, 9]])
    assert exact_recall_at_k(returned, truth) == pytest.approx(2 / 6)
    assert exact_recall_at_k(returned, truth, k=2) == pytest.approx(1 / 4)


def test_recall_validation():
    with pytest.raises(ValueError):
        exact_recall_at_k(np.zeros((2, 3)), np.zeros((3, 3)))
