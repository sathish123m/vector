"""Distance kernels must agree with direct NumPy computations."""

from __future__ import annotations

import numpy as np
import pytest

from churnvec.core.distance import (
    cosine_distance,
    cosine_distance_batch,
    distance_batch,
    distance_one,
    inner_product_distance,
    l2_sqr,
    l2_sqr_batch,
    l2_sqr_batch_from_norms,
    normalize_rows,
)


def test_l2_scalar_matches_naive(rng):
    a = rng.normal(size=32).astype(np.float32)
    b = rng.normal(size=32).astype(np.float32)
    expected = float(np.sum((a - b) ** 2))
    assert l2_sqr(a, b) == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_l2_shape_mismatch():
    with pytest.raises(ValueError):
        l2_sqr(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_l2_batch_matches_naive(rng):
    x = rng.normal(size=(17, 5)).astype(np.float32)
    q = rng.normal(size=5).astype(np.float32)
    expected = np.sum((x - q) ** 2, axis=1)
    np.testing.assert_allclose(l2_sqr_batch(x, q), expected, rtol=1e-5, atol=1e-5)


def test_l2_batch_from_norms_matches_naive(rng):
    x = rng.normal(size=(17, 5)).astype(np.float32)
    q = rng.normal(size=5).astype(np.float32)
    norms = np.einsum("ij,ij->i", x, x)
    q_norm = float(np.dot(q, q))
    expected = np.sum((x - q) ** 2, axis=1)
    np.testing.assert_allclose(
        l2_sqr_batch_from_norms(x, q, norms, q_norm), expected, rtol=1e-4, atol=1e-4
    )


def test_l2_batch_identical_vectors_is_non_negative(rng):
    x = rng.normal(size=(10, 16)).astype(np.float32)
    q = x[3].copy()
    dists = l2_sqr_batch_from_norms(x, q, np.einsum("ij,ij->i", x, x))
    assert np.all(dists >= 0)
    assert dists[3] == pytest.approx(0.0, abs=1e-4)


def test_cosine_zero_vector_is_defined():
    zero = np.zeros(4, dtype=np.float32)
    other = np.ones(4, dtype=np.float32)
    assert cosine_distance(zero, other) == 1.0
    dists = cosine_distance_batch(
        np.stack([zero, other]), other, np.array([0.0, 4.0], dtype=np.float32)
    )
    assert dists[0] == pytest.approx(1.0)
    assert dists[1] == pytest.approx(0.0, abs=1e-6)


def test_cosine_rank_order(rng):
    x = rng.normal(size=(20, 8)).astype(np.float32)
    q = rng.normal(size=8).astype(np.float32)
    batch = cosine_distance_batch(x, q, np.einsum("ij,ij->i", x, x))
    scalar = np.array([cosine_distance(row, q) for row in x])
    np.testing.assert_allclose(batch, scalar, rtol=1e-5, atol=1e-6)


def test_inner_product_returns_negative_dot(rng):
    a = rng.normal(size=8).astype(np.float32)
    b = rng.normal(size=8).astype(np.float32)
    assert inner_product_distance(a, b) == pytest.approx(-float(np.dot(a, b)))


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_distance_one_and_batch_agree(metric, rng):
    x = rng.normal(size=(13, 6)).astype(np.float32)
    q = rng.normal(size=6).astype(np.float32)
    norms = np.einsum("ij,ij->i", x, x)
    batch = distance_batch(x, q, metric, x_norms=norms, q_norm=float(np.dot(q, q)))
    scalar = np.array(
        [
            distance_one(row, q, metric, a_norm=float(n), b_norm=float(np.dot(q, q)))
            for row, n in zip(x, norms)
        ]
    )
    np.testing.assert_allclose(batch, scalar, rtol=1e-4, atol=1e-4)


def test_unknown_metric_rejected(rng):
    with pytest.raises(ValueError):
        distance_batch(rng.normal(size=(2, 3)), rng.normal(size=3), "hamming")  # type: ignore[arg-type]


def test_normalize_rows(rng):
    x = rng.normal(size=(9, 5)).astype(np.float32)
    normalized = normalize_rows(x)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(9), rtol=1e-6)
    zero_rows = normalize_rows(np.zeros((2, 5), dtype=np.float32))
    assert np.all(np.isfinite(zero_rows))
