"""Distance kernels.

All distances are "smaller is closer".  The HNSW implementation uses squared
L2 as its fast path because SIFT1M is an L2 benchmark and because it avoids a
square root in the inner loop.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Metric = Literal["l2", "cosine", "ip"]

_EPS = 1e-12


def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def l2_sqr(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two vectors."""
    aa = _as_float32(a)
    bb = _as_float32(b)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch: {aa.shape} vs {bb.shape}")
    diff = aa - bb
    return float(np.dot(diff, diff))


def l2_sqr_batch(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Squared L2 distances from every row of ``x`` to ``q``."""
    xx = _as_float32(x)
    qq = _as_float32(q)
    if xx.ndim != 2:
        raise ValueError("x must be 2-D")
    if qq.shape != (xx.shape[1],):
        raise ValueError(f"q must have shape ({xx.shape[1]},), got {qq.shape}")
    diff = xx - qq
    return np.einsum("ij,ij->i", diff, diff, optimize=True).astype(np.float32, copy=False)


def l2_sqr_batch_from_norms(
    x: np.ndarray,
    q: np.ndarray,
    x_norms: np.ndarray,
    q_norm: float | None = None,
) -> np.ndarray:
    """Squared L2 using precomputed squared norms.

    This is the hot-path kernel used by HNSW.  It avoids materialising an
    ``(n, d)`` difference matrix and lets BLAS handle the dot products.
    """
    xx = _as_float32(x)
    qq = _as_float32(q)
    if q_norm is None:
        q_norm = float(np.dot(qq, qq))
    dots = xx @ qq
    d2 = x_norms.astype(np.float32, copy=False) - 2.0 * dots + np.float32(q_norm)
    # Round-off can produce tiny negative values for identical vectors.
    return np.maximum(d2, 0.0, dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = _as_float32(a)
    bb = _as_float32(b)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= _EPS:
        return 1.0
    return float(1.0 - np.dot(aa, bb) / denom)


def cosine_distance_batch(x: np.ndarray, q: np.ndarray, x_norms: np.ndarray | None = None) -> np.ndarray:
    xx = _as_float32(x)
    qq = _as_float32(q)
    if x_norms is None:
        x_norms = np.einsum("ij,ij->i", xx, xx)
    q_norm = float(np.dot(qq, qq))
    denom = np.sqrt(np.maximum(x_norms * q_norm, _EPS))
    return (1.0 - (xx @ qq) / denom).astype(np.float32)


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Negative inner product, so that smaller is closer."""
    return float(-np.dot(_as_float32(a), _as_float32(b)))


def inner_product_distance_batch(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (-(_as_float32(x) @ _as_float32(q))).astype(np.float32)


def distance_batch(
    x: np.ndarray,
    q: np.ndarray,
    metric: Metric,
    x_norms: np.ndarray | None = None,
    q_norm: float | None = None,
) -> np.ndarray:
    if metric == "l2":
        if x_norms is not None:
            return l2_sqr_batch_from_norms(x, q, x_norms, q_norm)
        return l2_sqr_batch(x, q)
    if metric == "cosine":
        return cosine_distance_batch(x, q, x_norms)
    if metric == "ip":
        return inner_product_distance_batch(x, q)
    raise ValueError(f"unknown metric: {metric}")


def distance_one(
    a: np.ndarray,
    b: np.ndarray,
    metric: Metric,
    a_norm: float | None = None,
    b_norm: float | None = None,
) -> float:
    if metric == "l2":
        if a_norm is not None and b_norm is not None:
            d2 = float(a_norm) - 2.0 * float(np.dot(a, b)) + float(b_norm)
            return max(d2, 0.0)
        return l2_sqr(a, b)
    if metric == "cosine":
        return cosine_distance(a, b)
    if metric == "ip":
        return inner_product_distance(a, b)
    raise ValueError(f"unknown metric: {metric}")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = _as_float32(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, _EPS)
