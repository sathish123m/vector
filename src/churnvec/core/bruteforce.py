"""Exact top-k search, used as ground truth and as a correctness oracle."""

from __future__ import annotations

import numpy as np

from .distance import Metric


def _auto_chunk_size(n_base: int, target_bytes: int = 256 * 1024 * 1024) -> int:
    # float32 matrix is n_query_chunk * n_base * 4 bytes.
    per_row = max(1, 4 * n_base)
    return max(1, min(2048, target_bytes // per_row))


def brute_force_topk(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    metric: Metric = "l2",
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k IDs and distances.

    Queries are processed in chunks so that SIFT1M (10K x 1M) never
    materialises a 40 GB distance matrix.
    """
    x = np.ascontiguousarray(base, dtype=np.float32)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if x.ndim != 2 or q.ndim != 2:
        raise ValueError("base and queries must be 2-D")
    if x.shape[1] != q.shape[1]:
        raise ValueError(f"dimension mismatch: {x.shape[1]} vs {q.shape[1]}")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), x.shape[0])

    n_base = x.shape[0]
    x_norms = np.einsum("ij,ij->i", x, x).astype(np.float32)
    q_norms = np.einsum("ij,ij->i", q, q).astype(np.float32)
    if chunk_size is None:
        chunk_size = _auto_chunk_size(n_base)
    chunk_size = max(1, int(chunk_size))

    out_ids = np.empty((q.shape[0], k), dtype=np.int64)
    out_dists = np.empty((q.shape[0], k), dtype=np.float32)

    for start in range(0, q.shape[0], chunk_size):
        stop = min(start + chunk_size, q.shape[0])
        qc = q[start:stop]
        if metric == "l2":
            dots = qc @ x.T
            dists = q_norms[start:stop, None] + x_norms[None, :] - 2.0 * dots
            np.maximum(dists, 0.0, out=dists)
        elif metric == "cosine":
            denom = np.sqrt(np.maximum(x_norms[None, :] * q_norms[start:stop, None], 1e-12))
            dists = 1.0 - (qc @ x.T) / denom
        elif metric == "ip":
            dists = -(qc @ x.T)
        else:
            raise ValueError(f"unknown metric: {metric}")

        if k == n_base:
            idx = np.argsort(dists, axis=1)[:, :k]
        else:
            idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        selected = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(selected, axis=1)
        out_ids[start:stop] = np.take_along_axis(idx, order, axis=1)
        out_dists[start:stop] = np.take_along_axis(selected, order, axis=1)

    return out_ids, out_dists


def exact_recall_at_k(returned_ids: np.ndarray, truth_ids: np.ndarray, k: int | None = None) -> float:
    if k is None:
        k = returned_ids.shape[1]
    if returned_ids.shape[0] != truth_ids.shape[0]:
        raise ValueError("returned and truth must have the same number of queries")
    k = min(int(k), returned_ids.shape[1], truth_ids.shape[1])
    hits = 0
    for returned, truth in zip(returned_ids[:, :k], truth_ids[:, :k]):
        hits += len(set(int(x) for x in returned) & set(int(x) for x in truth))
    return float(hits / (returned_ids.shape[0] * k))
