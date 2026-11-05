"""Readers and adapters for the TEXMEX SIFT benchmark files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Read a ``.fvecs`` file into a contiguous ``(n, d) float32`` array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    first = np.fromfile(path, dtype="<i4", count=1)
    if first.size != 1:
        raise ValueError(f"empty or truncated .fvecs file: {path}")
    dim = int(first[0])
    if dim <= 0:
        raise ValueError(f"invalid dimension {dim} in {path}")
    dtype = np.dtype([("d", "<i4"), ("v", "<f4", (dim,))])
    if limit is None:
        data = np.fromfile(path, dtype=dtype)
    else:
        data = np.fromfile(path, dtype=dtype, count=limit)
    if data.size == 0:
        return np.empty((0, dim), dtype=np.float32)
    if not np.all(data["d"] == dim):
        raise ValueError(f"inconsistent record dimensions in {path}")
    return np.ascontiguousarray(data["v"], dtype=np.float32)


def read_ivecs(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Read a ``.ivecs`` file into a contiguous ``(n, d) int32`` array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    first = np.fromfile(path, dtype="<i4", count=1)
    if first.size != 1:
        raise ValueError(f"empty or truncated .ivecs file: {path}")
    dim = int(first[0])
    if dim <= 0:
        raise ValueError(f"invalid dimension {dim} in {path}")
    dtype = np.dtype([("d", "<i4"), ("v", "<i4", (dim,))])
    if limit is None:
        data = np.fromfile(path, dtype=dtype)
    else:
        data = np.fromfile(path, dtype=dtype, count=limit)
    if data.size == 0:
        return np.empty((0, dim), dtype=np.int32)
    if not np.all(data["d"] == dim):
        raise ValueError(f"inconsistent record dimensions in {path}")
    return np.ascontiguousarray(data["v"], dtype=np.int32)


def write_fvecs(path: str | Path, vectors: np.ndarray) -> None:
    """Write ``(n, d)`` vectors in TEXMEX ``.fvecs`` format (for tests)."""
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("vectors must be 2-D")
    dim = arr.shape[1]
    dtype = np.dtype([("d", "<i4"), ("v", "<f4", (dim,))])
    out = np.empty(arr.shape[0], dtype=dtype)
    out["d"] = dim
    out["v"] = arr
    out.tofile(path)


def write_ivecs(path: str | Path, values: np.ndarray) -> None:
    arr = np.ascontiguousarray(values, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("values must be 2-D")
    dim = arr.shape[1]
    dtype = np.dtype([("d", "<i4"), ("v", "<i4", (dim,))])
    out = np.empty(arr.shape[0], dtype=dtype)
    out["d"] = dim
    out["v"] = arr
    out.tofile(path)


@dataclass
class Dataset:
    name: str
    base: np.ndarray
    queries: np.ndarray
    groundtruth: np.ndarray | None
    dim: int
    learn: np.ndarray | None = None
    base_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.base.ndim != 2 or self.queries.ndim != 2:
            raise ValueError("base and queries must be 2-D")
        if self.base.shape[1] != self.queries.shape[1]:
            raise ValueError("base/query dimension mismatch")
        if self.dim != self.base.shape[1]:
            raise ValueError("declared dim does not match data")
        if self.base_ids is None:
            self.base_ids = np.arange(self.base.shape[0], dtype=np.int64)

    @property
    def n_base(self) -> int:
        return self.base.shape[0]

    @property
    def n_queries(self) -> int:
        return self.queries.shape[0]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_base": self.n_base,
            "n_queries": self.n_queries,
            "dim": self.dim,
            "groundtruth_shape": None if self.groundtruth is None else list(self.groundtruth.shape),
            "base_bytes": int(self.base.nbytes),
            "query_bytes": int(self.queries.nbytes),
            "learn_shape": None if self.learn is None else list(self.learn.shape),
        }


_PATHS = {
    "siftsmall": {
        "base": "siftsmall/siftsmall_base.fvecs",
        "query": "siftsmall/siftsmall_query.fvecs",
        "gt": "siftsmall/siftsmall_groundtruth.ivecs",
        "learn": "siftsmall/siftsmall_learn.fvecs",
    },
    "sift1m": {
        "base": "sift/sift_base.fvecs",
        "query": "sift/sift_query.fvecs",
        "gt": "sift/sift_groundtruth.ivecs",
        "learn": "sift/sift_learn.fvecs",
    },
}


def load_dataset(
    name: str,
    root: str | Path = "data/raw",
    base_limit: int | None = None,
    query_limit: int | None = None,
    include_learn: bool = False,
) -> Dataset:
    """Load one of the supported SIFT sets.

    ``sift100k`` is a prefix of SIFT1M's base file; its ground truth is not
    shipped and must be computed by the caller.
    """
    root = Path(root)
    name = name.lower()
    if name in {"sift10k", "siftsmall"}:
        spec = _PATHS["siftsmall"]
        dataset_name = "siftsmall"
        compute_gt = False
    elif name in {"sift100k", "sift1m"}:
        spec = _PATHS["sift1m"]
        dataset_name = "sift1m"
        compute_gt = name == "sift100k"
        if name == "sift100k":
            base_limit = 100_000 if base_limit is None else min(base_limit, 100_000)
    else:
        raise ValueError(f"unknown dataset: {name}")

    base = read_fvecs(root / spec["base"], limit=base_limit)
    queries = read_fvecs(root / spec["query"], limit=query_limit)
    gt = None
    if not compute_gt and (root / spec["gt"]).exists():
        gt = read_ivecs(root / spec["gt"], limit=query_limit)
    learn = read_fvecs(root / spec["learn"]) if include_learn else None
    return Dataset(
        name=dataset_name if not compute_gt else "sift100k",
        base=base,
        queries=queries,
        groundtruth=gt,
        dim=base.shape[1],
        learn=learn,
    )


def load_or_compute_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    cache_path: str | Path,
    k: int = 100,
    metric: str = "l2",
) -> np.ndarray:
    """Load cached exact top-k IDs or compute and cache them."""
    from ..core.bruteforce import brute_force_topk

    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (queries.shape[0], k):
            return cached.astype(np.int32, copy=False)
    ids, _ = brute_force_topk(base, queries, k=k, metric=metric)  # type: ignore[arg-type]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, ids.astype(np.int32))
    return ids.astype(np.int32, copy=False)
