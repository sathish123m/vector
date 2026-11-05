"""TEXMEX readers and synthetic dataset adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churnvec.bench.datasets import (
    Dataset,
    load_dataset,
    load_or_compute_groundtruth,
    read_fvecs,
    read_ivecs,
    write_fvecs,
    write_ivecs,
)


def test_fvecs_roundtrip(tmp_path, rng):
    data = rng.normal(size=(11, 7)).astype(np.float32)
    path = tmp_path / "x.fvecs"
    write_fvecs(path, data)
    loaded = read_fvecs(path)
    np.testing.assert_array_equal(loaded, data)


def test_ivecs_roundtrip(tmp_path, rng):
    data = rng.integers(0, 100, size=(13, 5), dtype=np.int32)
    path = tmp_path / "x.ivecs"
    write_ivecs(path, data)
    loaded = read_ivecs(path)
    np.testing.assert_array_equal(loaded, data)


def test_limit_and_empty(tmp_path):
    path = tmp_path / "x.fvecs"
    write_fvecs(path, np.ones((5, 3), dtype=np.float32))
    assert read_fvecs(path, limit=2).shape == (2, 3)
    empty = tmp_path / "empty.fvecs"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        read_fvecs(empty)


def test_reader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fvecs(tmp_path / "missing.fvecs")


def _make_siftsmall(root: Path, n=30, d=6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(n, d)).astype(np.float32)
    queries = rng.normal(size=(4, d)).astype(np.float32)
    folder = root / "siftsmall"
    folder.mkdir(parents=True)
    write_fvecs(folder / "siftsmall_base.fvecs", base)
    write_fvecs(folder / "siftsmall_query.fvecs", queries)
    write_fvecs(folder / "siftsmall_learn.fvecs", base[:10])
    write_ivecs(folder / "siftsmall_groundtruth.ivecs", np.tile(np.arange(10), (4, 1)))
    return base, queries


def test_load_dataset_siftsmall(tmp_path):
    base, queries = _make_siftsmall(tmp_path)
    ds = load_dataset("siftsmall", root=tmp_path, include_learn=True)
    assert ds.name == "siftsmall"
    assert ds.n_base == 30
    assert ds.n_queries == 4
    assert ds.dim == 6
    np.testing.assert_array_equal(ds.base, base)
    np.testing.assert_array_equal(ds.queries, queries)
    assert ds.groundtruth is not None
    assert ds.learn is not None
    assert ds.summary()["n_base"] == 30


def test_load_dataset_sift100k_uses_prefix(tmp_path):
    root = tmp_path
    folder = root / "sift"
    folder.mkdir()
    rng = np.random.default_rng(4)
    base = rng.normal(size=(200, 5)).astype(np.float32)
    queries = rng.normal(size=(3, 5)).astype(np.float32)
    write_fvecs(folder / "sift_base.fvecs", base)
    write_fvecs(folder / "sift_query.fvecs", queries)
    write_fvecs(folder / "sift_learn.fvecs", base[:20])
    # No ground truth: sift100k must not silently use the 1M GT.
    ds = load_dataset("sift100k", root=root)
    assert ds.name == "sift100k"
    assert ds.n_base == 200  # min(100_000, file size)
    assert ds.groundtruth is None


def test_dataset_validation():
    with pytest.raises(ValueError):
        Dataset(
            name="bad",
            base=np.zeros((3, 4), dtype=np.float32),
            queries=np.zeros((2, 5), dtype=np.float32),
            groundtruth=None,
            dim=4,
        )
    with pytest.raises(ValueError):
        load_dataset("unknown")


def test_groundtruth_cache(tmp_path, rng):
    base = rng.normal(size=(40, 5)).astype(np.float32)
    queries = rng.normal(size=(6, 5)).astype(np.float32)
    cache = tmp_path / "gt" / "gt.npy"
    first = load_or_compute_groundtruth(base, queries, cache, k=7)
    assert cache.exists()
    second = load_or_compute_groundtruth(base, queries, cache, k=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (6, 7)


@pytest.mark.dataset
def test_real_siftsmall_if_available():
    root = Path("data/raw")
    if not (root / "siftsmall" / "siftsmall_base.fvecs").exists():
        pytest.skip("SIFT data not downloaded")
    ds = load_dataset("siftsmall", root=root)
    assert ds.n_base == 10_000
    assert ds.dim == 128
    assert ds.groundtruth is not None
    assert ds.groundtruth.shape[0] == ds.n_queries
