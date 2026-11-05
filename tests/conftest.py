"""Shared test helpers."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def clustered_data():
    """Small, well-separated clusters make exact-neighbour assertions stable."""
    rng = np.random.default_rng(7)
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    points = np.concatenate(
        [center + 0.05 * rng.normal(size=(20, 4)).astype(np.float32) for center in centers]
    )
    queries = centers + 0.01 * rng.normal(size=(5, 4)).astype(np.float32)
    return points.astype(np.float32), queries.astype(np.float32)
