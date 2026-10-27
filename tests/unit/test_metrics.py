"""Metrics and reporting helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from churnvec.bench.metrics import (
    RSSMonitor,
    current_rss_mb,
    latency_stats,
    recall_at_k,
    write_json,
)


def test_recall_at_k():
    returned = np.array([[1, 2, 3], [4, 5, 6]])
    truth = np.array([[1, 9, 3], [7, 8, 9]])
    assert recall_at_k(returned, truth) == pytest.approx(2 / 6)
    assert recall_at_k(returned, truth, k=2) == pytest.approx(1 / 4)


def test_latency_stats():
    samples = np.array([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    stats = latency_stats(samples)
    assert stats["count"] == 4
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert stats["max_ms"] == pytest.approx(4.0)
    assert stats["qps"] == pytest.approx(4000 / 10)
    assert latency_stats([]) == {"count": 0}


def test_rss_monitor():
    start = current_rss_mb()
    assert start > 0
    with RSSMonitor(interval_s=0.001) as monitor:
        _ = np.zeros((1024, 1024), dtype=np.float32)
    assert monitor.start_mb > 0
    assert monitor.peak_mb >= monitor.start_mb


def test_write_json_handles_numpy(tmp_path):
    path = tmp_path / "nested" / "result.json"
    write_json(
        path,
        {
            "int": np.int64(3),
            "float": np.float32(1.5),
            "array": np.arange(3),
            "path": tmp_path,
        },
    )
    payload = json.loads(path.read_text())
    assert payload["int"] == 3
    assert payload["array"] == [0, 1, 2]
    assert payload["path"] == str(tmp_path)


def test_write_json_rejects_unknown(tmp_path):
    with pytest.raises(TypeError):
        write_json(tmp_path / "x.json", {"bad": object()})
