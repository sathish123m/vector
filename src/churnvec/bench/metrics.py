"""Measurement utilities: recall, latency, and memory."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from ..core.bruteforce import exact_recall_at_k


def recall_at_k(returned_ids: np.ndarray, truth_ids: np.ndarray, k: int | None = None) -> float:
    return exact_recall_at_k(returned_ids, truth_ids, k=k)


def latency_stats(samples_ns: np.ndarray | list[float]) -> dict:
    arr = np.asarray(samples_ns, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    arr_ms = arr / 1e6
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr_ms)),
        "p50_ms": float(np.percentile(arr_ms, 50)),
        "p95_ms": float(np.percentile(arr_ms, 95)),
        "p99_ms": float(np.percentile(arr_ms, 99)),
        "min_ms": float(np.min(arr_ms)),
        "max_ms": float(np.max(arr_ms)),
        "qps": float(arr.size / max(float(np.sum(arr)) / 1e9, 1e-12)),
    }


def current_rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


class RSSMonitor:
    """Sample process RSS in a background thread and retain the peak."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = float(interval_s)
        self.peak_mb = 0.0
        self.start_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, float(rss))
            except psutil.Error:
                break
            self._stop.wait(self.interval_s)
        try:
            self.peak_mb = max(self.peak_mb, float(process.memory_info().rss / (1024 * 1024)))
        except psutil.Error:
            pass

    def __enter__(self) -> "RSSMonitor":
        self.start_mb = current_rss_mb()
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def write_json(path: str | Path, payload: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter_ns() - start
    return result, elapsed
