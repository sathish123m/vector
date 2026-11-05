"""Snapshot isolation and concurrent reader/writer tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from churnvec import ChurnIndex


def _build_index(n: int = 600, dim: int = 8) -> tuple[ChurnIndex, np.ndarray]:
    rng = np.random.default_rng(2024)
    base = rng.normal(size=(n, dim)).astype(np.float32)
    index = ChurnIndex(dim=dim, M=8, ef_construction=60, flush_threshold=128, seed=9)
    index.add_many(base, np.arange(n))
    index.flush()
    return index, base


def test_parallel_readers_match_serial_results():
    index, base = _build_index()
    queries = base[::7][:60]
    expected = [index.search_ids(query, k=10, ef=64) for query in queries]

    def worker(seed: int):
        order = np.random.default_rng(seed).permutation(len(queries))
        results = []
        for position in order:
            results.append((int(position),) + index.search_ids(queries[position], k=10, ef=64))
        return results

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, seed) for seed in range(8)]
        for future in futures:
            for position, ids, distances in future.result():
                expected_ids, expected_distances = expected[position]
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_allclose(distances, expected_distances, rtol=1e-6, atol=1e-6)


def test_reader_snapshot_is_self_consistent_during_writes():
    index, base = _build_index(n=300)
    stop = threading.Event()
    errors: list[str] = []
    observations = {"snapshots": 0}

    def reader(seed: int):
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            snapshot = index.snapshot()
            query = base[int(rng.integers(0, base.shape[0]))]
            ids, distances = snapshot.search_ids(query, k=10, ef=64)
            if ids.size != min(10, snapshot.live_count):
                errors.append(f"expected 10 results, got {ids.size}")
            if not np.all(np.isfinite(distances)):
                errors.append("non-finite distance")
            for external_id in ids.tolist():
                if not snapshot.contains(int(external_id)):
                    errors.append(f"snapshot returned non-live id {external_id}")
            live_ids, _ = snapshot.live_arrays()
            if len(set(live_ids.tolist())) != snapshot.live_count:
                errors.append("live_count does not match live arrays")
            observations["snapshots"] += 1

    threads = [threading.Thread(target=reader, args=(seed,)) for seed in range(4)]
    for thread in threads:
        thread.start()

    rng = np.random.default_rng(77)
    for step in range(180):
        external_id = int(rng.integers(0, 300))
        if step % 3 == 0:
            index.delete(external_id)
        elif step % 3 == 1:
            index.upsert(external_id, rng.normal(size=8).astype(np.float32))
        else:
            index.add(10_000 + step, rng.normal(size=8).astype(np.float32))
        if step % 40 == 0:
            index.flush()

    stop.set()
    for thread in threads:
        thread.join()
    assert not errors, errors[:5]
    assert observations["snapshots"] > 0


def test_concurrent_writers_are_serialised():
    index = ChurnIndex(dim=4, M=4, ef_construction=20, flush_threshold=50, seed=4)
    rng = np.random.default_rng(5)
    vectors = {i: rng.normal(size=4).astype(np.float32) for i in range(200)}

    def writer(start: int):
        for external_id in range(start, start + 50):
            index.add(external_id, vectors[external_id])

    threads = [threading.Thread(target=writer, args=(start,)) for start in range(0, 200, 50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(index) == 200
    live_ids, _ = index.snapshot().live_arrays()
    assert set(live_ids.tolist()) == set(range(200))
    for external_id in range(200):
        assert external_id in index


def test_readers_do_not_block_each_other_or_writer_progress():
    index, base = _build_index(n=200)
    stop = threading.Event()
    read_count = {"value": 0}

    def reader():
        while not stop.is_set():
            index.search_ids(base[0], k=10, ef=32)
            read_count["value"] += 1

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for thread in threads:
        thread.start()

    rng = np.random.default_rng(1)
    for step in range(100):
        index.add(5000 + step, rng.normal(size=8).astype(np.float32))
    stop.set()
    for thread in threads:
        thread.join()

    assert len(index) == 300
    assert read_count["value"] > 0
