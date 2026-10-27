"""End-to-end SIFT10K smoke test (requires downloaded data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churnvec import HNSWIndex, brute_force_topk
from churnvec.bench.datasets import load_dataset
from churnvec.bench.metrics import recall_at_k

SIFT_ROOT = Path("data/raw")


@pytest.fixture(scope="module")
def siftsmall():
    if not (SIFT_ROOT / "siftsmall" / "siftsmall_base.fvecs").exists():
        pytest.skip("SIFT data not downloaded; run make data")
    return load_dataset("siftsmall", root=SIFT_ROOT)


@pytest.mark.slow
@pytest.mark.dataset
def test_sift10k_recall_smoke(siftsmall):
    index = HNSWIndex(
        dim=siftsmall.dim,
        M=16,
        ef_construction=100,
        seed=42,
        capacity=siftsmall.n_base,
    )
    index.add_items(siftsmall.base, siftsmall.base_ids)
    report = index.check_invariants()
    assert report.ok
    assert report.unreachable == 0

    query_count = min(1000, siftsmall.n_queries)
    returned = []
    for query in siftsmall.queries[:query_count]:
        ids, _ = index.search_ids(query, k=10, ef=64)
        returned.append(ids)
    recall = recall_at_k(np.stack(returned), siftsmall.groundtruth[:query_count], k=10)
    assert recall >= 0.95, f"SIFT10K recall@10={recall:.4f}"


@pytest.mark.dataset
def test_sift10k_bruteforce_matches_shipped_groundtruth(siftsmall):
    # Distances are compared rather than IDs so equal-distance ties are safe.
    query_count = 20
    k = 10
    _, dists = brute_force_topk(
        siftsmall.base, siftsmall.queries[:query_count], k=k
    )
    truth_ids = siftsmall.groundtruth[:query_count, :k]
    truth_vecs = siftsmall.base[truth_ids]
    truth_dists = np.sum(
        (truth_vecs - siftsmall.queries[:query_count, None, :]) ** 2, axis=2
    )
    np.testing.assert_allclose(dists, truth_dists, rtol=1e-3, atol=1e-2)
