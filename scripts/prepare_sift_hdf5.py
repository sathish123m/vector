#!/usr/bin/env python
"""Convert an ann-benchmarks SIFT HDF5 file into TEXMEX .fvecs/.ivecs files.

Kaggle blocks some outbound FTP traffic, so the official TEXMEX host may be
unreachable there.  The ann-benchmarks project publishes the same SIFT1M
vectors over HTTPS:

    https://ann-benchmarks.com/sift-128-euclidean.hdf5

That file contains ``train``, ``test``, ``neighbors``, and ``distances``.  This
script writes the layout the ChurnVec harness expects:

    data/raw/sift/sift_base.fvecs
    data/raw/sift/sift_query.fvecs
    data/raw/sift/sift_groundtruth.ivecs

With ``--siftsmall`` it also derives a 10K SIFT subset with recomputed exact
ground truth for the test suite.  The vectors are unchanged; only the
container format differs from the official archive.

Requires ``h5py`` (``pip install h5py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnvec.bench.datasets import write_fvecs, write_ivecs  # noqa: E402


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SystemExit("h5py is required: pip install h5py") from exc
    return h5py


def convert(source: Path, out_dir: Path, siftsmall: bool) -> None:
    h5py = _require_h5py()
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        for key in ("train", "test", "neighbors"):
            if key not in handle:
                raise SystemExit(f"{source} has no '{key}' dataset")
        train = np.asarray(handle["train"], dtype=np.float32)
        test = np.asarray(handle["test"], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"], dtype=np.int32)

    if train.ndim != 2 or test.ndim != 2:
        raise SystemExit("train/test must be 2-D")
    if train.shape[1] != test.shape[1]:
        raise SystemExit("train/test dimension mismatch")

    print(f"[hdf5] base={train.shape} queries={test.shape} gt={neighbors.shape}")
    write_fvecs(out_dir / "sift_base.fvecs", train)
    write_fvecs(out_dir / "sift_query.fvecs", test)
    write_ivecs(out_dir / "sift_groundtruth.ivecs", neighbors)
    print(f"[hdf5] wrote {out_dir}")

    if not siftsmall:
        return

    # The test suite only needs a small SIFT set with exact ground truth.
    n_base = min(10_000, train.shape[0])
    n_query = min(100, test.shape[0])
    small_dir = out_dir.parent / "siftsmall"
    small_dir.mkdir(parents=True, exist_ok=True)
    base = train[:n_base]
    queries = test[:n_query]
    print(f"[hdf5] deriving siftsmall: base={base.shape} queries={queries.shape}")

    # Exact L2 top-100, chunked over queries to bound memory.
    k = 100
    ids = np.empty((n_query, k), dtype=np.int32)
    for start in range(0, n_query, 20):
        chunk = queries[start : start + 20]
        dists = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            - 2.0 * (chunk @ base.T)
            + np.sum(base * base, axis=1, keepdims=True).T
        )
        ids[start : start + chunk.shape[0]] = np.argsort(dists, axis=1)[:, :k]

    write_fvecs(small_dir / "siftsmall_base.fvecs", base)
    write_fvecs(small_dir / "siftsmall_query.fvecs", queries)
    write_fvecs(small_dir / "siftsmall_learn.fvecs", base[:10_000])
    write_ivecs(small_dir / "siftsmall_groundtruth.ivecs", ids)
    print(f"[hdf5] wrote {small_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ann-benchmarks .hdf5 file")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "raw" / "sift",
        help="destination directory for sift_*.fvecs",
    )
    parser.add_argument(
        "--siftsmall",
        action="store_true",
        help="also derive data/raw/siftsmall for the test suite",
    )
    args = parser.parse_args()
    if not args.source.exists():
        raise SystemExit(f"no such file: {args.source}")
    convert(args.source, args.out, args.siftsmall)


if __name__ == "__main__":
    main()
