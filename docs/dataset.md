# SIFT dataset notes

## Source

The official IRISA/CNRS TEXMEX corpus:

```text
ftp://ftp.irisa.fr/local/texmex/corpus/
```

The project uses the official `sift.tar.gz` and `siftsmall.tar.gz` archives.
The site’s HTTPS certificate is broken; the FTP source and official MD5s are
used instead of a mirror.

## Checksums

| Archive | MD5 |
|---|---|
| `sift.tar.gz` | `b23d1b3b2ee8469d819b61ca900ef0ed` |
| `siftsmall.tar.gz` | `0b8324a7a82d7f2663d7dcbd57642df7` |

`scripts/download_data.sh` downloads, resumes, verifies, and extracts both.

## Shapes

| Set | Base | Queries | Learn | Ground truth |
|---|---:|---:|---:|---:|
| SIFT10K (`siftsmall`) | 10,000 × 128 | 100 × 128 | 25,000 × 128 | 100 × 100 IDs |
| SIFT1M (`sift`) | 1,000,000 × 128 | 10,000 × 128 | 100,000 × 128 | 10,000 × 100 IDs |

SIFT10K has **100 queries**, not 10,000. SIFT1M has 10,000 queries.

## File format

TEXMEX `.fvecs`:

- Each record is `int32 dimension` followed by `dimension` little-endian
  float32 values.
- SIFT dimension is 128, so each base record is 516 bytes.
- `.ivecs` is the same layout with int32 payloads.

Readers live in `src/churnvec/bench/datasets.py`.

## Ground-truth semantics

- IDs are 0-based positions in the base file.
- The metric is squared L2.
- The shipped GT is valid only for the exact original base snapshot.
- It is invalid for a 100K prefix, for any base after inserts/deletes, or for
  cosine/inner product.
- The benchmark harness recomputes exact GT for a different snapshot and
  caches it under `data/processed/`.
- The SIFT100K cache is `gt_sift100k_100000_q10000_k100.npy`, shape
  `(10000, 100)`, 3.8 MB.

A materialised 10K × 1M float32 distance matrix would be 40 GB. Exact GT is
therefore computed in query chunks, keeping only top-k. The current cache is
small because only IDs are stored.

## Why SIFT

- Standard, public-domain, citable benchmark.
- 128-d L2 workload with official queries and GT.
- 100K is small enough for repeated runs on an 8 GB laptop and large enough
  to expose graph-saturation issues (the 15K–20K fragmentation only appeared
  above 10K).
- SIFT1M remains the eventual primary static benchmark; it was not built in
  this milestone.

## Churn implications

SIFT1M is static. Future churn experiments should:

1. hold out base vectors for inserts;
2. delete by stable external ID;
3. recompute exact GT over the live snapshot at checkpoints;
4. never compare against the shipped static GT after mutation.

The current HNSW external-ID map and snapshot-based GT cache are the starting
points for that work, but deletion itself is not implemented yet.
