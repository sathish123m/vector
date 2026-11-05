"""Create a small JSON manifest for the downloaded SIFT files."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".fvecs", ".ivecs", ".tar.gz"})
    manifest = {
        "root": str(root),
        "files": [
            {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in files
        ],
    }
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
