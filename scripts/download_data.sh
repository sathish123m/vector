#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

FTP="ftp://ftp.irisa.fr/local/texmex/corpus"

# Official TEXMEX checksums, from ftp://ftp.irisa.fr/local/texmex/corpus/MD5SUM
SIFT_MD5="b23d1b3b2ee8469d819b61ca900ef0ed"
SMALL_MD5="0b8324a7a82d7f2663d7dcbd57642df7"

md5_of() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{print $1}'
  fi
}

fetch() {
  local name="$1" expected="$2"
  local path="$RAW/$name"
  if [[ -f "$path" ]] && [[ "$(md5_of "$path")" == "$expected" ]]; then
    echo "[data] $name already present and verified"
    return
  fi
  echo "[data] downloading $name"
  curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
    --output "$path" "$FTP/$name"
  local actual
  actual="$(md5_of "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "[data] checksum mismatch for $name: expected $expected, got $actual" >&2
    exit 1
  fi
  echo "[data] $name md5 ok ($actual)"
}

fetch "siftsmall.tar.gz" "$SMALL_MD5"
fetch "sift.tar.gz" "$SIFT_MD5"

if [[ ! -d "$RAW/siftsmall" ]]; then
  echo "[data] extracting siftsmall.tar.gz"
  tar -xzf "$RAW/siftsmall.tar.gz" -C "$RAW"
fi
if [[ ! -d "$RAW/sift" ]]; then
  echo "[data] extracting sift.tar.gz"
  tar -xzf "$RAW/sift.tar.gz" -C "$RAW"
fi

echo "[data] files:"
ls -lh "$RAW/sift" "$RAW/siftsmall"
