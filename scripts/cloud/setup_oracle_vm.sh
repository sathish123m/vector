#!/usr/bin/env bash
# One-shot bootstrap for an Oracle Cloud Always Free ARM VM (Ubuntu 24.04).
# Safe to re-run: it updates the clone and re-checks the environment.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sathish123m/vector.git}"
SWAP_GB="${SWAP_GB:-8}"
TARGET="${TARGET:-$HOME/churnvec}"

log() { printf '\n\033[1;36m[churnvec]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[churnvec:error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This script targets Linux (Ubuntu on Oracle Cloud)."

if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y git make build-essential python3-dev python3-venv curl ca-certificates

if ! swapon --show | grep -q .; then
  log "Creating ${SWAP_GB} GB swap"
  $SUDO fallocate -l "${SWAP_GB}G" /swapfile 2>/dev/null \
    || $SUDO dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024)) status=none
  $SUDO chmod 600 /swapfile
  $SUDO mkswap /swapfile >/dev/null
  $SUDO swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null
else
  log "Swap already active"
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv installation failed; check your PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../../Makefile" && -d "$SCRIPT_DIR/../../src" ]]; then
  TARGET="$(cd "$SCRIPT_DIR/../.." && pwd)"
  log "Using existing repository at $TARGET"
elif [[ -d "$TARGET/.git" ]]; then
  log "Updating repository at $TARGET"
  git -C "$TARGET" pull --ff-only
else
  log "Cloning $REPO_URL into $TARGET"
  git clone "$REPO_URL" "$TARGET"
fi

cd "$TARGET"

log "Installing Python dependencies"
make setup

log "Downloading and verifying SIFT data"
make data

log "Running the test suite"
make test

log "Running the SIFT10K smoke benchmark"
make e0

log "Environment"
.venv/bin/python - <<'PY'
import platform

import numpy
import psutil

print("Python :", platform.python_version())
print("NumPy  :", numpy.__version__)
print("Machine:", platform.machine())
print("CPU    :", platform.processor() or "unknown")
print("RAM GB :", round(psutil.virtual_memory().total / 1e9, 1))
PY
free -h || true
df -h "$TARGET" || true

log "Setup complete."
log "Next: make e1, make e2, make e2-100k, make e3"
