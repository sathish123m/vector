#!/usr/bin/env bash
# One-shot bootstrap for an Ubuntu benchmark host.
# Works on Oracle Cloud, GitHub Codespaces, and other Ubuntu VMs/containers.
# Safe to re-run: it updates the clone and re-checks the environment.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sathish123m/vector.git}"
SWAP_GB="${SWAP_GB:-8}"
TARGET="${TARGET:-$HOME/churnvec}"

log() { printf '\n\033[1;36m[churnvec]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[churnvec:warn]\033[0m %s\n' "$*" >&2; }
die() { printf '\n\033[1;31m[churnvec:error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This script targets Ubuntu Linux hosts (Oracle, Codespaces, or another VM)."

if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y git make build-essential python3-dev python3-venv curl ca-certificates tmux rsync

swap_active() {
  swapon --show 2>/dev/null | grep -q .
}

if swap_active; then
  log "Swap already active"
else
  log "Creating ${SWAP_GB} GB swap"
  if ! $SUDO fallocate -l "${SWAP_GB}G" /swapfile 2>/dev/null; then
    $SUDO dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024)) status=none \
      || { warn "Could not create a swap file; continuing without swap"; $SUDO rm -f /swapfile; }
  fi
  if [[ -f /swapfile ]]; then
    $SUDO chmod 600 /swapfile
    if ! $SUDO mkswap /swapfile >/dev/null 2>&1; then
      warn "mkswap failed; continuing without swap"
      $SUDO rm -f /swapfile
      $SUDO sed -i '\#^/swapfile #d' /etc/fstab 2>/dev/null || true
    elif ! $SUDO swapon /swapfile 2>/dev/null; then
      # Containers (including GitHub Codespaces) usually lack CAP_SYS_ADMIN.
      warn "Swap is not permitted on this host; continuing without it"
      $SUDO rm -f /swapfile
      $SUDO sed -i '\#^/swapfile #d' /etc/fstab 2>/dev/null || true
    else
      grep -q '^/swapfile ' /etc/fstab \
        || echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null
    fi
  fi
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

log "Preparing SIFT data"
if [[ -f data/raw/sift/sift_base.fvecs ]]; then
  log "SIFT data already present"
elif ! make data; then
  log "FTP download failed; using the ann-benchmarks HTTPS mirror"
  HDF5="$TARGET/sift-128-euclidean.hdf5"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
    -o "$HDF5" https://ann-benchmarks.com/sift-128-euclidean.hdf5
  uv pip install h5py
  .venv/bin/python scripts/prepare_sift_hdf5.py \
    "$HDF5" --out data/raw/sift --siftsmall
  rm -f "$HDF5"
fi

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
