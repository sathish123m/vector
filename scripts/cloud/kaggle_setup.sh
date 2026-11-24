#!/usr/bin/env bash
# One-shot Kaggle Notebook bootstrap + benchmark runner.
#
# Why Kaggle: it needs no credit card, provides ~30 GB RAM, a 20 GB working
# directory, and up to 12-hour sessions -- enough for SIFT100K churn and
# SIFT1M.  Internet must be enabled in the notebook settings (Kaggle asks for
# phone verification once).
#
# Usage, in one Kaggle cell:
#   !git clone https://github.com/sathish123m/vector.git /kaggle/working/churnvec
#   !bash /kaggle/working/churnvec/scripts/cloud/kaggle_setup.sh
#
# Re-running is safe: the repository is updated, data is reused, and finished
# results are archived after every target.
#
# Environment overrides:
#   TARGETS="e2-100k e3"   make targets to run (default)
#   TARGETS="e1 e2 e2-100k e3"   full cloud sweep
#   RUN_TESTS=0            skip pytest
#   CHURNVEC_DIR=...       repository location (default /kaggle/working/churnvec)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sathish123m/vector.git}"
TARGETS="${TARGETS:-e2-100k e3}"
RUN_TESTS="${RUN_TESTS:-1}"

log() { printf '\n\033[1;36m[kaggle]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[kaggle:warn]\033[0m %s\n' "$*" >&2; }

if [[ -n "${CHURNVEC_DIR:-}" ]]; then
  WORK="$CHURNVEC_DIR"
elif [[ -d /kaggle/working ]]; then
  WORK="/kaggle/working/churnvec"
else
  WORK="$PWD"
fi
OUT_DIR="/kaggle/working"
[[ -d /kaggle/working ]] || OUT_DIR="$WORK"

log "Repository: $WORK"
if [[ -d "$WORK/.git" ]]; then
  git -C "$WORK" fetch --depth 1 origin main
  git -C "$WORK" reset --hard FETCH_HEAD
else
  git clone --depth 1 "$REPO_URL" "$WORK"
fi
cd "$WORK"

export PYTHON="$(command -v python3)"
export MPLBACKEND="Agg"
log "Python: $PYTHON ($($PYTHON --version 2>&1))"

log "Installing the package and test dependencies"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q -e . pytest hypothesis

have_sift() {
  [[ -f data/raw/sift/sift_base.fvecs && -f data/raw/sift/sift_query.fvecs ]]
}

# First matching file under /kaggle/input, or an empty string.
find_input() {
  find /kaggle/input -type f -name "$1" 2>/dev/null | head -1
}

prepare_from_hdf5() {
  local url="https://ann-benchmarks.com/sift-128-euclidean.hdf5"
  local file="$WORK/sift-128-euclidean.hdf5"
  log "Downloading the ann-benchmarks SIFT1M HDF5 mirror"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$file" "$url"
  "$PYTHON" -m pip install -q h5py
  "$PYTHON" scripts/prepare_sift_hdf5.py "$file" --out data/raw/sift --siftsmall
  rm -f "$file"
}

if have_sift; then
  log "SIFT data already present"
else
  # 1. A Kaggle Dataset with extracted .fvecs files avoids all downloads.
  SIFT_BASE="$(find_input 'sift_base.fvecs')"
  if [[ -n "$SIFT_BASE" ]]; then
    SRC_DIR="$(dirname "$SIFT_BASE")"
    log "Copying SIFT files from Kaggle input: $SRC_DIR"
    mkdir -p data/raw/sift
    for name in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs sift_learn.fvecs; do
      if [[ -f "$SRC_DIR/$name" ]]; then
        cp "$SRC_DIR/$name" data/raw/sift/
      fi
    done
  fi

  # 2. Or a Kaggle Dataset with the official archives.
  if ! have_sift; then
    mapfile -t SIFT_ARCHIVES < <(find_input 'sift.tar.gz')
    mapfile -t SMALL_ARCHIVES < <(find_input 'siftsmall.tar.gz')
    if [[ ${#SIFT_ARCHIVES[@]} -gt 0 ]]; then
      log "Extracting the SIFT archive from /kaggle/input"
      mkdir -p data/raw
      cp "${SIFT_ARCHIVES[0]}" data/raw/sift.tar.gz
      tar -xzf data/raw/sift.tar.gz -C data/raw
      if [[ ${#SMALL_ARCHIVES[@]} -gt 0 ]]; then
        cp "${SMALL_ARCHIVES[0]}" data/raw/siftsmall.tar.gz
        tar -xzf data/raw/siftsmall.tar.gz -C data/raw
      fi
    fi
  fi

  # 3. Or the official TEXMEX FTP host, if this host allows FTP.
  if ! have_sift; then
    if curl -sI --connect-timeout 15 --max-time 30 \
        "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" >/dev/null 2>&1; then
      log "Downloading the official SIFT archives over FTP"
      if ! bash scripts/download_data.sh; then
        warn "FTP download failed; falling back to the HTTPS HDF5 mirror"
        prepare_from_hdf5
      fi
    else
      log "FTP is blocked here; using the HTTPS HDF5 mirror"
      prepare_from_hdf5
    fi
  fi
fi
have_sift || { warn "SIFT data is missing; cannot run benchmarks"; exit 1; }

log "Host resources"
free -h || true
df -h "$WORK" || true

if [[ "$RUN_TESTS" == "1" ]]; then
  log "Running the test suite"
  make test
fi

archive_results() {
  if [[ "$OUT_DIR" != "$WORK" ]]; then
    mkdir -p "$OUT_DIR/results"
    cp -R results/. "$OUT_DIR/results/" 2>/dev/null || true
  fi
  if command -v zip >/dev/null 2>&1 && [[ -d results ]]; then
    (cd "$WORK" && zip -qr "$OUT_DIR/churnvec_results.zip" results logs 2>/dev/null) || true
  fi
  log "Archived results to $OUT_DIR/results and $OUT_DIR/churnvec_results.zip"
}

status=0
for target in $TARGETS; do
  log "Running make $target"
  mkdir -p logs/cloud
  set +e
  make "$target" 2>&1 | tee "logs/cloud/${target}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    warn "make $target failed with exit code $rc; continuing"
    status=1
  fi
  archive_results
done

{
  echo "ChurnVec Kaggle run"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "targets: $TARGETS"
  echo "python: $($PYTHON --version 2>&1)"
  uname -a
  free -h || true
} > "$OUT_DIR/RUN_SUMMARY.txt"

log "Done (exit $status). Download results from the notebook Output panel:"
log "  $OUT_DIR/results/"
log "  $OUT_DIR/churnvec_results.zip"
exit "$status"
