#!/usr/bin/env bash
# Run the benchmark suite on the cloud VM and keep a log per target.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/cloud

run() {
  local target="$1"
  printf '\n\033[1;36m[bench]\033[0m make %s\n' "$target"
  make "$target" 2>&1 | tee "logs/cloud/${target}.log"
}

run e1
run e2

if [[ "${RUN_100K:-0}" == "1" ]]; then
  run e2-100k
else
  printf '\n[bench] skipping e2-100k (set RUN_100K=1 to include it)\n'
fi

if [[ "${RUN_1M:-0}" == "1" ]]; then
  run e3
else
  printf '\n[bench] skipping e3 (set RUN_1M=1 to include it)\n'
fi

printf '\n[bench] done. Logs: logs/cloud/\n'
