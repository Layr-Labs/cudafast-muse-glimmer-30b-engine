#!/usr/bin/env bash
# Builds every kernel package under harness/kernels/<name>/build.sh into $KERNELS_OUT.
# Runs in the pinned image at setup (no network, checkout read-only); writes only to $KERNELS_OUT.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
mkdir -p "$KERNELS_OUT"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pids=()
names=()
for sub in "$here"/*/build.sh; do
  test -f "$sub" || continue
  name=$(basename "$(dirname "$sub")")
  echo "kernels: building $name"
  bash "$sub" &
  pids+=("$!")
  names+=("$name")
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "kernels: FAILED ${names[$i]}" >&2
    fail=1
  fi
done
exit "$fail"
