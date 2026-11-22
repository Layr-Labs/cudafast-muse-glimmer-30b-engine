#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/muse_bf16_lane"
mkdir -p "$out"
nvcc -O3 -arch=sm_121a -Xcompiler=-fPIC -shared \
  "$here/lane.cu" -o "$out/muse_bf16_lane.so"
cp "$here/py/__init__.py" "$out/__init__.py"
