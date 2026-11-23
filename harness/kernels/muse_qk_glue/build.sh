#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/muse_qk_glue"
mkdir -p "$out"
nvcc -O3 -std=c++17 --fmad=false -arch=sm_121a -Xcompiler=-fPIC -shared \
  "$here/qk_glue.cu" -o "$out/muse_qk_glue.so"
cp "$here/py/__init__.py" "$out/__init__.py"
