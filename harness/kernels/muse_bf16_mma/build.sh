#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out="${KERNELS_OUT:?}/muse_bmma"
mkdir -p "$out"
nvcc -O3 -std=c++17 -arch=sm_121a --cudart shared -Xcompiler=-fPIC --shared "$here/tile.cu" -o "$out/libmuse_bmma.so"
cp "$here/__init__.py" "$out/"
