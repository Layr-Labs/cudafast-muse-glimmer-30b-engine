#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
nvcc -O3 -std=c++17 --fmad=false -lineinfo --shared --cudart shared \
    -Xcompiler=-fPIC -gencode arch=compute_121a,code=sm_121a \
    "$here/embed.cu" -o "$KERNELS_OUT/libmuse_embed.so"
install -m 0644 "$here/muse_embed.py" "$KERNELS_OUT/muse_embed.py"
