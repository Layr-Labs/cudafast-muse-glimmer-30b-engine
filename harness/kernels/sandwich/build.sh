#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
nvcc -O3 -std=c++17 --fmad=false -lineinfo --shared --cudart shared \
    -Xcompiler=-fPIC -gencode "arch=compute_${sm},code=sm_${sm}" \
    "$here/sandwich.cu" -o "$KERNELS_OUT/libmuse_sandwich.so"
install -m 0644 "$here/muse_sandwich.py" "$KERNELS_OUT/muse_sandwich.py"
