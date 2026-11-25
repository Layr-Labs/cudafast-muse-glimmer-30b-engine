#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
arch=${arch%+ptx}
sm=${arch/./}
nvcc -O3 -std=c++17 --fmad=false -lineinfo --shared --cudart shared \
    -Xcompiler=-fPIC -Xptxas=-v -gencode "arch=compute_${sm},code=sm_${sm}" \
    "$here/embed.cu" -o "$KERNELS_OUT/libmuse_embed.so"
install -m 0644 "$here/muse_embed.py" "$KERNELS_OUT/muse_embed.py"
