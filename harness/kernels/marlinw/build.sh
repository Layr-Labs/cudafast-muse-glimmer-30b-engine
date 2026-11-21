#!/usr/bin/env bash
# marlinw: libmarlinw.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/marlinw.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/marlinw
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "marlinw: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# core/scalar_type.hpp includes <torch/headeronly/util/Exception.h> (header only).
torch_inc=$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
"$nvcc" --version | tail -n 2
# Same numerics as the stock Marlin build: no --use_fast_math (IEEE fp32 adds
# and multiplies, no flush to zero), default contraction.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  --expt-relaxed-constexpr -DENABLE_FP8 \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -I "$here/csrc" -I "$torch_inc" \
  -Xptxas -v \
  -o "$out/libmarlinw.so" "$here/csrc/marlinw.cu"
test -f "$out/libmarlinw.so" || { echo "marlinw: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "marlinw: built $out/libmarlinw.so for sm_${sm}"
