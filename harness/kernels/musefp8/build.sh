#!/usr/bin/env bash
# musefp8: libmusefp8.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/musefp8.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/musefp8
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "musefp8: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# IEEE fp32 multiplies in the epilogue (no --use_fast_math), as cutlass_scaled_mm.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libmusefp8.so" "$here/csrc/musefp8.cu"
test -s "$out/libmusefp8.so" || { echo "musefp8: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "musefp8: built $out/libmusefp8.so for sm_${sm}"
