#!/usr/bin/env bash
# fp8m512: libfp8m512.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8m512.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8m512
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8m512: nvcc not found (skipped; cutlass_scaled_mm stays)" >&2; exit 0; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libfp8m512.so"
# IEEE fp32 multiplies in the epilogue (no --use_fast_math), as the source submission builds it.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libfp8m512.so" "$here/csrc/fp8m512.cu" || true
# Non-fatal: without the library every layer keeps cutlass_scaled_mm.
test -s "$out/libfp8m512.so" || { echo "fp8m512: build failed (skipped; cutlass_scaled_mm stays)" >&2; exit 0; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8m512: built $out/libfp8m512.so for sm_${sm}"
