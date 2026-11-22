#!/usr/bin/env bash
# fp8z: libfp8z.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8z.
# Optional: a compiler failure leaves the package out (the engine then keeps CUTLASS) and does
# not fail the other kernel packages.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8z
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8z: nvcc not found; skipped" >&2; rm -rf "$out"; exit 0; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libfp8z.so"
# No --use_fast_math: the epilogue must stay plain IEEE fp32 multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8z.so" "$here/csrc/fp8z.cu" 2>&1 || true
if ! test -s "$out/libfp8z.so"; then
  echo "fp8z: build failed; skipped (the engine keeps CUTLASS)" >&2
  rm -rf "$out"
  exit 0
fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8z: built $out/libfp8z.so for sm_${sm}"
