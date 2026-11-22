#!/usr/bin/env bash
# fp8ws: libfp8ws.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8ws.
# Zero-VRAM dual-TMA M=512 prefill FP8 GEMM for MuseGlimmer-30B-NVFP4.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8ws
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8ws: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libfp8ws.so"
# No --use_fast_math: keep IEEE fp32 adds and multiplies identical to CUTLASS.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8ws.so" "$here/csrc/fp8ws.cu" 2>&1 || true
test -f "$out/libfp8ws.so" || { echo "fp8ws: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8ws: built $out/libfp8ws.so for sm_${sm}"
