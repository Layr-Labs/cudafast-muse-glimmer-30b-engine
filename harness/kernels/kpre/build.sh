#!/usr/bin/env bash
# kpre: libkpre.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/kpre.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/kpre
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "kpre: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# torch header-only include (core/scalar_type.hpp uses STD_TORCH_CHECK); no torch link.
py=$(command -v python3 || command -v python)
tinc=$("$py" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
"$nvcc" --version | tail -n 2
# No --use_fast_math: plain IEEE fp32 / bf16, as kmarlin.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  --expt-relaxed-constexpr \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -I "$here/csrc" -I "$here/csrc/marlin" -I "$tinc" \
  -Xptxas -v \
  -o "$out/libkpre.so" "$here/csrc/kpre.cu"
test -f "$out/libkpre.so" || { echo "kpre: build failed" >&2; exit 1; }
# Test-only twin with the stock slab-major part-1 order (exactness test at any SM count).
if [ "${KPRE_BUILD_REF:-0}" = 1 ]; then
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo --expt-relaxed-constexpr \
    -gencode "arch=compute_${sm},code=sm_${sm}" -DKPRE_STOCK_ORDER \
    -I "$here/csrc" -I "$here/csrc/marlin" -I "$tinc" \
    -o "$out/libkpre_ref.so" "$here/csrc/kpre.cu"
fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "kpre: built $out/libkpre.so for sm_${sm}"
