#!/usr/bin/env bash
# l2pf: libl2pf.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/l2pf.
# Offline; a compiler failure fails the build.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/l2pf
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "l2pf: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libl2pf.so" "$here/csrc/l2pf.cu"
test -s "$out/libl2pf.so" || { echo "l2pf: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$here/py/sites.py" "$out/"
echo "l2pf: built $out/libl2pf.so for sm_${sm}"
