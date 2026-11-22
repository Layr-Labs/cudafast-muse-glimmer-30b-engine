#!/usr/bin/env bash
# lmpost: liblmpost.so (ctypes, static cudart) + Python module into $KERNELS_OUT/lmpost.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/lmpost
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "lmpost: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/liblmpost.so"
# No --use_fast_math: softcap LUT comes verbatim from PyTorch ops; keep strict IEEE ops.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/liblmpost.so" "$here/csrc/lmpost.cu" 2>&1 || true
test -f "$out/liblmpost.so" || { echo "lmpost: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "lmpost: built $out/liblmpost.so for sm_${sm}"
