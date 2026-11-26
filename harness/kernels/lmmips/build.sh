#!/usr/bin/env bash
# lmmips: liblmmips.so (ctypes) + Python module into $KERNELS_OUT/lmmips (H2 exact lm_head MIPS pruning, MUSE_LM_MIPS).
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/lmmips
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "lmmips: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/liblmmips.so"
# No --use_fast_math: the bounds rely on IEEE fp64/fp32 semantics and directed rounding intrinsics.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/liblmmips.so" "$here/csrc/lmmips.cu"
test -f "$out/liblmmips.so" || { echo "lmmips: build failed" >&2; exit 1; }
rm -rf "$out/__pycache__"
cp "$here/py/__init__.py" "$out/__init__.py"
grep -q "^def tma4_tile_states" "$out/__init__.py" || { echo "lmmips: tma4_tile_states missing from module" >&2; exit 1; }
echo "lmmips: built $out/liblmmips.so for sm_${sm}"
