#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/muse_bf16_tc_rans"
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "muse_bf16_tc_rans: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" -O3 -std=c++17 -Xcompiler=-fPIC -shared -lineinfo \
  --expt-relaxed-constexpr -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  "$here/tc_rans.cu" -o "$out/muse_bf16_tc_rans.so"
cp "$here/__init__.py" "$out/__init__.py"
echo "muse_bf16_tc_rans: built $out/muse_bf16_tc_rans.so for sm_${sm}"
