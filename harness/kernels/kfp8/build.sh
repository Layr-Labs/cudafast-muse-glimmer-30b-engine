#!/usr/bin/env bash
# kfp8: libkfp8.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/kfp8.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/kfp8
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "kfp8: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# kind::f8f6f4 MMA is an arch-specific feature: sm_12x needs the "a" target.
case "$sm" in 120|121) sm=${sm}a ;; esac
"$nvcc" --version | tail -n 2
# No --use_fast_math: the epilogue must stay plain IEEE fp32 multiplies and RN bf16.
# KFP8_NVCC_FLAGS (optional) adds flags, e.g. -DKFP8_ALL_VARIANTS for the research set.
# shellcheck disable=SC2086
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v ${KFP8_NVCC_FLAGS:-} \
  -o "$out/libkfp8.so" "$here/csrc/kfp8.cu"
test -f "$out/libkfp8.so" || { echo "kfp8: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "kfp8: built $out/libkfp8.so for sm_${sm}"
