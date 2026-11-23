#!/usr/bin/env bash
# marlinw: libmarlinw.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/marlinw.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/marlinw
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "marlinw: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# core/scalar_type.hpp includes <torch/headeronly/util/Exception.h> (header only).
torch_inc=$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
"$nvcc" --version | tail -n 2
# Same numerics as the stock Marlin build: no --use_fast_math (IEEE fp32 adds
# and multiplies, no flush to zero), default contraction.
build_library() {  # build_library <library name> [<source name> [<extra flags>...]]
  local name=$1 source=${2:-$1}
  shift; shift || true
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  --expt-relaxed-constexpr -DENABLE_FP8 "$@" \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -I "$here/csrc" -I "$torch_inc" \
  -Xptxas -v \
  -o "$out/lib${name}.so" "$here/csrc/${source}.cu"
}
pids=()
libs=(marlinw marlinw_swiglu marlinw_swiglu2 marlinw_swiglu3 marlinw_swiglu3_fp8 marlinw_swiglu_fp8)
build_library marlinw & pids+=("$!")
build_library marlinw_swiglu & pids+=("$!")
build_library marlinw_swiglu2 & pids+=("$!")
# The 128-row gate/up + SwiGLU tiles fed by TMA; marlinw_down128 builds the down projection from the same source.
build_library marlinw_swiglu3 marlinw_tma -DMARLINW_TMA_GATEUP & pids+=("$!")
build_library marlinw_swiglu3_fp8 marlinw_tma -DMARLINW_TMA_GATEUP -DMARLINW_TMA_FP8 & pids+=("$!")
build_library marlinw_swiglu_fp8 & pids+=("$!")
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "marlinw: FAILED lib${libs[$i]}.so" >&2
    fail=1
  fi
done
test "$fail" -eq 0
for lib in "${libs[@]}"; do
  test -s "$out/lib${lib}.so" || { echo "marlinw: missing $out/lib${lib}.so" >&2; exit 1; }
done
cp "$here/py/__init__.py" "$here/py/swiglu.py" "$here/py/swiglu_fp8.py" "$out/"
echo "marlinw: built $out/libmarlinw.so for sm_${sm}"
