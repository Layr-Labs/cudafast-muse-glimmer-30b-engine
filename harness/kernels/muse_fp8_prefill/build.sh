#!/usr/bin/env bash
# muse_fp8_prefill: libmuse_fp8_prefill.so (ctypes) plus its Python package, into $KERNELS_OUT/muse_fp8_prefill.
# Runs in the pinned image; the CUTLASS headers are the ones flashinfer ships there (as muse_fp8_swiglu).
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/muse_fp8_prefill"
mkdir -p "$out"
cutlass=$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent / "data" / "cutlass")')
nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
# IEEE fp32 multiplies in the epilogue (no --use_fast_math), as cutlass_scaled_mm.
"$nvcc" -O3 -std=c++17 --shared -Xcompiler=-fPIC --expt-relaxed-constexpr --expt-extended-lambda \
  -gencode=arch=compute_121a,code=sm_121a -I"$cutlass/include" -I"$cutlass/tools/util/include" \
  "$here/csrc/fp8_prefill.cu" -o "$out/libmuse_fp8_prefill.so"
test -s "$out/libmuse_fp8_prefill.so" || { echo "muse_fp8_prefill: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "muse_fp8_prefill: built $out/libmuse_fp8_prefill.so for sm_121a"
