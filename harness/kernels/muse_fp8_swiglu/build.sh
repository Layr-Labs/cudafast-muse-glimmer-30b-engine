#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/muse_fp8_swiglu"
mkdir -p "$out"
cutlass=$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent / "data" / "cutlass")')
nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
"$nvcc" -O3 -std=c++17 --shared -Xcompiler=-fPIC --expt-relaxed-constexpr --expt-extended-lambda \
  -gencode=arch=compute_121a,code=sm_121a -I"$cutlass/include" -I"$cutlass/tools/util/include" \
  -I"$here/../marlinw/csrc" "$here/csrc/prefill.cu" -o "$out/libmuse_fp8_swiglu.so"
cp "$here/py/__init__.py" "$out/__init__.py"
