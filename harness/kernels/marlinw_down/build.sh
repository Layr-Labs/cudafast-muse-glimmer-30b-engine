#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
torch_include=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    --expt-relaxed-constexpr -DENABLE_FP8 -gencode arch=compute_121a,code=sm_121a \
    -I "$here/../marlinw/csrc" -I "$torch_include" -Xptxas -v \
    -o "$KERNELS_OUT/libmarlinw_down.so" "$here/marlinw_down.cu"
install -m 644 "$here/marlinw_down.py" "$KERNELS_OUT/"
