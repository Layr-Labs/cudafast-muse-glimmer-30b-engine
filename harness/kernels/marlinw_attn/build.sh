#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
arch="${TORCH_CUDA_ARCH_LIST:-12.1a}"
arch="${arch%%;*}"
arch="${arch%% *}"
sm="${arch//./}"
torch_include=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    --expt-relaxed-constexpr -DENABLE_FP8 -gencode "arch=compute_${sm},code=sm_${sm}" \
    -I "$here/../marlinw/csrc" -I "$torch_include" -Xptxas -v \
    -o "$KERNELS_OUT/libmarlinw_attn.so" "$here/marlinw_attn.cu"
install -m 644 "$here/marlinw_attn.py" "$KERNELS_OUT/"
