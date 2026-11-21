#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
torch_include=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')
export TMPDIR="$KERNELS_OUT"
nvcc -std=c++17 --expt-relaxed-constexpr -O3 -arch=sm_121a --shared -Xcompiler=-fPIC \
    -I"$src/marlin" -I"$torch_include" "$src/marlin_prefill.cu" \
    -o "$KERNELS_OUT/libgb10_marlin_prefill.so"
install -m 644 "$src/glimmer_marlin_prefill.py" "$KERNELS_OUT/"
