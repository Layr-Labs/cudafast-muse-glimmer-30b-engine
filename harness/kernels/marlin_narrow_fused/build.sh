#!/usr/bin/env bash
# Separate experiment: no symbol or library is shared with the verified narrow path.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
export TMPDIR="$KERNELS_OUT"
torch_include=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')
nvcc -std=c++17 --expt-relaxed-constexpr -O3 -arch=sm_121a --shared -Xcompiler=-fPIC \
    -I"$here/../decode/marlin" -I"$torch_include" "$here/marlin_prefill.cu" \
    -o "$KERNELS_OUT/libgb10_marlin_narrow_fused.so"
install -m 644 "$here/marlin_narrow_fused.py" "$KERNELS_OUT/"
