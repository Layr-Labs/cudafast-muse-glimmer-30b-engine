#!/usr/bin/env bash
# Build the measured 52c7159 narrow Marlin prefill library in the pinned image.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
export TMPDIR="$KERNELS_OUT"
torch_include=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')
# The shared headers under decode/marlin are byte-identical to the donor's.
nvcc -std=c++17 --expt-relaxed-constexpr -O3 -arch=sm_121a --shared -Xcompiler=-fPIC \
    -I"$here/../decode/marlin" -I"$torch_include" "$here/marlin_prefill.cu" \
    -o "$KERNELS_OUT/libgb10_marlin_prefill.so"
install -m 644 "$here/marlin_narrow.py" "$KERNELS_OUT/"
