#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?}"
here=$(cd "$(dirname "$0")" && pwd)
inc=$(python3 -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "include")')
for mode in gate down; do
  nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo --expt-relaxed-constexpr \
    -DENABLE_FP8 -DMARLINW_SINGLE_SLAB -gencode arch=compute_121a,code=sm_121a \
    -I"$here/../marlinw/csrc" -I"$inc" "$here/$mode.cu" -o "$KERNELS_OUT/libmuse_sample_$mode.so"
done
install -m 644 "$here/muse_final_slab.py" "$KERNELS_OUT/"
