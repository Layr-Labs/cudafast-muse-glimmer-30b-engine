#!/usr/bin/env bash
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -lineinfo \
    -gencode arch=compute_121a,code=sm_121a -Xptxas=-v \
    "$src/qk_glue.cu" -o "$KERNELS_OUT/libmuse_qk_glue.so"
install -m 644 "$src/muse_qk_glue.py" "$KERNELS_OUT/muse_qk_glue.py"
