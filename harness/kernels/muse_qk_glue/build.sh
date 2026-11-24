#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$KERNELS_OUT"

nvcc_bin="$(command -v nvcc || true)"
test -n "$nvcc_bin" || nvcc_bin="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
cuobjdump_bin="$(command -v cuobjdump || true)"
test -n "$cuobjdump_bin" || cuobjdump_bin="${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump"

so_out="$KERNELS_OUT/libmuse_qk_glue.so"
rm -f "$so_out"

ptx_log="$("$nvcc_bin" -O3 -std=c++17 --shared -Xcompiler -fPIC -lineinfo \
    -gencode arch=compute_121a,code=sm_121a -Xptxas=-v,-Werror \
    "$src/qk_glue.cu" -o "$so_out" 2>&1)"
echo "$ptx_log"

if echo "$ptx_log" | grep -E "spill stores|spill loads" | grep -qv "0 bytes spill stores, 0 bytes spill loads"; then
  echo "muse_qk_glue: register spill detected in ptxas output!" >&2
  exit 1
fi

if test -x "$cuobjdump_bin" && command -v nvdisasm >/dev/null 2>&1; then
  sass="$("$cuobjdump_bin" --dump-sass "$so_out")"
  if echo "$sass" | grep -Eq 'desc\[UR[0-9]*[13579]\]'; then
    echo "muse_qk_glue: odd-UR descriptor detected in SASS!" >&2
    exit 1
  fi
  if echo "$sass" | grep -Eq '\b(STL|LDL)\b'; then
    echo "muse_qk_glue: local-memory spill instruction detected in SASS!" >&2
    exit 1
  fi
fi

install -m 644 "$src/muse_qk_glue.py" "$KERNELS_OUT/muse_qk_glue.py"
echo "muse_qk_glue: built $so_out for sm_121a (0 spills, 0 odd-UR)"
