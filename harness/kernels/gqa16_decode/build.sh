#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$KERNELS_OUT"

nvcc_bin="$(command -v nvcc || true)"
test -n "$nvcc_bin" || nvcc_bin="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
cuobjdump_bin="$(command -v cuobjdump || true)"
test -n "$cuobjdump_bin" || cuobjdump_bin="${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump"

sm="121a"

so_out="$KERNELS_OUT/libgqa16_decode.so"
rm -f "$so_out"

ptx_log="$("$nvcc_bin" -O3 -std=c++17 --shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas=-v,-Werror \
  "$src/gqa16_decode.cu" -o "$so_out" 2>&1)"
echo "$ptx_log"

if echo "$ptx_log" | grep -E "spill stores|spill loads" | grep -qv "0 bytes spill stores, 0 bytes spill loads"; then
  echo "gqa16_decode: register spill detected in ptxas output!" >&2
  exit 1
fi

if test -x "$cuobjdump_bin" && command -v nvdisasm >/dev/null 2>&1; then
  sass="$("$cuobjdump_bin" --dump-sass "$so_out")"
  if echo "$sass" | grep -Eq 'desc\[UR[0-9]*[13579]\]'; then
    echo "gqa16_decode: odd-UR descriptor detected in SASS!" >&2
    exit 1
  fi
  if echo "$sass" | grep -Eq '\b(STL|LDL)\b'; then
    echo "gqa16_decode: local-memory spill instruction detected in SASS!" >&2
    exit 1
  fi
fi

install -m 0644 "$src/muse_gqa16_attn.py" "$KERNELS_OUT/muse_gqa16_attn.py"
echo "gqa16_decode: built $so_out for sm_${sm} (0 spills, 0 odd-UR)"
