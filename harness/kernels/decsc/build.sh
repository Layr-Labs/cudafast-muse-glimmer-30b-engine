#!/usr/bin/env bash
# decsc: libdecsc.so (the decode package's one-row tile-major Marlin GEMV compiled with MARLIN_SC6) plus its Python
# module, into $KERNELS_OUT. Optional: a compiler failure leaves it out (the engine keeps the decode package's GEMV).
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
tmp="$(mktemp -d "$KERNELS_OUT/build.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
torch_include="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "include")')"
rm -f "$KERNELS_OUT/libdecsc.so" "$KERNELS_OUT/decsc.py"
# Same flags as harness/kernels/decode/build.sh (no --use_fast_math).
TMPDIR="$tmp" nvcc -O3 --expt-relaxed-constexpr -std=c++17 --shared --cudart shared \
    -Xcompiler=-fPIC -gencode arch=compute_121a,code=sm_121a -I"$torch_include" \
    -I"$src/../decode" -Xptxas -v \
    "$src/csrc/gemv_sc6.cu" "$src/csrc/gemv_sc5.cu" -o "$KERNELS_OUT/libdecsc.so" 2>&1 || true
if ! test -s "$KERNELS_OUT/libdecsc.so"; then
  echo "decsc: build failed; skipped (the engine keeps the decode GEMV)" >&2
  rm -f "$KERNELS_OUT/libdecsc.so"
  exit 0
fi
install -m 0644 "$src/py/decsc.py" "$KERNELS_OUT/decsc.py"
echo "decsc: built $KERNELS_OUT/libdecsc.so"
