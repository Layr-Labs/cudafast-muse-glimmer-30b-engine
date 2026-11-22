#!/usr/bin/env bash
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
tmp="$(mktemp -d "$KERNELS_OUT/build.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
torch_include="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "include")')"
TMPDIR="$tmp" nvcc -O3 --expt-relaxed-constexpr -std=c++17 --shared --cudart shared \
    -Xcompiler=-fPIC -gencode arch=compute_121a,code=sm_121a -I"$torch_include" \
    "$src/gemv.cu" -o "$KERNELS_OUT/libmuse_marlin.so"
# Coded-scale variant (same kernel; scale bytes rebuilt from 6-bit codes). Optional: decode falls back if absent.
TMPDIR="$tmp" nvcc -O3 --expt-relaxed-constexpr -std=c++17 --shared --cudart shared \
    -Xcompiler=-fPIC -gencode arch=compute_121a,code=sm_121a -I"$torch_include" \
    "$src/gemv_coded.cu" -o "$KERNELS_OUT/libmuse_marlin_coded.so" || echo "decode: coded variant failed to build" >&2
install -m 0644 "$src/muse_marlin_gemv.py" "$KERNELS_OUT/muse_marlin_gemv.py"
