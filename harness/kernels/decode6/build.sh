#!/usr/bin/env bash
# libmuse_marlin6.so: the decode GEMV with 6-bit palette block scales (same flags as harness/kernels/decode/build.sh),
# plus gemv6_srans.cu, the fixed gate/up grid reading rANS-coded block scales, and gemv6_srans_down.cu, the 48-CTA
# stream-K down projection reading rANS-coded block scales (each in its own namespace with its own entry point).
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
tmp="$(mktemp -d "$KERNELS_OUT/build.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
torch_include="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "include")')"
TMPDIR="$tmp" nvcc -O3 --expt-relaxed-constexpr -std=c++17 --shared --cudart shared \
    -Xcompiler=-fPIC -gencode arch=compute_121a,code=sm_121a -I"$torch_include" \
    "$src/gemv6.cu" "$src/gemv6_srans.cu" "$src/gemv6_srans_down.cu" -o "$KERNELS_OUT/libmuse_marlin6.so"
install -m 0644 "$src/muse_marlin6_gemv.py" "$KERNELS_OUT/muse_marlin6_gemv.py"
nvcc -O3 -std=c++17 --shared --cudart none -Xcompiler=-fPIC \
    -I"$torch_include" "$src/compressed_vmm.cu" \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" -lcuda -o "$KERNELS_OUT/libmuse_compressed_vmm.so"
install -m 0644 "$src/muse_compressed_vmm.py" "$KERNELS_OUT/muse_compressed_vmm.py"
