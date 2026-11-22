#!/usr/bin/env bash
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${KERNELS_OUT:?}"
tmp="$(mktemp -d "$KERNELS_OUT/build.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
torch_include="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "include")')"
TMPDIR="$tmp" nvcc -O3 --expt-relaxed-constexpr -std=c++17 --shared --cudart shared \
    -Xcompiler=-fPIC -gencode "arch=compute_${sm},code=sm_${sm}" -I"$torch_include" \
    "$src/gemv.cu" -o "$KERNELS_OUT/libmuse_marlin.so"
install -m 0644 "$src/muse_marlin_gemv.py" "$KERNELS_OUT/muse_marlin_gemv.py"
