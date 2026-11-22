#!/usr/bin/env bash
# fp8tile: libfp8tile.so (ctypes, static cudart) and its Python package, into
# $KERNELS_OUT/fp8tile. Builds inside the pinned image with the CUTLASS headers
# that ship in it (flashinfer's bundled copy). Any compiler failure fails the build.
# FP8TILE_CONFIGS: comma-separated configuration indices (csrc/fp8tile_gemm.cuh).
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${KERNELS_OUT}/fp8tile"
mkdir -p "${out}"
configs="${FP8TILE_CONFIGS:-6}"  # shipped: 16x32x512
nvcc="$(command -v nvcc || true)"
[[ -n "${nvcc}" ]] || nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
[[ -x "${nvcc}" ]] || { echo "fp8tile: nvcc not found" >&2; exit 1; }
py="$(command -v python3)"
cutlass="$("${py}" -c 'import importlib.util, os; s = importlib.util.find_spec("flashinfer"); print(os.path.join(os.path.dirname(s.origin), "data", "cutlass"))')"
[[ -f "${cutlass}/include/cutlass/gemm/collective/sm120_mma_tma.hpp" ]] \
  || { echo "fp8tile: CUTLASS SM120 headers not found under ${cutlass}" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch="${TORCH_CUDA_ARCH_LIST:-12.1a}"
arch="${arch%%;*}"
arch="${arch// /}"
arch="${arch%+PTX}"
sm="${arch/./}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
# No --use_fast_math: the epilogue products must stay IEEE fp32 round-to-nearest.
flags=(-O3 -std=c++17 --expt-relaxed-constexpr -DNDEBUG -Xcompiler -fPIC
       -gencode "arch=compute_${sm},code=sm_${sm}"
       -I"${cutlass}/include" -I"${cutlass}/tools/util/include")
have=()
pids=()
for i in ${configs//,/ }; do
  [[ "${i}" =~ ^[013467]$ ]] || { echo "fp8tile: bad configuration '${i}'" >&2; exit 1; }
  "${nvcc}" "${flags[@]}" -DFP8TILE_CFG="${i}" -c "${here}/csrc/fp8tile_cfg.cu" -o "${tmp}/cfg${i}.o" &
  pids+=("$!")
  have+=(-DFP8TILE_HAVE_"${i}")
done
for p in "${pids[@]}"; do wait "${p}"; done
"${nvcc}" "${flags[@]}" "${have[@]}" -c "${here}/csrc/fp8tile.cu" -o "${tmp}/fp8tile.o"
"${nvcc}" -shared -gencode "arch=compute_${sm},code=sm_${sm}" -o "${out}/libfp8tile.so" "${tmp}"/*.o
cp "${here}/py/__init__.py" "${out}/__init__.py"
echo "fp8tile: built ${out}/libfp8tile.so for sm_${sm} (configurations ${configs})"
