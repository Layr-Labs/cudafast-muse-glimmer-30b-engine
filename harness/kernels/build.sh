#!/usr/bin/env bash
# harness/kernels/build.sh: builds the exact glimmer kernels into $KERNELS_OUT, in the pinned image, offline:
#   glimmer_marlin_ext  vLLM v0.27.1's Marlin (NVFP4 W4A16, bf16, thread_m_blocks 4) with the exact prefill tile-order remap
#   glimmer_fp8_ext     CUTLASS sm120 FP8 GEMM with vLLM's ScaledEpilogue arithmetic, 16x32x128 decode tile   [cand5 only]
# A failed build is NOT fatal: the matching Python op then finds no extension and runs vLLM's stock kernel everywhere.
set -uo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}" PYTHONDONTWRITEBYTECODE=1 MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_EXTENSIONS_DIR="${KERNELS_OUT}/.torch_ext" TMPDIR="${KERNELS_OUT}/.tmp"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"
build_ext() {  # build_ext NAME SRC_DIR SOURCE EXTRA_FLAGS...
  local name=$1 src=$2 file=$3; shift 3
  python3 - "${name}" "${src}" "${file}" "${KERNELS_OUT}" "$@" <<'PY'
import os, shutil, sys, torch
from torch.utils.cpp_extension import load
name, src, file, out, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
inc = os.path.join(os.path.dirname(torch.__file__), "include")
flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_121a,code=sm_121a",
         "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
         "-U__CUDA_NO_HALF2_OPERATORS__", f"-I{src}", f"-I{inc}"] + extra
bdir = os.path.join(out, "build_" + name); os.makedirs(bdir, exist_ok=True)
load(name=name, sources=[os.path.join(src, file)], extra_cuda_cflags=flags, extra_cflags=["-O3", f"-I{src}"],
     build_directory=bdir, verbose=False)
shutil.copy2(os.path.join(bdir, name + ".so"), os.path.join(out, name + ".so"))
print(name, "built into", out)
PY
}
build_ext glimmer_marlin_ext "${HERE}/glimmer_marlin" gm_ext.cu -DMARLIN_NAMESPACE_NAME=glimmer_marlin_ns -DGM_REMAP \
  || echo "build.sh: WARNING glimmer_marlin_ext failed to build; stock Marlin will be used everywhere" >&2
if [[ -d "${HERE}/glimmer_fp8" ]]; then
  CUT="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent / "data" / "cutlass")')"
  build_ext glimmer_fp8_ext "${HERE}/glimmer_fp8" fp8x_ext.cu --expt-extended-lambda "-I${CUT}/include" "-I${CUT}/tools/util/include" \
    || echo "build.sh: WARNING glimmer_fp8_ext failed to build; stock cutlass_scaled_mm will be used everywhere" >&2
fi
# leave only the importable modules in KERNELS_OUT (no build trees, locks or temp files)
rm -rf "${KERNELS_OUT}"/build_* "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"
ls -la "${KERNELS_OUT}" >&2
exit 0
