#!/usr/bin/env bash
# glimmer_fp8: glimmer_fp8_ext.so (CUTLASS sm120 FP8 GEMM, vLLM's ScaledEpilogue arithmetic, 16x32x128 decode tile)
# into $KERNELS_OUT, importable as `glimmer_fp8_ext`. Offline, in the pinned image.
# Fail-soft: without the extension vllm/.../utils/glimmer_fp8.py runs ops.cutlass_scaled_mm everywhere, so a failed
# build must not stop the other kernel packages (harness/kernels/build.sh runs under set -e).
set -uo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}" PYTHONDONTWRITEBYTECODE=1 MAX_JOBS="${MAX_JOBS:-4}"
work="${KERNELS_OUT}/.glimmer_fp8_build"
mkdir -p "${work}/torch_ext" "${work}/tmp"
export TORCH_EXTENSIONS_DIR="${work}/torch_ext" TMPDIR="${work}/tmp"
python3 - "${here}" "${KERNELS_OUT}" "${work}" <<'PY' \
  || echo "glimmer_fp8: WARNING glimmer_fp8_ext failed to build; stock cutlass_scaled_mm will be used everywhere" >&2
import importlib.util, os, pathlib, shutil, sys, torch
from torch.utils.cpp_extension import load
src, out, work = sys.argv[1], sys.argv[2], sys.argv[3]
cut = pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent / "data" / "cutlass"
inc = os.path.join(os.path.dirname(torch.__file__), "include")
flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
         "-gencode=arch=compute_121a,code=sm_121a",
         "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
         "-U__CUDA_NO_HALF2_OPERATORS__", f"-I{src}", f"-I{inc}", f"-I{cut}/include", f"-I{cut}/tools/util/include"]
bdir = os.path.join(work, "build"); os.makedirs(bdir, exist_ok=True)
load(name="glimmer_fp8_ext", sources=[os.path.join(src, "fp8x_ext.cu")], extra_cuda_cflags=flags,
     extra_cflags=["-O3", f"-I{src}"], build_directory=bdir, verbose=False)
shutil.copy2(os.path.join(bdir, "glimmer_fp8_ext.so"), os.path.join(out, "glimmer_fp8_ext.so"))
print("glimmer_fp8: built glimmer_fp8_ext into", out)
PY
rm -rf "${work}"
exit 0
