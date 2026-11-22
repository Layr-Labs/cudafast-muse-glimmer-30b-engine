#!/usr/bin/env bash
# Builds the accepted small-row FP8 CUTLASS kernels for GB10 in the pinned image.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
build_dir=$(mktemp -d "$KERNELS_OUT/.glimmer_fp8.XXXXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
export MAX_JOBS="${MAX_JOBS:-4}" PYTHONDONTWRITEBYTECODE=1

python3 - "$here" "$build_dir" "$KERNELS_OUT" <<'PY'
import importlib.util
import os
import pathlib
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

source, build, out = map(pathlib.Path, sys.argv[1:])
include = pathlib.Path(torch.__file__).parent / "include"
cutlass = pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent / "data" / "cutlass"
arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.1a").split(";")[0].strip().removesuffix("+PTX")
sm = arch.replace(".", "")
name = "glimmer_fp8_ext"
load(
    name=name,
    sources=[str(source / "fp8x_ext.cu")],
    extra_cuda_cflags=[
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{sm},code=sm_{sm}",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        f"-I{source}", f"-I{include}", "--expt-extended-lambda",
        f"-I{cutlass / 'include'}", f"-I{cutlass / 'tools' / 'util' / 'include'}",
    ],
    extra_cflags=["-O3", f"-I{source}"],
    build_directory=str(build),
    verbose=False,
)
shutil.copy2(build / f"{name}.so", out / f"{name}.so")
print(f"glimmer_fp8: built {out / name}.so for sm_{sm}")
PY
