#!/usr/bin/env bash
# muserans: muse_rans_ext.so (torch extension), the lossless-FP8 one-token GEMV (csrc/muse_rans.cu), into $KERNELS_OUT.
# Same numerics flags as the stock kernels: no --use_fast_math.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
build_dir=$(mktemp -d "$KERNELS_OUT/.muserans.XXXXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
export MAX_JOBS="${MAX_JOBS:-4}" PYTHONDONTWRITEBYTECODE=1

python3 - "$here" "$build_dir" "$KERNELS_OUT" <<'PY'
import os
import pathlib
import shutil
import sys

from torch.utils.cpp_extension import load

source, build, out = map(pathlib.Path, sys.argv[1:])
arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.1a").split(";")[0].strip().removesuffix("+PTX")
sm = arch.replace(".", "")
name = "muse_rans_ext"
load(
    name=name,
    sources=[str(source / "csrc" / "muse_rans.cu")],
    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{sm},code=sm_{sm}", "-DMUSE_RANS_STANDALONE_MODULE", "-lineinfo"],
    extra_cflags=["-O3"],
    build_directory=str(build),
    verbose=False,
)
shutil.copy2(build / f"{name}.so", out / f"{name}.so")
print(f"muserans: built {out / name}.so for sm_{sm}")
PY
