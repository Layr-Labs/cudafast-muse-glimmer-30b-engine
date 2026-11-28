#!/usr/bin/env bash
# musebfx: lossless 12-bit coded BF16 small-M GEMM (drafter + BF16 target layers at verify width).
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$KERNELS_OUT"
build_dir=$(mktemp -d "$KERNELS_OUT/.musebfx.XXXXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
export MAX_JOBS="${MAX_JOBS:-4}" PYTHONDONTWRITEBYTECODE=1
python3 - "$here" "$build_dir" "$KERNELS_OUT" <<'PY'
import pathlib, shutil, sys
from torch.utils.cpp_extension import load
source, build, out = map(pathlib.Path, sys.argv[1:])
name = "muse_bfx_ext"
load(name=name, sources=[str(source / "bfx.cu"), str(source / "bf16_rans_mma.cu"), str(source / "int3_mma.cu"), str(source / "vq4d_mma.cu"), str(source / "int4_mma.cu"), str(source / "dflash_fuse.cu"), str(source / "draft_attn.cu"), str(source / "draft_vocab_tail.cu"), str(source / "draft_head_fast.cu"), str(source / "native_leaf.cu")],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
     extra_cflags=["-O3"], build_directory=str(build), verbose=False)
shutil.copy2(build / f"{name}.so", out / f"{name}.so")
print(f"musebfx: built {out / name}.so for sm_121a")
PY
