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
import pathlib
import shutil
import sys

from torch.utils.cpp_extension import load

source, build, out = map(pathlib.Path, sys.argv[1:])
name = "muse_rans_ext"
load(
    name=name,
    sources=[str(source / "csrc" / "muse_rans.cu")],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-DMUSE_RANS_STANDALONE_MODULE", "-lineinfo"],
    extra_cflags=["-O3"],
    build_directory=str(build),
    verbose=False,
)
shutil.copy2(build / f"{name}.so", out / f"{name}.so")
print(f"muserans: built {out / name}.so for sm_121a")
PY

# muse_rans_rows_ext: the same source built with -DRANS_MR=7, the 7-row DFlash verify GEMM (rans_gemv_rows) over the
# blobs packed above. Optional: if it does not build, every verify-block FP8 GEMM stays on the small-row CUTLASS op.
rows_dir=$(mktemp -d "$KERNELS_OUT/.muserans_rows.XXXXXXXX")
trap 'rm -rf -- "$build_dir" "$rows_dir"' EXIT
if ! python3 - "$here" "$rows_dir" "$KERNELS_OUT" <<'PY'
import pathlib
import shutil
import sys

from torch.utils.cpp_extension import load

source, build, out = map(pathlib.Path, sys.argv[1:])
name = "muse_rans_rows_ext"
load(
    name=name,
    sources=[str(source / "csrc" / "muse_rans.cu")],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-DMUSE_RANS_STANDALONE_MODULE", "-DRANS_MR=7",
                       "-lineinfo"],
    extra_cflags=["-O3"],
    build_directory=str(build),
    verbose=False,
)
shutil.copy2(build / f"{name}.so", out / f"{name}.so")
print(f"muserans: built {out / name}.so (7 verify rows) for sm_121a")
PY
then
  rm -f "$KERNELS_OUT/muse_rans_rows_ext.so"
  echo "muserans: verify-rows extension not built; verify-block FP8 GEMMs stay on the small-row CUTLASS op" >&2
fi
