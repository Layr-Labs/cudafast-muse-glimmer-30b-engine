#!/usr/bin/env bash
# muse_draft_rows: libmuse_draft_rows.so (ctypes) plus its Python package, into $KERNELS_OUT/muse_draft_rows.
# Optional: the DFlash drafter keeps its stock BF16 GEMMs if this does not build, so a failure is not fatal.
set -uo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/muse_draft_rows
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_121a,code=sm_121a" -o "$out/libmuse_draft_rows.so" "$here/rows.cu"; then
  rm -rf "$out"
  echo "muse_draft_rows: not built; the drafter keeps its stock BF16 GEMMs" >&2
  exit 0
fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "muse_draft_rows: built $out/libmuse_draft_rows.so for sm_121a"
