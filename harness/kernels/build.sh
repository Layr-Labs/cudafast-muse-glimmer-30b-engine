#!/usr/bin/env bash
# harness/kernels/build.sh: runs inside the pinned image (no network) with KERNELS_OUT and
# TORCH_CUDA_ARCH_LIST set; writes only under $KERNELS_OUT (on the resident's PYTHONPATH).
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$here/kpre/build.sh"
