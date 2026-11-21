#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must identify the writable build directory}"
mkdir -p "${KERNELS_OUT}"
export TMPDIR="${KERNELS_OUT}"
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS=4
python3 "$(dirname "${BASH_SOURCE[0]}")/build.py"
