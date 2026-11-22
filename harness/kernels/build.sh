#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
OUT_DIR="${KERNELS_OUT:?KERNELS_OUT must be set}"
mkdir -p "${OUT_DIR}"

readarray -t TORCH_CFG < <(python3 - <<'PY'
import torch
from torch.utils.cpp_extension import include_paths, library_paths
print(" ".join(f"-I{p}" for p in include_paths(device_type="cuda")))
print(" ".join(f"-L{p}" for p in library_paths(device_type="cuda")))
print("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
PY
)

INC_FLAGS="${TORCH_CFG[0]}"
LIB_FLAGS="${TORCH_CFG[1]}"
CXX11_ABI="${TORCH_CFG[2]}"

TMP_BUILD="$(mktemp -d /tmp/muse-kernels-XXXXXX)"
trap 'rm -rf "${TMP_BUILD}"' EXIT

nvcc -O3 -std=c++17 \
  -gencode arch=compute_121a,code=sm_121a \
  --shared -Xcompiler -fPIC \
  "-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" \
  ${INC_FLAGS} \
  "${SCRIPT_DIR}/muse_kernels.cu" \
  -o "${TMP_BUILD}/_muse_kernels.so" \
  ${LIB_FLAGS} -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

install -m 0755 "${TMP_BUILD}/_muse_kernels.so" "${OUT_DIR}/_muse_kernels.so"
install -m 0644 "${SCRIPT_DIR}/muse_kernels.py" "${OUT_DIR}/muse_kernels.py"
