# SPDX-License-Identifier: Apache-2.0
"""Narrow whole-K Marlin for the measured widening prefill geometry."""

import ctypes
import os
from pathlib import Path

import torch

ROWS = 512
OUTPUT_COLUMNS = 39936
INPUT_COLUMNS = 6656
SM_COUNT = 48
ENABLED = os.environ.get("MARLIN_NARROW_PREFILL", "1") == "1"

_lib = ctypes.CDLL(str(Path(__file__).with_name("libgb10_marlin_prefill.so")))
_launch = _lib.marlin_prefill
_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5
_launch.restype = ctypes.c_int


def eligible(n: int, k: int, device: torch.device, props) -> bool:
    return (
        (n, k) == (OUTPUT_COLUMNS, INPUT_COLUMNS)
        and device.type == "cuda"
        and props.multi_processor_count == SM_COUNT
        and torch.cuda.get_device_capability(device) == (12, 1)
    )


_PARTIAL_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor | None:
    if (
        (n, k) != (OUTPUT_COLUMNS, INPUT_COLUMNS)
        or x.shape != (ROWS, INPUT_COLUMNS)
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        return None

    out = x.new_empty((ROWS, n))
    key = (x.device, SM_COUNT)
    partial = _PARTIAL_CACHE.get(key)
    if partial is None:
        partial = x.new_empty((SM_COUNT, 64, 256), dtype=torch.float32)
        _PARTIAL_CACHE[key] = partial
    with torch.cuda.device(x.device):
        # This geometry selects the complete-K branch, which supplies its own
        # dynamic shared-memory size. The last argument is for the wide branch.
        status = _launch(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), partial.data_ptr(),
            workspace.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream,
            ROWS, n, k, SM_COUNT, 0,
        )
    if status:
        raise RuntimeError(f"Narrow Marlin prefill launch returned CUDA error {status}")
    return out
