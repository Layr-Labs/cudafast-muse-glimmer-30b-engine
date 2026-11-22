# SPDX-License-Identifier: Apache-2.0
"""Paired whole-K gate/up Marlin with the validated BF16 SiLU epilogue."""
import ctypes
from pathlib import Path
import torch

ROWS = 512
OUTPUT_COLUMNS = 39936
INPUT_COLUMNS = 6656
SM_COUNT = 48

_lib = ctypes.CDLL(str(Path(__file__).with_name("libgb10_marlin_narrow_fused.so")))
_launch = _lib.marlin_narrow_fused
_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
_launch.restype = ctypes.c_int


def eligible(n, k, device, props):
    return (
        (n, k) == (OUTPUT_COLUMNS, INPUT_COLUMNS)
        and device.type == "cuda"
        and props.multi_processor_count == SM_COUNT
        and torch.cuda.get_device_capability(device) == (12, 1)
    )


def gemm(x, weight, scales, global_scale, workspace, n, k):
    if (
        (n, k) != (OUTPUT_COLUMNS, INPUT_COLUMNS)
        or x.shape != (ROWS, INPUT_COLUMNS)
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, n // 2))
    # The complete-K geometry never touches the stock reduction scratch. Reuse
    # the persistent workspace for that ABI slot rather than allocate 12.6 MiB
    # of dead scratch on every captured-call argument construction.
    partial = workspace
    with torch.cuda.device(x.device):
        status = _launch(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), scales.data_ptr(),
            global_scale.data_ptr(), partial.data_ptr(), workspace.data_ptr(),
            torch.cuda.current_stream(x.device).cuda_stream, ROWS, n, k,
        )
    if status:
        raise RuntimeError(f"Fused narrow Marlin launch returned CUDA error {status}")
    return out
