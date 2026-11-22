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
_silu = _lib.marlin_narrow_fused_silu
_silu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
_silu.restype = ctypes.c_int


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


def silu_mul(y):
    """Compiled-stock-compatible BF16 activation for the non-fused row shapes."""
    if (y.dtype != torch.bfloat16 or y.ndim != 2 or y.shape[-1] % 16
            or y.stride(-1) != 1 or y.stride(0) % 8 or y.data_ptr() % 16
            or not y.is_cuda):
        raise RuntimeError("marlin_narrow_fused.silu_mul: unsupported input")
    out = y.new_empty((y.shape[0], y.shape[1] // 2))
    with torch.cuda.device(y.device):
        status = _silu(y.data_ptr(), out.data_ptr(), y.shape[0], out.shape[1],
                       y.stride(0), torch.cuda.current_stream(y.device).cuda_stream)
    if status:
        raise RuntimeError(f"Standalone SiLU launch returned CUDA error {status}")
    return out
