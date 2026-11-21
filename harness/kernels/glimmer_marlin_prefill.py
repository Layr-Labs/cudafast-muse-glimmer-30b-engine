"""A stock Marlin W4A16 specialization with weight reuse across activation rows."""

import ctypes
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op


_library = ctypes.CDLL(str(Path(__file__).with_name("libgb10_marlin_prefill.so")))
_launch = _library.marlin_prefill
_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5
_launch.restype = ctypes.c_int


def _linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
    sms: int,
    shared_bytes: int,
) -> torch.Tensor:
    rows = x.numel() // k
    # These shapes select the 64x256x64 Marlin tile and one CTA per SM.
    # The stock launcher splits larger batches and uses smaller tiles for decode.
    if not (
        128 <= rows <= 1024
        and rows % 64 == 0
        and (rows // 64) * (n // 256) > sms
        and x.is_contiguous()
    ):
        return apply_fp4_marlin_linear(x, weight, scales, global_scale, workspace, n, k)
    out = x.new_empty((*x.shape[:-1], n))
    partial = torch.empty((sms, 64, 256), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        status = _launch(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), scales.data_ptr(),
            global_scale.data_ptr(), partial.data_ptr(), workspace.data_ptr(),
            torch.cuda.current_stream().cuda_stream, rows, n, k, sms, shared_bytes,
        )
    if status:
        raise RuntimeError(f"GB10 Marlin prefill launch returned CUDA error {status}")
    return out


def _fake(x, weight, scales, global_scale, workspace, n, k, sms, shared_bytes):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("gb10_marlin_prefill", _linear, fake_impl=_fake)
