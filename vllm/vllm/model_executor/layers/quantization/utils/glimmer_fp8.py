# SPDX-License-Identifier: Apache-2.0
"""Decode (M <= 16) FP8 w8a8 GEMMs through glimmer_fp8_ext (harness/kernels/glimmer_fp8): CUTLASS sm120 FP8 kernels whose
epilogue is vLLM's ScaledEpilogue arithmetic, D = bf16(scale_a * fp32(scale_b * acc)).
The one-row measurements select a 16x32x512 tile. M=2..16 uses the 16x32x128 tile.
The per-element k order is the same for both tiles (no split-K). Only
per-tensor scales (both scalars), no bias, bf16 output and the shapes listed in _CFG take this path; everything else, and
every prefill, calls ops.cutlass_scaled_mm unchanged. The shape test is inside an opaque custom op."""
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import glimmer_fp8_ext as _ext  # built into KERNELS_OUT
except Exception:
    _ext = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.fp8ws import (
        maybe_gemm as _fp8ws_maybe_gemm,
    )
except Exception:
    _fp8ws_maybe_gemm = None

# (N, K) -> fp8x config id (0 = vLLM's own M16 config; 1..4 alternatives), chosen by quiet-GPU timing on GB10.
_CFG: dict[tuple[int, int], int] = {
    (4608, 6656): 2,   # qkv
    (4096, 6656): 2,   # attention output gate
    (6656, 4096): 2,   # o_proj
    (39936, 6656): 2,  # gate_up
    (6656, 19968): 2,  # down
}

# On SM120 (170 SMs), cutlass_scaled_mm beats glimmer_fp8 16x32x512 on o_proj and down.
_CFG_M1_SM120: dict[tuple[int, int], int] = {
    (4608, 6656): 512,
    (4096, 6656): 512,
    (39936, 6656): 2,
}
_IS_SM120: dict[int, bool] = {}


def _is_sm120(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _IS_SM120.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_capability(device) == (12, 0)
        _IS_SM120[idx] = cached
    return cached


@torch.library.custom_op("glimmer::fp8_scaled_mm", mutates_args=())
def fp8_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if a.size(0) == 512 and _fp8ws_maybe_gemm is not None:
        out = _fp8ws_maybe_gemm(a, b, scale_a, scale_b)
        if out is not None:
            return out
    if (_ext is not None and 0 < a.size(0) <= 16
            and a.is_cuda and a.device == b.device == scale_a.device == scale_b.device
            and scale_a.numel() == scale_b.numel() == 1
            and scale_a.dtype == scale_b.dtype == torch.float32
            and a.dtype == b.dtype == torch.float8_e4m3fn
            and a.is_contiguous() and b.stride(0) == 1 and b.stride(1) == b.size(0)):
        key = (b.size(1), b.size(0))
        if a.size(0) == 1:
            if _is_sm120(a.device):
                cfg = _CFG_M1_SM120.get(key)
            else:
                cfg = 512 if key in _CFG else None
        else:
            cfg = _CFG.get(key)
        if cfg is not None:
            return _ext.gemm(a, b, scale_a.reshape(1), scale_b.reshape(1), cfg)
    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None)


@fp8_scaled_mm.register_fake
def _(a, b, scale_a, scale_b):
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.bfloat16)


try:
    import muse_fp8_swiglu as _swiglu
except ImportError:
    _swiglu = None


def prepare_fp8_swiglu_weight(weight: torch.Tensor) -> torch.Tensor | None:
    return _swiglu.prepare(weight) if _swiglu is not None else None


_swiglu_logged = False


@torch.library.custom_op("glimmer::fp8_gate_swiglu", mutates_args=())
def fp8_gate_swiglu(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    global _swiglu_logged
    if not _swiglu_logged:
        # First call of the running engine's forward (the load-time check
        # calls _swiglu.apply directly).
        _swiglu_logged = True
        logger.info(
            "FP8 gate/up SwiGLU prefill in use: one CUTLASS FP8 GEMM with SiLU-and-mul and the "
            "down projection's FP8 quantization in the epilogue (muse_fp8_swiglu)"
        )
    return _swiglu.apply(a, b, scale_a, scale_b, down_scale)


@fp8_gate_swiglu.register_fake
def _(a, b, scale_a, scale_b, down_scale):
    return a.new_empty((a.size(0), b.size(1) // 2))
