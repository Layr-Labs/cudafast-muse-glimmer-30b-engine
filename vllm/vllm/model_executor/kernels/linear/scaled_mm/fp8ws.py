# SPDX-License-Identifier: Apache-2.0
"""Zero-VRAM dual-TMA M=512 prefill dispatch for MuseGlimmer FP8 linear layers.

From SP5D's Yukon submission 98c1361e-289f-45a3-8471-5df0259bd3b2 (this file and harness/kernels/fp8ws). Muse Glimmer
routing (integration3): only the attention qkv (4608, 6656) and output-gate (4096, 6656) projections, where fp8ws beat
stock CUTLASS in sustained capped timing on GB10 (opus-prefill2 ws-w1: 199.5 vs 208.2 us and 189.4 vs 195.7 us per
call, 6/6 rounds each). Every other shape keeps its current route (glimmer_fp8._prefill_gemm). The rival's 8-repetition
timing gate is replaced by that fixed routing, so the kernel choice does not depend on a load-time race; admission is
the rival's bitwise check against ops.cutlass_scaled_mm on the live activation, a random one and a bit-pattern one
(each output element is one ascending-K mma.sync chain from +0, then bf16_rn(fmul(fmul(acc, scale_b), scale_a)), the
CUTLASS SM120 order). The first call of each shape outside graph capture runs it; FP8WS=0 disables the kernel."""

from __future__ import annotations

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("FP8WS", "1") != "0"
SHAPES = frozenset({
    (4608, 6656),
    (4096, 6656),
})
_admitted: dict[tuple[int, int, int], bool] = {}

try:
    import fp8ws as _fp8ws
except ImportError:
    _fp8ws = None


def _admit_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    n: int,
    k: int,
) -> bool:
    assert _fp8ws is not None
    dev = a.device
    if not _fp8ws.ready(dev):
        return False
    try:
        # Bitwise equality on the live input and synthetic / exhaustive bit-pattern activations.
        g = torch.Generator(device=dev).manual_seed(20261121)
        bits = (
            torch.arange(512 * k, device=dev, dtype=torch.int64) * 7
            + torch.arange(512, device=dev, dtype=torch.int64).repeat_interleave(k)
        )
        a_bits = (bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(512, k)
        a_rand = (
            torch.randn((512, k), device=dev, generator=g, dtype=torch.float32) * 20.0
        ).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        for act in (a, a_rand, a_bits):
            ref = ops.cutlass_scaled_mm(act, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            got = _fp8ws.gemm(act, b, sa, sb)
            if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                logger.warning("fp8ws: bitwise mismatch vs CUTLASS for (%d, %d); keeping CUTLASS", n, k)
                return False
        logger.info("fp8ws: (%d, %d) bit-exact vs cutlass_scaled_mm on live, random and bit-pattern activations "
                    "-> ADMITTED (fixed routing)", n, k)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("fp8ws: admission error for (%d, %d): %r; keeping CUTLASS", n, k, exc)
        return False


def maybe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor | None:
    """Return the fp8ws result for admitted M=512 FP8 GEMMs, or None to use the current route."""
    if not ENABLED or _fp8ws is None:
        return None
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 512:
        return None
    k, n = b.shape
    if (
        (n, k) not in SHAPES
        or a.shape[1] != k
        or not a.is_cuda
        or b.device != a.device
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != torch.float8_e4m3fn
        or not a.is_contiguous()
        or b.stride() != (1, k)
        or a.data_ptr() % 16 != 0
        or b.data_ptr() % 16 != 0
        or sa.device != a.device
        or sb.device != a.device
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
        or sa.numel() != 1
        or sb.numel() != 1
        or not sa.is_contiguous()
        or not sb.is_contiguous()
    ):
        return None

    dev_idx = a.device.index if a.device.index is not None else torch.cuda.current_device()
    key = (dev_idx, n, k)
    admitted = _admitted.get(key)
    if admitted is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        admitted = _admit_shape(a, b, sa, sb, n, k)
        _admitted[key] = admitted
    if not admitted:
        return None
    return _fp8ws.gemm(a, b, sa, sb)
