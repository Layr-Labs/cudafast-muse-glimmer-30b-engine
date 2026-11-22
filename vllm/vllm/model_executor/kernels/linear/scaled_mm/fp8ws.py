# SPDX-License-Identifier: Apache-2.0
"""Zero-VRAM dual-TMA M=512 prefill dispatch for MuseGlimmer FP8 linear layers."""

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
    (6656, 4096),
    (39936, 6656),
    (6656, 19968),
})
_admitted: dict[tuple[int, int, int], bool] = {}
_admitted_dual: dict[tuple[int, int, int, int], bool] = {}

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
        # 1. Bitwise equality on live inputs and synthetic / exhaustive bit-pattern activations.
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

        if os.environ.get("FP8WS_FORCE", "0") == "1":
            logger.info("fp8ws: admitted (%d, %d) (forced)", n, k)
            return True

        # 2. Timing gate against CUTLASS on this GPU.
        for _ in range(3):
            ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            _fp8ws.gemm(a, b, sa, sb)
        torch.cuda.synchronize(dev)

        def _measure_us(fn) -> float:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(8):
                fn()
            e.record()
            torch.cuda.synchronize(dev)
            return s.elapsed_time(e) * 1e3 / 8.0

        t_cut = _measure_us(lambda: ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        t_ws = _measure_us(lambda: _fp8ws.gemm(a, b, sa, sb))
        ok = t_ws < t_cut
        logger.info(
            "fp8ws: (%d, %d) bit-exact; cutlass=%.1fus fp8ws=%.1fus -> %s",
            n,
            k,
            t_cut,
            t_ws,
            "ADMITTED" if ok else "KEPT CUTLASS",
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("fp8ws: admission error for (%d, %d): %r; keeping CUTLASS", n, k, exc)
        return False


def maybe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor | None:
    """Return fp8ws result for admitted M=512 FP8 GEMMs, or None to use CUTLASS."""
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


def _admit_shape_dual(
    a: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    sa: torch.Tensor,
    sb0: torch.Tensor,
    sb1: torch.Tensor,
    n0: int,
    n1: int,
    k: int,
) -> bool:
    assert _fp8ws is not None
    dev = a.device
    if not _fp8ws.ready(dev):
        return False
    try:
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
            ref0 = ops.cutlass_scaled_mm(act, b0, scale_a=sa, scale_b=sb0, out_dtype=torch.bfloat16)
            ref1 = ops.cutlass_scaled_mm(act, b1, scale_a=sa, scale_b=sb1, out_dtype=torch.bfloat16)
            got0, got1 = _fp8ws.gemm_dual(act, b0, b1, sa, sb0, sb1)
            if not (
                torch.equal(ref0.view(torch.int16), got0.view(torch.int16))
                and torch.equal(ref1.view(torch.int16), got1.view(torch.int16))
            ):
                logger.warning("fp8ws: dual bitwise mismatch vs CUTLASS for (%d+%d, %d); keeping CUTLASS", n0, n1, k)
                return False

        if os.environ.get("FP8WS_FORCE", "0") == "1":
            logger.info("fp8ws: admitted dual (%d+%d, %d) (forced)", n0, n1, k)
            return True

        for _ in range(3):
            ops.cutlass_scaled_mm(a, b0, scale_a=sa, scale_b=sb0, out_dtype=torch.bfloat16)
            ops.cutlass_scaled_mm(a, b1, scale_a=sa, scale_b=sb1, out_dtype=torch.bfloat16)
            _fp8ws.gemm_dual(a, b0, b1, sa, sb0, sb1)
        torch.cuda.synchronize(dev)

        def _measure_us(fn) -> float:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(8):
                fn()
            e.record()
            torch.cuda.synchronize(dev)
            return s.elapsed_time(e) * 1e3 / 8.0

        t_cut = _measure_us(lambda: (
            ops.cutlass_scaled_mm(a, b0, scale_a=sa, scale_b=sb0, out_dtype=torch.bfloat16),
            ops.cutlass_scaled_mm(a, b1, scale_a=sa, scale_b=sb1, out_dtype=torch.bfloat16),
        ))
        t_ws = _measure_us(lambda: _fp8ws.gemm_dual(a, b0, b1, sa, sb0, sb1))
        ok = t_ws < t_cut
        logger.info(
            "fp8ws: dual (%d+%d, %d) bit-exact; cutlass=%.1fus fp8ws_dual=%.1fus -> %s",
            n0,
            n1,
            k,
            t_cut,
            t_ws,
            "ADMITTED" if ok else "KEPT CUTLASS",
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("fp8ws: dual admission error for (%d+%d, %d): %r; keeping CUTLASS", n0, n1, k, exc)
        return False


def maybe_gemm_dual(
    a: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    sa: torch.Tensor,
    sb0: torch.Tensor,
    sb1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return dual fp8ws result for admitted M=512 (4608+4096, 6656) FP8 GEMMs."""
    if not ENABLED or _fp8ws is None:
        return None
    if a.ndim != 2 or b0.ndim != 2 or b1.ndim != 2 or a.shape[0] != 512:
        return None
    k, n0 = b0.shape
    k1, n1 = b1.shape
    if (
        n0 != 4608
        or n1 != 4096
        or k != 6656
        or k1 != k
        or a.shape[1] != k
        or not a.is_cuda
        or b0.device != a.device
        or b1.device != a.device
        or a.dtype != torch.float8_e4m3fn
        or b0.dtype != torch.float8_e4m3fn
        or b1.dtype != torch.float8_e4m3fn
        or not a.is_contiguous()
        or b0.stride() != (1, k)
        or b1.stride() != (1, k)
        or a.data_ptr() % 16 != 0
        or b0.data_ptr() % 16 != 0
        or b1.data_ptr() % 16 != 0
        or sa.device != a.device
        or sb0.device != a.device
        or sb1.device != a.device
        or sa.dtype != torch.float32
        or sb0.dtype != torch.float32
        or sb1.dtype != torch.float32
        or sa.numel() != 1
        or sb0.numel() != 1
        or sb1.numel() != 1
        or not sa.is_contiguous()
        or not sb0.is_contiguous()
        or not sb1.is_contiguous()
    ):
        return None

    dev_idx = a.device.index if a.device.index is not None else torch.cuda.current_device()
    key = (dev_idx, n0, n1, k)
    admitted = _admitted_dual.get(key)
    if admitted is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        admitted = _admit_shape_dual(a, b0, b1, sa, sb0, sb1, n0, n1, k)
        _admitted_dual[key] = admitted
    if not admitted:
        return None
    return _fp8ws.gemm_dual(a, b0, b1, sa, sb0, sb1)
