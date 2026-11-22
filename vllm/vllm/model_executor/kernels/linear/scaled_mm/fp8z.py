# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (glimmer dec2): the single-row (decode) FP8 w8a8 GEMM of static
per-tensor FP8 linears reads a lossless 7-bit coding of the weight
(harness/kernels/fp8z) instead of the 8-bit weight.

The coding is a load-time, input-independent repack: fp8z.prepare decodes the
whole buffer with the kernel's own decoder and keeps it only if every byte
equals the weight; otherwise the layer keeps its 8-bit path. The kernel computes
what ops.cutlass_scaled_mm (and the glimmer_fp8 small-row tile) computes for one
row (one ascending-K e4m3 m16n8k32 chain per output from +0, fp32 * scale_b,
bf16_rn(* scale_a)), so the BF16 output bits are unchanged. The original weight
stays for every other row count. glimmer_fp8.fp8_scaled_mm looks the code up by
the weight's address (REGISTRY) for one-row calls of the shapes in SHAPES.
FP8Z=0 disables it.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
_logged = False

# (N, K) of the FP8 linears whose one-row call reads the code; a shape is listed
# only where fp8z measured faster than the glimmer_fp8 tile on GB10.
SHAPES: set[tuple[int, int]] = {
    (4608, 6656),   # qkv
    (4096, 6656),   # attention output gate
    (6656, 4096),   # o_proj
    (39936, 6656),  # gate_up
    (6656, 19968),  # down
}
# B (= W^T, column-major [K, N]) data_ptr -> code buffer
REGISTRY: dict[int, torch.Tensor] = {}


def prepare_weight(b: torch.Tensor) -> torch.Tensor | None:
    """Build, verify and register the fp8z buffer of B (column-major [K, N] e4m3),
    or None to keep the 8-bit path for every row count."""
    if os.environ.get("FP8Z", "1") == "0":
        return None
    k, n = b.shape
    if (n, k) not in SHAPES:
        return None
    try:
        import fp8z
    except ImportError:
        return None
    rejected = fp8z.stats["rejected"]
    try:
        comp = fp8z.prepare(b.t())
    except Exception as exc:  # noqa: BLE001 (a failed encoder or decoder keeps the 8-bit path)
        logger.warning("fp8z: prepare failed (%r); this layer keeps the 8-bit weight", exc)
        return None
    if comp is None:
        if fp8z.stats["rejected"] > rejected:
            logger.warning("fp8z: %s; this layer keeps the 8-bit weight", fp8z.stats["last_reason"])
        return None
    REGISTRY[b.data_ptr()] = comp
    return comp


def _log_once(fp8z) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    s = fp8z.stats
    raw = max(s["raw_bytes"], 1)
    logger.info(
        "fp8z active: %d FP8 linears read from the lossless 7-bit code at one row "
        "(%.1f MB -> %.1f MB, %.3f bits/weight, %d kept the 8-bit weight); FP8Z=0 disables",
        s["tensors"], s["raw_bytes"] / 1e6, s["bytes"] / 1e6, 8.0 * s["bytes"] / raw,
        s["rejected"],
    )


def lookup(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor | None:
    """The code buffer for a one-row call that fp8z serves, else None."""
    if a.size(0) != 1 or not REGISTRY:
        return None
    comp = REGISTRY.get(b.data_ptr())
    if (
        comp is None
        or a.dtype != torch.float8_e4m3fn
        or not a.is_contiguous()
        or a.data_ptr() % 16
        or a.size(1) != b.size(0)
        or sa.numel() != 1
        or sb.numel() != 1
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
        or a.device != comp.device
    ):
        return None
    return comp


def _gemm(a: torch.Tensor, comp: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, n: int, k: int) -> torch.Tensor:
    import fp8z

    _log_once(fp8z)
    return fp8z.gemm(a, comp, sa, sb, n, k)


def _gemm_fake(a: torch.Tensor, comp: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, n: int,
               k: int) -> torch.Tensor:
    return torch.empty((a.shape[0], n), device=a.device, dtype=torch.bfloat16)


# The launch as its own op, so the l2pf capture hooks see it (and its code buffer).
direct_register_custom_op("fp8z_gemm", _gemm, fake_impl=_gemm_fake)
