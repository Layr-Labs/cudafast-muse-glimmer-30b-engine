# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M FP8 w8a8 GEMM (harness/kernels/fp8tile) behind a custom op.

fp8tile is the CUTLASS SM120 FP8 GEMM that ops.cutlass_scaled_mm runs for
M <= 16, instantiated with a narrower output tile (same instruction sequence
per output element, see harness/kernels/fp8tile). The custom op keeps the
dispatch on the real M inside the operation, so one symbolic compiled graph
serves both decode (fp8tile) and prefill (stock), and CUDA graph capture
records whichever kernel the captured M selects.

A layer is routed here only after fp8tile.prepare compared it bit for bit with
the stock kernel on the layer's own weight.
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    import fp8tile as _fp8tile
except ImportError:
    _fp8tile = None

_counts = {"admitted": 0, "rejected": 0}
_tiles: set[str] = set()
_logged = False


def _stock(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def prepare(b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> int | None:
    """Load-time admission of one layer (weight operand b = [K, N] column-major).
    Returns the fp8tile configuration, or None to keep the stock path."""
    cfg = None
    if _fp8tile is not None:
        try:
            cfg = _fp8tile.prepare(b, sa, sb, _stock)
        except Exception:  # noqa: BLE001 - any failure keeps the stock path
            cfg = None
    _counts["admitted" if cfg is not None else "rejected"] += 1
    if cfg is not None:
        _tiles.add("x".join(str(v) for v in _fp8tile.TILES[cfg]))
    return cfg


def _log_once() -> None:
    global _logged
    if not _logged:
        _logged = True
        logger.info(
            "fp8tile: %d of %d FP8 linear layers admitted (CUTLASS FP8 GEMM M<=16, tile %s)",
            _counts["admitted"],
            _counts["admitted"] + _counts["rejected"],
            ",".join(sorted(_tiles)) or "-",
        )


def _scaled_mm(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, cfg: int
) -> torch.Tensor:
    _log_once()
    # Real tensor metadata: decided per call, never at trace time.
    if (
        1 <= a.shape[0] <= 16
        and a.dtype == torch.float8_e4m3fn
        and a.is_contiguous()
        and a.data_ptr() % 16 == 0
        and a.shape[1] == b.shape[0]
        and sa.numel() == 1
        and sb.numel() == 1
        and sa.dtype == torch.float32
        and sb.dtype == torch.float32
    ):
        return _fp8tile.gemm(a, b, sa, sb, cfg)
    return _stock(a, b, sa, sb)


def _scaled_mm_fake(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, cfg: int
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device)


direct_register_custom_op("fp8tile_scaled_mm", _scaled_mm, fake_impl=_scaled_mm_fake)
