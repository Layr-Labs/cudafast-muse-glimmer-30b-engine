# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    should_use_atomic_add_reduce,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

try:
    import marlinw
except ImportError:
    marlinw = None

try:
    from marlinw import swiglu as marlin_swiglu
except (ImportError, OSError):
    marlin_swiglu = None

try:
    from marlinw import swiglu_fp8 as marlin_swiglu_fp8
except (ImportError, OSError):
    marlin_swiglu_fp8 = None

try:
    import marlin_narrow
except (ImportError, OSError):
    marlin_narrow = None

try:
    import marlinw_down
except (ImportError, OSError):
    marlinw_down = None

try:
    import marlinw_down128
except (ImportError, OSError):
    marlinw_down128 = None

_down128_weights: set[int] = set()
_attn128_weights: set[int] = set()


# Which 512-row MLP kernel each call ran (Python runs on eager forwards and on graph capture, not on replay), logged
# once a full forward's worth (45 paired gate/up, 40 down) has been routed: shows that placement kept the routes.
_mlp_routes = {"gate_128": 0, "gate_64": 0, "gate_slab": 0, "down128": 0, "slab": 0, "down64": 0, "logged": False}


def _count_mlp_route(kind: str) -> None:
    _mlp_routes[kind] += 1
    gate = _mlp_routes["gate_128"] + _mlp_routes["gate_64"] + _mlp_routes["gate_slab"]
    down = _mlp_routes["down128"] + _mlp_routes["slab"] + _mlp_routes["down64"]
    if not _mlp_routes["logged"] and gate >= 45 and down >= 40:
        _mlp_routes["logged"] = True
        logger.info(
            "512-row MLP routes over the first %d gate/up and %d down calls: gate/up %d on the 128-row paired "
            "kernel, %d on the 64-row one, %d on the final slab; down %d on the 128x128 tile, %d on the final slab, "
            "%d on marlinw_down",
            gate, down, _mlp_routes["gate_128"], _mlp_routes["gate_64"], _mlp_routes["gate_slab"], _mlp_routes["down128"],
            _mlp_routes["slab"], _mlp_routes["down64"],
        )


def down128_admitted(weight: torch.Tensor) -> bool:
    """Muse Glimmer prefill weight placement: True if this down weight runs the 128x128-tile kernel."""
    return marlinw_down128 is not None and weight.data_ptr() in _down128_weights


def down128_moved(old_ptr: int, new_ptr: int) -> None:
    """A placed down weight moved to another buffer with the same bytes: it keeps its kernel."""
    if old_ptr in _down128_weights:
        _down128_weights.discard(old_ptr)
        _down128_weights.add(new_ptr)
        if hasattr(marlinw_down128, "moved"):
            marlinw_down128.moved(old_ptr, new_ptr)  # its TMA-fed kernel, if admitted
    if old_ptr in _attn128_weights:
        _attn128_weights.discard(old_ptr)
        _attn128_weights.add(new_ptr)

try:
    import muse_final_slab
except (ImportError, OSError):
    muse_final_slab = None

try:
    import marlinw_attn
except (ImportError, OSError):
    marlinw_attn = None

try:
    from muse_marlin_gemv import prepare_decode_layout
except (ImportError, OSError):
    prepare_decode_layout = None

try:
    # 6-bit block-scale codes for the tiled decode GEMV (harness/kernels/decode6)
    from muse_marlin6_gemv import STATS as _scale6_stats
    from muse_marlin6_gemv import prepare_decode_layout6
except (ImportError, OSError):
    prepare_decode_layout6 = None
    _scale6_stats = None

logger = init_logger(__name__)

# Prompt-sized calls of NVFP4 weights much larger than L2 (MLP gate_up and
# down) take one of two schedules that give the same bits as the single
# ops.marlin_gemm call, chosen per layer at load:
#   1. marlinw (harness/kernels/marlinw), when built: the Marlin kernel body
#      on the same work slices as ops.marlin_gemm, handed out column-major, for
#      every call of 128 rows or more;
#   2. otherwise, or if a layer fails marlinw's load-time check, the
#      column-sliced schedule below (stock ops.marlin_gemm on column slices of
#      a slice-major copy of the same weight), for the row counts where it is
#      exact.
# Every other call, and every layer that fails both checks, runs the single
# ops.marlin_gemm call.
_PROMPT_CHECK_M = (512, 4096)
_prompt_stats = {
    "wide": 0, "narrow": 0, "narrow_down": 0, "down128": 0, "down128_rejected": 0, "sliced": 0, "single": 0, "logged": False,
    # Narrow 512-row NVFP4 attention projections: [admitted, rejected] per shape.
    "attn": {"qkv": [0, 0], "gate": [0, 0], "o": [0, 0]},
}
_attn_route_logged: set = set()

# Column-sliced schedule for NVFP4 weights much larger than L2 (prompt-sized M).
#
# ops.marlin_gemm walks its output tiles chunk-major: every 256-column tile of
# rows 0-63, then of rows 64-127, and so on. A weight much larger than L2 is
# therefore streamed from DRAM again for each group of row chunks. Calling the
# same kernel on column slices of the same weight keeps one slice's codes and
# scales in L2 while all row chunks use them.
#
# Both schedules give bitwise identical outputs when every launch of both
# computes each (64-row chunk, 256-column tile) in a single CTA over the whole
# K range with the same thread config: the summation order of an output element
# is then set by the K loop alone and does not depend on the column range of
# the call. marlin_mm issues launches of up to `max_par` 64-row chunks (16 if
# n > 4096, else 128), and its partition gives every CTA whole tiles exactly
# when a launch's chunks * n / 256 is a multiple of the SM count (no tile is
# split along K, so there is no global reduction). _marlin_fp4_col_sliced_rows
# checks this for both schedules; a load-time comparison against the single
# call confirms it for every layer that uses the sliced path.
_COL_TILE_N = 256
_COL_SLICE_REF_M = 512


def _log_prompt_stats() -> None:
    if not _prompt_stats["logged"]:
        _prompt_stats["logged"] = True
        if _scale6_stats is not None:
            logger.info("NVFP4 decode palette admission: %d compressed, %d kept uncompressed",
                        _scale6_stats["admitted"], _scale6_stats["kept"])
            logger.info("NVFP4 fixed decode gate/up grid admitted on %d layers", _scale6_stats["fixed_grid"])
            coded = _scale6_stats.get("srans_scales", 0)
            logger.info("NVFP4 rANS-coded gate/up block scales: %d layers admitted, %d rejected, %.3f bits per scale",
                        _scale6_stats.get("gate_up_srans", 0), _scale6_stats.get("gate_up_srans_rejected", 0),
                        8.0 * _scale6_stats.get("srans_bytes", 0) / coded if coded else 0.0)
            coded = _scale6_stats.get("down_srans_scales", 0)
            logger.info("NVFP4 rANS-coded down block scales: %d layers admitted, %d rejected, %.3f bits per scale",
                        _scale6_stats.get("down_srans", 0), _scale6_stats.get("down_srans_rejected", 0),
                        8.0 * _scale6_stats.get("down_srans_bytes", 0) / coded if coded else 0.0)
        logger.info(
            "Marlin NVFP4 prompt GEMM: %d layers on the wide schedule, %d on "
            "column slices, %d on the single call",
            _prompt_stats["wide"],
            _prompt_stats["sliced"],
            _prompt_stats["single"],
        )
        if _prompt_stats["narrow"]:
            logger.info(
                "Narrow Marlin prefill passed bitwise validation on %d layers",
                _prompt_stats["narrow"],
            )
        if _prompt_stats["narrow_down"]:
            logger.info(
                "Narrow split-K Marlin prefill passed bitwise validation on %d layers",
                _prompt_stats["narrow_down"],
            )
        if _prompt_stats["down128"] or _prompt_stats["down128_rejected"]:
            logger.info(
                "128x128-tile down prefill GEMM admitted on %d layers, rejected on %d",
                _prompt_stats["down128"],
                _prompt_stats["down128_rejected"],
            )
        tma = []
        if marlinw_down128 is not None and hasattr(marlinw_down128, "counts"):
            offered, passed = marlinw_down128.counts()
            if offered:
                tma.append(f"down {passed} of {offered} layers")
        if marlin_swiglu is not None and hasattr(marlin_swiglu, "counts"):
            fast, passed = marlin_swiglu.counts()
            if fast:
                tma.append(f"gate/up {passed} of {fast} layers")
        if marlin_swiglu_fp8 is not None and hasattr(marlin_swiglu_fp8, "counts"):
            fp8_passed = marlin_swiglu_fp8.counts()
            if fp8_passed:
                tma.append(f"gate/up_fp8 {fp8_passed} layers")
        if tma:
            logger.info("TMA-fed 128-row prefill GEMM admitted: %s", ", ".join(tma))
        attn = _prompt_stats["attn"]
        if any(a + r for a, r in attn.values()):
            logger.info(
                "Narrow NVFP4 attention prefill GEMM (admitted/rejected): qkv %d/%d, "
                "gate %d/%d, o %d/%d (128x128 admitted: %d)",
                *attn["qkv"], *attn["gate"], *attn["o"], len(_attn128_weights),
            )


def _marlin_fp4_single(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    if (
        x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and decode_weight is not None
        and decode_scales is not None
    ):
        if decode_palette is not None:
            # decode_scales holds the layer's 6-bit codes into decode_palette
            return torch.ops.vllm.marlin_tiled_decode6.default(
                x, decode_weight, decode_scales, decode_palette, weight_global_scale, workspace,
                size_n, size_k, decode_fixed_grid,
            )
        return torch.ops.vllm.marlin_tiled_decode.default(
            x, decode_weight, decode_scales, weight_global_scale, workspace, size_n, size_k
        )
    return apply_fp4_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        weight_global_scale=weight_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
    )


def _marlin_fp4_rows_into(
    a: torch.Tensor,
    c: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> None:
    ops.marlin_gemm(
        a=a,
        c=c,
        b_q_weight=weight,
        b_bias=None,
        b_scales=weight_scale,
        a_scales=None,
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=a.shape[0],
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=False,
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
    )


def _marlin_fp4_wide_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor | None:
    if marlinw is None or x.shape[0] < marlinw.MIN_M:
        return None

    def rows(a: torch.Tensor, c: torch.Tensor) -> None:
        _marlin_fp4_rows_into(
            a, c, weight, weight_scale, weight_global_scale, workspace, size_n, size_k
        )

    return marlinw.gemm(
        x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k, rows
    )


def _marlin_fp4_wide(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    use_narrow: bool = False,
    down_work: torch.Tensor | None = None,
    down_work_ptr: torch.Tensor | None = None,
    down_slots: int = 0,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    _log_prompt_stats()
    args = (x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k)
    if use_narrow and marlin_narrow is not None:
        out = marlin_narrow.gemm(*args)
        if out is not None:
            return out
    if marlinw_down is not None and down_work is not None and down_work_ptr is not None:
        if muse_final_slab is not None and x.shape[0] == muse_final_slab.ROWS:
            out = muse_final_slab.down_if_active(x, weight, weight_scale, weight_global_scale)
            if out is not None:
                _count_mlp_route("slab")
                return out
        if marlinw_down128 is not None and weight.data_ptr() in _down128_weights:
            out = marlinw_down128.gemm_auto(x, weight, weight_scale, weight_global_scale)
            if out is not None:
                _count_mlp_route("down128")
                return out
        out = marlinw_down.gemm(
            x, weight, weight_scale, weight_global_scale, down_work, down_work_ptr, down_slots
        )
        if out is not None:
            _count_mlp_route("down64")
            return out
    out = _marlin_fp4_wide_gemm(*args)
    return out if out is not None else _marlin_fp4_single(
        *args, decode_weight, decode_scales, decode_palette, decode_fixed_grid
    )


def _marlin_fp4_wide_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    use_narrow: bool = False,
    down_work: torch.Tensor | None = None,
    down_work_ptr: torch.Tensor | None = None,
    down_slots: int = 0,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


def _marlin_fp4_single_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(
    op_name="marlin_fp4_single_or_tiled",
    op_func=_marlin_fp4_single,
    fake_impl=_marlin_fp4_single_fake,
)

direct_register_custom_op(
    op_name="marlin_fp4_wide",
    op_func=_marlin_fp4_wide,
    fake_impl=_marlin_fp4_wide_fake,
)


def _marlin_fp4_narrow_attn(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    attn_work: torch.Tensor | None = None,
    attn_work_ptr: torch.Tensor | None = None,
    attn_slots: int = 0,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    """An admitted NVFP4 attention projection: the narrow plan for exactly
    512 rows, else (decode and every other row count) the same call as
    marlin_fp4_single_or_tiled."""
    if (
        marlinw_attn is not None
        and attn_work is not None
        and attn_work_ptr is not None
        and x.shape[0] == marlinw_attn.ROWS
    ):
        _log_prompt_stats()
        if marlinw_down128 is not None and weight.data_ptr() in _attn128_weights:
            out = marlinw_down128.gemm_shape(
                x, weight, weight_scale, weight_global_scale, size_n, size_k
            )
            if out is not None:
                return out
        if muse_final_slab is not None and x.shape[0] == muse_final_slab.ROWS:
            out = muse_final_slab.attn_if_active(
                x, weight, weight_scale, weight_global_scale, size_n, size_k
            )
            if out is not None:
                return out
        out = marlinw_attn.gemm(
            x, weight, weight_scale, weight_global_scale, attn_work, attn_work_ptr, attn_slots
        )
        kind = marlinw_attn.SHAPES.get((size_n, size_k), "?")
        if out is not None:
            if kind not in _attn_route_logged:
                _attn_route_logged.add(kind)
                logger.info("Narrow NVFP4 attention prefill GEMM in use for %s (%dx%d)",
                            kind, size_n, size_k)
            return out
        logger.warning_once(
            "Narrow NVFP4 attention prefill GEMM declined a %s input (shape %s, strides %s, "
            "dtype %s); using ops.marlin_gemm",
            kind, tuple(x.shape), tuple(x.stride()), x.dtype,
        )
    return _marlin_fp4_single(
        x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k,
        decode_weight, decode_scales, decode_palette, decode_fixed_grid,
    )


def _marlin_fp4_narrow_attn_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    attn_work: torch.Tensor | None = None,
    attn_work_ptr: torch.Tensor | None = None,
    attn_slots: int = 0,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(
    op_name="marlin_fp4_narrow_attn",
    op_func=_marlin_fp4_narrow_attn,
    fake_impl=_marlin_fp4_narrow_attn_fake,
)


def _marlin_fp4_swiglu(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    if muse_final_slab is not None and x.shape[0] == muse_final_slab.ROWS:
        out = muse_final_slab.gate_if_active(x, weight)
        if out is not None:
            _count_mlp_route("gate_slab")
            return out
    uses_128 = getattr(marlin_swiglu, "uses_128", None)
    _count_mlp_route("gate_128" if uses_128 is not None and uses_128(weight) else "gate_64")
    return marlin_swiglu.gemm(x, weight, scales, global_scale)


def _marlin_fp4_swiglu_fake(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], scales.shape[-1] // 2))


direct_register_custom_op(
    op_name="marlin_fp4_swiglu",
    op_func=_marlin_fp4_swiglu,
    fake_impl=_marlin_fp4_swiglu_fake,
)


def _marlin_fp4_swiglu_fp8(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    return marlin_swiglu_fp8.gemm(x, weight, scales, global_scale, down_scale)


def _marlin_fp4_swiglu_fp8_fake(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], scales.shape[-1] // 2), dtype=torch.float8_e4m3fn)


direct_register_custom_op(
    op_name="marlin_fp4_swiglu_fp8",
    op_func=_marlin_fp4_swiglu_fp8,
    fake_impl=_marlin_fp4_swiglu_fp8_fake,
)


def _marlin_launch_chunks(m: int, n: int) -> list[int]:
    max_par = 16 * 8 if n <= 4096 else 16
    rest = m // 64
    chunks = []
    while rest:
        chunks.append(min(rest, max_par))
        rest -= chunks[-1]
    return chunks


def _marlin_whole_tiles(m: int, n: int, sms: int) -> bool:
    return (
        m > 0
        and m % 64 == 0
        and n % _COL_TILE_N == 0
        and all(
            par * (n // _COL_TILE_N) % sms == 0
            for par in _marlin_launch_chunks(m, n)
        )
    )


def _marlin_fp4_col_sliced_rows(m: int, size_n: int, slice_n: int, sms: int) -> bool:
    """Whether an m-row call takes the column-sliced schedule."""
    return _marlin_whole_tiles(m, size_n, sms) and _marlin_whole_tiles(
        m, slice_n, sms
    )


def _marlin_fp4_col_sliced_gemm(
    x: torch.Tensor,
    weight_slices: torch.Tensor,
    weight_scale_slices: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    num_slices, _, slice_n = weight_scale_slices.shape
    m = x.shape[0]
    out = torch.empty((m, size_n), dtype=x.dtype, device=x.device)
    part = torch.empty((m, slice_n), dtype=x.dtype, device=x.device)
    for i in range(num_slices):
        _marlin_fp4_rows_into(
            x,
            part,
            weight_slices[i],
            weight_scale_slices[i],
            weight_global_scale,
            workspace,
            slice_n,
            size_k,
        )
        out[:, i * slice_n : (i + 1) * slice_n].copy_(part)
    return out


def _marlin_fp4_col_sliced(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_slices: torch.Tensor,
    weight_scale_slices: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    sms: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    _log_prompt_stats()
    if _marlin_fp4_col_sliced_rows(
        x.shape[0], size_n, weight_scale_slices.shape[2], sms
    ):
        return _marlin_fp4_col_sliced_gemm(
            x,
            weight_slices,
            weight_scale_slices,
            weight_global_scale,
            workspace,
            size_n,
            size_k,
        )
    return _marlin_fp4_single(
        x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k,
        decode_weight, decode_scales, decode_palette, decode_fixed_grid,
    )


def _marlin_fp4_col_sliced_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_slices: torch.Tensor,
    weight_scale_slices: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    sms: int,
    decode_weight: torch.Tensor | None = None,
    decode_scales: torch.Tensor | None = None,
    decode_palette: torch.Tensor | None = None,
    decode_fixed_grid: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(
    op_name="marlin_fp4_col_sliced",
    op_func=_marlin_fp4_col_sliced,
    fake_impl=_marlin_fp4_col_sliced_fake,
)


def _prompt_check_input(m: int, size_k: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, size_k, generator=gen, device=gen.device)
    # Large and cancelling terms make a change of summation order show.
    x[:, ::3] *= 64
    x[1::2, 1::2] = -x[1::2, ::2]
    return x.to(torch.bfloat16)


def _prompt_check(layer: torch.nn.Module, fn, ms) -> bool:
    """Compares fn(x) with the single ops.marlin_gemm call bitwise (as int16)
    on this layer's weights; any difference or error returns False."""
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(size_n * 31 + size_k)
    try:
        for m in ms:
            x = _prompt_check_input(m, size_k, gen)
            got = fn(x)
            ref = _marlin_fp4_single(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                size_n,
                size_k,
            )
            if got is None or not torch.equal(
                got.view(torch.int16), ref.view(torch.int16)
            ):
                return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Marlin NVFP4 prompt GEMM self-check failed: %s", e)
        return False


def _prepare_narrow_attn(
    layer: torch.nn.Module, global_scale: torch.Tensor, size_n: int, size_k: int, sms: int
) -> None:
    """Admits the narrow 512-row plan (harness/kernels/marlinw_attn) for an
    L2-sized NVFP4 attention projection after a bitwise comparison with the
    single ops.marlin_gemm call on the layer's own weights; any difference or
    error keeps the stock call."""
    if marlinw_attn is None or not marlinw_attn.ENABLED:
        return
    kind = marlinw_attn.SHAPES.get((size_n, size_k))
    if (
        kind is None
        or sms != marlinw_attn.SMS
        or torch.cuda.get_device_capability(layer.weight.device) != (12, 1)
    ):
        return
    weight, weight_scale = layer.weight, layer.weight_scale
    try:
        plan = marlinw_attn.prepare(weight.device, size_n, size_k)
    except (RuntimeError, ValueError) as exc:
        logger.warning_once("Narrow NVFP4 attention prefill GEMM unavailable: %s", exc)
        _prompt_stats["attn"][kind][1] += 1
        return
    buffers = (plan.work, plan.work_ptr, plan.scratch_slots)
    if _prompt_check(
        layer,
        lambda x: marlinw_attn.gemm(x, weight, weight_scale, global_scale, *buffers),
        (marlinw_attn.ROWS,),
    ):
        layer.marlin_prompt_path = ("narrow_attn", *buffers)
        _prompt_stats["attn"][kind][0] += 1
        if (
            marlinw_down128 is not None
            and getattr(marlinw_down128, "ATTN_ENABLED", False)
            and (size_n, size_k) in getattr(marlinw_down128, "ATTN_SHAPES", {})
            and _prompt_check(
                layer,
                lambda x: marlinw_down128.gemm_shape(
                    x, weight, weight_scale, global_scale, size_n, size_k
                ),
                (marlinw_attn.ROWS,),
            )
        ):
            _attn128_weights.add(weight.data_ptr())
    else:
        _prompt_stats["attn"][kind][1] += 1


def _prepare_prompt_gemm(layer: torch.nn.Module) -> None:
    """Picks the prompt-sized schedule of a layer: marlinw, column slices or
    the single call (see the comment at the top of this file)."""
    layer.marlin_prompt_path = None
    if os.environ.get("VLLM_MARLIN_NVFP4_PROMPT_GEMM", "1") == "0":
        return
    weight, weight_scale = layer.weight, layer.weight_scale
    global_scale = getattr(layer, "weight_global_scale", None)
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    if (
        not USE_FP32_REDUCE_DEFAULT
        or global_scale is None
        or getattr(layer, "bias", None) is not None
        or weight.dtype != torch.int32
        or tuple(weight.shape) != (size_k // 16, 2 * size_n)
        or tuple(weight_scale.shape) != (size_k // 16, size_n)
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        return
    props = torch.cuda.get_device_properties(weight.device)
    sms = props.multi_processor_count
    weight_bytes = weight.numel() * 4 + weight_scale.numel()
    if weight_bytes <= 2 * props.L2_cache_size:
        _prepare_narrow_attn(layer, global_scale, size_n, size_k, sms)
    if weight_bytes <= 2 * props.L2_cache_size or should_use_atomic_add_reduce(
        m=_COL_SLICE_REF_M,
        n=size_n,
        k=size_k,
        device=weight.device,
        dtype=torch.bfloat16,
    ):
        return
    args = (weight, weight_scale, global_scale, layer.workspace, size_n, size_k)

    if (
        marlinw is not None
        and marlinw.prepare(layer)
        and _prompt_check(
            layer, lambda x: _marlin_fp4_wide_gemm(x, *args), _PROMPT_CHECK_M
        )
    ):
        use_narrow = (
            marlin_narrow is not None
            and marlin_narrow.ENABLED
            and marlin_narrow.eligible(size_n, size_k, weight.device, props)
            and _prompt_check(
                layer, lambda x: marlin_narrow.gemm(x, *args), (marlin_narrow.ROWS,)
            )
        )
        down_plan = (None, None, 0)
        if (
            marlinw_down is not None
            and marlinw_down.ENABLED
            and (size_n, size_k) == (marlinw_down.COLUMNS, marlinw_down.INNER)
            and sms == marlinw_down.SMS
            and torch.cuda.get_device_capability(weight.device) == (12, 1)
        ):
            try:
                plan = marlinw_down.prepare(weight.device)
            except (RuntimeError, ValueError) as exc:
                logger.warning_once("Narrow split-K Marlin prefill unavailable: %s", exc)
            else:
                buffers = (plan.work, plan.work_ptr, plan.scratch_slots)
                if _prompt_check(
                    layer,
                    lambda x: marlinw_down.gemm(x, weight, weight_scale, global_scale, *buffers),
                    (marlinw_down.ROWS,),
                ):
                    down_plan = buffers
                    _prompt_stats["narrow_down"] += 1
                    if (
                        marlinw_down128 is not None
                        and marlinw_down128.ENABLED
                        and _prompt_check(
                            layer,
                            lambda x: marlinw_down128.gemm_auto(
                                x, weight, weight_scale, global_scale
                            ),
                            (marlinw_down.ROWS,),
                        )
                    ):
                        _down128_weights.add(weight.data_ptr())
                        _prompt_stats["down128"] += 1
                        # The same tiles fed by TMA (Muse Glimmer): admitted per layer by the same comparison
                        # with the single stock call; refused or unavailable -> the cp.async-fed 128x128 kernel.
                        if getattr(marlinw_down128, "TMA_ENABLED", False):
                            marlinw_down128.admit_tma(
                                weight,
                                _prompt_check(
                                    layer,
                                    lambda x: marlinw_down128.gemm_tma(
                                        x, weight, weight_scale, global_scale
                                    ),
                                    (marlinw_down.ROWS,),
                                ),
                            )
                    else:
                        _prompt_stats["down128_rejected"] += 1
        if use_narrow and marlin_swiglu is not None:
            layer.marlin_swiglu_ready = marlin_swiglu.prepare(layer)
        if use_narrow and marlin_swiglu_fp8 is not None:
            layer.marlin_swiglu_fp8_ready = marlin_swiglu_fp8.prepare(layer)
        layer.marlin_prompt_path = ("wide", use_narrow, down_plan)
        _prompt_stats["narrow"] += use_narrow
        _prompt_stats["wide"] += 1
        return

    # Narrowest slice for which a prompt-sized call fills each SM with whole
    # tiles.
    slice_n = next(
        (
            n
            for n in range(_COL_TILE_N, size_n, _COL_TILE_N)
            if size_n % n == 0
            and _marlin_fp4_col_sliced_rows(_COL_SLICE_REF_M, size_n, n, sms)
        ),
        None,
    )
    if slice_n is not None:
        num_slices = size_n // slice_n
        k_rows = weight.shape[0]
        # Marlin's layout is local to 64-column tiles, so a column slice of
        # the prepared tensors is exactly the prepared form of that slice.
        weight_slices = (
            weight.view(k_rows, num_slices, 2 * slice_n).transpose(0, 1).contiguous()
        )
        weight_scale_slices = (
            weight_scale.view(k_rows, num_slices, slice_n)
            .transpose(0, 1)
            .contiguous()
        )
        check_m = [
            m
            for m in _PROMPT_CHECK_M
            if _marlin_fp4_col_sliced_rows(m, size_n, slice_n, sms)
        ]
        if check_m and _prompt_check(
            layer,
            lambda x: _marlin_fp4_col_sliced_gemm(
                x,
                weight_slices,
                weight_scale_slices,
                global_scale,
                layer.workspace,
                size_n,
                size_k,
            ),
            check_m,
        ):
            layer.marlin_prompt_path = (
                "sliced",
                weight_slices,
                weight_scale_slices,
                sms,
            )
            _prompt_stats["sliced"] += 1
            return
        del weight_slices, weight_scale_slices
    _prompt_stats["single"] += 1


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        prepare_fp4_layer_for_marlin(layer)
        _prepare_prompt_gemm(layer)
        # Loading replacement weights must also replace their private scale codes.
        layer._buffers.pop("_marlin_decode_scales6", None)
        layer._buffers.pop("_marlin_decode_palette", None)
        if prepare_decode_layout is not None and USE_FP32_REDUCE_DEFAULT:
            prepare_decode_layout(layer)
            if (
                prepare_decode_layout6 is not None
                and os.environ.get("MUSE_SCALE6", "1") != "0"
                and getattr(layer, "_marlin_decode_weight", None) is not None
            ):
                # 6-bit codes of the tiled block scales; admitted per layer after a
                # bitwise comparison with the uncompressed tiled decode GEMV
                from muse_marlin_gemv import decode_linear as _tiled_decode

                try:
                    prepare_decode_layout6(layer, _tiled_decode)
                except (RuntimeError, ValueError) as exc:
                    layer._buffers.pop("_marlin_decode_scales6", None)
                    layer._buffers.pop("_marlin_decode_palette", None)
                    logger.warning_once("NVFP4 scale palette admission failed; retaining tiled decode: %s", exc)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        path = getattr(layer, "marlin_prompt_path", None)
        decode_weight = getattr(layer, "_marlin_decode_weight", None)
        if bias is None and (path is not None or decode_weight is not None):
            size_n = layer.output_size_per_partition
            args = (x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale)
            tail = (layer.weight_global_scale, layer.workspace, size_n, layer.input_size_per_partition)
            palette = getattr(layer, "_marlin_decode_palette", None)
            scales = getattr(layer, "_marlin_decode_scales6", None) if palette is not None else None
            if scales is None:
                palette = None
                scales = getattr(layer, "_marlin_decode_scales", None)
            decode = (decode_weight, scales)
            fixed_grid = palette is not None and getattr(layer, "marlin_decode6_fixed_grid", False)
            if path is None:
                out = torch.ops.vllm.marlin_fp4_single_or_tiled(*args, *tail, *decode, palette, fixed_grid)
            elif path[0] == "wide":
                out = torch.ops.vllm.marlin_fp4_wide(*args, *tail, *decode, path[1], *path[2], palette, fixed_grid)
            elif path[0] == "narrow_attn":
                out = torch.ops.vllm.marlin_fp4_narrow_attn(
                    *args, *tail, *decode, *path[1:], palette, fixed_grid
                )
            else:
                out = torch.ops.vllm.marlin_fp4_col_sliced(
                    *args, path[1], path[2], *tail, path[3], *decode, palette, fixed_grid
                )
            return out.reshape(x.shape[:-1] + (size_n,))
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
