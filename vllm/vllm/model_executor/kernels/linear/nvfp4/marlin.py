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
    "wide": 0, "narrow": 0, "narrow_down": 0, "sliced": 0, "single": 0, "logged": False,
    "swiglu": 0, "swiglu_rejected": 0, "swiglu_fp8": 0, "swiglu_fp8_rejected": 0,
}

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
            logger.info(
                "Lossless NVFP4 scale check: %d of %d palette layers decoded in full equal to the "
                "loaded scales (%d scale bytes, %d tile-major weight bytes compared) and to "
                "ops.marlin_gemm at one row",
                _lossless6["full"], _lossless6["full"] + _lossless6["bad"],
                _lossless6["scale_bytes"], _lossless6["weight_bytes"],
            )
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
        if _prompt_stats["swiglu"] or _prompt_stats["swiglu_rejected"]:
            logger.info(
                "Paired SwiGLU prefill matched the single Marlin call plus the "
                "compiled SiLU-and-mul bitwise on %d layers, %d rejected",
                _prompt_stats["swiglu"],
                _prompt_stats["swiglu_rejected"],
            )
        if _prompt_stats["swiglu_fp8"] or _prompt_stats["swiglu_fp8_rejected"]:
            logger.info(
                "Paired SwiGLU FP8 prefill matched the single Marlin call plus the "
                "compiled SiLU-and-mul FP8 quantization bitwise on %d layers, %d rejected",
                _prompt_stats["swiglu_fp8"],
                _prompt_stats["swiglu_fp8_rejected"],
            )


_decode_routes_seen: set[str] = set()


def _log_decode_route(route: str) -> None:
    # Once per route, at the first one-row call of the running engine (decode
    # warm-up / graph capture): shows which GEMV the decode step records.
    if route not in _decode_routes_seen:
        _decode_routes_seen.add(route)
        logger.info("NVFP4 one-row GEMM in use: tile-major GEMV with %s", route)


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
) -> torch.Tensor:
    if (
        x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and decode_weight is not None
        and decode_scales is not None
    ):
        if decode_palette is not None:
            _log_decode_route("6-bit scale codes expanded inside the GEMV (marlin_tiled_decode6)")
            # decode_scales holds the layer's 6-bit codes into decode_palette
            return torch.ops.vllm.marlin_tiled_decode6.default(
                x, decode_weight, decode_scales, decode_palette, weight_global_scale, workspace, size_n, size_k
            )
        _log_decode_route("8-bit scales (marlin_tiled_decode)")
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
) -> torch.Tensor:
    _log_prompt_stats()
    args = (x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k)
    if use_narrow and marlin_narrow is not None:
        out = marlin_narrow.gemm(*args)
        if out is not None:
            return out
    if marlinw_down is not None and down_work is not None and down_work_ptr is not None:
        out = marlinw_down.gemm(
            x, weight, weight_scale, weight_global_scale, down_work, down_work_ptr, down_slots
        )
        if out is not None:
            return out
    out = _marlin_fp4_wide_gemm(*args)
    return out if out is not None else _marlin_fp4_single(*args, decode_weight, decode_scales, decode_palette)


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


def _marlin_fp4_swiglu(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
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
        decode_weight, decode_scales, decode_palette,
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


_silu_ref_kernel = None


def _silu_mul_reference(y: torch.Tensor) -> torch.Tensor:
    """The compiled graph's SiLU-and-mul kernel body (TorchInductor's Triton
    code for SiluAndMul.forward_native on bf16 ahead of a bf16-input
    down_proj), launched directly: fp32 silu(g) = g / (exp(-g) + 1) times up,
    one bf16 rounding at the store."""
    global _silu_ref_kernel
    import triton

    if _silu_ref_kernel is None:
        import triton.language as tl
        from torch._inductor.runtime.triton_helpers import libdevice

        @triton.jit
        def _kernel(in_ptr0, out_ptr0, xnumel, D: tl.constexpr, XBLOCK: tl.constexpr):
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex % D
            x1 = xindex // D
            tmp0 = tl.load(in_ptr0 + (x0 + 2 * D * x1), xmask).to(tl.float32)
            tmp8 = tl.load(in_ptr0 + (D + x0 + 2 * D * x1), xmask).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = -tmp1
            tmp3 = libdevice.exp(tmp2)
            tmp4 = tl.full([1], 1.0, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tmp1 / tmp5
            tmp7 = tmp6.to(tl.float32)
            tmp9 = tmp7 * tmp8
            tl.store(out_ptr0 + (xindex), tmp9, xmask)

        _silu_ref_kernel = _kernel

    m, d = y.shape[0], y.shape[1] // 2
    out = torch.empty((m, d), dtype=torch.bfloat16, device=y.device)
    _silu_ref_kernel[(triton.cdiv(m * d, 512),)](
        y, out, m * d, D=d, XBLOCK=512, num_warps=8
    )
    return out


def _swiglu_check(layer: torch.nn.Module) -> bool:
    """Load-time check of the paired gate/up SwiGLU (marlinw/swiglu.py, the
    mechanism of Yukon entry 95faadcd) on this layer's weights: its output
    must equal the single ops.marlin_gemm call followed by the compiled
    SiLU-and-mul body bitwise (as int16); otherwise the layer keeps the
    projection and the compiled glue."""
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(size_n * 43 + size_k)
    try:
        for i in range(3):
            if i == 0:
                x = _prompt_check_input(512, size_k, gen)
            else:
                # Smaller activations put gate values in silu's curved range.
                x = torch.randn(512, size_k, generator=gen, device=gen.device)
                x = (x * (0.05 if i == 1 else 0.5)).to(torch.bfloat16)
            got = marlin_swiglu.gemm(
                x, layer.weight, layer.weight_scale, layer.weight_global_scale
            )
            ref = _silu_mul_reference(
                _marlin_fp4_single(
                    x,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    layer.workspace,
                    size_n,
                    size_k,
                )
            )
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Paired SwiGLU prefill self-check failed: %s", e)
        return False


_silu_fp8_ref_kernel = None


def _silu_mul_fp8_reference(y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The compiled graph's SiLU-and-mul plus static FP8 quantization kernel
    body (TorchInductor's Triton code ahead of an FP8 down_proj that takes a
    per-tensor input scale), launched directly: fp32 silu(g) = g / (exp(-g) + 1)
    times up, times 1 / scale, clamped to [-448, 448], one e4m3 rounding at
    the store (no bf16 rounding of the product)."""
    global _silu_fp8_ref_kernel
    import triton

    if _silu_fp8_ref_kernel is None:
        import triton.language as tl
        from torch._inductor.runtime import triton_helpers
        from torch._inductor.runtime.triton_helpers import libdevice

        @triton.jit
        def _kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, D: tl.constexpr, XBLOCK: tl.constexpr):
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex % D
            x1 = xindex // D
            tmp0 = tl.load(in_ptr0 + (x0 + 2 * D * x1), xmask).to(tl.float32)
            tmp8 = tl.load(in_ptr0 + (D + x0 + 2 * D * x1), xmask).to(tl.float32)
            tmp11 = tl.load(in_ptr1 + (0))
            tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
            tmp1 = tmp0.to(tl.float32)
            tmp2 = -tmp1
            tmp3 = libdevice.exp(tmp2)
            tmp4 = tl.full([1], 1.0, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tmp1 / tmp5
            tmp7 = tmp6.to(tl.float32)
            tmp9 = tmp7 * tmp8
            tmp10 = tmp9.to(tl.float32)
            tmp13 = tmp4 / tmp12
            tmp14 = tmp10 * tmp13
            tmp15 = tl.full([1], -448.0, tl.float32)
            tmp16 = triton_helpers.maximum(tmp14, tmp15)
            tmp17 = tl.full([1], 448.0, tl.float32)
            tmp18 = triton_helpers.minimum(tmp16, tmp17)
            tmp19 = tmp18.to(tl.float8e4nv)
            tl.store(out_ptr0 + (xindex), tmp19, xmask)

        _silu_fp8_ref_kernel = _kernel

    m, d = y.shape[0], y.shape[1] // 2
    out = torch.empty((m, d), dtype=torch.float8_e4m3fn, device=y.device)
    _silu_fp8_ref_kernel[(triton.cdiv(m * d, 512),)](
        y, scale, out, m * d, D=d, XBLOCK=512, num_warps=8
    )
    return out


def _swiglu_fp8_check(layer: torch.nn.Module, scale: torch.Tensor) -> bool:
    """Load-time check of the paired gate/up SwiGLU with FP8 output
    (marlinw/swiglu_fp8.py, the mechanism of Yukon entry 67957adf) on this
    layer's weights and the next FP8 linear's own activation scale: its e4m3
    output must equal the single ops.marlin_gemm call followed by the compiled
    SiLU-and-mul FP8 quantization body bitwise; otherwise the layer keeps the
    projection and the compiled glue."""
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(size_n * 47 + size_k)
    try:
        for i in range(4):
            if i == 0:
                x = _prompt_check_input(512, size_k, gen)
            else:
                # Smaller activations put gate values in silu's curved range;
                # larger ones reach the +-448 clamp.
                x = torch.randn(512, size_k, generator=gen, device=gen.device)
                x = (x * (0.05, 0.5, 4.0)[i - 1]).to(torch.bfloat16)
            got = marlin_swiglu_fp8.gemm(
                x, layer.weight, layer.weight_scale, layer.weight_global_scale, scale
            )
            ref = _silu_mul_fp8_reference(
                _marlin_fp4_single(
                    x,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    layer.workspace,
                    size_n,
                    size_k,
                ),
                scale,
            )
            if not torch.equal(got.view(torch.uint8), ref.view(torch.uint8)):
                return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Paired SwiGLU FP8 prefill self-check failed: %s", e)
        return False


# The FP8 variant needs the next linear's activation scale, which belongs to the
# sibling down projection: the two halves meet here by layer prefix, in either
# processing order, and the pair is admitted only after _swiglu_fp8_check.
_swiglu_fp8_gate_up: dict[str, torch.nn.Module] = {}
_swiglu_fp8_down: dict[str, torch.nn.Module] = {}


def _mlp_prefix(layer: torch.nn.Module, leaf: str) -> str | None:
    prefix = getattr(layer, "prefix", "") or ""
    return prefix[: -len(leaf)] if prefix.endswith("." + leaf) else None


def _pair_swiglu_fp8(key: str) -> None:
    gate_up = _swiglu_fp8_gate_up.get(key)
    down = _swiglu_fp8_down.get(key)
    if gate_up is None or down is None:
        return
    del _swiglu_fp8_gate_up[key], _swiglu_fp8_down[key]
    ok = _swiglu_fp8_check(gate_up, down.input_scale)
    _prompt_stats["swiglu_fp8" if ok else "swiglu_fp8_rejected"] += 1
    gate_up.marlin_swiglu_fp8_ready = ok
    down.marlin_swiglu_fp8_input = ok


def offer_swiglu_fp8_down(down: torch.nn.Module) -> None:
    """An FP8 down projection that can take the fused FP8 activation (see
    ModelOptFp8LinearMethod): stays off until its gate_up passes the check."""
    down.marlin_swiglu_fp8_input = False
    key = _mlp_prefix(down, "down_proj")
    if key is not None:
        _swiglu_fp8_down[key] = down
        _pair_swiglu_fp8(key)


def _offer_swiglu_fp8_gate_up(layer: torch.nn.Module) -> None:
    layer.marlin_swiglu_fp8_ready = False
    key = _mlp_prefix(layer, "gate_up_proj")
    if key is not None:
        _swiglu_fp8_gate_up[key] = layer
        _pair_swiglu_fp8(key)


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
        if use_narrow and marlin_swiglu is not None:
            ready = marlin_swiglu.prepare(layer)
            if ready:
                ready = _swiglu_check(layer)
                _prompt_stats["swiglu" if ready else "swiglu_rejected"] += 1
            layer.marlin_swiglu_ready = ready
        if use_narrow and marlin_swiglu_fp8 is not None:
            layer.marlin_swiglu_fp8_ready = False
            if marlin_swiglu_fp8.prepare(layer):
                _offer_swiglu_fp8_gate_up(layer)
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


# Full decode of the 6-bit scale codes of every admitted layer (decode6, the
# representation of Yukon entries cdefbb34 / 67957adf): all codes through the
# palette back to the Marlin scale layout, compared byte for byte with the
# loaded scales (and the tile-major weight with the loaded weight), then the
# one-row GEMV against ops.marlin_gemm itself, not only the tiled kernel.
_lossless6 = {"full": 0, "bad": 0, "scale_bytes": 0, "weight_bytes": 0}


def _decode6_full_check(layer: torch.nn.Module) -> bool:
    from muse_marlin6_gemv import decode_linear6

    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    tile_n = 128 if n % 128 == 0 else 64
    codes = layer._marlin_decode_scales6
    palette = layer._marlin_decode_palette
    q = codes.view(-1, 3).to(torch.int32)
    idx = torch.stack(
        (q[:, 0] & 63, (q[:, 0] >> 6) | ((q[:, 1] & 15) << 2),
         (q[:, 1] >> 4) | ((q[:, 2] & 3) << 4), q[:, 2] >> 2),
        1,
    ).reshape(-1)
    scales = palette[idx.long()].view(n // tile_n, k // 16, tile_n).transpose(0, 1).reshape(k // 16, n)
    weight = layer._marlin_decode_weight.transpose(0, 1).reshape(k // 16, 2 * n)
    if not (
        torch.equal(scales, layer.weight_scale.view(torch.uint8).reshape(k // 16, n))
        and torch.equal(weight.view(torch.int32), layer.weight.view(torch.int32))
    ):
        return False
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(n * 29 + k)
    for i in range(3):
        x = torch.randn(1, k, generator=gen, device=gen.device)
        if i == 1:
            x[:, ::3] *= 64
        elif i == 2:
            x *= torch.exp(torch.randn(k, generator=gen, device=gen.device) * 2)
        x = x.to(torch.bfloat16)
        got = decode_linear6(x, layer._marlin_decode_weight, codes, palette,
                             layer.weight_global_scale, layer.workspace, n, k)
        ref = _marlin_fp4_single(x, layer.weight, layer.weight_scale, layer.weight_global_scale,
                                 layer.workspace, n, k)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    _lossless6["scale_bytes"] += scales.numel()
    _lossless6["weight_bytes"] += weight.numel() * weight.element_size()
    return True


def _admit_decode6(layer: torch.nn.Module, reference) -> None:
    if not prepare_decode_layout6(layer, reference):
        return
    try:
        ok = _decode6_full_check(layer)
    except Exception as e:  # noqa: BLE001
        logger.warning("NVFP4 scale palette full check failed: %s", e)
        ok = False
    _lossless6["full" if ok else "bad"] += 1
    if not ok:
        layer._buffers.pop("_marlin_decode_scales6", None)
        layer._buffers.pop("_marlin_decode_palette", None)
        _scale6_stats["admitted"] -= 1
        _scale6_stats["kept"] += 1


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
                    _admit_decode6(layer, _tiled_decode)
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
            if path is None:
                out = torch.ops.vllm.marlin_fp4_single_or_tiled(*args, *tail, *decode, palette)
            elif path[0] == "wide":
                out = torch.ops.vllm.marlin_fp4_wide(*args, *tail, *decode, path[1], *path[2], palette)
            else:
                out = torch.ops.vllm.marlin_fp4_col_sliced(
                    *args, path[1], path[2], *tail, path[3], *decode, palette
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
