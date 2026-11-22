# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import weakref

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
    import marlin_narrow
except (ImportError, OSError):
    marlin_narrow = None

try:
    import marlinw_down
except (ImportError, OSError):
    marlinw_down = None

try:
    import muse_marlin_gemv
    from muse_marlin_gemv import prepare_decode_layout
except (ImportError, OSError):
    muse_marlin_gemv = None
    prepare_decode_layout = None

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
    "wide": 0, "narrow": 0, "narrow_down": 0, "sliced": 0, "single": 0, "logged": False
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
) -> torch.Tensor:
    if (
        x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and decode_weight is not None
        and decode_scales is not None
    ):
        _log_marlint_stats()
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
    return out if out is not None else _marlin_fp4_single(*args, decode_weight, decode_scales)


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
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(
    op_name="marlin_fp4_single_or_tiled",
    op_func=_marlin_fp4_single,
    fake_impl=_marlin_fp4_wide_fake,
)

direct_register_custom_op(
    op_name="marlin_fp4_wide",
    op_func=_marlin_fp4_wide,
    fake_impl=_marlin_fp4_wide_fake,
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
        decode_weight, decode_scales,
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


# SiLU-and-mul of the gated MLP fused into the gate_up GEMM.
#
# The model's compiled graph computes act = silu(gate) * up on the [M, 2D]
# bf16 output of gate_up in one Inductor (Triton) kernel: fp32 math on the bf16
# values, silu(g) = g / (exp(-g) + 1) without rounding to bf16 in between, one
# round to bf16 at the end. marlinw.silu_mul and marlinw's fused epilogue run
# the same instruction chain (harness/kernels/marlinw/csrc/silu_mul.cuh) on the
# same bf16 gate and up values that ops.marlin_gemm writes, so the output is the
# same bits: the gate tile is rounded to bf16 by the unchanged Marlin epilogue,
# stored, and read back by the up tile's epilogue. For a layer admitted here
# the model calls marlin_fp4_silu_mul instead of gate_up_proj + act_fn; calls
# without a fused plan (fewer than 128 rows, rows not a multiple of 256) run the
# layer's GEMM path followed by marlinw.silu_mul.
_silu_stats = {"fused": 0, "rejected": 0, "logged": False, "ref_ok": None}
_SILU_CHECK_M = (512, 4096, 640)


_silu_layers: list = []


def _log_silu_stats() -> None:
    if not _silu_stats["logged"]:
        _silu_stats["logged"] = True
        # The model fuses only ahead of a down_proj that reads bf16 (an FP8
        # down_proj keeps the compiled SiLU-and-mul + quantization glue).
        fused = glue = 0
        for ref in _silu_layers:
            layer = ref()
            consumer = getattr(layer, "silu_mul_consumer", None) if layer else None
            down = consumer() if consumer is not None else None
            if down is not None and getattr(down, "marlin_bf16_input", False):
                fused += 1
            else:
                glue += 1
        logger.info(
            "Marlin NVFP4 gate_up with fused SiLU-and-mul: %d layers fused, %d on "
            "the stock glue (down_proj not bf16-input), %d rejected",
            fused,
            glue,
            _silu_stats["rejected"],
        )


def _silu_mul_impl(
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
) -> torch.Tensor:
    # Narrow gate/up has its own per-layer stock-bitwise admission. Its output
    # feeds the separately validated standalone activation below. The wide
    # fused epilogue remains preferable when the narrow path does not apply.
    y = marlin_narrow.gemm(
        x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k
    ) if use_narrow and marlin_narrow is not None else None
    if y is not None:
        return marlinw.silu_mul(y)
    out = marlinw.gemm_silu_mul(
        x, weight, weight_scale, weight_global_scale, size_n, size_k
    )
    if out is not None:
        return out
    args = (x, weight, weight_scale, weight_global_scale, workspace, size_n, size_k)
    y = _marlin_fp4_wide_gemm(*args)
    return marlinw.silu_mul(
        y if y is not None else _marlin_fp4_single(*args, decode_weight, decode_scales)
    )


def _marlin_fp4_silu_mul(
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
) -> torch.Tensor:
    _log_prompt_stats()
    _log_silu_stats()
    return _silu_mul_impl(
        x,
        weight,
        weight_scale,
        weight_global_scale,
        workspace,
        size_n,
        size_k,
        decode_weight,
        decode_scales,
        use_narrow,
    )


def _marlin_fp4_silu_mul_fake(
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
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n // 2))


direct_register_custom_op(
    op_name="marlin_fp4_silu_mul",
    op_func=_marlin_fp4_silu_mul,
    fake_impl=_marlin_fp4_silu_mul_fake,
)


def marlin_fp4_silu_mul(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up of a merged [gate | up] layer admitted at load
    (layer.marlin_silu_mul)."""
    size_n = layer.output_size_per_partition
    out = torch.ops.vllm.marlin_fp4_silu_mul(
        x.reshape(-1, x.shape[-1]),
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        size_n,
        layer.input_size_per_partition,
        getattr(layer, "_marlin_decode_weight", None),
        getattr(layer, "_marlin_decode_scales", None),
        bool(getattr(layer, "marlin_prompt_path", (None, False))[1]),
    )
    return out.reshape(x.shape[:-1] + (size_n // 2,))


def _silu_mul_reference(y: torch.Tensor) -> torch.Tensor:
    """The compiled graph's SiLU-and-mul kernel body (TorchInductor's Triton
    code for SiluAndMul.forward_native on bf16), launched directly."""
    import triton
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

    m, d = y.shape[0], y.shape[1] // 2
    out = torch.empty((m, d), dtype=torch.bfloat16, device=y.device)
    _kernel[(triton.cdiv(m * d, 512),)](y, out, m * d, D=d, XBLOCK=512, num_warps=8)
    return out


def _silu_reference_check(device: torch.device) -> bool:
    """marlinw.silu_mul against the reference on every bf16 gate value (and
    random, tiny, huge and special up values); once per process."""
    if _silu_stats["ref_ok"] is None:
        try:
            gate = torch.arange(-32768, 32768, dtype=torch.int32, device=device)
            gate = gate.to(torch.int16).view(torch.bfloat16)
            gen = torch.Generator(device=device)
            gen.manual_seed(1)
            rows = 48
            up = torch.randn(rows, gate.numel(), generator=gen, device=device)
            up[1::4] *= 1e-30
            up[2::4] *= 1e30
            up[3::8] = torch.tensor(
                [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 3.0],
                device=device,
            ).repeat(gate.numel() // 8)
            y = torch.cat([gate.expand(rows, -1), up.to(torch.bfloat16)], 1).contiguous()
            got = marlinw.silu_mul(y)
            ref = _silu_mul_reference(y)
            _silu_stats["ref_ok"] = bool(
                torch.equal(got.view(torch.int16), ref.view(torch.int16))
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("SiLU-and-mul reference check failed: %s", e)
            _silu_stats["ref_ok"] = False
    return _silu_stats["ref_ok"]


def _prepare_silu_mul(layer: torch.nn.Module) -> None:
    """Admits a merged [gate | up] layer on the wide schedule to the fused
    SiLU-and-mul op after a bitwise comparison with the single stock call
    followed by marlinw.silu_mul (itself checked against the reference)."""
    layer.marlin_silu_mul = False
    if os.environ.get("VLLM_MARLIN_NVFP4_SILU_MUL", "1") == "0":
        return
    path = getattr(layer, "marlin_prompt_path", None)
    sizes = getattr(layer, "output_partition_sizes", None)
    if (
        path is None
        or path[0] != "wide"
        or sizes is None
        or len(sizes) != 2
        or sizes[0] != sizes[1]
        or sizes[0] % 256
    ):
        return
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    ok = _silu_reference_check(layer.weight.device)
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(size_n * 37 + size_k)
    try:
        for m in _SILU_CHECK_M if ok else ():
            x = _prompt_check_input(m, size_k, gen)
            got = _silu_mul_impl(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                size_n,
                size_k,
            )
            ref = marlinw.silu_mul(
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
                ok = False
                break
    except Exception as e:  # noqa: BLE001
        logger.warning("Marlin NVFP4 fused SiLU-and-mul self-check failed: %s", e)
        ok = False
    if ok:
        layer.marlin_silu_mul = True
        _silu_stats["fused"] += 1
        _silu_layers.append(weakref.ref(layer))
    else:
        _silu_stats["rejected"] += 1


# One-row calls on the tile-major copy run harness/kernels/marlint when a
# layer's outputs equal ops.marlin_gemm bitwise at load: the same Marlin
# template, template arguments and tile-major addresses as the M=1 tiled
# kernel, with reduction scratch that persists across calls.
_marlint_stats = {"marlint": 0, "tiled": 0, "logged": False}


def _log_marlint_stats() -> None:
    if not _marlint_stats["logged"]:
        _marlint_stats["logged"] = True
        logger.info(
            "Marlin NVFP4 M=1 tile-major decode: %d layers on marlint, %d on the "
            "tiled kernel",
            _marlint_stats["marlint"],
            _marlint_stats["tiled"],
        )


def _admit_marlint(layer: torch.nn.Module) -> None:
    weight_tm = getattr(layer, "_marlin_decode_weight", None)
    scale_tm = getattr(layer, "_marlin_decode_scales", None)
    if weight_tm is None or scale_tm is None:
        return
    ok = False
    try:
        import marlint

        size_n = layer.output_size_per_partition
        size_k = layer.input_size_per_partition
        tn = marlint.eligible(layer) if os.environ.get("MARLINT", "1") != "0" else None
        if tn is not None and weight_tm.shape[-1] == 2 * tn:
            gen = torch.Generator(device=layer.weight.device)
            gen.manual_seed(size_n * 17 + size_k)
            ok = True
            for i in range(3):
                x = torch.randn(1, size_k, generator=gen, device=gen.device)
                if i == 1:
                    x[:, ::3] *= 64
                elif i == 2:
                    x *= torch.exp(
                        torch.randn(size_k, generator=gen, device=gen.device) * 2
                    )
                x = x.to(torch.bfloat16)
                got = marlint.gemm(
                    x, weight_tm, scale_tm, layer.weight_global_scale, size_n, size_k
                )
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
                    ok = False
                    break
    except Exception as e:  # noqa: BLE001
        logger.warning("Marlin NVFP4 marlint self-check failed: %s", e)
        ok = False
    if ok:
        muse_marlin_gemv.use_marlint(weight_tm)
        _marlint_stats["marlint"] += 1
    else:
        _marlint_stats["tiled"] += 1


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
        # weight-only: the input is read as the bf16/fp16 tensor it is given
        layer.marlin_bf16_input = True
        _prepare_prompt_gemm(layer)
        if prepare_decode_layout is not None and USE_FP32_REDUCE_DEFAULT:
            prepare_decode_layout(layer)
            _admit_marlint(layer)
        _prepare_silu_mul(layer)

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
            decode = (decode_weight, getattr(layer, "_marlin_decode_scales", None))
            if path is None:
                out = torch.ops.vllm.marlin_fp4_single_or_tiled(*args, *tail, *decode)
            elif path[0] == "wide":
                out = torch.ops.vllm.marlin_fp4_wide(*args, *tail, *decode, path[1], *path[2])
            else:
                out = torch.ops.vllm.marlin_fp4_col_sliced(
                    *args, path[1], path[2], *tail, path[3], *decode
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
