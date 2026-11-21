# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

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
_COL_SLICE_CHECK_M = (512, 4096)
_col_slice_stats = {"admitted": 0, "rejected": 0, "logged": False}


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
        ops.marlin_gemm(
            a=x,
            c=part,
            b_q_weight=weight_slices[i],
            b_bias=None,
            b_scales=weight_scale_slices[i],
            a_scales=None,
            global_scale=weight_global_scale,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=m,
            size_n=slice_n,
            size_k=size_k,
            use_atomic_add=False,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
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
) -> torch.Tensor:
    if not _col_slice_stats["logged"]:
        _col_slice_stats["logged"] = True
        logger.info(
            "Marlin NVFP4 column-sliced prompt GEMM: %d layers admitted, "
            "%d rejected",
            _col_slice_stats["admitted"],
            _col_slice_stats["rejected"],
        )
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
    return apply_fp4_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        weight_global_scale=weight_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
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
) -> torch.Tensor:
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(
    op_name="marlin_fp4_col_sliced",
    op_func=_marlin_fp4_col_sliced,
    fake_impl=_marlin_fp4_col_sliced_fake,
)


def _prepare_col_slices(layer: torch.nn.Module) -> None:
    """Adds slice-major copies of the Marlin weight and scales to layers whose
    weight is much larger than L2 and whose prompt-sized calls can be sliced,
    after checking the sliced result against the single call bitwise."""
    layer.marlin_col_slices = None
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
    if slice_n is None:
        return

    num_slices = size_n // slice_n
    k_rows = weight.shape[0]
    # Marlin's layout is local to 64-column tiles, so a column slice of the
    # prepared tensors is exactly the prepared form of that slice.
    weight_slices = (
        weight.view(k_rows, num_slices, 2 * slice_n).transpose(0, 1).contiguous()
    )
    weight_scale_slices = (
        weight_scale.view(k_rows, num_slices, slice_n).transpose(0, 1).contiguous()
    )

    gen = torch.Generator(device=weight.device)
    gen.manual_seed(size_n * 31 + size_k)
    admitted = True
    try:
        for m in _COL_SLICE_CHECK_M:
            if not _marlin_fp4_col_sliced_rows(m, size_n, slice_n, sms):
                continue
            x = torch.randn(m, size_k, generator=gen, device=weight.device)
            # Large and cancelling terms make a change of summation order show.
            x[:, ::3] *= 64
            x[1::2, 1::2] = -x[1::2, ::2]
            x = x.to(torch.bfloat16)
            got = _marlin_fp4_col_sliced_gemm(
                x,
                weight_slices,
                weight_scale_slices,
                global_scale,
                layer.workspace,
                size_n,
                size_k,
            )
            ref = apply_fp4_marlin_linear(
                input=x,
                weight=weight,
                weight_scale=weight_scale,
                weight_global_scale=global_scale,
                workspace=layer.workspace,
                size_n=size_n,
                size_k=size_k,
            )
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                admitted = False
                break
    except Exception as e:  # noqa: BLE001
        logger.warning("Marlin NVFP4 column-sliced self-check failed: %s", e)
        admitted = False

    _col_slice_stats["admitted" if admitted else "rejected"] += 1
    if admitted:
        layer.marlin_col_slices = (weight_slices, weight_scale_slices, sms)


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
        _prepare_col_slices(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        col_slices = getattr(layer, "marlin_col_slices", None)
        if bias is None and col_slices is not None:
            size_n = layer.output_size_per_partition
            out = torch.ops.vllm.marlin_fp4_col_sliced(
                x.reshape(-1, x.shape[-1]),
                layer.weight,
                layer.weight_scale,
                col_slices[0],
                col_slices[1],
                layer.weight_global_scale,
                layer.workspace,
                size_n,
                layer.input_size_per_partition,
                col_slices[2],
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
