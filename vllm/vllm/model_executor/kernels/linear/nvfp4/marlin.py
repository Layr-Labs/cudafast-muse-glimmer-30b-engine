# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# Registry of NVFP4 layers handled by custom bit-exact SM121 CUDA kernels:
#  - gate_up (N=39936, K=6656): streaming M=1 decode (exact part1/part2) and
#    column-major L2-resident BM=64, BN=256 prefill (exact zero-split-K when
#    M >= 256 and M % 256 == 0, with zero duplicate weight copies).
#  - down (N=6656, K=19968): chunked-SMEM 96-block M=1 decode (exact iters=169).
#  - lm_head (N=202048, K=6656): streaming M=1 decode (exact part1/part2).
_MUSE_KERNELS_READY = False
try:
    from muse_kernels import load_muse_kernels as _load_muse_kernels

    _MUSE_KERNELS_READY = _load_muse_kernels()
except Exception:
    _MUSE_KERNELS_READY = False

_FAST_GATE_UP_LAYERS: list[torch.nn.Module] = []
_FAST_DOWN_LAYERS: list[torch.nn.Module] = []
_FAST_LMHEAD_LAYERS: list[torch.nn.Module] = []


def _muse_nvfp4_gate_up_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_GATE_UP_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    out_shape = x.shape[:-1] + (39936,)
    if (
        _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
        and m >= 256
        and m % 256 == 0
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        out = torch.ops.muse_fast.nvfp4_prefill_bm64_bn256(
            reshaped_x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            39936,
        )
        return out.reshape(out_shape)

    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=39936,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_gate_up_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (39936,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_down_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_DOWN_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    if (
        reshaped_x.shape[0] == 1
        and _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        out = torch.ops.muse_fast.nvfp4_down_m1(
            reshaped_x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
        )
        return out.reshape(x.shape[:-1] + (6656,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=6656,
        size_k=19968,
        bias=None,
    )


def _muse_nvfp4_down_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (6656,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_lmhead_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_LMHEAD_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    if (
        reshaped_x.shape[0] == 1
        and _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        out = torch.ops.muse_fast.nvfp4_lmhead_m1(
            reshaped_x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
        )
        return out.reshape(x.shape[:-1] + (202048,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=202048,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_lmhead_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (202048,), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="muse_nvfp4_gate_up",
    op_func=_muse_nvfp4_gate_up_impl,
    fake_impl=_muse_nvfp4_gate_up_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_down",
    op_func=_muse_nvfp4_down_impl,
    fake_impl=_muse_nvfp4_down_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_lmhead",
    op_func=_muse_nvfp4_lmhead_impl,
    fake_impl=_muse_nvfp4_lmhead_fake,
)


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
        if not _MUSE_KERNELS_READY or getattr(layer, "bias", None) is not None:
            return
        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        if (
            n == 39936
            and k == 6656
            and layer.weight.shape == (416, 79872)
            and layer.weight_scale.shape == (416, 39936)
        ):
            idx = len(_FAST_GATE_UP_LAYERS)
            _FAST_GATE_UP_LAYERS.append(layer)
            layer._muse_fast_gate_up_id = idx
        elif (
            n == 6656
            and k == 19968
            and layer.weight.shape == (1248, 13312)
            and layer.weight_scale.shape == (1248, 6656)
        ):
            idx = len(_FAST_DOWN_LAYERS)
            _FAST_DOWN_LAYERS.append(layer)
            layer._muse_fast_down_id = idx
        elif (
            n == 202048
            and k == 6656
            and layer.weight.shape == (416, 404096)
            and layer.weight_scale.shape == (416, 202048)
        ):
            idx = len(_FAST_LMHEAD_LAYERS)
            _FAST_LMHEAD_LAYERS.append(layer)
            layer._muse_fast_lmhead_id = idx

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            gu_id = getattr(layer, "_muse_fast_gate_up_id", -1)
            if gu_id >= 0:
                return torch.ops.vllm.muse_nvfp4_gate_up(x, gu_id)
            down_id = getattr(layer, "_muse_fast_down_id", -1)
            if down_id >= 0:
                return torch.ops.vllm.muse_nvfp4_down(x, down_id)
            lmhead_id = getattr(layer, "_muse_fast_lmhead_id", -1)
            if lmhead_id >= 0:
                return torch.ops.vllm.muse_nvfp4_lmhead(x, lmhead_id)
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
