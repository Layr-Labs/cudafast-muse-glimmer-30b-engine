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

# LOCAL PATCH (pfx, harness/kernels/pfx): a Marlin-exact NVFP4 W4A16 GEMM for the
# 512-row prompt pass that streams each weight from DRAM once instead of once per
# 64-row Marlin sub-problem. Bit-identical to stock Marlin on every NVFP4 shape of
# the checkpoint (pfx/tests/pfx_exact.py). Verified layers call the opaque custom
# op below, which runs pfx only for a verified row count (512) and stock Marlin for
# every other call (decode, warm-ups, teacher-forced phases). VLLM_PFX=0, or a
# missing/unloadable library, keeps stock Marlin everywhere.
try:
    import pfx as _pfx
except ImportError:
    _pfx = None


def _pfx_marlin_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    if _pfx is not None:
        x2 = x.reshape(-1, x.shape[-1])
        out = _pfx.maybe_gemm(
            x2, weight, weight_scale, weight_global_scale, size_n, size_k
        )
        if out is not None:
            return out.reshape(x.shape[:-1] + (size_n,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        weight_global_scale=weight_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
        bias=None,
    )


def _pfx_marlin_nvfp4_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return x.new_empty(x.shape[:-1] + (size_n,))


if _pfx is not None:
    direct_register_custom_op(
        op_name="pfx_marlin_nvfp4",
        op_func=_pfx_marlin_nvfp4,
        fake_impl=_pfx_marlin_nvfp4_fake,
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
        if _pfx is not None and _pfx.prepare(layer):
            logger.info_once("pfx: Marlin-exact NVFP4 GEMM enabled for M = 512")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and getattr(layer, "_pfx_ok", False):
            return torch.ops.vllm.pfx_marlin_nvfp4(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
            )
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
