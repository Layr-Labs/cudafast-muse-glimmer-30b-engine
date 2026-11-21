# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


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
        self._decode = None
        if (
            current_platform.is_device_capability(121)
            and layer.params_dtype == torch.bfloat16
            # The larger grid improved these projection shapes in cold-cache
            # measurements. Other shapes retain Marlin's usual tile schedule.
            and (layer.input_size_per_partition, layer.output_size_per_partition)
            in {(6656, 4608), (6656, 4096), (19968, 6656), (6656, 202048)}
        ):
            from muse_marlin import decode

            self._decode = decode

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self._decode is not None
            and x.numel() == layer.input_size_per_partition
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and bias is None
        ):
            output = self._decode(
                x.reshape(1, layer.input_size_per_partition),
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
            )
            return output.reshape(*x.shape[:-1], layer.output_size_per_partition)
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
