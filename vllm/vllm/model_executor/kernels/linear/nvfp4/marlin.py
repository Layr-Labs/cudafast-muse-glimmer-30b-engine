# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import find_spec

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
        n, k = layer.output_size_per_partition, layer.input_size_per_partition
        if (
            current_platform.is_cuda()
            and current_platform.is_device_capability(121)
            and layer.params_dtype == torch.bfloat16
            and layer.weight_global_scale is not None
            and n % 256 == 0
            and k % 128 == 0
            and layer.weight.shape == (k // 16, n * 2)
            and find_spec("glimmer_marlin_prefill") is not None
        ):
            import glimmer_marlin_prefill  # noqa: F401 - registers the CUDA op

            props = torch.cuda.get_device_properties(layer.weight.device)
            layer.marlin_prefill_config = (
                props.multi_processor_count,
                props.shared_memory_per_block_optin,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and hasattr(layer, "marlin_prefill_config"):
            return torch.ops.vllm.gb10_marlin_prefill(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                *layer.marlin_prefill_config,
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
