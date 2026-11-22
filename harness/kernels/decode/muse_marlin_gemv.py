# SPDX-License-Identifier: Apache-2.0
import ctypes
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin.so")))
_gemv = _lib.muse_marlin_gemv
_gemv.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv.restype = ctypes.c_int


def prepare_decode_layout(layer: torch.nn.Module) -> None:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if (
        not layer.weight.is_cuda
        or layer.params_dtype != torch.bfloat16
        or layer.weight_global_scale is None
        or k % 128
        or n % 64
        or n < 2048
        or layer.weight.dtype != torch.int32
        or layer.weight_scale.element_size() != 1
        or tuple(layer.weight.shape) != (k // 16, n * 2)
        or tuple(layer.weight_scale.shape) != (k // 16, n)
        or torch.cuda.get_device_capability(layer.weight.device) != (12, 1)
        or torch.cuda.get_device_properties(layer.weight.device).multi_processor_count != 48
    ):
        return
    tile_n = 128 if n % 128 == 0 else 64
    # Marlin has already permuted and scaled these tensors. Move complete
    # output tiles next to their next K stripe without changing any bits.
    for name, original, width in (
        ("_marlin_decode_weight", layer.weight, 2 * tile_n),
        ("_marlin_decode_scales", layer.weight_scale, tile_n),
    ):
        packed = original.reshape(k // 16, n // tile_n, width).transpose(0, 1).contiguous()
        previous = getattr(layer, name, None)
        if previous is None:
            layer.register_buffer(name, packed, persistent=False)
        else:
            previous.copy_(packed)


def decode_linear(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    tiled_scales: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    # Stock's stream-K schedule uses one block per SM and one FP32 tile per lock.
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _gemv(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            tiled_scales.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV launch failed with CUDA error {status}")
    return output


def _decode_fake(x, tiled_weight, tiled_scales, global_scale, workspace, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode", decode_linear, fake_impl=_decode_fake)
