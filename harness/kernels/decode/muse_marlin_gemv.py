# SPDX-License-Identifier: Apache-2.0
import ctypes
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin.so")))
_gemv = _lib.muse_marlin_gemv
_gemv.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv.restype = ctypes.c_int


_SMS_CACHE: dict[int, int] = {}
_PARTIALS_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SMS_CACHE.get(idx)
    if sms is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        _SMS_CACHE[idx] = sms
    return sms


def _get_partials(device: torch.device, blocks: int, tile_n: int) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, blocks, tile_n)
    buf = _PARTIALS_CACHE.get(key)
    if buf is None:
        buf = torch.empty((blocks, 16, tile_n), dtype=torch.float32, device=device)
        _PARTIALS_CACHE[key] = buf
    return buf


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
    ):
        return
    cap = torch.cuda.get_device_capability(layer.weight.device)
    sms = _get_sms(layer.weight.device)
    if cap == (12, 1) and sms == 48:
        pass
    elif cap == (12, 0) and n == 202048:
        # On 32 GiB SM120 (170 SMs), only tile lm_head in-place (M=1 always ->
        # original Marlin buffers can be released once decode6 admits it).
        pass
    else:
        return
    tile_n = 128 if (n % 128 == 0 and (n // 128) * 4 > sms) else 64
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
    blocks = _get_sms(x.device)
    tile_n = 128 if (n % 128 == 0 and (n // 128) * 4 > blocks) else 64
    partials = _get_partials(x.device, blocks, tile_n)
    with torch.cuda.device(x.device):
        status = _gemv(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            tiled_scales.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"Marlin GEMV launch failed with CUDA error {status}")
    return output


def _decode_fake(x, tiled_weight, tiled_scales, global_scale, workspace, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode", decode_linear, fake_impl=_decode_fake)
