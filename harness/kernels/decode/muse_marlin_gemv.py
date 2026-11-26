# SPDX-License-Identifier: Apache-2.0
import ctypes
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin.so")))
_gemv = _lib.muse_marlin_gemv
_gemv.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv.restype = ctypes.c_int
_gemv_m = _lib.muse_marlin_gemv_m
_gemv_m.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
_gemv_m.restype = ctypes.c_int

_M32_ADMITTED: set[int] = set()
M32_STATS: dict[str, int] = {"admitted": 0, "rejected": 0}


def is_m32_admitted(tiled_scales: torch.Tensor) -> bool:
    return tiled_scales.data_ptr() in _M32_ADMITTED


def verify_decode_layout_m32(layer: torch.nn.Module, ref_fn) -> bool:
    """Bitwise 0-ULP boot check for uncompressed tiled decode across M=1, 16, 32."""
    w = getattr(layer, "_marlin_decode_weight", None)
    s = getattr(layer, "_marlin_decode_scales", None)
    gs = getattr(layer, "weight_global_scale", None)
    ws = getattr(layer, "workspace", None)
    if w is None or s is None or gs is None or ws is None:
        return False
    n, k = int(layer.output_size_per_partition), int(layer.input_size_per_partition)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(n * 37 + k + 7)
    try:
        for m in (1, 7, 8, 16, 32):
            x = torch.randn((m, k), generator=gen, device=w.device, dtype=torch.float32)
            x[:, ::3] *= 64.0
            if m > 1:
                x[1::2, 1::2] = -x[1::2, ::2]
            x = x.to(torch.bfloat16)
            got = decode_linear(x, w, s, gs, ws, n, k)
            if m <= 16:
                ref = ref_fn(x)
            else:
                ref = torch.cat([ref_fn(x[:16]), ref_fn(x[16:])], dim=0)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)) or int(torch.count_nonzero(ws).item()) != 0:
                M32_STATS["rejected"] += 1
                return False
        _M32_ADMITTED.add(s.data_ptr())
        M32_STATS["admitted"] += 1
        return True
    except Exception:  # noqa: BLE001
        M32_STATS["rejected"] += 1
        return False


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
    m = int(x.shape[0])
    if m < 1 or m > 32:
        raise ValueError(f"marlin_tiled_decode supports 1 <= M <= 32, got M={m}")
    output = x.new_empty((*x.shape[:-1], n))
    # Stock's stream-K schedule uses one block per SM and one FP32 tile per lock.
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    tb_m = 32 if m > 16 else 16
    partials = x.new_empty((blocks, tb_m, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _gemv_m(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            tiled_scales.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), m, n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (M={m}) launch failed with CUDA error {status}")
    return output


def _decode_fake(x, tiled_weight, tiled_scales, global_scale, workspace, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode", decode_linear, fake_impl=_decode_fake)
