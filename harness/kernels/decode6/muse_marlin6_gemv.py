# SPDX-License-Identifier: Apache-2.0
"""Decode (M == 1) NVFP4 GEMV with 6-bit block scales: the board's tiled Marlin GEMV (muse_marlin_gemv) reading each
layer's group-16 FP8 block scales as 6-bit codes into a 64-entry palette, 3/4 of the scale bytes. The decoded bytes are
the same bytes, in the same shared-memory fragment, so the kernel computes the same bits; every layer is admitted only
after a bitwise comparison with the uncompressed tiled GEMV on its own weight."""
import ctypes
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
STATS = {"admitted": 0, "kept": 0, "logged": False}

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin6.so")))
_gemv6 = _lib.muse_marlin6_gemv
_gemv6.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv6.restype = ctypes.c_int


def pack_scales6(tiled_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """[n/tile_n, k/16, tile_n] one-byte scales (any 1-byte dtype) -> (codes uint8 [3/4 bytes], palette uint8 [64]),
    or None when the layer has more than 64 distinct scale bytes. The code of byte b of the flat tensor sits at bits
    [6b, 6b + 6) of the code stream, least significant bit first: four codes per three bytes."""
    flat = tiled_scales.contiguous().view(torch.uint8).reshape(-1)
    if flat.numel() % 4:
        return None
    values = torch.unique(flat)
    if values.numel() > 64:
        return None
    palette = torch.zeros(64, dtype=torch.uint8, device=flat.device)
    palette[: values.numel()] = values
    lut = torch.zeros(256, dtype=torch.uint8, device=flat.device)
    lut[values.long()] = torch.arange(values.numel(), dtype=torch.uint8, device=flat.device)
    codes = lut[flat.long()].view(-1, 4).to(torch.int32)
    c0, c1, c2, c3 = codes.unbind(1)
    packed = torch.stack(
        (c0 | ((c1 & 3) << 6), (c1 >> 2) | ((c2 & 15) << 4), (c2 >> 4) | (c3 << 2)), dim=1
    ).to(torch.uint8).reshape(-1)
    # round trip on the device before anyone relies on it
    q = packed.view(-1, 3).to(torch.int32)
    d = torch.stack((q[:, 0] & 63, (q[:, 0] >> 6) | ((q[:, 1] & 15) << 2), (q[:, 1] >> 4) | ((q[:, 2] & 3) << 4), q[:, 2] >> 2), 1)
    if not torch.equal(palette[d.reshape(-1).long()], flat):
        return None
    return packed, palette


def decode_linear6(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    if not STATS["logged"]:
        STATS["logged"] = True
        logger.info("NVFP4 decode GEMV with 6-bit block scales: %d layers, %d kept uncompressed",
                    STATS["admitted"], STATS["kept"])
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _gemv6(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (scale6) launch failed with CUDA error {status}")
    return output


def _decode6_fake(x, tiled_weight, scales6, palette, global_scale, workspace, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode6", decode_linear6, fake_impl=_decode6_fake)


def self_check6(layer: torch.nn.Module, reference) -> bool:
    """decode_linear6 == reference (the uncompressed tiled GEMV) bitwise on the layer's own weight, for activations with
    large and cancelling entries at several magnitudes."""
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    dev = layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 7 + k)
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        x = x.to(torch.bfloat16)
        got = decode_linear6(x, layer._marlin_decode_weight, layer._marlin_decode_scales6, layer._marlin_decode_palette,
                             layer.weight_global_scale, layer.workspace, n, k)
        ref = reference(x, layer._marlin_decode_weight, layer._marlin_decode_scales, layer.weight_global_scale,
                        layer.workspace, n, k)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    return True


def prepare_decode_layout6(layer: torch.nn.Module, reference) -> bool:
    """After the board's prepare_decode_layout: add the 6-bit scale codes and palette, check, register. True if admitted."""
    tiled = getattr(layer, "_marlin_decode_scales", None)
    if tiled is None or getattr(layer, "_marlin_decode_weight", None) is None:
        return False
    k = layer.input_size_per_partition
    packed = pack_scales6(tiled) if k % 128 == 0 else None
    if packed is None:
        STATS["kept"] += 1
        return False
    codes, palette = packed
    layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)
    layer.register_buffer("_marlin_decode_palette", palette, persistent=False)
    if not self_check6(layer, reference):
        del layer._marlin_decode_scales6, layer._marlin_decode_palette
        STATS["kept"] += 1
        return False
    STATS["admitted"] += 1
    return True
