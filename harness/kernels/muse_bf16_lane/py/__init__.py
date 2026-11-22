# SPDX-License-Identifier: Apache-2.0
"""Lossless one-row BF16 attention projections for Muse Glimmer."""

import ctypes
from functools import cache
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op


logger = init_logger("vllm.muse_bf16_lane")
_WIDTH = 6656
_ROWS = (4096, 4608)
_INDEX_BYTES = 36
_stats = dict(admitted=0, rejected=0, bytes=0, logged=False)


@cache
def _lib():
    try:
        native = ctypes.CDLL(str(Path(__file__).with_name("muse_bf16_lane.so")))
    except OSError:
        return None
    ptr = ctypes.c_void_p
    native.muse_bf16_lane_pack.argtypes = [ptr, ctypes.c_int, ptr, ctypes.c_size_t, ptr, ctypes.c_int]
    native.muse_bf16_lane_pack.restype = ctypes.c_size_t
    native.muse_bf16_lane_verify.argtypes = [ptr, ctypes.c_int, ptr, ptr, ctypes.c_size_t]
    native.muse_bf16_lane_verify.restype = ctypes.c_int
    native.muse_bf16_lane_gemv.argtypes = [ctypes.c_int, ctypes.c_int] + [ptr] * 5
    native.muse_bf16_lane_gemv.restype = ctypes.c_int
    return native


def _launch(x, payload, index, n):
    out = x.new_empty((1, n))
    stream = torch.cuda.current_stream(x.device).cuda_stream
    error = _lib().muse_bf16_lane_gemv(
        1, n, x.data_ptr(), payload.data_ptr(), index.data_ptr(),
        out.data_ptr(), stream,
    )
    if error:
        raise RuntimeError(f"muse_bf16_lane launch returned CUDA status {error}")
    return out


def _prepare(weight):
    if (weight.ndim != 2 or weight.shape[0] not in _ROWS or weight.shape[1] != _WIDTH
            or not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or torch.cuda.get_device_capability(weight.device) not in ((12, 0), (12, 1))):
        return None
    native = _lib()
    n, k = weight.shape
    dense = weight.detach().cpu()
    payload = torch.empty(dense.nbytes, dtype=torch.uint8, device="cpu")
    index = torch.empty((n, _INDEX_BYTES), dtype=torch.uint8, device="cpu")
    size = native.muse_bf16_lane_pack(
        dense.data_ptr(), n, payload.data_ptr(), payload.numel(), index.data_ptr(), 1,
    )
    if not 0 < size < dense.nbytes:
        return None
    if not native.muse_bf16_lane_verify(dense.data_ptr(), n, payload.data_ptr(), index.data_ptr(), size):
        return None
    packed = payload[:size].to(weight.device)
    device_index = index.to(weight.device)
    rng = torch.Generator(device=weight.device).manual_seed(720397)
    for magnitude in (0.02, 1.0, 32.0, 1000.0):
        x = torch.randn((1, k), generator=rng, device=weight.device)
        x[:, ::2].mul_(-magnitude)
        x = x.to(torch.bfloat16)
        expected = F.linear(x, weight)
        result = _launch(x, packed, device_index, n)
        if not torch.equal(expected.view(torch.int16), result.view(torch.int16)):
            return None
    return packed, device_index


def _decode(x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor,
            index: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    if (x.shape != (1, _WIDTH) or x.dtype != torch.bfloat16
            or not x.is_cuda or not x.is_contiguous() or x.device != packed.device
            or n not in _ROWS or index.shape != (n, _INDEX_BYTES) or index.device != x.device
            or index.dtype != torch.uint8 or not index.is_contiguous()
            or packed.dtype != torch.uint8 or not packed.is_contiguous() or _lib() is None):
        return F.linear(x, weight)
    if not _stats["logged"]:
        _stats["logged"] = True
        logger.info("muse_bf16_lane decode GEMV: %d admitted, %d rejected, %d packed bytes",
                    _stats["admitted"], _stats["rejected"], _stats["bytes"])
    return _launch(x, packed, index, n)


def _decode_fake(x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor,
                 index: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="muse_bf16_lane_linear",
    op_func=_decode,
    mutates_args=[],
    fake_impl=_decode_fake,
)


def _packed_branch(x, weight, packed, index):
    return torch.ops.vllm.muse_bf16_lane_linear.default(x, weight, packed, index)


def _dense_branch(x, weight, packed, index):
    return F.linear(x, weight)


class _MuseBF16Lane(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        layer.register_buffer("muse_bf16_lane_weight", None, persistent=False)
        layer.register_buffer("muse_bf16_lane_index", None, persistent=False)
        try:
            with torch.no_grad():
                buffers = _prepare(layer.weight)
        except (AttributeError, OSError, RuntimeError, TypeError) as error:
            logger.warning("muse_bf16_lane admission failed for %s (%s)", layer.prefix, error)
            buffers = None
        if buffers is None:
            _stats["rejected"] += 1
            return
        layer.muse_bf16_lane_weight, layer.muse_bf16_lane_index = buffers
        _stats["admitted"] += 1
        _stats["bytes"] += sum(t.nbytes for t in buffers)

    def apply(self, layer, x, bias=None):
        packed = getattr(layer, "muse_bf16_lane_weight", None)
        if (packed is None or bias is not None or x.ndim != 2 or not x.is_cuda
                or x.dtype != torch.bfloat16 or not x.is_contiguous()):
            return super().apply(layer, x, bias)
        args = (x, layer.weight, packed, layer.muse_bf16_lane_index)
        if torch.compiler.is_compiling():
            return torch.cond(x.shape[0] == 1, _packed_branch, _dense_branch, args)
        if x.shape[0] == 1:
            return _packed_branch(*args)
        return super().apply(layer, x, bias)


def enable(layer):
    if (type(layer.quant_method) is UnquantizedLinearMethod
            and not envs.VLLM_BATCH_INVARIANT
            and os.environ.get("MUSE_BF16_LANE", "1") != "0" and _lib() is not None):
        layer.quant_method = _MuseBF16Lane()
