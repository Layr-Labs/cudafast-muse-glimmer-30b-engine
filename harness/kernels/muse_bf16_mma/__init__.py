# SPDX-License-Identifier: Apache-2.0
"""Lossless BF16 gate/up with an exact small-row admission on the owning weight."""
import ctypes as C
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

log = init_logger(__name__)
lib = C.CDLL(str(Path(__file__).with_name("libmuse_bmma.so")))
P, I = C.c_void_p, C.c_int
lib.muse_bmma_prepare.argtypes = [P, P, P, P, I, I, P]
lib.muse_bmma_run.argtypes = [P, P, P, P, P, I, I, I, P]


def _project(x: torch.Tensor, w: torch.Tensor, bits: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    if (x.ndim != 2 or not (2 <= x.shape[0] <= 8) or x.dtype != torch.bfloat16
            or x.device != w.device or not x.is_contiguous() or x.data_ptr() % 4):
        return F.linear(x, w)
    n, k = w.shape
    if x.shape[1] != k:
        return F.linear(x, w)
    with torch.cuda.device(x.device):
        out = x.new_empty((x.shape[0], n))
        rc = lib.muse_bmma_run(x.data_ptr(), w.data_ptr(), bits.data_ptr(), meta.data_ptr(),
                               out.data_ptr(), x.shape[0], n, k, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"BF16 tile launch failed: CUDA {rc}")
        return out


def _fake(x: torch.Tensor, w: torch.Tensor, bits: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w.shape[0]))


direct_register_custom_op(op_name="muse_bmma_gate", op_func=_project, mutates_args=[], fake_impl=_fake)


@torch.no_grad()
def prepare(layer) -> None:
    # Called after the normal loader, only for unquantized, bias-free gate/up.
    w = layer.weight
    n, k = w.shape
    if (w.dtype != torch.bfloat16 or not w.is_cuda or not w.is_contiguous() or w.data_ptr() % 4
            or (n, k) != (39936, 6656) or torch.cuda.get_device_capability(w.device) != (12, 1)):
        return
    with torch.cuda.device(w.device):
        tiles = n * k // 256
        bits = torch.empty((tiles, 96), device=w.device, dtype=torch.int32)
        meta = torch.empty((tiles,), device=w.device, dtype=torch.int32)
        bad = torch.zeros((), device=w.device, dtype=torch.int32)
        rc = lib.muse_bmma_prepare(w.data_ptr(), bits.data_ptr(), meta.data_ptr(), bad.data_ptr(),
                                   n, k, torch.cuda.current_stream().cuda_stream)
        if rc or bad.item():
            log.warning("BF16 tile gate kept dense: pack/round trip failed (CUDA %d)", rc)
            return
        gen = torch.Generator(device=w.device).manual_seed(n + k + 179)
        for m in range(2, 9):
            for cancel in (False, True):
                x = torch.randn((m, k), device=w.device, generator=gen)
                if cancel:
                    x[:, 1::2] = -x[:, ::2]
                    x[:, ::5] *= 32
                x = x.to(torch.bfloat16)
                ref = F.linear(x, w)
                out = _project(x, w, bits, meta)
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    log.warning("BF16 tile gate kept dense: %d-row result differed", m)
                    return
        layer.register_buffer("_bmma_bits", bits, persistent=False)
        layer.register_buffer("_bmma_meta", meta, persistent=False)
        log.info("BF16 tile gate admitted for %s; exact round trip and rows 2..8", layer.prefix)
