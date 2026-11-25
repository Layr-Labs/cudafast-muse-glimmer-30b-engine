# SPDX-License-Identifier: Apache-2.0
"""Lossless 10-bit exponent rANS Tensor Core GEMM for BF16 DFlash drafter layers."""

import ctypes
from functools import cache
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm.muse_bf16_tc_rans")
_TW = 384
_stats = dict(admitted=0, freed_bytes=0, packed_bytes=0, raw_bytes=0)


@cache
def _lib():
    try:
        native = ctypes.CDLL(str(Path(__file__).with_name("muse_bf16_tc_rans.so")))
    except OSError:
        return None
    ptr = ctypes.c_void_p
    c_int = ctypes.c_int
    c_sz = ctypes.c_size_t
    native.muse_bf16_tc_rans_hist.argtypes = [ptr, c_sz, ptr, ptr]
    native.muse_bf16_tc_rans_hist.restype = c_int
    native.muse_bf16_tc_rans_pass1.argtypes = [ptr, c_int, c_int, ptr, ptr, ptr, ptr]
    native.muse_bf16_tc_rans_pass1.restype = c_int
    native.muse_bf16_tc_rans_pass2.argtypes = [ptr, c_int, c_int, c_int, ptr, ptr, ptr, ptr, ptr, ptr]
    native.muse_bf16_tc_rans_pass2.restype = c_int
    native.muse_bf16_tc_rans_decode.argtypes = [ptr, ptr, c_int, c_int, ptr]
    native.muse_bf16_tc_rans_decode.restype = c_int
    native.muse_bf16_tc_rans_gemm.argtypes = [c_int, c_int, c_int, c_sz, ptr, ptr, ptr, ptr, ptr]
    native.muse_bf16_tc_rans_gemm.restype = c_int
    native.muse_bf16_tc_rans_set_pdl.argtypes = [c_int]
    native.muse_bf16_tc_rans_set_pdl.restype = c_int
    return native


def available(device: torch.device | int | None = None) -> bool:
    env = os.environ.get("MUSE_DFLASH_TC_RANS", "1").strip().lower()
    if env in ("0", "", "false", "no") or envs.VLLM_BATCH_INVARIANT or _lib() is None:
        return False
    if env in ("force", "2"):
        return True
    if not torch.cuda.is_available():
        return False
    dev = torch.cuda.current_device() if device is None else device
    return (
        torch.cuda.get_device_capability(dev) == (12, 1)
        and torch.cuda.get_device_properties(dev).multi_processor_count == 48
    )


def set_pdl(mode: int) -> int | None:
    native = _lib()
    if native is None:
        return None
    return int(native.muse_bf16_tc_rans_set_pdl(int(mode)))


def pack(weight: torch.Tensor) -> torch.Tensor | None:
    native = _lib()
    if native is None:
        return None
    if (
        weight.ndim != 2
        or not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        return None
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n <= 0 or k <= 0 or (n % 16) != 0 or (k % 16) != 0:
        return None
    dev = weight.device
    with torch.cuda.device(dev):
        st = torch.cuda.current_stream(dev).cuda_stream
        cnt = torch.zeros(256, dtype=torch.int64, device=dev)
        if native.muse_bf16_tc_rans_hist(weight.data_ptr(), weight.numel(), cnt.data_ptr(), st):
            return None
        ch = cnt.cpu().tolist()
        tot = sum(ch)
        if tot <= 0:
            return None
        f = [max(1, round(c * 1024 / tot)) if c else 0 for c in ch]
        diff = 1024 - sum(f)
        while diff:
            idx_m = max(range(256), key=lambda i: f[i])
            step = diff if diff > 0 else max(diff, -(f[idx_m] - 1))
            if step == 0:
                return None
            f[idx_m] += step
            diff -= step
        cum = [0] * 256
        acc = 0
        for i in range(256):
            cum[i] = acc
            acc += f[i]
        lut = [0] * 1024
        for s in range(256):
            fs = f[s]
            cs = cum[s]
            for j in range(fs):
                lut[cs + j] = s | (fs << 8) | (j << 19)
        fd = torch.tensor(f, dtype=torch.int32, device=dev)
        cd = torch.tensor(cum, dtype=torch.int32, device=dev)
        wb = torch.empty(_TW, dtype=torch.uint32, device=dev)
        if native.muse_bf16_tc_rans_pass1(weight.data_ptr(), n, k, fd.data_ptr(), cd.data_ptr(), wb.data_ptr(), st):
            return None
        wb_h = wb.cpu().tolist()
        woff16 = [0] * (_TW + 1)
        for i in range(_TW):
            woff16[i + 1] = woff16[i] + ((wb_h[i] + 15) // 16)
        G = [(wb_h[i] + 511) // 512 for i in range(_TW)]
        gmin = min(G)
        tab = woff16 + [0] * _TW
        t = gmin * _TW
        for i in range(_TW):
            tab[_TW + 1 + i] = t
            t += G[i] - gmin
        stream_bytes = t * 512
        soff = ((4112 + len(tab) * 4) + 511) & ~511
        total = soff + stream_bytes + 64
        if total >= weight.nbytes:
            return None
        blob = torch.zeros(total, dtype=torch.uint8, device=dev)
        blob[:4096].copy_(torch.tensor(lut, dtype=torch.int32, device=dev).view(torch.uint8))
        blob[4096:4112].copy_(torch.tensor([n, _TW, gmin, soff], dtype=torch.int32, device=dev).view(torch.uint8))
        blob[4112 : 4112 + len(tab) * 4].copy_(torch.tensor(tab, dtype=torch.int32, device=dev).view(torch.uint8))
        bp = blob.data_ptr()
        if native.muse_bf16_tc_rans_pass2(
            weight.data_ptr(), n, k, gmin, fd.data_ptr(), cd.data_ptr(),
            bp + 4112, wb.data_ptr(), bp + soff, st,
        ):
            return None
    return blob


def decode(blob: torch.Tensor, n: int, k: int) -> torch.Tensor:
    native = _lib()
    if native is None:
        raise RuntimeError("muse_bf16_tc_rans native library unavailable")
    out = torch.empty((n, k), dtype=torch.bfloat16, device=blob.device)
    with torch.cuda.device(blob.device):
        st = torch.cuda.current_stream(blob.device).cuda_stream
        err = native.muse_bf16_tc_rans_decode(blob.data_ptr(), out.data_ptr(), int(n), int(k), st)
    if err:
        raise RuntimeError(f"muse_bf16_tc_rans_decode failed with CUDA error {err}")
    return out


def gemm(
    x: torch.Tensor,
    blob: torch.Tensor,
    n: int,
    k: int,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    native = _lib()
    if native is None:
        raise RuntimeError("muse_bf16_tc_rans native library unavailable")
    if not x.is_contiguous():
        x = x.contiguous()
    if x.data_ptr() & 3:
        x = x.clone()
    m = int(x.shape[0])
    if out is None:
        out = x.new_empty((m, n), dtype=torch.bfloat16)
    if m == 0:
        return out
    bias_ptr = 0
    if bias is not None:
        if not bias.is_contiguous() or bias.dtype != torch.bfloat16:
            bias = bias.to(dtype=torch.bfloat16).contiguous()
        bias_ptr = bias.data_ptr()
    with torch.cuda.device(x.device):
        st = torch.cuda.current_stream(x.device).cuda_stream
        err = native.muse_bf16_tc_rans_gemm(
            m, int(n), int(k), int(blob.numel()),
            x.data_ptr(), blob.data_ptr(), bias_ptr, out.data_ptr(), st,
        )
    if err:
        raise RuntimeError(f"muse_bf16_tc_rans_gemm failed with CUDA error {err}")
    return out


def _tc_rans_linear(
    x: torch.Tensor,
    blob: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
) -> torch.Tensor:
    return gemm(x, blob, n, k, bias=bias)


def _tc_rans_linear_fake(
    x: torch.Tensor,
    blob: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="muse_bf16_tc_rans_linear",
    op_func=_tc_rans_linear,
    mutates_args=[],
    fake_impl=_tc_rans_linear_fake,
)


class _MuseBF16TcRansMethod(UnquantizedLinearMethod):
    def __init__(self, free_raw_weight: bool) -> None:
        super().__init__()
        self.free_raw_weight = free_raw_weight

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        layer._already_called_process_weights_after_loading = True
        super().process_weights_after_loading(layer)
        layer.register_buffer("_tc_rans_blob", None, persistent=False)
        layer._tc_rans_n = 0
        layer._tc_rans_k = 0
        w = getattr(layer, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 2 or not w.is_cuda or w.dtype != torch.bfloat16:
            return
        w_c = w.contiguous()
        n, k = int(w_c.shape[0]), int(w_c.shape[1])
        blob = pack(w_c)
        if blob is None:
            return
        # Bitwise weight decode self-check before possibly freeing raw weights
        rec = decode(blob, n, k)
        if not torch.equal(rec.view(torch.int16), w_c.view(torch.int16)):
            logger.warning("muse_bf16_tc_rans decode mismatch on %s; keeping dense weight", getattr(layer, "prefix", ""))
            return
        del rec
        layer._tc_rans_blob = blob
        layer._tc_rans_n = n
        layer._tc_rans_k = k
        _stats["admitted"] += 1
        _stats["packed_bytes"] += blob.nbytes
        _stats["raw_bytes"] += w.nbytes
        if self.free_raw_weight:
            _stats["freed_bytes"] += w.nbytes
            layer.weight = nn.Parameter(
                torch.empty((0,), dtype=w.dtype, device=w.device),
                requires_grad=False,
            )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        blob = getattr(layer, "_tc_rans_blob", None)
        if blob is not None and x.ndim == 2 and x.is_cuda and x.dtype == torch.bfloat16:
            n = int(layer._tc_rans_n)
            k = int(layer._tc_rans_k)
            if x.shape[1] == k:
                use_tc = self.free_raw_weight or (1 <= x.shape[0] <= 8)
                if use_tc:
                    b = bias.to(dtype=torch.bfloat16).contiguous() if (bias is not None and (bias.dtype != torch.bfloat16 or not bias.is_contiguous())) else bias
                    return torch.ops.vllm.muse_bf16_tc_rans_linear.default(x, blob, b, n, k)
        return super().apply(layer, x, bias)


def enable_layer(layer: nn.Module, free_raw_weight: bool) -> bool:
    w = getattr(layer, "weight", None)
    dev = w.device if isinstance(w, torch.Tensor) and w.is_cuda else None
    if not available(dev):
        return False
    qm = getattr(layer, "quant_method", None)
    if type(qm) is not UnquantizedLinearMethod:
        return False
    layer.quant_method = _MuseBF16TcRansMethod(free_raw_weight=free_raw_weight)
    return True
