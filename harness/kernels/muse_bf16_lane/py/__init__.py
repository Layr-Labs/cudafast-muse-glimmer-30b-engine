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
_stats = dict(admitted=0, rejected=0, bytes=0, rans=0, logged=False)


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
    if hasattr(native, "muse_bf16_rans_gemv"):
        native.muse_bf16_rans_hist.argtypes = [ptr, ctypes.c_size_t, ptr, ptr]
        native.muse_bf16_rans_hist.restype = ctypes.c_int
        native.muse_bf16_rans_pass1.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr]
        native.muse_bf16_rans_pass1.restype = ctypes.c_int
        native.muse_bf16_rans_pass2.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr, ptr]
        native.muse_bf16_rans_pass2.restype = ctypes.c_int
        native.muse_bf16_rans_gemv.argtypes = [ctypes.c_int, ctypes.c_size_t, ptr, ptr, ptr, ptr]
        native.muse_bf16_rans_gemv.restype = ctypes.c_int
    if hasattr(native, "muse_bf16_lane_set_pdl"):
        native.muse_bf16_lane_set_pdl.argtypes = [ctypes.c_int]
        native.muse_bf16_lane_set_pdl.restype = ctypes.c_int
    return native


def set_pdl(mode: int) -> int | None:
    """Set SM121 PDL mode for BF16 rANS decode launches (0=off, 1=consumer, 2=all, <0=env default)."""
    native = _lib()
    fn = getattr(native, "muse_bf16_lane_set_pdl", None) if native is not None else None
    if fn is None:
        return None
    return int(fn(int(mode)))


def _launch(x, payload, index, n):
    out = x.new_empty((1, n))
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        if index.numel() == 1:
            error = _lib().muse_bf16_rans_gemv(
                n, payload.numel(), x.data_ptr(), payload.data_ptr(), out.data_ptr(), stream.cuda_stream,
            )
        else:
            error = _lib().muse_bf16_lane_gemv(
                1, n, x.data_ptr(), payload.data_ptr(), index.data_ptr(),
                out.data_ptr(), stream.cuda_stream,
            )
    if error:
        raise RuntimeError(f"muse_bf16_lane launch returned CUDA status {error}")
    return out


def _pack_rans(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    native = _lib()
    if native is None or not hasattr(native, "muse_bf16_rans_gemv"):
        return None
    rows, k = weight.shape
    tw = 576 if rows == 4608 else (512 if rows == 4096 else rows // 8)
    if tw <= 0:
        return None
    dev = weight.device
    with torch.cuda.device(dev):
        st = torch.cuda.current_stream(dev).cuda_stream
        cnt = torch.zeros(256, dtype=torch.int64, device=dev)
        if native.muse_bf16_rans_hist(weight.data_ptr(), weight.numel(), cnt.data_ptr(), st):
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
        wb = torch.empty(tw, dtype=torch.int32, device=dev)
        if native.muse_bf16_rans_pass1(weight.data_ptr(), rows, tw, fd.data_ptr(), cd.data_ptr(), wb.data_ptr(), st):
            return None
        wb_h = wb.cpu().tolist()
        woff16 = [0] * (tw + 1)
        for i in range(tw):
            woff16[i + 1] = woff16[i] + wb_h[i]
        G = [(wb_h[i] * 16 + 511) // 512 for i in range(tw)]
        gmin = min(G)
        tab = woff16 + [0] * tw
        t = gmin * tw
        for i in range(tw):
            tab[tw + 1 + i] = t
            t += G[i] - gmin
        stream_bytes = t * 512
        soff = ((4112 + len(tab) * 4) + 511) & ~511
        total = soff + stream_bytes + 64
        if total >= weight.nbytes:
            return None
        blob = torch.zeros(total, dtype=torch.uint8, device=dev)
        blob[:4096].copy_(torch.tensor(lut, dtype=torch.int32, device=dev).view(torch.uint8))
        blob[4096:4112].copy_(torch.tensor([rows, tw, gmin, soff], dtype=torch.int32, device=dev).view(torch.uint8))
        blob[4112 : 4112 + len(tab) * 4].copy_(torch.tensor(tab, dtype=torch.int32, device=dev).view(torch.uint8))
        bp = blob.data_ptr()
        if native.muse_bf16_rans_pass2(
            weight.data_ptr(), rows, tw, gmin, fd.data_ptr(), cd.data_ptr(), bp + 4112, bp + soff, st
        ):
            return None
        torch.cuda.synchronize(dev)
    marker = torch.zeros((1,), dtype=torch.uint8, device=dev)
    return blob, marker


def _verify_probes(weight: torch.Tensor, packed: torch.Tensor, device_index: torch.Tensor) -> bool:
    n, k = weight.shape
    rng = torch.Generator(device=weight.device).manual_seed(720397)
    for magnitude in (0.02, 1.0, 32.0, 1000.0):
        x = torch.randn((1, k), generator=rng, device=weight.device)
        x[:, ::2].mul_(-magnitude)
        x = x.to(torch.bfloat16)
        expected = F.linear(x, weight)
        result = _launch(x, packed, device_index, n)
        if not torch.equal(expected.view(torch.int16), result.view(torch.int16)):
            return False
    return True


def _prepare(weight):
    if (weight.ndim != 2 or weight.shape[0] not in _ROWS or weight.shape[1] != _WIDTH
            or not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or torch.cuda.get_device_capability(weight.device) != (12, 1)):
        return None
    if os.environ.get("MUSE_BF16_RANS", "1") != "0":
        packed_rans = _pack_rans(weight)
        if packed_rans is not None and _verify_probes(weight, *packed_rans):
            _stats["rans"] += 1
            return packed_rans
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
    if not _verify_probes(weight, packed, device_index):
        return None
    return packed, device_index


_M_RANS: dict[int, tuple[torch.Tensor, int]] = {}
_M_RANS_STATS = {"admitted": 0, "rejected": 0, "logged": False}


def _prepare_m_rans(weight: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    if os.environ.get("MUSE_BF16_RANS_M", "1") == "0":
        return None
    try:
        import muse_bfx_ext as _bfx
    except Exception:  # noqa: BLE001
        return None
    if not hasattr(_bfx, "rans_pack") or not hasattr(_bfx, "rans_gemm"):
        return None
    n, k = weight.shape
    if n not in _ROWS or k != _WIDTH or not weight.is_cuda or weight.dtype != torch.bfloat16:
        return None
    wc = weight.detach().contiguous()
    blob = _bfx.rans_pack(wc)
    unp = _bfx.rans_unpack(blob, n, k)
    if not torch.equal(unp.view(torch.int16), wc.view(torch.int16)):
        del unp, blob
        return None
    del unp
    bexp = int(_bfx.rans_base(blob))
    gen = torch.Generator(device=weight.device)
    gen.manual_seed(n * 193 + k + 17)
    for scale in (0.05, 0.5, 2.0, 12.0):
        x = (torch.randn((7, k), generator=gen, device=weight.device) * scale).to(torch.bfloat16)
        ref = F.linear(x, wc)
        got = _bfx.rans_gemm(x, blob, n, k, bexp)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            del blob
            return None
    for cm in (12, 16):
        xm = (torch.randn((cm, k), generator=gen, device=weight.device) * 1.5).to(torch.bfloat16)
        got_m = _bfx.rans_gemm(xm, blob, n, k, bexp)
        ref_lo = _bfx.rans_gemm(xm[:8].contiguous(), blob, n, k, bexp)
        ref_hi = _bfx.rans_gemm(xm[8:].contiguous(), blob, n, k, bexp)
        if (
            not torch.equal(got_m[:8].view(torch.int16), ref_lo.view(torch.int16))
            or not torch.equal(got_m[8:].view(torch.int16), ref_hi.view(torch.int16))
        ):
            del blob
            return None
    return blob, bexp


def get_m_rans(index: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    return _M_RANS.get(index.data_ptr())


def update_m_rans(index: torch.Tensor, blob: torch.Tensor) -> None:
    entry = _M_RANS.get(index.data_ptr())
    if entry is not None:
        _M_RANS[index.data_ptr()] = (blob, entry[1])


def _decode(x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor,
            index: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    if (2 <= x.shape[0] <= 16 and x.ndim == 2 and x.shape[1] == _WIDTH
            and x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
            and n in _ROWS):
        entry = _M_RANS.get(index.data_ptr())
        if entry is not None:
            import muse_bfx_ext as _bfx
            blob, bexp = entry
            if not _M_RANS_STATS["logged"]:
                _M_RANS_STATS["logged"] = True
                logger.info("muse_bf16_lane multi-M rANS MMA: %d target BF16 layers admitted bitwise (0-ULP)",
                            _M_RANS_STATS["admitted"])
            return _bfx.rans_gemm(x, blob, n, _WIDTH, bexp)
    idx_ok = index.numel() == 1 or index.shape == (n, _INDEX_BYTES)
    if (x.shape != (1, _WIDTH) or x.dtype != torch.bfloat16
            or not x.is_cuda or not x.is_contiguous() or x.device != packed.device
            or n not in _ROWS or not idx_ok or index.device != x.device
            or index.dtype != torch.uint8 or not index.is_contiguous()
            or packed.dtype != torch.uint8 or not packed.is_contiguous() or _lib() is None):
        return F.linear(x, weight)
    if not _stats["logged"]:
        _stats["logged"] = True
        tpf_kb = os.environ.get("MUSE_BF16_TLB_PREFETCH", "2048")
        gate_blks = 32 if os.environ.get("MUSE_BF16_GATE_32_BLOCKS", "1") != "0" else 48
        logger.info("muse_bf16_lane decode GEMV: %d admitted (%d rANS), %d rejected, %d packed bytes "
                    "(translation prefetch %s KiB, gate %d blocks)",
                    _stats["admitted"], _stats["rans"], _stats["rejected"], _stats["bytes"], tpf_kb, gate_blks)
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
        layer.register_buffer("muse_bf16_rans_m", None, persistent=False)
        try:
            with torch.no_grad():
                m_res = _prepare_m_rans(layer.weight)
        except Exception as error:  # noqa: BLE001
            logger.warning("muse_bf16_lane multi-M rANS admission failed for %s (%s)", layer.prefix, error)
            m_res = None
        if m_res is not None:
            blob_m, bexp_m = m_res
            layer.muse_bf16_rans_m = blob_m
            _M_RANS[layer.muse_bf16_lane_index.data_ptr()] = (blob_m, bexp_m)
            _M_RANS_STATS["admitted"] += 1
        else:
            _M_RANS_STATS["rejected"] += 1

    def apply(self, layer, x, bias=None):
        packed = getattr(layer, "muse_bf16_lane_weight", None)
        if (packed is None or bias is not None or x.ndim != 2 or not x.is_cuda
                or x.dtype != torch.bfloat16 or not x.is_contiguous()):
            return super().apply(layer, x, bias)
        args = (x, layer.weight, packed, layer.muse_bf16_lane_index)
        if torch.compiler.is_compiling():
            return torch.cond(x.shape[0] <= 16, _packed_branch, _dense_branch, args)
        if 1 <= x.shape[0] <= 16:
            return _packed_branch(*args)
        return super().apply(layer, x, bias)


def enable(layer):
    if (type(layer.quant_method) is UnquantizedLinearMethod
            and not envs.VLLM_BATCH_INVARIANT
            and os.environ.get("MUSE_BF16_LANE", "1") != "0" and _lib() is not None):
        layer.quant_method = _MuseBF16Lane()
