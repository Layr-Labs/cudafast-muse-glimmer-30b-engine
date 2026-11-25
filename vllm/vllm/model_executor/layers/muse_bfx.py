# SPDX-License-Identifier: Apache-2.0
"""Lossless compressed BF16 weights for small-row (M <= 8) GEMMs (the DFlash drafter at verify width).

Primary path (MUSE_BFX_RANS=1, default): 12-bit exponent-pair rANS + inline escape + raw (sign, mantissa)
(~10.6 bits/weight, 100% bit-exact lossless verified via rans_unpack at load) with K32 2x mma.sync.m16n8k16
Tensor-Core GEMM, 48-CTA persistent cp.async (L2::evict_first), 2 MiB TLB prefetch, dual-sided SM121 PDL,
and load-time DRAM placement across GB10 UMA banks.

Fallback path (MUSE_BFX_RANS=0): original 12-bit coded BF16 + CSR escape kernel (bfx_linear).
Rows > 8 always use dense F.linear unchanged.
"""
from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import muse_bfx_ext as _ext
except ImportError:  # kernels not built: dense path
    _ext = None

CH = 512
MAX_M = 8
_USE_RANS = os.environ.get("MUSE_BFX_RANS", "1") != "0"
_PLACE_RANS = os.environ.get("MUSE_BFX_PLACE", "1") != "0"


def _encode_rows(w: torch.Tensor):
    N, K = w.shape
    u = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    base = e.max(dim=1, keepdim=True).values.clamp_min(15) - 15
    e -= base
    inwin = (e >= 1) & (e <= 15)
    e.masked_fill_(~inwin, 0)
    lo = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    del u
    nib = (e[:, 0::2] | (e[:, 1::2] << 4)).to(torch.uint8)
    lo32 = lo.to(torch.int32)
    bits = ((lo32 & 0x80) << 8) | ((base + e) << 7) | (lo32 & 0x7F)
    del lo32
    bits.masked_fill_(e == 0, 0)
    del e
    esc = (~inwin) & ((w.view(torch.int16) & 0x7FFF) != 0)
    rows, cols = esc.nonzero(as_tuple=True)
    rec = bits.to(torch.int16)
    del bits
    rec[rows, cols] = w.view(torch.int16)[rows, cols]
    ok = torch.equal(rec, w.view(torch.int16))
    return ok, lo, nib, base.squeeze(1).to(torch.int32), rows, cols


def _encode(w: torch.Tensor) -> dict | None:
    if _ext is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % CH or not w.is_cuda:
        return None
    w = w.contiguous()
    N, K = w.shape
    lo = torch.empty(N, K, dtype=torch.uint8, device=w.device)
    nib = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    base = torch.empty(N, dtype=torch.int32, device=w.device)
    R, C = [], []
    step = max(1, (64 << 20) // K)
    for r0 in range(0, N, step):
        ok, l, nb, b, rr, cc = _encode_rows(w[r0:r0 + step])
        if not ok:
            return None
        lo[r0:r0 + step] = l
        nib[r0:r0 + step] = nb
        base[r0:r0 + step] = b
        R.append(rr + r0)
        C.append(cc)
        del l, nb, b
    rows = torch.cat(R)
    cols = torch.cat(C)
    rowptr = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    rowptr[1:] = torch.bincount(rows, minlength=N).cumsum(0).to(torch.int32)
    ecol = cols.to(torch.int32).contiguous()
    evals = w[rows, cols].contiguous()
    if ecol.numel() == 0:
        ecol = torch.zeros(1, dtype=torch.int32, device=w.device)
        evals = torch.zeros(1, dtype=torch.bfloat16, device=w.device)
    return dict(lo=lo, nib=nib, base=base, rowptr=rowptr, ecol=ecol, eval=evals)


def _encode_rans(w: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    if (
        _ext is None
        or not _USE_RANS
        or not hasattr(_ext, "rans_pack")
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_cuda
    ):
        return None
    N, K = w.shape
    if N % 8 != 0 or K % 32 != 0:
        return None
    wc = w.contiguous()
    blob = _ext.rans_pack(wc)
    unp = _ext.rans_unpack(blob, N, K)
    ok = torch.equal(unp.view(torch.int16), wc.view(torch.int16))
    del unp
    if not ok:
        del blob
        return None
    base_exp = int(_ext.rans_base(blob))
    return blob, base_exp


@torch.library.custom_op("muse::bfx_linear", mutates_args=())
def bfx_linear(x: torch.Tensor, w: torch.Tensor, lo: torch.Tensor, nib: torch.Tensor, base: torch.Tensor,
               rowptr: torch.Tensor, ecol: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _ext.gemm(x2.contiguous(), lo, nib, base, rowptr, ecol, ev)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], w.shape[0])


@bfx_linear.register_fake
def _(x, w, lo, nib, base, rowptr, ecol, ev):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@torch.library.custom_op("muse::bfx_rans_linear", mutates_args=())
def bfx_rans_linear(x: torch.Tensor, w: torch.Tensor, blob: torch.Tensor, base_exp: int) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _ext.rans_gemm(x2.contiguous(), blob, w.shape[0], w.shape[1], base_exp)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], w.shape[0])


@bfx_rans_linear.register_fake
def _(x, w, blob, base_exp):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def _register(mod: torch.nn.Module, prefix: str, enc: dict) -> None:
    for k, v in enc.items():
        mod.register_buffer(f"{prefix}_{k}", v, persistent=False)


def attach_linear(layer: torch.nn.Module) -> bool:
    w = getattr(layer, "weight", None)
    if w is None or getattr(layer, "bias", None) is not None or getattr(layer, "_bfx", False) \
            or getattr(layer, "_mfp8", None) is not None:
        return False
    rans = _encode_rans(w.data)
    if rans is not None:
        blob, base_exp = rans
        layer.register_buffer("_bfx_rans_blob", blob, persistent=False)
        layer._bfx_rans_base = base_exp
        layer._bfx_rans = True
        layer._bfx = True
        return True
    enc = _encode(w.data)
    if enc is None:
        return False
    _register(layer, "_bfx", enc)
    layer._bfx_rans = False
    layer._bfx = True
    return True


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "_bfx_rans", False):
        return torch.ops.muse.bfx_rans_linear(x, layer.weight, layer._bfx_rans_blob, layer._bfx_rans_base)
    return torch.ops.muse.bfx_linear(x, layer.weight, layer._bfx_lo, layer._bfx_nib, layer._bfx_base,
                                     layer._bfx_rowptr, layer._bfx_ecol, layer._bfx_eval)


def attach_tensor(owner: torch.nn.Module, name: str, w: torch.Tensor) -> bool:
    rans = _encode_rans(w)
    if rans is not None:
        blob, base_exp = rans
        owner.register_buffer(f"_bfx_{name}_rans_blob", blob, persistent=False)
        setattr(owner, f"_bfx_{name}_rans_base", base_exp)
        return True
    enc = _encode(w)
    if enc is None:
        return False
    _register(owner, f"_bfx_{name}", enc)
    return True


def tensor_apply(owner: torch.nn.Module, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p = f"_bfx_{name}_"
    if hasattr(owner, p + "rans_blob"):
        return torch.ops.muse.bfx_rans_linear(x, w, getattr(owner, p + "rans_blob"), getattr(owner, p + "rans_base"))
    if not hasattr(owner, p + "lo"):
        return F.linear(x, w)
    return torch.ops.muse.bfx_linear(x, w, getattr(owner, p + "lo"), getattr(owner, p + "nib"), getattr(owner, p + "base"),
                                     getattr(owner, p + "rowptr"), getattr(owner, p + "ecol"), getattr(owner, p + "eval"))


def place_drafter_rans(model: torch.nn.Module) -> None:
    """Place drafter BF16 rANS blobs in fast GB10 UMA DRAM regions via cold-L2 spare-buffer selection."""
    if not _PLACE_RANS or _ext is None or not hasattr(_ext, "rans_gemm_out"):
        return
    entries: list[tuple[torch.nn.Module, str, int, int, int]] = []
    for mod in model.modules():
        if getattr(mod, "_bfx_rans", False) and hasattr(mod, "_bfx_rans_blob"):
            w = mod.weight
            entries.append((mod, "_bfx_rans_blob", int(w.shape[0]), int(w.shape[1]), int(mod._bfx_rans_base)))
        if hasattr(mod, "_bfx_fkv_rans_blob") and hasattr(mod, "_fused_kv_weight"):
            w = mod._fused_kv_weight
            entries.append((mod, "_bfx_fkv_rans_blob", int(w.shape[0]), int(w.shape[1]), int(mod._bfx_fkv_rans_base)))
    if not entries:
        return

    began = time.perf_counter()
    dev = entries[0][0]._buffers[entries[0][1]].device
    evict_buf = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    by_shape: dict[tuple[int, int], list[tuple[torch.nn.Module, str, int]]] = {}
    for mod, attr, n, k, bexp in entries:
        by_shape.setdefault((n, k), []).append((mod, attr, bexp))

    total_saved = 0.0
    total_moved = 0
    # Largest groups first so big blobs get first pick of fast DRAM regions
    ordered = sorted(
        by_shape.items(),
        key=lambda kv: -(len(kv[1]) * max(m._buffers[a].numel() for m, a, _ in kv[1])),
    )
    for (n, k), items in ordered:
        homes = [m._buffers[a] for m, a, _ in items]
        ref_bexp = items[0][2]
        max_len = max(h.numel() for h in homes)
        # Keep temporary spare allocation <= ~5 GiB per group
        max_spares = max(2, min(16, (5 << 30) // max(1, max_len)))
        n_spares = min(max_spares, max(3, len(items) * 2))
        x7 = torch.zeros((7, k), dtype=torch.bfloat16, device=dev)
        out7 = torch.empty((7, n), dtype=torch.bfloat16, device=dev)

        def _time_blob(buf: torch.Tensor, bexp: int, reps: int = 4) -> float:
            ts = []
            for _ in range(reps):
                evict_buf.zero_()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                _ext.rans_gemm_out(x7, buf, out7, n, k, bexp)
                e1.record()
                torch.cuda.synchronize(dev)
                ts.append(e0.elapsed_time(e1))
            ts.sort()
            return ts[len(ts) // 2]

        # Warm up kernel once
        _ext.rans_gemm_out(x7, homes[0], out7, n, k, ref_bexp)
        torch.cuda.synchronize(dev)

        home_t = [_time_blob(h, items[i][2]) for i, h in enumerate(homes)]
        spares: list[torch.Tensor] = []
        spare_t: list[tuple[float, int]] = []
        ref_blob = homes[0]
        for s_idx in range(n_spares):
            try:
                sb = torch.empty(max_len, dtype=torch.uint8, device=dev)
            except RuntimeError:
                break
            sb[:ref_blob.numel()].copy_(ref_blob)
            t_s = _time_blob(sb[:ref_blob.numel()], ref_bexp)
            spares.append(sb)
            spare_t.append((t_s, s_idx))

        spare_t.sort(key=lambda pair: pair[0])
        # Assign fastest spares to slowest homes if strictly faster by > 1%
        order_homes = sorted(range(len(items)), key=lambda i: -home_t[i])
        after_t = list(home_t)
        sp_ptr = 0
        moved = 0
        for hi in order_homes:
            if sp_ptr >= len(spare_t):
                break
            t_sp, si = spare_t[sp_ptr]
            if t_sp < home_t[hi] * 0.99:
                mod, attr, bexp = items[hi]
                h = homes[hi]
                dst = spares[si][:h.numel()]
                dst.copy_(h)
                t_new = _time_blob(dst, bexp)
                if t_new < home_t[hi]:
                    mod._buffers[attr] = dst
                    after_t[hi] = t_new
                    moved += 1
                    sp_ptr += 1
                else:
                    sp_ptr += 1
        del spares, x7, out7
        before_sum = sum(home_t)
        after_sum = sum(after_t)
        total_saved += before_sum - after_sum
        total_moved += moved
        logger.info(
            "Drafter BF16 rANS placement %dx%d: %d blobs (~%.0f MiB), %d moved; sum %.3f -> %.3f ms",
            n, k, len(items), max_len / (1 << 20), moved, before_sum, after_sum,
        )
    del evict_buf
    torch.cuda.empty_cache()
    logger.info(
        "Drafter BF16 rANS placement done: %d moved, saved %.3f ms/step in %.2f s",
        total_moved, total_saved, time.perf_counter() - began,
    )


def attach_drafter(model: torch.nn.Module) -> int:
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc"):
            n += attach_linear(mod)
    return n
