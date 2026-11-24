"""Token embedding + weightless embedding RMSNorm of MuseGlimmerModel.embed_input_ids, written straight into the
prefill graph's inputs_embeds buffer (prefill3).

The eager path is eight kernels (int64 index copy, gather, to-fp32 copy, pow, mean, add, rsqrt, broadcast mul, to-bf16
copy) plus the copy into inputs_embeds. Here: one gather+square kernel (x * x of the fp32 value: ATen's pow(x, 2) on
float is base * base), ATen's OWN mean / add / rsqrt on the same fp32 tensor (same reduction kernel and order), and one
gather+scale kernel (fp32 x * inv, then round-to-nearest-even to bf16 as c10::BFloat16) writing the destination.

Admitted once, on the first eager call outside graph capture: (1) the square kernel equals torch's pow(x.float(), 2)
on every non-NaN bf16 bit pattern, (2) the whole fused path equals the eager path bit for bit on the call's real ids and
on random ids. Otherwise (or MUSE_EMBED_FUSED=0) the caller keeps the eager path.
"""

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("MUSE_EMBED_FUSED", "1") == "1"
FAST_INDEXED = os.environ.get("MUSE_EMBED_INDEXED", "1") != "0"
_lib = None
_scale_indexed = None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_embed.so")))
    _p = ctypes.c_void_p
    _lib.muse_embed_square.argtypes = [_p, ctypes.c_int, _p, _p, ctypes.c_int, ctypes.c_int, _p]
    _lib.muse_embed_square.restype = ctypes.c_int
    _lib.muse_embed_scale.argtypes = [_p, ctypes.c_int, _p, _p, _p, ctypes.c_int, ctypes.c_int, _p]
    _lib.muse_embed_scale.restype = ctypes.c_int
    _scale_indexed = getattr(_lib, "muse_embed_scale_indexed", None)
    if _scale_indexed is not None:
        _scale_indexed.argtypes = [_p, ctypes.c_int, _p, _p, _p, ctypes.c_int, ctypes.c_int, _p]
        _scale_indexed.restype = ctypes.c_int
except OSError:
    _lib = None
    _scale_indexed = None

state = {"admitted": None, "fused_calls": 0, "indexed_calls": 0, "indexed_admitted": False, "reason": ""}
_inv_rms_cache: dict[tuple[int, int, int, float, int], torch.Tensor] = {}


def _fused_5kernel(ids, table, eps, out):
    rows, width = out.shape
    stream = torch.cuda.current_stream(out.device).cuda_stream
    x2 = torch.empty((rows, width), dtype=torch.float32, device=out.device)
    status = _lib.muse_embed_square(ids.data_ptr(), int(ids.dtype == torch.int64), table.data_ptr(), x2.data_ptr(),
                                    rows, width, stream)
    if status:
        raise RuntimeError(f"muse_embed square launch returned CUDA error {status}")
    inv = torch.rsqrt(x2.mean(-1, keepdim=True) + eps)
    status = _lib.muse_embed_scale(ids.data_ptr(), int(ids.dtype == torch.int64), table.data_ptr(), inv.data_ptr(),
                                   out.data_ptr(), rows, width, stream)
    if status:
        raise RuntimeError(f"muse_embed scale launch returned CUDA error {status}")


def _fused_indexed(ids, table, inv_table, out):
    rows, width = out.shape
    stream = torch.cuda.current_stream(out.device).cuda_stream
    status = _scale_indexed(
        ids.data_ptr(),
        int(ids.dtype == torch.int64),
        table.data_ptr(),
        inv_table.data_ptr(),
        out.data_ptr(),
        rows,
        width,
        stream,
    )
    if status:
        raise RuntimeError(f"muse_embed scale_indexed launch returned CUDA error {status}")


def _build_inv_rms_table(table: torch.Tensor, eps: float) -> torch.Tensor:
    dev = table.device
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (table.data_ptr(), int(table.shape[0]), int(table.shape[1]), float(eps), int(dev_idx))
    cached = _inv_rms_cache.get(key)
    if cached is not None:
        return cached
    vocab, width = int(table.shape[0]), int(table.shape[1])
    inv_rms = torch.empty((vocab,), dtype=torch.float32, device=dev)
    chunk = 512
    x2_buf = torch.empty((chunk, width), dtype=torch.float32, device=dev)
    with torch.no_grad():
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            n = end - start
            if n == chunk:
                x2_buf.copy_(table[start:end]).pow_(2)
                inv_chunk = torch.rsqrt(x2_buf.mean(-1, keepdim=True) + eps).squeeze(-1)
                inv_rms[start:end].copy_(inv_chunk)
            else:
                x2_buf[:n].copy_(table[start:end])
                x2_buf[n:].copy_(table[0:1].expand(chunk - n, width))
                x2_buf.pow_(2)
                inv_chunk = torch.rsqrt(x2_buf.mean(-1, keepdim=True) + eps).squeeze(-1)
                inv_rms[start:end].copy_(inv_chunk[:n])
    del x2_buf
    _inv_rms_cache[key] = inv_rms
    return inv_rms


def precompute_inv_rms(model) -> torch.Tensor | None:
    """Precompute and cache embed_inv_rms[vocab] FP32 without touching embed_tokens.weight."""
    if not ENABLED or not FAST_INDEXED or _lib is None or _scale_indexed is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    emb = getattr(model, "embed_tokens", None)
    norm = getattr(model, "embed_norm", None)
    table = getattr(emb, "weight", None)
    if (
        table is None or norm is None or getattr(emb, "tp_size", 1) != 1
        or type(emb.quant_method).__name__ != "UnquantizedEmbeddingMethod"
        or norm.with_scale or table.dtype != torch.bfloat16
        or not table.is_cuda or not table.is_contiguous() or table.dim() != 2
    ):
        return None
    return _build_inv_rms_table(table, float(norm.eps))


def _fused(ids, table, eps, out):
    if state["indexed_admitted"] and _scale_indexed is not None:
        dev = table.device
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        key = (table.data_ptr(), int(table.shape[0]), int(table.shape[1]), float(eps), int(dev_idx))
        inv_table = _inv_rms_cache.get(key)
        if inv_table is not None:
            _fused_indexed(ids, table, inv_table, out)
            state["indexed_calls"] += 1
            return
    _fused_5kernel(ids, table, eps, out)


def _eligible(model, ids, out):
    emb, norm = model.embed_tokens, model.embed_norm
    table = getattr(emb, "weight", None)
    if table is None or getattr(emb, "tp_size", 1) != 1 or type(emb.quant_method).__name__ != "UnquantizedEmbeddingMethod":
        return None
    if norm.with_scale or table.dtype != torch.bfloat16 or not table.is_contiguous() or table.dim() != 2:
        return None
    if (ids.dim() != 1 or ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous() or not ids.is_cuda
            or out.dtype != torch.bfloat16 or out.dim() != 2 or not out.is_contiguous() or out.shape[0] != ids.shape[0]
            or out.shape[1] != table.shape[1] or out.shape[1] % 8 or out.data_ptr() % 16 or table.data_ptr() % 16):
        return None
    return table


def _admit(model, ids, out, table) -> str:
    dev = out.device
    width = table.shape[1]
    eps = model.embed_norm.eps
    # (1) square on every non-NaN bf16 pattern
    pats = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    nrows = (65536 + width - 1) // width
    t = torch.zeros(nrows * width, dtype=torch.bfloat16, device=dev)
    t[:65536] = pats
    t = t.view(nrows, width)
    x2 = torch.empty((nrows, width), dtype=torch.float32, device=dev)
    rid = torch.arange(nrows, dtype=torch.int32, device=dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    if _lib.muse_embed_square(rid.data_ptr(), 0, t.data_ptr(), x2.data_ptr(), nrows, width, stream):
        return "square launch failed"
    ref = t.float().pow(2)
    finite = ~torch.isnan(t.float())
    if not torch.equal(x2.view(torch.int32)[finite], ref.view(torch.int32)[finite]):
        return "square differs from pow"
    if not bool(torch.isfinite(table).all()):
        return "non-finite embedding rows"
    # (2) whole path vs eager on the real ids and random ids
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261123)
    cases = [ids.clone()]
    for _ in range(4):
        cases.append(torch.randint(0, table.shape[0], ids.shape, generator=gen, device=dev, dtype=ids.dtype))
    for case in cases:
        mine = torch.empty_like(out)
        _fused_5kernel(case, table, eps, mine)
        theirs = model.embed_norm(model.embed_tokens(case))
        torch.cuda.synchronize()
        if not torch.equal(mine.view(torch.int16), theirs.view(torch.int16)):
            return "fused path differs from eager"
    # (3) Precompute embed_inv_rms[vocab] FP32 via exact ATen mean/add/rsqrt and verify 1-kernel
    #     muse_embed_scale_indexed bitwise against BOTH the 5-kernel path and eager on M=1 and M=512.
    state["indexed_admitted"] = False
    if FAST_INDEXED and _scale_indexed is not None:
        try:
            inv_table = _build_inv_rms_table(table, float(eps))
            idx_ok = bool(torch.isfinite(inv_table).all())
            if idx_ok:
                idx_cases = list(cases)
                for m_probe in (1, 512):
                    for dt in (torch.int32, torch.int64):
                        for _ in range(4):
                            idx_cases.append(
                                torch.randint(0, table.shape[0], (m_probe,), generator=gen, device=dev, dtype=dt)
                            )
                for c_ids in idx_cases:
                    ref_5k = torch.empty((c_ids.shape[0], width), dtype=torch.bfloat16, device=dev)
                    got_1k = torch.empty_like(ref_5k)
                    _fused_5kernel(c_ids, table, eps, ref_5k)
                    _fused_indexed(c_ids, table, inv_table, got_1k)
                    theirs_c = model.embed_norm(model.embed_tokens(c_ids))
                    torch.cuda.synchronize()
                    if (
                        not torch.equal(got_1k.view(torch.int16), ref_5k.view(torch.int16))
                        or not torch.equal(got_1k.view(torch.int16), theirs_c.view(torch.int16))
                    ):
                        idx_ok = False
                        break
            state["indexed_admitted"] = idx_ok
        except Exception:  # noqa: BLE001
            state["indexed_admitted"] = False
    return ""


def embed_into(model, ids, out) -> bool:
    """Write embed_norm(embed_tokens(ids)) into `out`. False when the caller must use the eager path."""
    if not ENABLED or _lib is None or state["admitted"] is False:
        return False
    table = _eligible(model, ids, out)
    if table is None:
        return False
    if state["admitted"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        reason = _admit(model, ids, out, table)
        state["admitted"] = not reason
        state["reason"] = reason or "admitted"
        if reason:
            return False
    _fused(ids, table, model.embed_norm.eps, out)
    state["fused_calls"] += 1
    return True
