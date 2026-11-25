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
ENABLED_M1 = os.environ.get("MUSE_EMBED_FUSED_M1", "1") != "0"
ENABLED_TAIL = os.environ.get("MUSE_FAST_DECODE_TAIL", "1") != "0"
_lib = None
_post_update_m1_fn = None
_single_decode_prep_fn = None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_embed.so")))
    _p = ctypes.c_void_p
    _lib.muse_embed_square.argtypes = [_p, ctypes.c_int, _p, _p, ctypes.c_int, ctypes.c_int, _p]
    _lib.muse_embed_square.restype = ctypes.c_int
    _lib.muse_embed_scale.argtypes = [_p, ctypes.c_int, _p, _p, _p, ctypes.c_int, ctypes.c_int, _p]
    _lib.muse_embed_scale.restype = ctypes.c_int
    _m1 = getattr(_lib, "muse_embed_norm_m1", None)
    if _m1 is not None:
        _m1.argtypes = [_p, ctypes.c_int, _p, ctypes.c_float, _p, ctypes.c_int, _p]
        _m1.restype = ctypes.c_int
    _mm = getattr(_lib, "muse_embed_norm_m", None)
    if _mm is not None:
        _mm.argtypes = [_p, ctypes.c_int, _p, ctypes.c_float, _p, ctypes.c_int, ctypes.c_int, _p]
        _mm.restype = ctypes.c_int
    _pu = getattr(_lib, "muse_post_update_m1", None)
    if _pu is not None:
        _pu.argtypes = [
            _p, _p, _p, _p, ctypes.c_int64, _p, _p, _p, _p, _p, ctypes.c_int64, _p, _p,
        ]
        _pu.restype = ctypes.c_int
        _post_update_m1_fn = _pu
    _dp = getattr(_lib, "muse_single_decode_prep", None)
    if _dp is not None:
        _dp.argtypes = [
            ctypes.c_int, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, ctypes.c_int64, _p, _p, ctypes.c_int64, _p,
        ]
        _dp.restype = ctypes.c_int
        _single_decode_prep_fn = _dp
except OSError:
    _lib = None


def fast_post_update_m1(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> bool:
    if (
        not ENABLED_TAIL
        or _post_update_m1_fn is None
        or idx_mapping.shape[0] != 1
        or idx_mapping.dtype != torch.int32
        or num_computed_tokens.dtype != torch.int32
        or last_sampled_tokens.dtype != torch.int64
        or not last_sampled_tokens.is_contiguous()
        or sampled_tokens.dtype != torch.int64
        or not sampled_tokens.is_contiguous()
        or num_sampled.dtype != torch.int32
        or num_rejected.dtype != torch.int32
        or all_token_ids.dtype != torch.int32
        or total_len.dtype != torch.int32
        or (query_start_loc is not None and query_start_loc.dtype != torch.int32)
        or (output_bin_counts is not None and output_bin_counts.dtype != torch.int32)
    ):
        return False
    rc = _post_update_m1_fn(
        idx_mapping.data_ptr(),
        num_computed_tokens.data_ptr(),
        last_sampled_tokens.data_ptr(),
        output_bin_counts.data_ptr() if output_bin_counts is not None else 0,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens.data_ptr(),
        num_sampled.data_ptr(),
        num_rejected.data_ptr(),
        query_start_loc.data_ptr() if query_start_loc is not None else 0,
        all_token_ids.data_ptr(),
        all_token_ids.stride(0),
        total_len.data_ptr(),
        torch.cuda.current_stream(idx_mapping.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"muse_post_update_m1 failed with CUDA error {rc}")
    return True


def fast_single_decode_prep(
    num_groups: int,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    src_bt_ptrs: torch.Tensor,
    dst_bt_ptrs: torch.Tensor,
    bt_strides: torch.Tensor,
    num_blocks: torch.Tensor,
    block_sizes: torch.Tensor,
    slot_mappings: torch.Tensor,
) -> bool:
    if (
        not ENABLED_TAIL
        or _single_decode_prep_fn is None
        or num_groups <= 0
        or idx_mapping.dtype != torch.int32
        or num_computed_tokens.dtype != torch.int32
        or last_sampled_tokens.dtype != torch.int64
        or not last_sampled_tokens.is_contiguous()
        or input_ids.dtype != torch.int32
        or positions.dtype != torch.int64
        or seq_lens.dtype != torch.int32
        or src_bt_ptrs.dtype != torch.uint64
        or dst_bt_ptrs.dtype != torch.uint64
        or bt_strides.dtype != torch.int64
        or num_blocks.dtype != torch.int32
        or block_sizes.dtype != torch.int32
        or slot_mappings.dtype != torch.int64
    ):
        return False
    rc = _single_decode_prep_fn(
        int(num_groups),
        idx_mapping.data_ptr(),
        num_computed_tokens.data_ptr(),
        last_sampled_tokens.data_ptr(),
        input_ids.data_ptr(),
        positions.data_ptr(),
        seq_lens.data_ptr(),
        src_bt_ptrs.data_ptr(),
        dst_bt_ptrs.data_ptr(),
        bt_strides.data_ptr(),
        num_blocks.data_ptr(),
        num_blocks.stride(0),
        block_sizes.data_ptr(),
        slot_mappings.data_ptr(),
        slot_mappings.stride(0),
        torch.cuda.current_stream(idx_mapping.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"muse_single_decode_prep failed with CUDA error {rc}")
    return True

state = {"admitted": None, "admitted_m1": None, "admitted_m": None, "fused_calls": 0, "m1_calls": 0, "reason": ""}


def _fused_m(ids, table, eps, out):
    rows, width = out.shape
    stream = torch.cuda.current_stream(out.device).cuda_stream
    status = _lib.muse_embed_norm_m(
        ids.data_ptr(), int(ids.dtype == torch.int64), table.data_ptr(), float(eps), out.data_ptr(), rows, width, stream
    )
    if status:
        raise RuntimeError(f"muse_embed norm_m launch returned CUDA error {status}")


def _admit_m(model, ids, out, table) -> bool:
    if not ENABLED_M1 or _lib is None or getattr(_lib, "muse_embed_norm_m", None) is None:
        return False
    rows, width = out.shape
    if rows <= 0 or rows > 16 or width <= 0 or (width & 7) != 0 or width > 8192:
        return False
    dev = out.device
    eps = model.embed_norm.eps
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261126)
    cases = [ids.clone()]
    for _ in range(4):
        cases.append(torch.randint(0, table.shape[0], (rows,), generator=gen, device=dev, dtype=ids.dtype))
    for case in cases:
        mine = torch.empty((rows, width), dtype=torch.bfloat16, device=dev)
        _fused_m(case, table, eps, mine)
        theirs = model.embed_norm(model.embed_tokens(case))
        torch.cuda.synchronize()
        if not torch.equal(mine.view(torch.int16), theirs.view(torch.int16)):
            return False
    return True


def is_single_kernel_m() -> bool:
    return bool(ENABLED and ENABLED_M1 and state["admitted_m"] is True)


def _fused_m1(ids, table, eps, out):
    _, width = out.shape
    stream = torch.cuda.current_stream(out.device).cuda_stream
    status = _lib.muse_embed_norm_m1(
        ids.data_ptr(), int(ids.dtype == torch.int64), table.data_ptr(), float(eps), out.data_ptr(), width, stream
    )
    if status:
        raise RuntimeError(f"muse_embed norm_m1 launch returned CUDA error {status}")


def _admit_m1(model, ids, out, table) -> bool:
    if not ENABLED_M1 or _lib is None or getattr(_lib, "muse_embed_norm_m1", None) is None:
        return False
    width = table.shape[1]
    if width <= 0 or (width & 7) != 0 or width > 8192:
        return False
    dev = out.device
    eps = model.embed_norm.eps
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261124)
    cases = [ids[:1].clone()]
    for _ in range(8):
        cases.append(torch.randint(0, table.shape[0], (1,), generator=gen, device=dev, dtype=ids.dtype))
    for case in cases:
        mine = torch.empty((1, width), dtype=torch.bfloat16, device=dev)
        _fused_m1(case, table, eps, mine)
        theirs = model.embed_norm(model.embed_tokens(case))
        torch.cuda.synchronize()
        if not torch.equal(mine.view(torch.int16), theirs.view(torch.int16)):
            return False
    return True


def is_single_kernel_m1() -> bool:
    return bool(ENABLED and ENABLED_M1 and state["admitted_m1"] is True)


def _fused(ids, table, eps, out):
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
        _fused(case, table, eps, mine)
        theirs = model.embed_norm(model.embed_tokens(case))
        torch.cuda.synchronize()
        if not torch.equal(mine.view(torch.int16), theirs.view(torch.int16)):
            return "fused path differs from eager"
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
    if out.shape[0] == 1 and ENABLED_M1:
        if state["admitted_m1"] is None and not torch.cuda.is_current_stream_capturing():
            state["admitted_m1"] = _admit_m1(model, ids, out, table)
        if state["admitted_m1"] is True:
            _fused_m1(ids, table, model.embed_norm.eps, out)
            state["fused_calls"] += 1
            state["m1_calls"] += 1
            return True
    if 2 <= out.shape[0] <= 16 and ENABLED_M1:
        if state["admitted_m"] is None and not torch.cuda.is_current_stream_capturing():
            state["admitted_m"] = _admit_m(model, ids, out, table)
        if state["admitted_m"] is True:
            _fused_m(ids, table, model.embed_norm.eps, out)
            state["fused_calls"] += 1
            state["m1_calls"] += 1
            return True
    _fused(ids, table, model.embed_norm.eps, out)
    state["fused_calls"] += 1
    return True
