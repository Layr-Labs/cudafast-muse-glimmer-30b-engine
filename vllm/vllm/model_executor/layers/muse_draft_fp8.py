# SPDX-License-Identifier: Apache-2.0
"""Weight-only quantized copies (NVFP4 TMA4-ILV / FP8 Marlin) of the pinned DFlash drafter's linears.

The drafter only PROPOSES tokens: every committed token is the target's own greedy token from the
verify step, and the per-depth oracles equal the serial tape. The drafter's arithmetic can therefore
change acceptance, never the emitted stream. A depth-6 speculative step reads the whole 5.1 GB BF16
drafter once; at verify width (M <= 8 rows) these GEMMs are weight-bandwidth bound on GB10.

At load, each bias-free BF16 drafter linear (qkv/o/gate_up/down of the 5 layers, `fc`, and the fused
context-KV projection `fkv`) gets a quantized weight-only copy:
  - `nvfp4` (default): group-16 NVFP4 (float4_e2m1f + 64-entry E4M3 block-scale palette) packed into
    decode6's 4-stage TMA interleaved stream-K layout (`muse_marlin6_sk_tma4_ilv_m`) for M = 1..8
    (~1.44 GB total, ~4.35 bits/weight, 0-ULP verified against ops.marlin_gemm at load), plus GB10
    UMA DRAM placement.
  - `hybrid`: MLP (gate_up, down) and `fc` on NVFP4 TMA4-ILV; attention (qkv, o_proj, fkv) on FP8
    Marlin W8A16.
  - `fp8`: all 22 drafter linears on FP8 e4m3 per-row Marlin W8A16.

Calls with M > MAX_M (e.g., the 512-row prompt context pass after prefill) use the original BF16
weight unchanged, so prefill numerics are untouched. Every layer is admitted only after a load-time
accuracy and finiteness check; any failure keeps the previous path. The target is never touched.
MUSE_DRAFT_FP8=0 disables it; MUSE_DRAFT_QUANT selects nvfp4 | hybrid | fp8.
"""
import os
import time

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M_FP8 = 16
MAX_M_NVFP4 = 8
_DECODER_NAMES = ("o_proj", "qkv_proj", "down_proj", "gate_up_proj")
_NAMES = (*_DECODER_NAMES, "fc")
_MLP_FC_NAMES = ("gate_up_proj", "down_proj", "fc")
_ATTN_NAMES = ("qkv_proj", "o_proj")
_ATTN_FC_NAMES = ("qkv_proj", "o_proj", "fc")
_ATTN_FC_DN_NAMES = ("qkv_proj", "o_proj", "down_proj", "fc")
_stats = {
    "admitted_int3": 0,
    "admitted_int4": 0,
    "admitted_nvfp4": 0,
    "admitted_fp8": 0,
    "admitted_m512_fp8": 0,
    "rejected": 0,
    "bytes_bf16": 0,
    "bytes_quant": 0,
    "bytes_freed": 0,
}

_SHARED_PARTIALS: dict[int, torch.Tensor] = {}
_BFX_EXT = None
_BFX_TRIED = False


def _get_bfx():
    global _BFX_EXT, _BFX_TRIED
    if not _BFX_TRIED:
        _BFX_TRIED = True
        try:
            import muse_bfx_ext as _bfx

            _BFX_EXT = _bfx
        except Exception:  # noqa: BLE001
            _BFX_EXT = None
    return _BFX_EXT


def enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_FP8", "1") != "0"


def _quant_mode() -> str:
    return os.environ.get("MUSE_DRAFT_QUANT", "int3").strip().lower()


def _get_partials(device: torch.device) -> torch.Tensor:
    idx = device.index or 0
    buf = _SHARED_PARTIALS.get(idx)
    if buf is None:
        buf = torch.empty((48, 16, 128), dtype=torch.float32, device=device)
        _SHARED_PARTIALS[idx] = buf
    return buf


# ---------------------------------------------------------------------------
# FP8 W8A16 (per-row e4m3 via stock Marlin)
# ---------------------------------------------------------------------------

def _quantize_rows_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] -> (e4m3 [N, K], fp32 scale [N]); per-row amax / 448, computed in fp32 in row chunks."""
    n, k = w.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n,), dtype=torch.float32, device=w.device)
    step = max(1, (32 << 20) // k)
    for r0 in range(0, n, step):
        wf = w[r0:r0 + step].float()
        sc = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-30)
        q[r0:r0 + step] = (wf / sc[:, None]).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        s[r0:r0 + step] = sc
        del wf, sc
    return q, s


def _pack_fp8(w: torch.Tensor):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
        pack_fp8_to_int32,
    )

    n, k = w.shape
    q, s = _quantize_rows_fp8(w)
    packed = pack_fp8_to_int32(q, size_k_first=False).T.contiguous()  # [K / 4, N] int32
    perm = torch.empty(0, dtype=torch.int, device=w.device)
    qweight = ops.gptq_marlin_repack(b_q_weight=packed, perm=perm, size_k=k, size_n=n, num_bits=8)
    del packed
    scales = marlin_permute_scales(s=s.to(w.dtype).view(1, n), size_k=k, size_n=n, group_size=-1)
    scales = fp8_fused_exponent_bias_into_scales(scales)
    ws = marlin_make_workspace_new(w.device)
    return q, s, qweight, scales, ws


def _marlin_fp8(x2: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor, n: int, k: int):
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    return ops.marlin_gemm(
        a=x2,
        c=None,
        b_q_weight=qweight,
        b_bias=None,
        b_scales=scales,
        a_scales=None,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=ws,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=x2.size(0),
        size_n=n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )


def _check_fp8(w: torch.Tensor, q: torch.Tensor, s: torch.Tensor, qweight, scales, ws) -> bool:
    n, k = w.shape
    gen = torch.Generator(device=w.device)
    gen.manual_seed(1009 + n + 7 * k)
    wd = None
    try:
        for m in (1, 7, 8, 16):
            x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
            got = _marlin_fp8(x, qweight, scales, ws, n, k)
            if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
                return False
            if wd is None:
                wd = (q.float() * s[:, None]).to(w.dtype)
            ref = F.linear(x, wd).float()
            err = (got.float() - ref).abs().max().item()
            tol = 2e-2 * ref.abs().max().item() + 1e-3
            if not err <= tol:
                return False
            dense = F.linear(x, w).float()
            cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
            if not cos >= 0.995:
                return False
        torch.cuda.synchronize(w.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_fp8: Marlin FP8 check failed (%s); layer keeps its BF16 path", e)
        return False
    return True


# ---------------------------------------------------------------------------
# NVFP4 W4A16 (group-16 float4_e2m1f via decode6 TMA4-ILV)
# ---------------------------------------------------------------------------

def _quantize_nvfp4_tma4(w: torch.Tensor, max_pal: int = 64):
    """Quantize BF16 [N, K] (N%128==0, K%128==0) to NVFP4 group-16 + decode6 TMA4-ILV buffers."""
    import muse_marlin6_gemv as d6
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    n, k = w.shape
    dev = w.device
    g_k = k // 16

    # Pass 1: compute block maxes in chunks to keep peak VRAM small.
    blk_max = torch.empty((n, g_k), dtype=torch.float32, device=dev)
    step = max(128, ((32 << 20) // k) & ~127)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        wf = w[r0:r1].float().view(r1 - r0, g_k, 16)
        blk_max[r0:r1] = wf.abs().amax(dim=-1)
        del wf

    gmax = blk_max.max().clamp_min(1e-30)
    gs = gmax / (6.0 * 448.0)
    sc_fp8 = (blk_max / (6.0 * gs)).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
    del blk_max

    # Constrain block scales to <= max_pal distinct E4M3 values so decode6's 6-bit rANS palette is lossless.
    u = sc_fp8.view(torch.uint8)
    uniq, cnts = torch.unique(u, return_counts=True)
    if uniq.numel() > max_pal:
        uf = uniq.view(torch.float8_e4m3fn).float()
        order_val = torch.argsort(uf, descending=True)
        # Always preserve the 8 largest distinct scale values (outlier blocks) + highest-energy remaining.
        n_top = min(8, max_pal)
        keep_top = order_val[:n_top]
        rest = order_val[n_top:]
        rest_score = cnts[rest].float() * (uf[rest] * uf[rest] + 1e-6)
        keep_rest = rest[torch.argsort(rest_score, descending=True)[: max_pal - n_top]]
        chosen = uniq[torch.cat((keep_top, keep_rest))]
        chosen_f = chosen.view(torch.float8_e4m3fn).float()
        sort_idx = torch.argsort(chosen_f)
        chosen = chosen[sort_idx]
        chosen_f = chosen_f[sort_idx]
        all_u = torch.arange(256, dtype=torch.uint8, device=dev)
        all_f = all_u.view(torch.float8_e4m3fn).float()
        all_f = torch.where(torch.isfinite(all_f) & (all_f >= 0), all_f, torch.zeros_like(all_f))
        nearest = torch.argmin((all_f[:, None] - chosen_f[None, :]).abs(), dim=1)
        lut = chosen[nearest]
        sc_fp8 = lut[u.long()].view(torch.float8_e4m3fn)

    # Pass 2: quantize weights to E2M1 nibbles packed into uint8 [N, K//2].
    fp4_u8 = torch.empty((n, k // 2), dtype=torch.uint8, device=dev)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        wf = w[r0:r1].float().view(r1 - r0, g_k, 16)
        sc_real = (sc_fp8[r0:r1].float() * gs).clamp_min(1e-30)
        scaled = wf / sc_real.unsqueeze(-1)
        sign = (scaled < 0).to(torch.uint8) << 3
        a = scaled.abs()
        code = (
            (a >= 0.25).to(torch.uint8)
            + (a >= 0.75).to(torch.uint8)
            + (a >= 1.25).to(torch.uint8)
            + (a >= 1.75).to(torch.uint8)
            + (a >= 2.50).to(torch.uint8)
            + (a >= 3.50).to(torch.uint8)
            + (a >= 5.00).to(torch.uint8)
        )
        nib = (sign | code).view(r1 - r0, k // 2, 2)
        fp4_u8[r0:r1] = nib[..., 0] | (nib[..., 1] << 4)
        del wf, sc_real, scaled, sign, a, code, nib

    perm = torch.empty(0, dtype=torch.int, device=dev)
    qw = ops.gptq_marlin_repack(
        b_q_weight=fp4_u8.view(torch.int32).T.contiguous(),
        perm=perm,
        size_k=k,
        size_n=n,
        num_bits=4,
    )
    del fp4_u8

    ms = marlin_permute_scales(
        s=sc_fp8.T.contiguous().to(torch.bfloat16),
        size_k=k,
        size_n=n,
        group_size=16,
    )
    del sc_fp8
    ms, sf = nvfp4_marlin_process_scales(ms, a_dtype=torch.bfloat16)
    mgs = nvfp4_marlin_process_global_scale(gs.view(1).to(torch.float32), torch.bfloat16) / sf

    # Tile + build decode6 TMA4-ILV layout.
    wt = qw.reshape(k // 16, n // 128, 256).transpose(0, 1).contiguous()
    st = ms.reshape(k // 16, n // 128, 128).transpose(0, 1).contiguous()
    packed6 = d6.pack_scales6(st)
    if packed6 is None:
        raise RuntimeError("pack_scales6 returned None (>64 distinct scales)")
    _, pal = packed6
    w_ilv = d6._build_ilv_weight(wt, n, k, False, 48, split2=False)
    del wt
    coded = d6.encode_tma4_scales(st, pal, n, k, False)
    del st
    ws = marlin_make_workspace_new(dev)
    return qw, ms, mgs, w_ilv, coded, pal, ws


def _run_nvfp4_tma4_into(
    x2: torch.Tensor,
    w_ilv: torch.Tensor,
    coded: torch.Tensor,
    pal: torch.Tensor,
    mgs: torch.Tensor,
    ws: torch.Tensor,
    out: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    import muse_marlin6_gemv as d6

    if not x2.is_contiguous() or x2.data_ptr() % 16:
        x2 = x2.clone(memory_format=torch.contiguous_format)
    m = int(x2.shape[0])
    partials = _get_partials(x2.device)
    st = d6._sk_tma4_ilv_m(
        x2.data_ptr(),
        w_ilv.data_ptr(),
        out.data_ptr(),
        partials.data_ptr(),
        coded.data_ptr(),
        pal.data_ptr(),
        mgs.data_ptr(),
        ws.data_ptr(),
        m,
        n,
        k,
        48,
        x2.device.index,
        torch.cuda.current_stream(x2.device).cuda_stream,
    )
    if st:
        raise RuntimeError(f"muse_marlin6_sk_tma4_ilv_m failed with CUDA error {st}")
    return out


def _run_nvfp4_tma4(
    x2: torch.Tensor,
    w_ilv: torch.Tensor,
    coded: torch.Tensor,
    pal: torch.Tensor,
    mgs: torch.Tensor,
    ws: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    out = x2.new_empty((int(x2.shape[0]), n))
    return _run_nvfp4_tma4_into(x2, w_ilv, coded, pal, mgs, ws, out, n, k)


def _check_nvfp4(w: torch.Tensor, qw, ms, mgs, w_ilv, coded, pal, ws) -> bool:
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    n, k = w.shape
    gen = torch.Generator(device=w.device)
    gen.manual_seed(2027 + n + 13 * k)
    try:
        for m in (1, 7, 8):
            x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
            got = _run_nvfp4_tma4(x, w_ilv, coded, pal, mgs, ws, n, k)
            if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
                return False
            ref = ops.marlin_gemm(
                a=x,
                c=None,
                b_q_weight=qw,
                b_bias=None,
                b_scales=ms,
                a_scales=None,
                global_scale=mgs,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=ws,
                b_q_type=scalar_types.float4_e2m1f,
                size_m=m,
                size_n=n,
                size_k=k,
                use_atomic_add=False,
                use_fp32_reduce=True,
            )
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                max_diff = (got.float() - ref.float()).abs().max().item()
                if max_diff > 1e-3:
                    logger.warning("muse_draft_nvfp4: tma4 vs marlin diff %.5f on (%d,%d) M=%d", max_diff, n, k, m)
                    return False
            dense = F.linear(x, w).float()
            cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
            if not cos >= 0.985:
                logger.warning("muse_draft_nvfp4: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
                return False
        if int(torch.count_nonzero(ws)) != 0:
            return False
        torch.cuda.synchronize(w.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_nvfp4: check failed (%s); falling back", e)
        return False
    return True


def _try_nvfp4_tma4_buffers(w: torch.Tensor):
    import muse_marlin6_gemv as d6

    if d6._sk_tma4_ilv_m is None:
        return None
    n, k = w.shape
    if n % 128 != 0 or k % 128 != 0:
        return None
    if torch.cuda.get_device_capability(w.device) != (12, 1):
        return None
    if torch.cuda.get_device_properties(w.device).multi_processor_count != 48:
        return None
    last_err = None
    for max_pal in (64, 48, 32):
        try:
            qw, ms, mgs, w_ilv, coded, pal, ws = _quantize_nvfp4_tma4(w.contiguous(), max_pal=max_pal)
            break
        except ValueError as e:
            last_err = e
    else:
        raise last_err if last_err is not None else RuntimeError("NVFP4 TMA4 scale encoding failed")
    ok = _check_nvfp4(w, qw, ms, mgs, w_ilv, coded, pal, ws)
    del qw, ms
    if not ok:
        return None
    return w_ilv, coded, pal, mgs, ws


# ---------------------------------------------------------------------------
# INT3 W3A16 (asymmetric group-64 via muse_bfx_ext.int3_gemm)
# ---------------------------------------------------------------------------

def _try_int3_buffer(w: torch.Tensor) -> torch.Tensor | None:
    import muse_bfx_ext as _bfx

    if not hasattr(_bfx, "int3_pack") or not hasattr(_bfx, "int3_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    blob = _bfx.int3_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(3037 + n + 17 * k)
    for m in (1, 7, 8):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = _bfx.int3_gemm(x, blob, n, k)
        if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w_c).float()
        cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.96:
            logger.warning("muse_draft_int3: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
            return None
    torch.cuda.synchronize(w.device)
    return blob


def _try_int4_buffer(w: torch.Tensor) -> torch.Tensor | None:
    bfx = _get_bfx()
    if bfx is None or not hasattr(bfx, "int4_pack") or not hasattr(bfx, "int4_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    blob = bfx.int4_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(4051 + n + 23 * k)
    for m in (1, 7, 8):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = bfx.int4_gemm(x, blob, n, k)
        if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w_c).float()
        cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.99:
            logger.warning("muse_draft_int4: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
            return None
    torch.cuda.synchronize(w.device)
    return blob


def _free_bf16_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_FREE_BF16", "1") != "0"


def _m512_fp8_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_M512_FP8", "1") != "0"


def _pack_fp8_large_m(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack BF16 [N, K] into column-major FP8 [K, N] + per-channel fp32 scale [1, N] for M > 8 cutlass_scaled_mm."""
    import vllm._custom_ops as ops

    n, k = w.shape
    if n % 16 != 0 or k % 16 != 0:
        return None
    q, s = _quantize_rows_fp8(w)
    b_t = q.t()  # [K, N], column-major stride (1, K), zero-copy view over row-major q [N, K]
    sb = s.view(1, n).contiguous()
    gen = torch.Generator(device=w.device)
    gen.manual_seed(4049 + n + 19 * k)
    try:
        x = (torch.randn((16, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        got = ops.cutlass_scaled_mm(xq, b_t, scale_a=xs, scale_b=sb, out_dtype=w.dtype)
        if got.shape != (16, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w).float()
        cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.995:
            logger.warning("muse_draft_m512_fp8: low cosine %.5f on (%d,%d)", cos, n, k)
            return None
        torch.cuda.synchronize(w.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_m512_fp8: check failed on (%d,%d) (%s); keeping BF16", n, k, e)
        return None
    return b_t, sb


def _run_fp8_large_m(x2: torch.Tensor, b_t: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    import vllm._custom_ops as ops

    xq, xs = ops.scaled_fp8_quant(x2.contiguous(), use_per_token_if_dynamic=True)
    return ops.cutlass_scaled_mm(xq, b_t, scale_a=xs, scale_b=sb, out_dtype=x2.dtype)


# ---------------------------------------------------------------------------
# Attachment (modules + fused KV tensor) and custom ops
# ---------------------------------------------------------------------------

def attach_linear(
    layer: torch.nn.Module,
    mode: str = "int3",
    *,
    free_bf16: bool = False,
    attach_m512_fp8: bool = False,
) -> bool:
    w = getattr(layer, "weight", None)
    if (
        w is None
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "_mfp8", None) is not None
        or getattr(layer, "_mnv4", None) is not None
        or getattr(layer, "_mi4", None) is not None
        or getattr(layer, "_mi3", None) is not None
        or not isinstance(w, torch.Tensor)
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_cuda
    ):
        return False
    n, k = w.shape
    if n % 64 or k % 128:
        return False

    admitted = False
    if mode == "int4":
        try:
            blob = _try_int4_buffer(w.data)
            if blob is not None:
                layer.register_buffer("_mi4_blob", blob, persistent=False)
                layer._mi4 = (n, k)
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                _stats["admitted_int4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += blob.numel() * blob.element_size()
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int4: attach failed for %s (%s); trying NVFP4", tuple(w.shape), e)

    if not admitted and mode == "int3":
        try:
            blob = _try_int3_buffer(w.data)
            if blob is not None:
                layer.register_buffer("_mi3_blob", blob, persistent=False)
                layer._mi3 = (n, k)
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                _stats["admitted_int3"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += blob.numel() * blob.element_size()
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int3: attach failed for %s (%s); trying NVFP4", tuple(w.shape), e)

    if not admitted and mode in ("int4", "int3", "nvfp4"):
        try:
            res = _try_nvfp4_tma4_buffers(w.data)
            if res is not None:
                w_ilv, coded, pal, mgs, ws = res
                layer.register_buffer("_mnv4_w_ilv", w_ilv, persistent=False)
                layer.register_buffer("_mnv4_coded", coded, persistent=False)
                layer.register_buffer("_mnv4_pal", pal, persistent=False)
                layer.register_buffer("_mnv4_gs", mgs, persistent=False)
                layer.register_buffer("_mnv4_ws", ws, persistent=False)
                layer._mnv4 = (n, k)
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                _stats["admitted_nvfp4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += (
                    w_ilv.numel() * w_ilv.element_size() + coded.numel() * coded.element_size()
                )
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_nvfp4: attach failed for %s (%s); trying FP8", tuple(w.shape), e)

    if not admitted:
        try:
            q, s, qweight, scales, ws = _pack_fp8(w.data.contiguous())
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_fp8: pack failed for %s (%s)", tuple(w.shape), e)
            _stats["rejected"] += 1
            return False
        if not _check_fp8(w.data, q, s, qweight, scales, ws):
            _stats["rejected"] += 1
            return False
        del q, s
        layer.register_buffer("_mfp8_qweight", qweight, persistent=False)
        layer.register_buffer("_mfp8_scales", scales, persistent=False)
        layer.register_buffer("_mfp8_ws", ws, persistent=False)
        layer._mfp8 = (n, k)
        _stats["admitted_fp8"] += 1
        _stats["bytes_bf16"] += n * k * 2
        _stats["bytes_quant"] += qweight.numel() * qweight.element_size()
        admitted = True

    has_m512_fp8 = False
    if attach_m512_fp8 and _m512_fp8_enabled():
        fp8_lm = _pack_fp8_large_m(w.data.contiguous())
        if fp8_lm is not None:
            b_t, sb = fp8_lm
            layer.register_buffer("_m512_fp8_bt", b_t, persistent=False)
            layer.register_buffer("_m512_fp8_sb", sb, persistent=False)
            _stats["admitted_m512_fp8"] += 1
            has_m512_fp8 = True

    if free_bf16 and _free_bf16_enabled() and (not attach_m512_fp8 or has_m512_fp8):
        bf16_bytes = w.numel() * w.element_size()
        w.data = torch.empty((0,), dtype=w.dtype, device=w.device)
        _stats["bytes_freed"] += bf16_bytes
    return True


def attach_tensor(
    mod: torch.nn.Module,
    prefix: str,
    w: torch.Tensor,
    mode: str = "int3",
    *,
    attach_m512_fp8: bool = False,
) -> bool:
    if not enabled() or w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_cuda:
        return False
    n, k = w.shape
    if n % 64 or k % 128:
        return False
    admitted = False
    if mode == "int4":
        try:
            blob = _try_int4_buffer(w.data)
            if blob is not None:
                mod.register_buffer(f"_{prefix}_mi4_blob", blob, persistent=False)
                setattr(mod, f"_{prefix}_mi4", (n, k))
                _stats["admitted_int4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += blob.numel() * blob.element_size()
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int4: tensor %s attach failed (%s); trying NVFP4", prefix, e)

    if not admitted and mode == "int3":
        try:
            blob = _try_int3_buffer(w.data)
            if blob is not None:
                mod.register_buffer(f"_{prefix}_mi3_blob", blob, persistent=False)
                setattr(mod, f"_{prefix}_mi3", (n, k))
                _stats["admitted_int3"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += blob.numel() * blob.element_size()
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int3: tensor %s attach failed (%s); trying NVFP4", prefix, e)

    if not admitted and mode in ("int4", "int3", "nvfp4"):
        try:
            res = _try_nvfp4_tma4_buffers(w.data)
            if res is not None:
                w_ilv, coded, pal, mgs, ws = res
                mod.register_buffer(f"_{prefix}_mnv4_w_ilv", w_ilv, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_coded", coded, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_pal", pal, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_gs", mgs, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_ws", ws, persistent=False)
                setattr(mod, f"_{prefix}_mnv4", (n, k))
                _stats["admitted_nvfp4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += (
                    w_ilv.numel() * w_ilv.element_size() + coded.numel() * coded.element_size()
                )
                admitted = True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_nvfp4: tensor %s attach failed (%s); trying FP8", prefix, e)

    if not admitted:
        try:
            q, s, qweight, scales, ws = _pack_fp8(w.data.contiguous())
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_fp8: tensor %s pack failed (%s)", prefix, e)
            _stats["rejected"] += 1
            return False
        if not _check_fp8(w.data, q, s, qweight, scales, ws):
            _stats["rejected"] += 1
            return False
        del q, s
        mod.register_buffer(f"_{prefix}_mfp8_qweight", qweight, persistent=False)
        mod.register_buffer(f"_{prefix}_mfp8_scales", scales, persistent=False)
        mod.register_buffer(f"_{prefix}_mfp8_ws", ws, persistent=False)
        setattr(mod, f"_{prefix}_mfp8", (n, k))
        _stats["admitted_fp8"] += 1
        _stats["bytes_bf16"] += n * k * 2
        _stats["bytes_quant"] += qweight.numel() * qweight.element_size()
        admitted = True

    if attach_m512_fp8 and _m512_fp8_enabled():
        fp8_lm = _pack_fp8_large_m(w.data.contiguous())
        if fp8_lm is not None:
            b_t, sb = fp8_lm
            mod.register_buffer(f"_{prefix}_m512_fp8_bt", b_t, persistent=False)
            mod.register_buffer(f"_{prefix}_m512_fp8_sb", sb, persistent=False)
            _stats["admitted_m512_fp8"] += 1
    return True


@torch.library.custom_op("muse::draft_int4_linear", mutates_args=())
def draft_int4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    blob: torch.Tensor,
    n: int,
    k: int,
    fp8_bt: torch.Tensor | None = None,
    fp8_sb: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = int(x2.size(0))
    if 1 <= m <= MAX_M_NVFP4 and x2.dtype == torch.bfloat16:
        bfx = _get_bfx()
        out = bfx.int4_gemm(x2.contiguous(), blob, n, k)
    elif fp8_bt is not None and fp8_sb is not None and m > 0 and x2.dtype == torch.bfloat16:
        out = _run_fp8_large_m(x2, fp8_bt, fp8_sb)
    elif w.numel() == n * k:
        out = F.linear(x2, w)
    elif m == 0:
        out = x2.new_empty((0, n))
    else:
        bfx = _get_bfx()
        xc = x2.contiguous()
        out = xc.new_empty((m, n))
        for r0 in range(0, m, MAX_M_NVFP4):
            r1 = min(m, r0 + MAX_M_NVFP4)
            bfx.int4_gemm_out(xc[r0:r1], blob, out[r0:r1], n, k)
    return out.reshape(*shape[:-1], n)


@draft_int4_linear.register_fake
def _(x, w, blob, n, k, fp8_bt=None, fp8_sb=None):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::draft_int3_linear", mutates_args=())
def draft_int3_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    blob: torch.Tensor,
    n: int,
    k: int,
    fp8_bt: torch.Tensor | None = None,
    fp8_sb: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = int(x2.size(0))
    if 1 <= m <= MAX_M_NVFP4 and x2.dtype == torch.bfloat16:
        bfx = _get_bfx()
        out = bfx.int3_gemm(x2.contiguous(), blob, n, k)
    elif fp8_bt is not None and fp8_sb is not None and m > 0 and x2.dtype == torch.bfloat16:
        out = _run_fp8_large_m(x2, fp8_bt, fp8_sb)
    elif w.numel() == n * k:
        out = F.linear(x2, w)
    elif m == 0:
        out = x2.new_empty((0, n))
    else:
        bfx = _get_bfx()
        xc = x2.contiguous()
        out = xc.new_empty((m, n))
        for r0 in range(0, m, MAX_M_NVFP4):
            r1 = min(m, r0 + MAX_M_NVFP4)
            bfx.int3_gemm_out(xc[r0:r1], blob, out[r0:r1], n, k)
    return out.reshape(*shape[:-1], n)


@draft_int3_linear.register_fake
def _(x, w, blob, n, k, fp8_bt=None, fp8_sb=None):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::draft_fp8_linear", mutates_args=())
def draft_fp8_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    ws: torch.Tensor,
    n: int,
    k: int,
    fp8_bt: torch.Tensor | None = None,
    fp8_sb: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = int(x2.size(0))
    if 1 <= m <= MAX_M_FP8 and x2.dtype == torch.bfloat16:
        out = _marlin_fp8(x2.contiguous(), qweight, scales, ws, n, k)
    elif fp8_bt is not None and fp8_sb is not None and m > 0 and x2.dtype == torch.bfloat16:
        out = _run_fp8_large_m(x2, fp8_bt, fp8_sb)
    elif w.numel() == n * k:
        out = F.linear(x2, w)
    elif m == 0:
        out = x2.new_empty((0, n))
    else:
        xc = x2.contiguous()
        out = xc.new_empty((m, n))
        for r0 in range(0, m, MAX_M_FP8):
            r1 = min(m, r0 + MAX_M_FP8)
            out[r0:r1] = _marlin_fp8(xc[r0:r1], qweight, scales, ws, n, k)
    return out.reshape(*shape[:-1], n)


@draft_fp8_linear.register_fake
def _(x, w, qweight, scales, ws, n, k, fp8_bt=None, fp8_sb=None):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::draft_nvfp4_linear", mutates_args=())
def draft_nvfp4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_ilv: torch.Tensor,
    coded: torch.Tensor,
    pal: torch.Tensor,
    mgs: torch.Tensor,
    ws: torch.Tensor,
    n: int,
    k: int,
    fp8_bt: torch.Tensor | None = None,
    fp8_sb: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = int(x2.size(0))
    if 1 <= m <= MAX_M_NVFP4 and x2.dtype == torch.bfloat16:
        out = _run_nvfp4_tma4(x2, w_ilv, coded, pal, mgs, ws, n, k)
    elif fp8_bt is not None and fp8_sb is not None and m > 0 and x2.dtype == torch.bfloat16:
        out = _run_fp8_large_m(x2, fp8_bt, fp8_sb)
    elif w.numel() == n * k:
        out = F.linear(x2, w)
    elif m == 0:
        out = x2.new_empty((0, n))
    else:
        xc = x2.contiguous()
        out = xc.new_empty((m, n))
        for r0 in range(0, m, MAX_M_NVFP4):
            r1 = min(m, r0 + MAX_M_NVFP4)
            _run_nvfp4_tma4_into(xc[r0:r1], w_ilv, coded, pal, mgs, ws, out[r0:r1], n, k)
    return out.reshape(*shape[:-1], n)


@draft_nvfp4_linear.register_fake
def _(x, w, w_ilv, coded, pal, mgs, ws, n, k, fp8_bt=None, fp8_sb=None):
    return x.new_empty((*x.shape[:-1], n))


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    fp8_bt = getattr(layer, "_m512_fp8_bt", None)
    fp8_sb = getattr(layer, "_m512_fp8_sb", None)
    mi4 = getattr(layer, "_mi4", None)
    if mi4 is not None:
        return torch.ops.muse.draft_int4_linear(
            x, layer.weight, layer._mi4_blob, mi4[0], mi4[1], fp8_bt, fp8_sb
        )
    mi3 = getattr(layer, "_mi3", None)
    if mi3 is not None:
        return torch.ops.muse.draft_int3_linear(
            x, layer.weight, layer._mi3_blob, mi3[0], mi3[1], fp8_bt, fp8_sb
        )
    mnv4 = getattr(layer, "_mnv4", None)
    if mnv4 is not None:
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            layer.weight,
            layer._mnv4_w_ilv,
            layer._mnv4_coded,
            layer._mnv4_pal,
            layer._mnv4_gs,
            layer._mnv4_ws,
            mnv4[0],
            mnv4[1],
            fp8_bt,
            fp8_sb,
        )
    mfp8 = layer._mfp8
    return torch.ops.muse.draft_fp8_linear(
        x,
        layer.weight,
        layer._mfp8_qweight,
        layer._mfp8_scales,
        layer._mfp8_ws,
        mfp8[0],
        mfp8[1],
        fp8_bt,
        fp8_sb,
    )


def tensor_apply(mod: torch.nn.Module, prefix: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    fp8_bt = getattr(mod, f"_{prefix}_m512_fp8_bt", None)
    fp8_sb = getattr(mod, f"_{prefix}_m512_fp8_sb", None)
    mi4 = getattr(mod, f"_{prefix}_mi4", None)
    if mi4 is not None:
        return torch.ops.muse.draft_int4_linear(
            x, w, getattr(mod, f"_{prefix}_mi4_blob"), mi4[0], mi4[1], fp8_bt, fp8_sb
        )
    mi3 = getattr(mod, f"_{prefix}_mi3", None)
    if mi3 is not None:
        return torch.ops.muse.draft_int3_linear(
            x, w, getattr(mod, f"_{prefix}_mi3_blob"), mi3[0], mi3[1], fp8_bt, fp8_sb
        )
    mnv4 = getattr(mod, f"_{prefix}_mnv4", None)
    if mnv4 is not None:
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mnv4_w_ilv"),
            getattr(mod, f"_{prefix}_mnv4_coded"),
            getattr(mod, f"_{prefix}_mnv4_pal"),
            getattr(mod, f"_{prefix}_mnv4_gs"),
            getattr(mod, f"_{prefix}_mnv4_ws"),
            mnv4[0],
            mnv4[1],
            fp8_bt,
            fp8_sb,
        )
    mfp8 = getattr(mod, f"_{prefix}_mfp8", None)
    if mfp8 is not None:
        return torch.ops.muse.draft_fp8_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mfp8_qweight"),
            getattr(mod, f"_{prefix}_mfp8_scales"),
            getattr(mod, f"_{prefix}_mfp8_ws"),
            mfp8[0],
            mfp8[1],
            fp8_bt,
            fp8_sb,
        )
    return None


def has_tensor(mod: torch.nn.Module, prefix: str) -> bool:
    return (
        getattr(mod, f"_{prefix}_mi4", None) is not None
        or getattr(mod, f"_{prefix}_mi3", None) is not None
        or getattr(mod, f"_{prefix}_mnv4", None) is not None
        or getattr(mod, f"_{prefix}_mfp8", None) is not None
    )


def tensor_apply_into(mod: torch.nn.Module, prefix: str, x: torch.Tensor, out: torch.Tensor) -> bool:
    if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    m = int(x.size(0))
    if m == 0:
        return True
    if m > MAX_M_NVFP4:
        return False
    mi4 = getattr(mod, f"_{prefix}_mi4", None)
    if mi4 is not None:
        bfx = _get_bfx()
        if bfx is None or not hasattr(bfx, "int4_gemm_out"):
            return False
        n, k = mi4
        if x.size(1) != k or out.shape != (m, n):
            return False
        bfx.int4_gemm_out(x.contiguous(), getattr(mod, f"_{prefix}_mi4_blob"), out, n, k)
        return True
    mi3 = getattr(mod, f"_{prefix}_mi3", None)
    if mi3 is not None:
        bfx = _get_bfx()
        if bfx is None or not hasattr(bfx, "int3_gemm_out"):
            return False
        n, k = mi3
        if x.size(1) != k or out.shape != (m, n):
            return False
        bfx.int3_gemm_out(x.contiguous(), getattr(mod, f"_{prefix}_mi3_blob"), out, n, k)
        return True
    mnv4 = getattr(mod, f"_{prefix}_mnv4", None)
    if mnv4 is not None:
        n, k = mnv4
        if x.size(1) != k or out.shape != (m, n):
            return False
        _run_nvfp4_tma4_into(
            x,
            getattr(mod, f"_{prefix}_mnv4_w_ilv"),
            getattr(mod, f"_{prefix}_mnv4_coded"),
            getattr(mod, f"_{prefix}_mnv4_pal"),
            getattr(mod, f"_{prefix}_mnv4_gs"),
            getattr(mod, f"_{prefix}_mnv4_ws"),
            out,
            n,
            k,
        )
        return True
    return False


def fc_aux5_apply(layer: torch.nn.Module, aux_list: list[torch.Tensor], out: torch.Tensor) -> bool:
    mi4 = getattr(layer, "_mi4", None)
    mi3 = getattr(layer, "_mi3", None)
    bfx = _get_bfx()
    if (
        (mi4 is None and mi3 is None)
        or bfx is None
        or len(aux_list) != 5
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        return False
    if mi4 is not None:
        if not hasattr(bfx, "int4_gemm_aux5_out"):
            return False
        n, k = mi4
        blob = layer._mi4_blob
        fn = bfx.int4_gemm_aux5_out
    else:
        if not hasattr(bfx, "int3_gemm_aux5_out"):
            return False
        n, k = mi3
        blob = layer._mi3_blob
        fn = bfx.int3_gemm_aux5_out
    m = int(out.size(0))
    if m == 0:
        return True
    if m > MAX_M_NVFP4 or out.size(1) != n or k % 5 != 0:
        return False
    k_each = k // 5
    if k_each % 64 != 0:
        return False
    sliced: list[torch.Tensor] = []
    for a in aux_list:
        if a.dtype != torch.bfloat16 or a.dim() != 2 or a.size(0) < m or a.size(1) != k_each:
            return False
        s = a[:m]
        if not s.is_contiguous() or (s.data_ptr() & 15) != 0:
            return False
        sliced.append(s)
    fn(sliced[0], sliced[1], sliced[2], sliced[3], sliced[4], blob, out, n, k_each)
    return True


def dflash_kv_post(
    all_kv_flat: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    kv_ptrs: list[int],
    block_size: int,
    stride_block: int,
    stride_kv_head: int,
    stride_slot: int,
    eps: float,
) -> bool:
    bfx = _get_bfx()
    if bfx is None or not hasattr(bfx, "dflash_kv_post"):
        return False
    bfx.dflash_kv_post(
        all_kv_flat,
        k_norm_weights,
        cos_sin_cache,
        positions,
        slot_mapping,
        kv_ptrs,
        block_size,
        stride_block,
        stride_kv_head,
        stride_slot,
        eps,
    )
    return True


def argmax_bf16_into(
    logits: torch.Tensor, sval: torch.Tensor, sidx: torch.Tensor, out_tok: torch.Tensor
) -> bool:
    bfx = _get_bfx()
    if (
        bfx is None
        or not hasattr(bfx, "dflash_argmax_bf16")
        or logits.dtype != torch.bfloat16
        or logits.dim() != 2
        or logits.stride(1) != 1
        or logits.stride(0) < logits.size(1)
        or (logits.stride(0) & 7) != 0
        or (logits.size(1) & 7) != 0
        or (logits.data_ptr() & 15) != 0
        or not (1 <= logits.size(0) <= 8)
        or out_tok.dtype != torch.int64
        or not out_tok.is_contiguous()
        or out_tok.numel() != logits.size(0)
    ):
        return False
    bfx.dflash_argmax_bf16(logits, sval, sidx, out_tok)
    return True


def prepare_draft_lmhead_buffers(
    lm_head: torch.nn.Module, org_vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, int, int] | None:
    try:
        import muse_marlin6_gemv as d6
    except Exception:  # noqa: BLE001
        return None
    if d6._sk_tma4_ilv_m is None:
        return None
    tiled_weight = getattr(lm_head, "_marlin_decode_weight", None)
    scales6 = getattr(lm_head, "_marlin_decode_scales6", None)
    palette = getattr(lm_head, "_marlin_decode_palette", None)
    gs = getattr(lm_head, "weight_global_scale", None)
    ws = getattr(lm_head, "workspace", None)
    n = int(getattr(lm_head, "output_size_per_partition", 0))
    k = int(getattr(lm_head, "input_size_per_partition", 0))
    if (
        tiled_weight is None
        or scales6 is None
        or palette is None
        or gs is None
        or ws is None
        or getattr(lm_head, "tp_size", 1) != 1
        or getattr(lm_head, "bias", None) is not None
        or (n, k) != (d6.LM_HEAD_N, d6.LM_HEAD_K)
        or not d6.is_tma4_ilv(scales6)
        or org_vocab <= 0
        or org_vocab > n
        or (org_vocab & 7) != 0
    ):
        return None
    eff_n = d6.LM_HEAD_PAD128_N if d6.is_lm_pad128(scales6) else n
    if (eff_n & 7) != 0:
        return None
    blocks = 96 if d6.is_lm_split2(scales6) else 48
    tile_n = 128 if eff_n % 128 == 0 else 64
    out_padded = torch.empty((8, eff_n), dtype=torch.bfloat16, device=device)
    partials = torch.empty((blocks, 16, tile_n), dtype=torch.float32, device=device)
    return out_padded, partials, eff_n, k, blocks


def launch_draft_lmhead_raw(
    x: torch.Tensor,
    lm_head: torch.nn.Module,
    org_vocab: int,
    out_padded: torch.Tensor,
    partials: torch.Tensor,
    eff_n: int,
    k: int,
    blocks: int,
) -> torch.Tensor | None:
    import muse_marlin6_gemv as d6

    m = int(x.shape[0])
    if (
        x.ndim != 2
        or not (1 < m <= 8)
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or (x.data_ptr() & 15) != 0
        or out_padded.shape[0] < m
        or out_padded.shape[1] != eff_n
    ):
        return None
    status = d6._sk_tma4_ilv_m(
        x.data_ptr(),
        lm_head._marlin_decode_weight.data_ptr(),
        out_padded.data_ptr(),
        partials.data_ptr(),
        lm_head._marlin_decode_scales6.data_ptr(),
        lm_head._marlin_decode_palette.data_ptr(),
        lm_head.weight_global_scale.data_ptr(),
        lm_head.workspace.data_ptr(),
        m,
        eff_n,
        k,
        blocks,
        x.device.index or 0,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"draft lm_head _sk_tma4_ilv_m failed with CUDA error {status}")
    return out_padded[:m, :org_vocab]


def place_drafter_quant(model: torch.nn.Module) -> int:
    """GB10 UMA DRAM placement for quantized drafter weights (INT3, NVFP4 w_ilv, and FP8 qweight)."""
    if os.environ.get("MUSE_DRAFT_PLACEMENT", "1") == "0":
        return 0
    groups: dict[tuple[str, int, int], list[tuple[torch.nn.Module, str, tuple]]] = {}
    for _, mod in model.named_modules():
        if getattr(mod, "_mi4", None) is not None:
            n, k = mod._mi4
            groups.setdefault(("int4", n, k), []).append((mod, "_mi4_blob", ()))
        elif getattr(mod, "_mi3", None) is not None:
            n, k = mod._mi3
            groups.setdefault(("int3", n, k), []).append((mod, "_mi3_blob", ()))
        elif getattr(mod, "_mnv4", None) is not None:
            n, k = mod._mnv4
            groups.setdefault(("nvfp4", n, k), []).append(
                (mod, "_mnv4_w_ilv", (mod._mnv4_coded, mod._mnv4_pal, mod._mnv4_gs, mod._mnv4_ws))
            )
        elif getattr(mod, "_mfp8_qweight", None) is not None:
            n, k = mod._mfp8
            groups.setdefault(("fp8", n, k), []).append(
                (mod, "_mfp8_qweight", (mod._mfp8_scales, mod._mfp8_ws))
            )
    if getattr(model, "_fkv_mi4", None) is not None:
        n, k = model._fkv_mi4
        groups.setdefault(("int4", n, k), []).append((model, "_fkv_mi4_blob", ()))
    elif getattr(model, "_fkv_mi3", None) is not None:
        n, k = model._fkv_mi3
        groups.setdefault(("int3", n, k), []).append((model, "_fkv_mi3_blob", ()))
    elif getattr(model, "_fkv_mnv4", None) is not None:
        n, k = model._fkv_mnv4
        groups.setdefault(("nvfp4", n, k), []).append(
            (model, "_fkv_mnv4_w_ilv", (model._fkv_mnv4_coded, model._fkv_mnv4_pal, model._fkv_mnv4_gs, model._fkv_mnv4_ws))
        )
    elif getattr(model, "_fkv_mfp8_qweight", None) is not None:
        n, k = model._fkv_mfp8
        groups.setdefault(("fp8", n, k), []).append(
            (model, "_fkv_mfp8_qweight", (model._fkv_mfp8_scales, model._fkv_mfp8_ws))
        )
    if not groups:
        return 0

    t0 = time.perf_counter()
    dev = next(model.parameters()).device
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=dev)
    total_moved = 0
    total_saved = 0.0

    for (kind, n, k), items in groups.items():
        first_buf = getattr(items[0][0], items[0][1])
        x = torch.randn((7, k), dtype=torch.bfloat16, device=dev)
        n_spares = min(len(items) * 2 + 2, 16)
        spares = []
        try:
            for _ in range(n_spares):
                spares.append(torch.empty_like(first_buf))
        except torch.cuda.OutOfMemoryError:
            pass
        if not spares:
            continue

        def _time_buf(buf: torch.Tensor, meta: tuple) -> float:
            if kind == "int4":
                bfx = _get_bfx()
                run = lambda: bfx.int4_gemm(x, buf, n, k)
            elif kind == "int3":
                bfx = _get_bfx()
                run = lambda: bfx.int3_gemm(x, buf, n, k)
            elif kind == "nvfp4":
                coded, pal, mgs, ws = meta
                run = lambda: _run_nvfp4_tma4(x, buf, coded, pal, mgs, ws, n, k)
            else:
                scales, ws = meta
                run = lambda: _marlin_fp8(x, buf, scales, ws, n, k)
            flush.zero_()
            run()
            torch.cuda.synchronize(dev)
            ts = []
            for _ in range(5):
                flush.zero_()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                run()
                e_ev.record()
                torch.cuda.synchronize(dev)
                ts.append(s_ev.elapsed_time(e_ev))
            ts.sort()
            return ts[2]

        meta0 = items[0][2]
        spares_t = []
        for sp in spares:
            sp.copy_(first_buf)
            spares_t.append((_time_buf(sp, meta0), sp))
        spares_t.sort(key=lambda pair: pair[0])

        item_t = []
        for mod, attr, meta in items:
            buf = getattr(mod, attr)
            item_t.append((_time_buf(buf, meta), mod, attr, meta))
        item_t.sort(key=lambda tup: tup[0], reverse=True)

        sp_idx = 0
        for cur_ms, mod, attr, meta in item_t:
            if sp_idx >= len(spares_t):
                break
            sp_ms, sp = spares_t[sp_idx]
            if cur_ms - sp_ms < 0.005:
                break
            buf = getattr(mod, attr)
            sp.copy_(buf)
            new_ms = _time_buf(sp, meta)
            if new_ms < cur_ms - 0.003:
                mod.register_buffer(attr, sp, persistent=False)
                total_moved += 1
                total_saved += cur_ms - new_ms
                sp_idx += 1
        del spares, spares_t

    del flush
    torch.cuda.empty_cache()
    logger.info(
        "Drafter quantized DRAM placement done: %d moved, saved %.3f ms/step in %.2f s",
        total_moved,
        total_saved,
        time.perf_counter() - t0,
    )
    return total_moved


def attach_drafter(model: torch.nn.Module) -> int:
    if not enabled():
        return 0
    mode = _quant_mode()
    n = 0
    by_leaf: dict[str, list[torch.nn.Module]] = {s: [] for s in _NAMES}
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in by_leaf:
            by_leaf[leaf].append(mod)

    def _pick_mode(leaf: str) -> str:
        if mode == "hybrid_int4_dn":
            return "int4" if leaf in _ATTN_FC_DN_NAMES else "int3"
        if mode == "hybrid_int4_fc":
            return "int4" if leaf in _ATTN_FC_NAMES else "int3"
        if mode == "hybrid_int4":
            return "int4" if leaf in _ATTN_NAMES else "int3"
        if mode == "hybrid_nvfp4":
            return "nvfp4" if leaf in _ATTN_NAMES else "int3"
        if mode == "int4":
            return "int4"
        if mode == "int3":
            return "int3"
        if mode == "nvfp4":
            return "nvfp4"
        if mode == "hybrid":
            return "nvfp4" if leaf in _MLP_FC_NAMES else "fp8"
        return "fp8"

    # Pack and release decoder-only BF16 weights smallest-to-largest first to free ~4.54 GB
    # before packing fc/fkv and running GB10 UMA DRAM placement.
    for leaf in _DECODER_NAMES:
        lm = _pick_mode(leaf)
        for mod in by_leaf[leaf]:
            if attach_linear(mod, mode=lm, free_bf16=True, attach_m512_fp8=False):
                n += 1
                if _free_bf16_enabled():
                    torch.cuda.empty_cache()

    for mod in by_leaf["fc"]:
        if attach_linear(mod, mode=_pick_mode("fc"), free_bf16=True, attach_m512_fp8=True):
            n += 1
            if _free_bf16_enabled():
                torch.cuda.empty_cache()

    fkv_w = getattr(model, "_fused_kv_weight", None)
    if fkv_w is not None and getattr(model, "_fused_kv_bias", None) is None:
        if mode in ("hybrid_int4_dn", "hybrid_int4_fc", "hybrid_int4", "int4"):
            fkv_mode = "int4"
        elif mode in ("hybrid_nvfp4", "nvfp4"):
            fkv_mode = "nvfp4"
        elif mode == "int3":
            fkv_mode = "int3"
        else:
            fkv_mode = "fp8"
        if attach_tensor(model, "fkv", fkv_w, mode=fkv_mode, attach_m512_fp8=True):
            n += 1
            if _free_bf16_enabled() and getattr(model, "_fkv_m512_fp8_bt", None) is not None:
                bf16_bytes = fkv_w.numel() * fkv_w.element_size()
                model._fused_kv_weight = torch.empty((0,), dtype=fkv_w.dtype, device=fkv_w.device)
                _stats["bytes_freed"] += bf16_bytes
                del fkv_w
                torch.cuda.empty_cache()

    place_drafter_quant(model)
    logger.info(
        "muse_draft_quant (%s): %d INT4-g64 + %d INT3-g64 + %d NVFP4-TMA4 + %d FP8-Marlin drafter linears "
        "(%d M>8 FP8, %d rejected), %.2f GB -> %.2f GB/step, %.2f GB BF16 freed",
        mode,
        _stats["admitted_int4"],
        _stats["admitted_int3"],
        _stats["admitted_nvfp4"],
        _stats["admitted_fp8"],
        _stats["admitted_m512_fp8"],
        _stats["rejected"],
        _stats["bytes_bf16"] / 1e9,
        _stats["bytes_quant"] / 1e9,
        _stats["bytes_freed"] / 1e9,
    )
    return n
