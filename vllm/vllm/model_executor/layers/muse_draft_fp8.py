# SPDX-License-Identifier: Apache-2.0
"""Weight-only quantized copies and fused inner-loop ops for the pinned DFlash drafter.

The drafter only PROPOSES tokens: every committed token is the target's own greedy token from the
verify step, and the per-depth oracles equal the serial tape. The drafter's arithmetic can therefore
change acceptance, never the emitted stream.

Default quantization (`MUSE_DRAFT_QUANT=hybrid_int4_dn`):
  - `qkv_proj`, `o_proj`, `down_proj`, `fc`, `fkv`: group-64 W4A16 (`int4_mma`, MSE/ALS packed, M <= 16).
  - `gate_up_proj`: paired group-64 W3A16 + fused SwiGLU epilogue (`int3_swiglu_gemm`, 0-ULP verified vs
    unfused `int3_gemm` + `ops.silu_and_mul`, MSE/ALS packed, M <= 16).
  - `M > 16` context calls (`fc`, `fkv` on 512-token prompt prefill): dynamic per-token W8A8 FP8 CUTLASS.
  - Per-layer query QKV split + weighted Q/K RMSNorm + NEOX RoPE + paged BF16 KV-cache scatter fused into
    one launch (`muse::dflash_qkv_norm_rope`, 0-ULP verified at load).
"""
import math
import os
import time

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M_FP8 = 16
MAX_M_INT = 16
MAX_M_NVFP4 = 8
_DECODER_NAMES = ("o_proj", "qkv_proj", "down_proj", "gate_up_proj")
_NAMES = (*_DECODER_NAMES, "fc")
_MLP_FC_NAMES = ("gate_up_proj", "down_proj", "fc")
_ATTN_NAMES = ("qkv_proj", "o_proj")
_ATTN_FC_NAMES = ("qkv_proj", "o_proj", "fc")
_ATTN_FC_DN_NAMES = ("qkv_proj", "o_proj", "down_proj", "fc")
_stats = {
    "admitted_int3": 0,
    "admitted_int3_swiglu": 0,
    "admitted_int4": 0,
    "admitted_nvfp4": 0,
    "admitted_fp8": 0,
    "admitted_m512_fp8": 0,
    "admitted_qkv_fused": 0,
    "rejected": 0,
    "bytes_bf16": 0,
    "bytes_quant": 0,
    "bytes_freed": 0,
}

_SHARED_PARTIALS: dict[int, torch.Tensor] = {}
_BFX_EXT = None
_BFX_TRIED = False
_QKV_FUSED_VERIFIED: dict[int, bool] = {}


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


def _int3_mse_enabled() -> bool:
    return os.environ.get("MUSE_INT3_MSE_PACK", "0") != "0"


def _int4_mse_enabled() -> bool:
    return os.environ.get("MUSE_INT4_MSE_PACK", "1") != "0"


def _mlp_swiglu_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_MLP_SWIGLU", "0") != "0"


def _qkv_fused_enabled() -> bool:
    return os.environ.get("MUSE_DFLASH_QKV_FUSED", "1") != "0"


def _snr_db(dense: torch.Tensor, got: torch.Tensor) -> float:
    sig = float((dense * dense).sum().item())
    diff = dense - got
    err = float((diff * diff).sum().item())
    if err <= 1e-30:
        return 99.0
    if sig <= 1e-30:
        return 0.0
    return 10.0 * math.log10(sig / err)


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
    n, k = w.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n,), dtype=torch.float32, device=w.device)
    step = max(1, (32 << 20) // k)
    for r0 in range(0, n, step):
        wf = w[r0 : r0 + step].float()
        sc = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-30)
        q[r0 : r0 + step] = (wf / sc[:, None]).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        s[r0 : r0 + step] = sc
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
    packed = pack_fp8_to_int32(q, size_k_first=False).T.contiguous()
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

    u = sc_fp8.view(torch.uint8)
    uniq, cnts = torch.unique(u, return_counts=True)
    if uniq.numel() > max_pal:
        uf = uniq.view(torch.float8_e4m3fn).float()
        order_val = torch.argsort(uf, descending=True)
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
# INT3 / INT4 / Paired INT3 SwiGLU (group-64 via muse_bfx_ext)
# ---------------------------------------------------------------------------

def _try_int3_buffer(w: torch.Tensor) -> torch.Tensor | None:
    bfx = _get_bfx()
    if bfx is None or not hasattr(bfx, "int3_pack") or not hasattr(bfx, "int3_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    mse_opt = _int3_mse_enabled()
    try:
        blob = bfx.int3_pack(w_c, mse_opt)
    except TypeError:
        blob = bfx.int3_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(3037 + n + 17 * k)
    for m in (1, 7, 8, 12, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = bfx.int3_gemm(x, blob, n, k)
        if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w_c).float()
        cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.96:
            logger.warning("muse_draft_int3: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
            return None
    torch.cuda.synchronize(w.device)
    return blob


def _try_int3_swiglu_buffer(w: torch.Tensor) -> torch.Tensor | None:
    bfx = _get_bfx()
    if (
        bfx is None
        or not _mlp_swiglu_enabled()
        or not hasattr(bfx, "int3_pack_swiglu")
        or not hasattr(bfx, "int3_swiglu_gemm")
        or not hasattr(bfx, "int3_pack")
        or not hasattr(bfx, "int3_gemm")
    ):
        return None
    n, k = w.shape
    if n % 64 != 0 or (n // 2) % 32 != 0 or k % 64 != 0:
        return None
    h = n // 2
    w_c = w.contiguous()
    mse_opt = _int3_mse_enabled()
    sw_blob = bfx.int3_pack_swiglu(w_c, mse_opt)
    ref_blob = bfx.int3_pack(w_c, mse_opt)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(5059 + n + 29 * k)
    for m in (1, 7, 8, 12, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = bfx.int3_swiglu_gemm(x, sw_blob, h, k)
        if got.shape != (m, h) or not bool(torch.isfinite(got).all()):
            del ref_blob
            return None
        ref_gu = bfx.int3_gemm(x, ref_blob, n, k)
        ref_out = torch.empty((m, h), dtype=w.dtype, device=w.device)
        torch.ops._C.silu_and_mul(ref_out, ref_gu)
        if not torch.equal(got.view(torch.int16), ref_out.view(torch.int16)):
            max_diff = (got.float() - ref_out.float()).abs().max().item()
            logger.warning(
                "muse_draft_int3_swiglu: bitwise check failed (max_diff=%.6f) on (%d,%d) M=%d; falling back",
                max_diff, n, k, m,
            )
            del ref_blob
            return None
    del ref_blob
    torch.cuda.synchronize(w.device)
    return sw_blob


def _try_int4_buffer(w: torch.Tensor) -> torch.Tensor | None:
    bfx = _get_bfx()
    if bfx is None or not hasattr(bfx, "int4_pack") or not hasattr(bfx, "int4_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    mse_opt = _int4_mse_enabled()
    try:
        blob = bfx.int4_pack(w_c, mse_opt)
    except TypeError:
        blob = bfx.int4_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(4051 + n + 23 * k)
    for m in (1, 7, 8, 12, 16):
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
    return os.environ.get("MUSE_DRAFT_M512_FP8", "0") != "0"


def _pack_fp8_large_m(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    import vllm._custom_ops as ops

    n, k = w.shape
    if n % 16 != 0 or k % 16 != 0:
        return None
    q, s = _quantize_rows_fp8(w)
    b_t = q.t()
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
# Attachment (modules + fused KV tensor + paired MLP SwiGLU + fused QKV norm/RoPE)
# ---------------------------------------------------------------------------

def attach_mlp_swiglu(mlp: torch.nn.Module, *, free_bf16: bool = False) -> bool:
    layer = getattr(mlp, "gate_up_proj", None)
    if layer is None or getattr(mlp, "_mi3_swiglu", None) is not None:
        return False
    w = getattr(layer, "weight", None)
    if (
        w is None
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "_mfp8", None) is not None
        or not isinstance(w, torch.Tensor)
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_cuda
    ):
        return False
    n, k = w.shape
    try:
        sw_blob = _try_int3_swiglu_buffer(w.data)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_int3_swiglu: attach failed for %s (%s); trying regular INT3", tuple(w.shape), e)
        return False
    if sw_blob is None:
        return False
    h = n // 2
    mlp.register_buffer("_mi3_swiglu_blob", sw_blob, persistent=False)
    mlp._mi3_swiglu = (h, k)
    layer._mfp8 = (n, k)  # sentinel so muse_bfx skips gate_up_proj
    _stats["admitted_int3_swiglu"] += 1
    _stats["bytes_bf16"] += n * k * 2
    _stats["bytes_quant"] += sw_blob.numel() * sw_blob.element_size()
    if free_bf16 and _free_bf16_enabled():
        bf16_bytes = w.numel() * w.element_size()
        w.data = torch.empty((0,), dtype=w.dtype, device=w.device)
        _stats["bytes_freed"] += bf16_bytes
    return True


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
                layer._mfp8 = (n, k)
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
                layer._mfp8 = (n, k)
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
                layer._mfp8 = (n, k)
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
    if 1 <= m <= MAX_M_INT and x2.dtype == torch.bfloat16:
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
        for r0 in range(0, m, MAX_M_INT):
            r1 = min(m, r0 + MAX_M_INT)
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
    if 1 <= m <= MAX_M_INT and x2.dtype == torch.bfloat16:
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
        for r0 in range(0, m, MAX_M_INT):
            r1 = min(m, r0 + MAX_M_INT)
            bfx.int3_gemm_out(xc[r0:r1], blob, out[r0:r1], n, k)
    return out.reshape(*shape[:-1], n)


@draft_int3_linear.register_fake
def _(x, w, blob, n, k, fp8_bt=None, fp8_sb=None):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::draft_int3_swiglu_linear", mutates_args=())
def draft_int3_swiglu_linear(
    x: torch.Tensor,
    blob: torch.Tensor,
    h: int,
    k: int,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = int(x2.size(0))
    bfx = _get_bfx()
    if 1 <= m <= MAX_M_INT and x2.dtype == torch.bfloat16:
        out = bfx.int3_swiglu_gemm(x2.contiguous(), blob, h, k)
    elif m == 0:
        out = x2.new_empty((0, h))
    else:
        xc = x2.contiguous()
        out = xc.new_empty((m, h))
        for r0 in range(0, m, MAX_M_INT):
            r1 = min(m, r0 + MAX_M_INT)
            bfx.int3_swiglu_gemm_out(xc[r0:r1], blob, out[r0:r1], h, k)
    return out.reshape(*shape[:-1], h)


@draft_int3_swiglu_linear.register_fake
def _(x, blob, h, k):
    return x.new_empty((*x.shape[:-1], h))


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


def _resolve_attn_kv_layout(kv_cache: torch.Tensor) -> tuple[int, int, int, int, int, int] | None:
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0 or kv_cache.dtype != torch.bfloat16:
        return None
    if kv_cache.dim() == 4 and kv_cache.stride(3) == 1 and kv_cache.shape[3] == 256:
        if kv_cache.shape[1] == 8:
            return (
                kv_cache.data_ptr(),
                int(kv_cache.shape[2]),
                int(kv_cache.stride(0)),
                int(kv_cache.stride(1)),
                int(kv_cache.stride(2)),
                128,
            )
        if kv_cache.shape[2] == 8:
            return (
                kv_cache.data_ptr(),
                int(kv_cache.shape[1]),
                int(kv_cache.stride(0)),
                int(kv_cache.stride(2)),
                int(kv_cache.stride(1)),
                128,
            )
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2 and kv_cache.shape[4] == 128 and kv_cache.stride(4) == 1:
        if kv_cache.shape[3] == 8:
            return (
                kv_cache.data_ptr(),
                int(kv_cache.shape[2]),
                int(kv_cache.stride(1)),
                int(kv_cache.stride(3)),
                int(kv_cache.stride(2)),
                int(kv_cache.stride(0)),
            )
        if kv_cache.shape[2] == 8:
            return (
                kv_cache.data_ptr(),
                int(kv_cache.shape[3]),
                int(kv_cache.stride(1)),
                int(kv_cache.stride(2)),
                int(kv_cache.stride(3)),
                int(kv_cache.stride(0)),
            )
    return None


@torch.library.custom_op("muse::dflash_qkv_norm_rope", mutates_args=())
def dflash_qkv_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    layer_name: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    shape = qkv.shape
    qkv2 = qkv.reshape(-1, shape[-1]).contiguous()
    m = int(qkv2.size(0))
    q_out = qkv2.new_empty((m, 4096))
    k_out = qkv2.new_empty((m, 1024))
    if m == 0:
        return q_out.reshape(*shape[:-1], 4096), k_out.reshape(*shape[:-1], 1024)

    pos_flat = positions.reshape(-1)
    if not pos_flat.is_contiguous():
        pos_flat = pos_flat.contiguous()

    attn_layer = None
    slot_tensor = None
    kv_meta = None
    if is_forward_context_available():
        fwd_ctx = get_forward_context()
        attn_layer = fwd_ctx.no_compile_layers.get(layer_name)
        if attn_layer is not None:
            attn_layer._qk_kv_cache_written = False
            attn_layer._qk_kv_cache_written_rows = 0
            if (
                fwd_ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
                and not attn_layer.calculate_kv_scales
                and attn_layer.kv_sharing_target_layer_name is None
                and not attn_layer.attn_backend.forward_includes_kv_cache_update
                and attn_layer.kv_cache_dtype in ("auto", "bfloat16")
                and getattr(attn_layer.impl, "dcp_world_size", 1) == 1
                and isinstance(fwd_ctx.slot_mapping, dict)
            ):
                cand_slot = fwd_ctx.slot_mapping.get(layer_name)
                if (
                    cand_slot is not None
                    and cand_slot.dtype == torch.int64
                    and cand_slot.is_contiguous()
                    and cand_slot.numel() >= m
                ):
                    kv_meta = _resolve_attn_kv_layout(attn_layer.kv_cache)
                    if kv_meta is not None:
                        slot_tensor = cand_slot

    bfx = _get_bfx()
    if kv_meta is not None and slot_tensor is not None and attn_layer is not None:
        kv_ptr, blk_sz, sb, sh, ss, v_off = kv_meta
        bfx.dflash_qkv_norm_rope_cache(
            qkv2, q_norm_w, k_norm_w, cos_sin_cache, pos_flat,
            q_out, None, slot_tensor, kv_ptr, blk_sz, sb, sh, ss, v_off, eps,
        )
        attn_layer._qk_kv_cache_written_rows = m
    else:
        bfx.dflash_qkv_norm_rope_cache(
            qkv2, q_norm_w, k_norm_w, cos_sin_cache, pos_flat,
            q_out, k_out, None, 0, 16, 0, 0, 0, 128, eps,
        )
    return q_out.reshape(*shape[:-1], 4096), k_out.reshape(*shape[:-1], 1024)


@dflash_qkv_norm_rope.register_fake
def _(qkv, positions, q_norm_w, k_norm_w, cos_sin_cache, layer_name, eps):
    return qkv.new_empty((*qkv.shape[:-1], 4096)), qkv.new_empty((*qkv.shape[:-1], 1024))


def verify_dflash_qkv_fused(device: torch.device) -> bool:
    import vllm._custom_ops as ops

    idx = device.index or 0
    if idx in _QKV_FUSED_VERIFIED:
        return _QKV_FUSED_VERIFIED[idx]
    if not _qkv_fused_enabled():
        _QKV_FUSED_VERIFIED[idx] = False
        return False
    bfx = _get_bfx()
    if bfx is None or not hasattr(bfx, "dflash_qkv_norm_rope_cache"):
        _QKV_FUSED_VERIFIED[idx] = False
        return False
    try:
        gen = torch.Generator(device=device).manual_seed(6073)
        qw = (torch.randn((128,), generator=gen, device=device) * 0.2 + 1.0).to(torch.bfloat16).contiguous()
        kw = (torch.randn((128,), generator=gen, device=device) * 0.2 + 1.0).to(torch.bfloat16).contiguous()
        cs = (torch.randn((256, 128), generator=gen, device=device) * 0.5).to(torch.bfloat16).contiguous()
        eps = 1e-5
        scale_one = torch.ones((1,), dtype=torch.float32, device=device)
        for m in (1, 7, 12, 16):
            qkv = (torch.randn((m, 6144), generator=gen, device=device) * 0.8).to(torch.bfloat16).contiguous()
            pos = torch.arange(10, 10 + m, dtype=torch.int64, device=device)
            slots = torch.arange(3, 3 + m, dtype=torch.int64, device=device)
            if m >= 4:
                slots[-1] = -1
            kv_got = torch.full((4, 8, 16, 256), -3.25, dtype=torch.bfloat16, device=device)
            kv_ref = kv_got.clone()
            q_got = torch.empty((m, 4096), dtype=torch.bfloat16, device=device)
            k_got = torch.empty((m, 1024), dtype=torch.bfloat16, device=device)
            kv_ptr, blk_sz, sb, sh, ss, v_off = _resolve_attn_kv_layout(kv_got)
            bfx.dflash_qkv_norm_rope_cache(
                qkv, qw, kw, cs, pos, q_got, k_got, slots, kv_ptr, blk_sz, sb, sh, ss, v_off, eps
            )
            q_ref = torch.empty((m, 32, 128), dtype=torch.bfloat16, device=device)
            k_ref = torch.empty((m, 8, 128), dtype=torch.bfloat16, device=device)
            ops.rms_norm(q_ref, qkv[:, :4096].contiguous().view(m, 32, 128), qw, eps)
            ops.rms_norm(k_ref, qkv[:, 4096:5120].contiguous().view(m, 8, 128), kw, eps)
            q_ref_2d = q_ref.view(m, 4096)
            k_ref_2d = k_ref.view(m, 1024)
            ops.rotary_embedding(pos, q_ref_2d, k_ref_2d, 128, cs, True)
            v_ref = qkv[:, 5120:].contiguous().view(m, 8, 128)
            kc_ref, vc_ref = kv_ref.transpose(1, 2).split(128, dim=-1)
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                k_ref_2d.view(m, 8, 128), v_ref, kc_ref, vc_ref, slots, "auto", scale_one, scale_one
            )
            if (
                not torch.equal(q_got.view(torch.int16), q_ref_2d.view(torch.int16))
                or not torch.equal(k_got.view(torch.int16), k_ref_2d.view(torch.int16))
                or not torch.equal(kv_got.view(torch.int16), kv_ref.view(torch.int16))
            ):
                logger.warning("muse_draft_qkv_fused: bitwise self-check failed at M=%d", m)
                _QKV_FUSED_VERIFIED[idx] = False
                return False
        torch.cuda.synchronize(device)
        _QKV_FUSED_VERIFIED[idx] = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("muse_draft_qkv_fused: self-check skipped (%s)", exc)
        _QKV_FUSED_VERIFIED[idx] = False
        return False


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


def mlp_swiglu_apply(mlp: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    h, k = mlp._mi3_swiglu
    return torch.ops.muse.draft_int3_swiglu_linear(x, mlp._mi3_swiglu_blob, h, k)


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
    if m > MAX_M_INT:
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
    if mnv4 is not None and m <= MAX_M_NVFP4:
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
    if m > MAX_M_INT or out.size(1) != n or k % 5 != 0:
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


def place_drafter_quant(model: torch.nn.Module) -> int:
    """GB10 UMA DRAM placement for quantized drafter weights (INT4, INT3, INT3-SwiGLU, NVFP4, FP8)."""
    if os.environ.get("MUSE_DRAFT_PLACEMENT", "1") == "0":
        return 0
    groups: dict[tuple[str, int, int], list[tuple[torch.nn.Module, str, tuple]]] = {}
    for _, mod in model.named_modules():
        if getattr(mod, "_mi3_swiglu", None) is not None:
            h, k = mod._mi3_swiglu
            groups.setdefault(("int3_swiglu", h, k), []).append((mod, "_mi3_swiglu_blob", ()))
        elif getattr(mod, "_mi4", None) is not None:
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
        x = torch.randn((16 if kind in ("int4", "int3", "int3_swiglu") else 7, k), dtype=torch.bfloat16, device=dev)
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
            if kind == "int3_swiglu":
                bfx = _get_bfx()
                run = lambda: bfx.int3_swiglu_gemm(x, buf, n, k)
            elif kind == "int4":
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
    t0 = time.perf_counter()
    mode = _quant_mode()
    n = 0
    by_leaf: dict[str, list[torch.nn.Module]] = {s: [] for s in _NAMES}
    mlps: list[torch.nn.Module] = []
    attns: list[torch.nn.Module] = []
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in by_leaf:
            by_leaf[leaf].append(mod)
        elif leaf == "mlp" and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj"):
            mlps.append(mod)
        elif leaf == "self_attn" and hasattr(mod, "q_norm") and hasattr(mod, "k_norm") and hasattr(mod, "rotary_emb"):
            attns.append(mod)

    def _pick_mode(leaf: str) -> str:
        if mode == "hybrid_int4_dn":
            return "int4" if leaf in _ATTN_FC_DN_NAMES else "int3"
        if mode in ("hybrid_int4_fc", "ctx_attn"):
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

    for leaf in ("o_proj", "qkv_proj", "down_proj"):
        lm = _pick_mode(leaf)
        for mod in by_leaf[leaf]:
            if attach_linear(mod, mode=lm, free_bf16=True, attach_m512_fp8=False):
                n += 1
                if _free_bf16_enabled():
                    torch.cuda.empty_cache()

    gu_mode = _pick_mode("gate_up_proj")
    if gu_mode == "int3" and _mlp_swiglu_enabled():
        for mlp in mlps:
            if attach_mlp_swiglu(mlp, free_bf16=True):
                n += 1
                if _free_bf16_enabled():
                    torch.cuda.empty_cache()
    for mod in by_leaf["gate_up_proj"]:
        if getattr(mod, "_mfp8", None) is not None:
            continue
        if attach_linear(mod, mode=gu_mode, free_bf16=True, attach_m512_fp8=False):
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
        if mode in ("hybrid_int4_dn", "hybrid_int4_fc", "ctx_attn", "hybrid_int4", "int4"):
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

    if attns:
        dev = attns[0].q_norm.weight.device
        if verify_dflash_qkv_fused(dev):
            for attn in attns:
                rot = attn.rotary_emb
                if (
                    getattr(attn, "q_size", 0) == 4096
                    and getattr(attn, "kv_size", 0) == 1024
                    and getattr(attn, "head_dim", 0) == 128
                    and getattr(rot, "head_size", 0) == 128
                    and getattr(rot, "rotary_dim", 0) == 128
                    and getattr(rot, "is_neox_style", False)
                ):
                    cs_bf16 = rot.cos_sin_cache.to(device=dev, dtype=torch.bfloat16).contiguous()
                    attn.register_buffer("_rope_cos_sin_bf16", cs_bf16, persistent=False)
                    attn._muse_qkv_fused = True
                    _stats["admitted_qkv_fused"] += 1

    pack_s = time.perf_counter() - t0
    place_drafter_quant(model)
    logger.info(
        "muse_draft_quant (%s, mse=%d/%d, pack=%.2fs): %d INT4-g64 + %d INT3-SwiGLU + %d INT3-g64 + "
        "%d NVFP4-TMA4 + %d FP8-Marlin (%d M>16 FP8, %d QKV-fused, %d rejected), %.2f GB -> %.2f GB/step, %.2f GB BF16 freed",
        mode,
        int(_int4_mse_enabled()),
        int(_int3_mse_enabled()),
        pack_s,
        _stats["admitted_int4"],
        _stats["admitted_int3_swiglu"],
        _stats["admitted_int3"],
        _stats["admitted_nvfp4"],
        _stats["admitted_fp8"],
        _stats["admitted_m512_fp8"],
        _stats["admitted_qkv_fused"],
        _stats["rejected"],
        _stats["bytes_bf16"] / 1e9,
        _stats["bytes_quant"] / 1e9,
        _stats["bytes_freed"] / 1e9,
    )
    return n
