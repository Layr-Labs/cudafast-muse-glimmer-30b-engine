# SPDX-License-Identifier: Apache-2.0
"""Weight-only quantized copies (INT3 / NVFP4 TMA4-ILV / FP8 Marlin) of the pinned DFlash drafter's linears.

The drafter only PROPOSES tokens: every committed token is the target's own greedy token from the
verify step, and the per-depth oracles equal the serial tape. The drafter's arithmetic can therefore
change acceptance, never the emitted stream. A depth-6 speculative step reads the whole 5.25 GB BF16
drafter once; at verify width (M <= 8 rows) these GEMMs are weight-bandwidth bound on GB10.

At load, each bias-free BF16 drafter linear (qkv/o/gate_up/down of the 5 layers, `fc`, and the fused
context-KV projection `fkv`) gets a quantized weight-only copy (`MUSE_DRAFT_QUANT=int3|nvfp4|hybrid|fp8|bf16`).
Context-path weights (`fc`, `fkv`) retain their BF16 weights for M > 8 (512-row prompt prefill).
Query-only weights (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`) release their unused BF16
storage after admission (`MUSE_DRAFT_FREE_BF16=1`, freeing ~4.67 GB UMA before DRAM placement).
DRAM placement uses a 48 MiB read-only L2 eviction buffer (`evict_buf.sum()`), largest-first group
ordering, round-robin median-of-5 timing, and slow-bank quarantine. Fused context-KV post-GEMM
(`dflash_kv_post`) is strictly self-checked at load before admission. The target is never touched.
"""
import math
import os
from statistics import median
import time

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M_FP8 = 16
MAX_M_NVFP4 = 8
MAX_M_INT3 = 16
MAX_M_INT4 = 16
_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")
_MLP_FC_NAMES = ("gate_up_proj", "down_proj", "fc")
_CTX_ATTN_NAMES = ("fc", "fkv", "qkv_proj", "o_proj")
_QUERY_ONLY_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
_HOST_HEADROOM_BYTES = 8 << 30
_SENSITIVE_PRESETS: dict[str, tuple[str, ...]] = {
    "1": ("fc", "fkv", "layers.0.self_attn", "layers.4.self_attn"),
    "default": ("fc", "fkv", "layers.0.self_attn", "layers.4.self_attn"),
    "ctx": ("fc", "fkv"),
    "ctx_edge_attn": ("fc", "fkv", "layers.0.self_attn", "layers.4.self_attn"),
    "ctx_attn": ("fc", "fkv", "qkv_proj", "o_proj"),
    "ctx_edge": ("fc", "fkv", "layers.0.self_attn", "layers.4.self_attn", "layers.4.mlp.down_proj"),
    "edge": ("layers.0", "layers.4"),
}
_stats = {
    "admitted_int3": 0,
    "admitted_int3_swiglu": 0,
    "admitted_int4": 0,
    "admitted_int4_swiglu": 0,
    "admitted_nvfp4": 0,
    "admitted_fp8": 0,
    "rejected": 0,
    "bytes_bf16": 0,
    "bytes_quant": 0,
    "bytes_freed_bf16": 0,
}
_layer_metrics: list[tuple[str, str, tuple[int, int], float, float, int]] = []

_SHARED_PARTIALS: dict[int, torch.Tensor] = {}


def _quant_mode() -> str:
    return os.environ.get("MUSE_DRAFT_QUANT", "int4").strip().lower()


def enabled() -> bool:
    if os.environ.get("MUSE_DRAFT_FP8", "1") == "0":
        return False
    return _quant_mode() not in ("bf16", "none", "off", "0", "false")


def _int3_mse_enabled() -> bool:
    return os.environ.get("MUSE_INT3_MSE_PACK", "1") != "0"


def _int4_mse_enabled() -> bool:
    return os.environ.get("MUSE_INT4_MSE_PACK", "1") != "0"


def _int3_swiglu_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_INT3_SWIGLU", "1") != "0"


def _free_bf16_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_FREE_BF16", "1") != "0"


def _parse_layer_selectors(env_name: str) -> tuple[str, ...]:
    raw = os.environ.get(env_name, "").strip()
    if not raw or raw in ("0", "none", "off", "false"):
        return ()
    tokens: list[str] = []
    for part in raw.split(","):
        tok = part.strip()
        if not tok:
            continue
        preset = _SENSITIVE_PRESETS.get(tok.lower())
        if preset is not None:
            tokens.extend(preset)
        else:
            tokens.append(tok)
    return tuple(tokens)


def _matches_selectors(name: str, leaf: str, selectors: tuple[str, ...]) -> bool:
    for sel in selectors:
        if sel == name or sel == leaf:
            return True
        if name.startswith(sel + ".") or name.endswith("." + sel) or (f".{sel}." in name):
            return True
    return False


def _resolve_layer_mode(
    name: str,
    leaf: str,
    base_mode: str,
    sens_fp8: tuple[str, ...],
    sens_nvfp4: tuple[str, ...],
    sens_int4: tuple[str, ...] = (),
) -> str:
    if base_mode in ("hybrid_int4", "hybrid_i4", "ctx_attn_int4"):
        lm = "int4" if leaf in _CTX_ATTN_NAMES else "int3"
    elif base_mode == "int4":
        lm = "int4"
    elif base_mode == "int3":
        lm = "int3"
    elif base_mode == "nvfp4":
        lm = "nvfp4"
    elif base_mode == "hybrid":
        lm = "nvfp4" if leaf in _MLP_FC_NAMES else "fp8"
    else:
        lm = "fp8"
    if sens_fp8 and _matches_selectors(name, leaf, sens_fp8):
        return "fp8"
    if sens_int4 and _matches_selectors(name, leaf, sens_int4) and lm == "int3":
        return "int4"
    if sens_nvfp4 and _matches_selectors(name, leaf, sens_nvfp4) and lm == "int3":
        return "nvfp4"
    return lm


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


def _run_chunked(x2: torch.Tensor, max_m: int, fn) -> torch.Tensor:
    m = x2.size(0)
    if m <= max_m:
        return fn(x2.contiguous())
    chunks = [fn(x2[i : i + max_m].contiguous()) for i in range(0, m, max_m)]
    return torch.cat(chunks, dim=0)


def _free_query_only_bf16(layer: torch.nn.Module, leaf: str) -> None:
    if not _free_bf16_enabled() or leaf not in _QUERY_ONLY_NAMES:
        return
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.numel() == 0:
        return
    freed = w.numel() * w.element_size()
    empty = torch.empty((0,), dtype=w.dtype, device=w.device)
    if isinstance(w, torch.nn.Parameter):
        layer.weight = torch.nn.Parameter(empty, requires_grad=False)
    else:
        layer.weight = empty
    _stats["bytes_freed_bf16"] += freed


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


def _check_fp8(
    w: torch.Tensor, q: torch.Tensor, s: torch.Tensor, qweight, scales, ws,
    metrics: dict[str, float] | None = None,
) -> bool:
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
            got_f = got.float()
            cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
            if not cos >= 0.995:
                return False
            if m == 7 and metrics is not None:
                metrics["cos"] = cos
                metrics["snr_db"] = _snr_db(dense, got_f)
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
    import muse_marlin6_gemv as d6

    if not x2.is_contiguous() or x2.data_ptr() % 16:
        x2 = x2.clone(memory_format=torch.contiguous_format)
    m = int(x2.shape[0])
    out = x2.new_empty((m, n))
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


def _check_nvfp4(
    w: torch.Tensor, qw, ms, mgs, w_ilv, coded, pal, ws,
    metrics: dict[str, float] | None = None,
) -> bool:
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
            got_f = got.float()
            cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
            if not cos >= 0.985:
                logger.warning("muse_draft_nvfp4: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
                return False
            if m == 7 and metrics is not None:
                metrics["cos"] = cos
                metrics["snr_db"] = _snr_db(dense, got_f)
        if int(torch.count_nonzero(ws)) != 0:
            return False
        torch.cuda.synchronize(w.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_nvfp4: check failed (%s); falling back", e)
        return False
    return True


def _try_nvfp4_tma4_buffers(w: torch.Tensor, metrics: dict[str, float] | None = None):
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
    ok = _check_nvfp4(w, qw, ms, mgs, w_ilv, coded, pal, ws, metrics=metrics)
    del qw, ms
    if not ok:
        return None
    return w_ilv, coded, pal, mgs, ws


# ---------------------------------------------------------------------------
# INT3 W3A16 (asymmetric group-64 via stock muse_bfx_ext.int3_gemm)
# ---------------------------------------------------------------------------

def _try_int3_buffer(
    w: torch.Tensor,
    metrics: dict[str, float] | None = None,
) -> torch.Tensor | None:
    import muse_bfx_ext as _bfx

    if not hasattr(_bfx, "int3_pack") or not hasattr(_bfx, "int3_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    mse_opt = _int3_mse_enabled()
    try:
        blob = _bfx.int3_pack(w_c, mse_opt)
    except TypeError:
        blob = _bfx.int3_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(3037 + n + 17 * k)
    for m in (1, 7, 8, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = _bfx.int3_gemm(x, blob, n, k)
        if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w_c).float()
        got_f = got.float()
        cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.96:
            logger.warning("muse_draft_int3: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
            return None
        if m == 7 and metrics is not None:
            metrics["cos"] = cos
            metrics["snr_db"] = _snr_db(dense, got_f)
    torch.cuda.synchronize(w.device)
    return blob


def _try_int3_swiglu_buffer(
    w: torch.Tensor,
    metrics: dict[str, float] | None = None,
) -> torch.Tensor | None:
    import muse_bfx_ext as _bfx

    if not hasattr(_bfx, "int3_pack_swiglu") or not hasattr(_bfx, "int3_gemm_swiglu"):
        return None
    n2, k = w.shape
    if n2 % 64 != 0 or k % 64 != 0:
        return None
    inter = n2 // 2
    w_c = w.contiguous()
    w_gate = w_c[:inter]
    w_up = w_c[inter:]
    mse_opt = _int3_mse_enabled()
    try:
        blob = _bfx.int3_pack_swiglu(w_c, mse_opt)
    except TypeError:
        blob = _bfx.int3_pack_swiglu(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(3571 + n2 + 23 * k)
    for m in (1, 7, 8, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = _bfx.int3_gemm_swiglu(x, blob, inter, k)
        if got.shape != (m, inter) or not bool(torch.isfinite(got).all()):
            return None
        dg = F.linear(x, w_gate).float()
        du = F.linear(x, w_up).float()
        dense = F.silu(dg) * du
        got_f = got.float()
        cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
        snr = _snr_db(dense, got_f)
        if not (cos >= 0.965 and snr >= 11.5):
            logger.warning(
                "muse_draft_int3_swiglu: quality check failed on (%d,%d) M=%d (SNR=%.2fdB cos=%.5f)",
                n2, k, m, snr, cos,
            )
            return None
        if m == 7 and metrics is not None:
            metrics["cos"] = cos
            metrics["snr_db"] = snr
    torch.cuda.synchronize(w.device)
    return blob


# ---------------------------------------------------------------------------
# INT4 W4A16 (asymmetric group-64 via muse_bfx_ext.int4_gemm / int4_gemm_aux5)
# ---------------------------------------------------------------------------

def _try_int4_buffer(
    w: torch.Tensor,
    metrics: dict[str, float] | None = None,
    check_aux5: bool = False,
    check_query_streamk: bool = False,
) -> tuple[torch.Tensor, bool, bool] | None:
    import muse_bfx_ext as _bfx

    if not hasattr(_bfx, "int4_pack") or not hasattr(_bfx, "int4_gemm"):
        return None
    n, k = w.shape
    if n % 64 != 0 or k % 64 != 0:
        return None
    w_c = w.contiguous()
    mse_opt = _int4_mse_enabled()
    try:
        blob = _bfx.int4_pack(w_c, mse_opt)
    except TypeError:
        blob = _bfx.int4_pack(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(4049 + n + 19 * k)
    for m in (1, 7, 8, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = _bfx.int4_gemm(x, blob, n, k)
        if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
            return None
        dense = F.linear(x, w_c).float()
        got_f = got.float()
        cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
        if not cos >= 0.99:
            logger.warning("muse_draft_int4: low cosine %.5f on (%d,%d) M=%d", cos, n, k, m)
            return None
        if m == 7 and metrics is not None:
            metrics["cos"] = cos
            metrics["snr_db"] = _snr_db(dense, got_f)
    aux5_ok = False
    if (
        check_aux5
        and os.environ.get("MUSE_DRAFT_AUX5", "1") != "0"
        and hasattr(_bfx, "int4_gemm_aux5")
        and k % 5 == 0
        and (k // 5) % 64 == 0
    ):
        k_sub = k // 5
        aux5_ok = True
        try:
            for m in (1, 7, 16):
                # Test with non-trivial row slice (larger backing buffer) to exercise stride handling.
                hs_full = [
                    (torch.randn((m + 3, k_sub), generator=gen, device=w.device) * 0.5).to(w.dtype)
                    for _ in range(5)
                ]
                hs = [h[:m] for h in hs_full]
                got_a5 = _bfx.int4_gemm_aux5(hs[0], hs[1], hs[2], hs[3], hs[4], blob, n, k)
                got_a5_b = _bfx.int4_gemm_aux5(hs[0], hs[1], hs[2], hs[3], hs[4], blob, n, k)
                ref_cat = _bfx.int4_gemm(torch.cat(hs, dim=-1).contiguous(), blob, n, k)
                max_diff = (got_a5.float() - ref_cat.float()).abs().max().item()
                if not torch.equal(got_a5.view(torch.int16), got_a5_b.view(torch.int16)):
                    aux5_ok = False
                    logger.warning("muse_draft_int4: aux5 non-deterministic on (%d,%d) M=%d", n, k, m)
                    break
                if not torch.allclose(got_a5.float(), ref_cat.float(), rtol=2e-2, atol=2e-3):
                    aux5_ok = False
                    logger.warning("muse_draft_int4: aux5 mismatch on (%d,%d) M=%d (max_diff=%.5f)", n, k, m, max_diff)
                    break
        except Exception as exc:  # noqa: BLE001
            aux5_ok = False
            logger.warning("muse_draft_int4: aux5 check failed (%s)", exc)
    query_sk_ok = False
    if (
        check_query_streamk
        and os.environ.get("MUSE_DRAFT_INT4_QUERY_SK", "1") != "0"
    ):
        query_sk_ok = True
        try:
            for m in (6, 7, 16):
                x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
                ref = _bfx.int4_gemm(x, blob, n, k, False)
                got_a = _bfx.int4_gemm(x, blob, n, k, True)
                got_b = _bfx.int4_gemm(x, blob, n, k, True)
                if not torch.equal(got_a.view(torch.int16), got_b.view(torch.int16)):
                    query_sk_ok = False
                    logger.warning("muse_draft_int4: query_streamk non-deterministic on (%d,%d) M=%d", n, k, m)
                    break
                if not torch.allclose(got_a.float(), ref.float(), rtol=2e-2, atol=2e-3):
                    max_diff = (got_a.float() - ref.float()).abs().max().item()
                    query_sk_ok = False
                    logger.warning("muse_draft_int4: query_streamk mismatch on (%d,%d) M=%d (max_diff=%.5f)", n, k, m, max_diff)
                    break
        except Exception as exc:  # noqa: BLE001
            query_sk_ok = False
            logger.warning("muse_draft_int4: query_streamk check failed (%s)", exc)
    torch.cuda.synchronize(w.device)
    return blob, aux5_ok, query_sk_ok


def _try_int4_swiglu_buffer(
    w: torch.Tensor,
    metrics: dict[str, float] | None = None,
) -> torch.Tensor | None:
    import muse_bfx_ext as _bfx

    if not hasattr(_bfx, "int4_pack_swiglu") or not hasattr(_bfx, "int4_gemm_swiglu"):
        return None
    n2, k = w.shape
    if n2 % 64 != 0 or k % 64 != 0:
        return None
    inter = n2 // 2
    w_c = w.contiguous()
    w_gate = w_c[:inter]
    w_up = w_c[inter:]
    mse_opt = _int4_mse_enabled()
    try:
        blob = _bfx.int4_pack_swiglu(w_c, mse_opt)
    except TypeError:
        blob = _bfx.int4_pack_swiglu(w_c)
    gen = torch.Generator(device=w.device)
    gen.manual_seed(4583 + n2 + 29 * k)
    for m in (1, 7, 8, 16):
        x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
        got = _bfx.int4_gemm_swiglu(x, blob, inter, k)
        if got.shape != (m, inter) or not bool(torch.isfinite(got).all()):
            return None
        dg = F.linear(x, w_gate).float()
        du = F.linear(x, w_up).float()
        dense = F.silu(dg) * du
        got_f = got.float()
        cos = F.cosine_similarity(got_f.flatten(), dense.flatten(), dim=0).item()
        snr = _snr_db(dense, got_f)
        if not (cos >= 0.985 and snr >= 15.0):
            logger.warning(
                "muse_draft_int4_swiglu: quality check failed on (%d,%d) M=%d (SNR=%.2fdB cos=%.5f)",
                n2, k, m, snr, cos,
            )
            return None
        if m == 7 and metrics is not None:
            metrics["cos"] = cos
            metrics["snr_db"] = snr
    torch.cuda.synchronize(w.device)
    return blob


# ---------------------------------------------------------------------------
# Load-time 0-ULP verification for dflash_kv_post & dflash_qkv_norm_rope
# ---------------------------------------------------------------------------

def _verify_dflash_kv_post(model: torch.nn.Module) -> None:
    if os.environ.get("MUSE_DFLASH_FUSE_KV_POST", "1") == "0":
        return
    if not (hasattr(model, "_k_norm_weights") and hasattr(model, "_rope_cos_sin_cache")):
        return
    hd = int(getattr(model, "_head_dim", 0))
    if hd != 128 or int(getattr(model, "_rope_head_size", 0)) != 128:
        return
    try:
        import muse_bfx_ext as _bfx
        from vllm import _custom_ops as ops
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_fuse: kv_post skipped (%s)", exc)
        return
    if not hasattr(_bfx, "dflash_kv_post"):
        return

    dev = model._k_norm_weights.device
    kv_ok = False
    kv_diff = -1.0
    try:
        L = int(model._num_attn_layers)
        nkv = int(model._num_kv_heads)
        eps = float(model._rms_norm_eps)
        is_neox = bool(model._rope_is_neox)
        bs = 16
        gen = torch.Generator(device=dev)
        gen.manual_seed(7009)
        all_kv_flat = (torch.randn((7, L * 2 * nkv * hd), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        pos = torch.tensor([0, 1, 15, 16, 31, 511, 512], dtype=torch.int64, device=dev)
        slots = torch.tensor([0, 1, 15, 16, 31, -1, 47], dtype=torch.int64, device=dev)
        cos_sin_cache = model._rope_cos_sin_cache
        if cos_sin_cache.dtype != torch.bfloat16:
            cos_sin_cache = cos_sin_cache.to(device=dev, dtype=torch.bfloat16)
        all_kv = all_kv_flat.view(7, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        all_k_ref = torch.empty_like(all_kv[0])
        ops.rms_norm(all_k_ref, all_kv[0], model._k_norm_weights, eps)
        all_k_flat = all_k_ref.view(L * 7, nkv * hd)
        ops.rotary_embedding(pos.repeat(L), all_k_flat, None, hd, cos_sin_cache, is_neox)
        all_k_final = all_k_flat.view(L, 7, nkv, hd)
        all_v_final = all_kv[1]
        kvcs_got = [torch.zeros((4, nkv, bs, 2 * hd), dtype=torch.bfloat16, device=dev) for _ in range(L)]
        kvcs_ref = [torch.zeros((4, nkv, bs, 2 * hd), dtype=torch.bfloat16, device=dev) for _ in range(L)]
        kc_got, vc_got = [], []
        for i in range(L):
            kg, vg = kvcs_got[i].transpose(1, 2).split(hd, dim=-1)
            kr, vr = kvcs_ref[i].transpose(1, 2).split(hd, dim=-1)
            kc_got.append(kg)
            vc_got.append(vg)
            for m_idx in range(7):
                sl = int(slots[m_idx].item())
                if sl >= 0:
                    kr[sl // bs, sl % bs] = all_k_final[i, m_idx]
                    vr[sl // bs, sl % bs] = all_v_final[i, m_idx]
        _bfx.dflash_kv_post(
            all_kv_flat, pos, model._k_norm_weights, cos_sin_cache,
            kc_got, vc_got, [slots] * L, nkv, bs, eps, is_neox,
        )
        max_k = 0.0
        v_exact = True
        for i in range(L):
            kg, vg = kvcs_got[i].transpose(1, 2).split(hd, dim=-1)
            kr, vr = kvcs_ref[i].transpose(1, 2).split(hd, dim=-1)
            max_k = max(max_k, float((kg.float() - kr.float()).abs().max().item()))
            if not torch.equal(vg, vr):
                v_exact = False
        kv_diff = max_k
        if v_exact and kv_diff <= 2e-2:
            kv_ok = True
            model._dflash_fuse_kv_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_fuse: kv_post self-check failed (%s); keeping stock path", exc)

    torch.cuda.synchronize(dev)
    logger.info("dflash_fuse: kv_post=%s (max_diff=%.5f)", "admitted" if kv_ok else "off", kv_diff)


def _verify_dflash_qkv_norm_rope(model: torch.nn.Module) -> None:
    if os.environ.get("MUSE_DFLASH_FUSE_QKV_ROPE", "1") == "0":
        return
    layers = getattr(model, "layers", None)
    if not layers:
        return
    try:
        import muse_bfx_ext as _bfx
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_fuse: qkv_norm_rope skipped (%s)", exc)
        return
    if not hasattr(_bfx, "dflash_qkv_norm_rope"):
        return
    try:
        from vllm import _custom_ops as ops
    except Exception:  # noqa: BLE001
        ops = None
    allow_all5 = ops is not None and os.environ.get("MUSE_DFLASH_QKV_ALL5", "1") != "0"
    cache_fuse = os.environ.get("MUSE_DFLASH_QKV_CACHE", "1") != "0"

    admitted = 0
    cache_admitted = 0
    worst_diff = 0.0
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            logger.warning("dflash_fuse: layer %d qkv_norm_rope skipped (no self_attn)", idx)
            continue
        hd = int(getattr(attn, "head_dim", 0))
        rope = getattr(attn, "rotary_emb", None)
        if hd != 128 or rope is None or int(getattr(rope, "head_size", 0)) != 128:
            logger.warning(
                "dflash_fuse: layer %d qkv_norm_rope skipped (hd=%d rope=%s rope_hd=%s)",
                idx,
                hd,
                type(rope).__name__ if rope is not None else "None",
                getattr(rope, "head_size", None),
            )
            continue
        qw = getattr(getattr(attn, "q_norm", None), "weight", None)
        kw = getattr(getattr(attn, "k_norm", None), "weight", None)
        if qw is None or kw is None or not (qw.is_cuda and kw.is_cuda):
            logger.warning("dflash_fuse: layer %d qkv_norm_rope skipped (missing CUDA q/k norm weight)", idx)
            continue
        if qw.dtype != torch.bfloat16 or kw.dtype != torch.bfloat16:
            logger.warning(
                "dflash_fuse: layer %d qkv_norm_rope skipped (non-bf16 norm weight: q=%s k=%s)",
                idx,
                qw.dtype,
                kw.dtype,
            )
            continue
        dev = qw.device
        nq = int(attn.num_heads)
        nkv = int(attn.num_kv_heads)
        eps = float(attn.q_norm.variance_epsilon)
        is_neox = bool(rope.is_neox_style)
        try:
            gen = torch.Generator(device=dev)
            gen.manual_seed(7127 + idx)
            qkv = (torch.randn((16, (nq + 2 * nkv) * hd), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            pos = torch.tensor([0, 1, 7, 15, 16, 31, 63, 127, 255, 511, 512, 513, 520, 525, 600, 640], dtype=torch.int64, device=dev)
            q_in, k_in, _ = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            if allow_all5:
                # Compiled-Inductor semantics: ops.rms_norm -> FP32 RoPE -> BF16.
                q_norm_bf16 = torch.empty((16, nq, hd), dtype=torch.bfloat16, device=dev)
                k_norm_bf16 = torch.empty((16, nkv, hd), dtype=torch.bfloat16, device=dev)
                ops.rms_norm(q_norm_bf16, q_in.contiguous().view(16, nq, hd), qw.contiguous(), eps)
                ops.rms_norm(k_norm_bf16, k_in.contiguous().view(16, nkv, hd), kw.contiguous(), eps)
                q_norm_f32 = q_norm_bf16.float()
                k_norm_f32 = k_norm_bf16.float()
            else:
                q_ref = attn.q_norm(q_in.contiguous().view(16, nq, hd)).view(16, nq * hd).contiguous()
                k_ref = attn.k_norm(k_in.contiguous().view(16, nkv, hd)).view(16, nkv * hd).contiguous()
                q_ref, k_ref = rope(pos, q_ref.clone(), k_ref.clone())

            # Try both BF16 and native (FP32) cos_sin_cache and pick the closest match.
            base_cache = rope.cos_sin_cache.to(device=dev)
            cands = (
                [base_cache.to(dtype=torch.bfloat16), base_cache]
                if base_cache.dtype != torch.bfloat16
                else [base_cache]
            )
            best_diff = 1e9
            best_ok = False
            best_cache = None
            for cand in cands:
                cand_c = cand.contiguous()
                if allow_all5:
                    half = hd // 2
                    cs = cand_c.index_select(0, pos).float()
                    c_half = cs[:, :half].unsqueeze(1)
                    s_half = cs[:, half:].unsqueeze(1)
                    if is_neox:
                        q1, q2 = q_norm_f32[..., :half], q_norm_f32[..., half:]
                        q_ref = torch.cat([q1 * c_half - q2 * s_half, q2 * c_half + q1 * s_half], dim=-1).to(torch.bfloat16).view(16, nq * hd)
                        k1, k2 = k_norm_f32[..., :half], k_norm_f32[..., half:]
                        k_ref = torch.cat([k1 * c_half - k2 * s_half, k2 * c_half + k1 * s_half], dim=-1).to(torch.bfloat16).view(16, nkv * hd)
                    else:
                        q1, q2 = q_norm_f32[..., 0::2], q_norm_f32[..., 1::2]
                        q_ref = torch.stack([q1 * c_half - q2 * s_half, q2 * c_half + q1 * s_half], dim=-1).to(torch.bfloat16).view(16, nq * hd)
                        k1, k2 = k_norm_f32[..., 0::2], k_norm_f32[..., 1::2]
                        k_ref = torch.stack([k1 * c_half - k2 * s_half, k2 * c_half + k1 * s_half], dim=-1).to(torch.bfloat16).view(16, nkv * hd)

                q_got, k_got = _bfx.dflash_qkv_norm_rope(
                    qkv, pos, qw.contiguous(), kw.contiguous(), cand_c, nq, nkv, eps, is_neox
                )
                d = max(
                    float((q_got.float() - q_ref.float()).abs().max().item()),
                    float((k_got.float() - k_ref.float()).abs().max().item()),
                )
                rtol_q = 2e-2 if allow_all5 else 1e-2
                atol_q = 3e-2 if allow_all5 else 2e-2
                ok = bool(
                    torch.allclose(q_got.float(), q_ref.float(), rtol=rtol_q, atol=atol_q)
                    and torch.allclose(k_got.float(), k_ref.float(), rtol=rtol_q, atol=atol_q)
                    and d <= 0.0625
                )
                if d < best_diff:
                    best_diff = d
                    best_ok = ok
                    best_cache = cand_c
                if d == 0.0:
                    break
            if best_cache is not None and best_ok:
                attn.register_buffer("_dflash_rope_cos_sin_cache", best_cache, persistent=False)
                attn._dflash_fuse_qkv_ok = True
                admitted += 1
                worst_diff = max(worst_diff, best_diff)
                attn._dflash_qkv_cache_ok = False
                if cache_fuse:
                    try:
                        attn._dflash_qkv_cache_ok = _check_dflash_qkv_cache(
                            _bfx, qkv, pos, qw.contiguous(), kw.contiguous(), best_cache,
                            nq, nkv, hd, eps, is_neox, dev,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("dflash_fuse: layer %d qkv kv-scatter check failed (%s); unfused", idx, exc)
                        attn._dflash_qkv_cache_ok = False
                    if attn._dflash_qkv_cache_ok:
                        cache_admitted += 1
                    else:
                        logger.warning("dflash_fuse: layer %d qkv kv-scatter rejected; unfused cache write", idx)
            else:
                logger.warning(
                    "dflash_fuse: layer %d qkv_norm_rope rejected (best_diff=%.5f, use_flashinfer=%s)",
                    idx,
                    best_diff,
                    getattr(rope, "use_flashinfer", False),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dflash_fuse: layer %d qkv_norm_rope check failed (%s); keeping stock path", idx, exc)

    logger.info(
        "dflash_fuse: qkv_norm_rope=%d/%d layers admitted (max_diff=%.5f, all5=%d, kv_scatter=%d)",
        admitted,
        len(layers),
        worst_diff,
        int(allow_all5),
        cache_admitted,
    )


def _check_dflash_qkv_cache(_bfx, qkv, pos, qw, kw, cos_sin, nq, nkv, hd, eps, is_neox, dev) -> bool:
    """Bitwise check: fused K/V paged scatter == qkv_norm_rope + reshape_and_cache_flash.

    Covers both KV layouts the runtime path accepts ([2,B,N,H,D] and the
    FlashAttention (B,H,N,2D) view), both block sizes (16 and 64), PAD slots (-1),
    out-of-order slots, and the returned Q/K tensors.
    """
    m = int(qkv.shape[0])
    q_ref, k_ref = _bfx.dflash_qkv_norm_rope(qkv, pos, qw, kw, cos_sin, nq, nkv, eps, is_neox)
    v_src = qkv[:, (nq + nkv) * hd :].contiguous().view(m, nkv, hd)
    scale_one = torch.ones((1,), dtype=torch.float32, device=dev)
    for bs, nblk, slot_list in (
        (16, 4, [5, 0, 17, 63, 16, 1, 40, 2, 33, 47, 9, 60, 31, 50, 22, -1]),
        (64, 2, [5, 0, 65, 127, 64, 1, 100, 2, 73, 111, 9, 120, 63, 95, 22, -1]),
    ):
        slots = torch.tensor(slot_list, dtype=torch.int64, device=dev)[:m].contiguous()
        layouts = []
        kvc5_t = torch.full((2, nblk, bs, nkv, hd), -1.75, dtype=torch.bfloat16, device=dev)
        kvc5_r = kvc5_t.clone()
        layouts.append((kvc5_t, kvc5_r, kvc5_t[0], kvc5_t[1], kvc5_r[0], kvc5_r[1]))
        kvc4_t = torch.full((nblk, nkv, bs, 2 * hd), 3.25, dtype=torch.bfloat16, device=dev)
        kvc4_r = kvc4_t.clone()
        kt, vt = kvc4_t.transpose(1, 2).split(hd, dim=-1)
        kr, vr = kvc4_r.transpose(1, 2).split(hd, dim=-1)
        layouts.append((kvc4_t, kvc4_r, kt, vt, kr, vr))
        for full_t, full_r, kc_t, vc_t, kc_r, vc_r in layouts:
            q_got, k_got = _bfx.dflash_qkv_norm_rope(
                qkv, pos, qw, kw, cos_sin, nq, nkv, eps, is_neox, slots, kc_t, vc_t, bs
            )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                k_ref.view(m, nkv, hd), v_src, kc_r, vc_r, slots, "auto", scale_one, scale_one
            )
            torch.cuda.synchronize(dev)
            if not torch.equal(full_t.view(torch.int16), full_r.view(torch.int16)):
                return False
            if not (
                torch.equal(q_got.view(torch.int16), q_ref.view(torch.int16))
                and torch.equal(k_got.view(torch.int16), k_ref.view(torch.int16))
            ):
                return False
    return True


def _verify_dflash_rms_norm(model: torch.nn.Module) -> None:
    if os.environ.get("MUSE_DFLASH_FUSE_NORM", "1") == "0":
        return
    try:
        import muse_bfx_ext as _bfx
        from vllm import _custom_ops as ops
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_fuse: rms_norm skipped (%s)", exc)
        return
    if not hasattr(_bfx, "dflash_rms_norm") or not hasattr(_bfx, "dflash_fused_add_rms_norm"):
        return
    norm_mod = getattr(model, "norm", None)
    w = getattr(norm_mod, "weight", None)
    if w is None or not w.is_cuda or w.dtype != torch.bfloat16 or w.dim() != 1:
        return
    k = int(w.shape[0])
    if k <= 0 or k > 8192 or (k % 8) != 0:
        return
    eps = float(getattr(norm_mod, "variance_epsilon", 1e-6))
    dev = w.device
    norm_ok = False
    max_diff = -1.0
    max_res_diff = -1.0
    try:
        gen = torch.Generator(device=dev)
        gen.manual_seed(7331 + k)
        w_c = w.contiguous()
        worst_d = 0.0
        worst_rd = 0.0
        all_ok = True
        for m in (6, 7, 16):
            x = (torch.randn((m, k), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            r = (torch.randn((m, k), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            ref1 = torch.empty_like(x)
            ops.rms_norm(ref1, x, w_c, eps)
            got1 = _bfx.dflash_rms_norm(x, w_c, eps)
            d1 = float((got1.float() - ref1.float()).abs().max().item())
            worst_d = max(worst_d, d1)
            if not (bool(torch.isfinite(got1).all()) and torch.allclose(got1.float(), ref1.float(), rtol=1e-2, atol=1e-2)):
                all_ok = False
                break
            x_ref = x.clone()
            r_ref = r.clone()
            ops.fused_add_rms_norm(x_ref, r_ref, w_c, eps)
            got2, got_r = _bfx.dflash_fused_add_rms_norm(x, r, w_c, eps)
            d2 = float((got2.float() - x_ref.float()).abs().max().item())
            dr = float((got_r.float() - r_ref.float()).abs().max().item())
            worst_d = max(worst_d, d2)
            worst_rd = max(worst_rd, dr)
            if not (
                bool(torch.isfinite(got2).all())
                and bool(torch.isfinite(got_r).all())
                and torch.allclose(got2.float(), x_ref.float(), rtol=1e-2, atol=1e-2)
                and torch.allclose(got_r.float(), r_ref.float(), rtol=1e-2, atol=1e-2)
            ):
                all_ok = False
                break
        max_diff = worst_d
        max_res_diff = worst_rd
        if all_ok and worst_d <= 0.03125 and worst_rd <= 0.03125:
            norm_ok = True
            model._dflash_fuse_norm_ok = True
            for layer in getattr(model, "layers", ()):
                layer._dflash_fuse_norm_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_fuse: rms_norm self-check failed (%s); keeping stock path", exc)
    torch.cuda.synchronize(dev)
    logger.info(
        "dflash_fuse: rms_norm=%s (max_diff=%.5f, res_diff=%.5f)",
        "admitted" if norm_ok else "off",
        max_diff,
        max_res_diff,
    )


# ---------------------------------------------------------------------------
# Attachment (modules + fused KV tensor) and custom ops
# ---------------------------------------------------------------------------

def attach_mlp_swiglu(mlp: torch.nn.Module, name: str = "", mode: str = "int3") -> bool:
    gate_up = getattr(mlp, "gate_up_proj", None)
    if gate_up is None or getattr(mlp, "_mi3_swiglu", None) is not None:
        return False
    w = getattr(gate_up, "weight", None)
    if (
        w is None
        or getattr(gate_up, "bias", None) is not None
        or getattr(gate_up, "_mfp8", None) is not None
        or not isinstance(w, torch.Tensor)
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_cuda
    ):
        return False
    n2, k = w.shape
    if n2 % 64 != 0 or k % 128 != 0:
        return False
    inter = n2 // 2
    label = name or f"mlp_swiglu_{n2}x{k}"
    use_i4 = (mode == "int4")
    try:
        m: dict[str, float] = {}
        blob = _try_int4_swiglu_buffer(w.data, metrics=m) if use_i4 else _try_int3_swiglu_buffer(w.data, metrics=m)
        if blob is None:
            return False
        mlp.register_buffer("_mi3_swiglu_blob", blob, persistent=False)
        mlp._mi3_swiglu = (inter, k)
        if use_i4:
            mlp._mi4_swiglu = (inter, k)
        gate_up._mfp8 = (n2, k)  # sentinel so attach_linear and muse_bfx skip gate_up_proj
        qbytes = blob.numel() * blob.element_size()
        if use_i4:
            _stats["admitted_int4_swiglu"] += 1
        else:
            _stats["admitted_int3_swiglu"] += 1
        _stats["bytes_bf16"] += n2 * k * 2
        _stats["bytes_quant"] += qbytes
        if "snr_db" in m:
            _layer_metrics.append((label, "i4swg" if use_i4 else "i3swg", (n2, k), m["snr_db"], m["cos"], qbytes))
        _free_query_only_bf16(gate_up, "gate_up_proj")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_%s_swiglu: attach failed for %s (%s); falling back", mode, label, e)
        return False


def attach_linear(layer: torch.nn.Module, mode: str = "int3", name: str = "", leaf: str = "") -> bool:
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
    label = name or f"linear_{n}x{k}"

    if mode == "int4":
        try:
            m: dict[str, float] = {}
            want_qsk = leaf in ("qkv_proj", "o_proj")
            res4 = _try_int4_buffer(
                w.data,
                metrics=m,
                check_aux5=(leaf == "fc"),
                check_query_streamk=want_qsk,
            )
            if res4 is not None:
                blob, aux5_ok, qsk_ok = res4
                layer.register_buffer("_mi4_blob", blob, persistent=False)
                layer._mi4 = (n, k)
                layer._mi4_aux5_ok = aux5_ok
                layer._mi4_query_streamk = qsk_ok
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                qbytes = blob.numel() * blob.element_size()
                _stats["admitted_int4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((label, "int4", (n, k), m["snr_db"], m["cos"], qbytes))
                _free_query_only_bf16(layer, leaf)
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int4: attach failed for %s (%s); trying NVFP4", tuple(w.shape), e)

    if mode == "int3":
        try:
            m = {}
            blob = _try_int3_buffer(w.data, metrics=m)
            if blob is not None:
                layer.register_buffer("_mi3_blob", blob, persistent=False)
                layer._mi3 = (n, k)
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                qbytes = blob.numel() * blob.element_size()
                _stats["admitted_int3"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((label, "int3", (n, k), m["snr_db"], m["cos"], qbytes))
                _free_query_only_bf16(layer, leaf)
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int3: attach failed for %s (%s); trying NVFP4", tuple(w.shape), e)

    if mode in ("int4", "int3", "nvfp4"):
        try:
            m = {}
            res = _try_nvfp4_tma4_buffers(w.data, metrics=m)
            if res is not None:
                w_ilv, coded, pal, mgs, ws = res
                layer.register_buffer("_mnv4_w_ilv", w_ilv, persistent=False)
                layer.register_buffer("_mnv4_coded", coded, persistent=False)
                layer.register_buffer("_mnv4_pal", pal, persistent=False)
                layer.register_buffer("_mnv4_gs", mgs, persistent=False)
                layer.register_buffer("_mnv4_ws", ws, persistent=False)
                layer._mnv4 = (n, k)
                layer._mfp8 = (n, k)  # sentinel so muse_bfx skips this layer
                qbytes = w_ilv.numel() * w_ilv.element_size() + coded.numel() * coded.element_size()
                _stats["admitted_nvfp4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((label, "nvfp4", (n, k), m["snr_db"], m["cos"], qbytes))
                _free_query_only_bf16(layer, leaf)
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_nvfp4: attach failed for %s (%s); trying FP8", tuple(w.shape), e)

    try:
        q, s, qweight, scales, ws = _pack_fp8(w.data.contiguous())
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_fp8: pack failed for %s (%s)", tuple(w.shape), e)
        _stats["rejected"] += 1
        return False
    m = {}
    if not _check_fp8(w.data, q, s, qweight, scales, ws, metrics=m):
        _stats["rejected"] += 1
        return False
    del q, s
    layer.register_buffer("_mfp8_qweight", qweight, persistent=False)
    layer.register_buffer("_mfp8_scales", scales, persistent=False)
    layer.register_buffer("_mfp8_ws", ws, persistent=False)
    layer._mfp8 = (n, k)
    qbytes = qweight.numel() * qweight.element_size()
    _stats["admitted_fp8"] += 1
    _stats["bytes_bf16"] += n * k * 2
    _stats["bytes_quant"] += qbytes
    if "snr_db" in m:
        _layer_metrics.append((label, "fp8", (n, k), m["snr_db"], m["cos"], qbytes))
    _free_query_only_bf16(layer, leaf)
    return True


def attach_tensor(mod: torch.nn.Module, prefix: str, w: torch.Tensor, mode: str = "int3") -> bool:
    if not enabled() or w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_cuda:
        return False
    n, k = w.shape
    if n % 64 or k % 128:
        return False
    if mode == "int4":
        try:
            m: dict[str, float] = {}
            want_qsk = (prefix == "fkv")
            res4 = _try_int4_buffer(
                w.data,
                metrics=m,
                check_aux5=False,
                check_query_streamk=want_qsk,
            )
            if res4 is not None:
                blob, _, qsk_ok = res4
                mod.register_buffer(f"_{prefix}_mi4_blob", blob, persistent=False)
                setattr(mod, f"_{prefix}_mi4", (n, k))
                setattr(mod, f"_{prefix}_mi4_query_streamk", qsk_ok)
                qbytes = blob.numel() * blob.element_size()
                _stats["admitted_int4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((prefix, "int4", (n, k), m["snr_db"], m["cos"], qbytes))
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int4: tensor %s attach failed (%s); trying NVFP4", prefix, e)

    if mode == "int3":
        try:
            m = {}
            blob = _try_int3_buffer(w.data, metrics=m)
            if blob is not None:
                mod.register_buffer(f"_{prefix}_mi3_blob", blob, persistent=False)
                setattr(mod, f"_{prefix}_mi3", (n, k))
                qbytes = blob.numel() * blob.element_size()
                _stats["admitted_int3"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((prefix, "int3", (n, k), m["snr_db"], m["cos"], qbytes))
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int3: tensor %s attach failed (%s); trying NVFP4", prefix, e)

    if mode in ("int4", "int3", "nvfp4"):
        try:
            m = {}
            res = _try_nvfp4_tma4_buffers(w.data, metrics=m)
            if res is not None:
                w_ilv, coded, pal, mgs, ws = res
                mod.register_buffer(f"_{prefix}_mnv4_w_ilv", w_ilv, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_coded", coded, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_pal", pal, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_gs", mgs, persistent=False)
                mod.register_buffer(f"_{prefix}_mnv4_ws", ws, persistent=False)
                setattr(mod, f"_{prefix}_mnv4", (n, k))
                qbytes = w_ilv.numel() * w_ilv.element_size() + coded.numel() * coded.element_size()
                _stats["admitted_nvfp4"] += 1
                _stats["bytes_bf16"] += n * k * 2
                _stats["bytes_quant"] += qbytes
                if "snr_db" in m:
                    _layer_metrics.append((prefix, "nvfp4", (n, k), m["snr_db"], m["cos"], qbytes))
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_nvfp4: tensor %s attach failed (%s); trying FP8", prefix, e)

    try:
        q, s, qweight, scales, ws = _pack_fp8(w.data.contiguous())
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_fp8: tensor %s pack failed (%s)", prefix, e)
        _stats["rejected"] += 1
        return False
    m = {}
    if not _check_fp8(w.data, q, s, qweight, scales, ws, metrics=m):
        _stats["rejected"] += 1
        return False
    del q, s
    mod.register_buffer(f"_{prefix}_mfp8_qweight", qweight, persistent=False)
    mod.register_buffer(f"_{prefix}_mfp8_scales", scales, persistent=False)
    mod.register_buffer(f"_{prefix}_mfp8_ws", ws, persistent=False)
    setattr(mod, f"_{prefix}_mfp8", (n, k))
    qbytes = qweight.numel() * qweight.element_size()
    _stats["admitted_fp8"] += 1
    _stats["bytes_bf16"] += n * k * 2
    _stats["bytes_quant"] += qbytes
    if "snr_db" in m:
        _layer_metrics.append((prefix, "fp8", (n, k), m["snr_db"], m["cos"], qbytes))
    return True


@torch.library.custom_op("muse::draft_int4_linear", mutates_args=())
def draft_int4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    blob: torch.Tensor,
    n: int,
    k: int,
    query_streamk: bool = False,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.dtype == torch.bfloat16 and (1 <= x2.size(0) <= MAX_M_INT4 or w.numel() == 0):
        import muse_bfx_ext as _bfx

        out = _run_chunked(x2, MAX_M_INT4, lambda xc: _bfx.int4_gemm(xc, blob, n, k, query_streamk))
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_int4_linear.register_fake
def _(x, w, blob, n, k, query_streamk=False):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::dflash_rms_norm", mutates_args=())
def dflash_rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import muse_bfx_ext as _bfx

    xc = x if x.is_contiguous() else x.contiguous()
    return _bfx.dflash_rms_norm(xc, weight, eps)


@dflash_rms_norm_op.register_fake
def _(x, weight, eps):
    return torch.empty_like(x)


@torch.library.custom_op("muse::dflash_fused_add_rms_norm", mutates_args=())
def dflash_fused_add_rms_norm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    import muse_bfx_ext as _bfx

    xc = x if x.is_contiguous() else x.contiguous()
    rc = residual if residual.is_contiguous() else residual.contiguous()
    return _bfx.dflash_fused_add_rms_norm(xc, rc, weight, eps)


@dflash_fused_add_rms_norm_op.register_fake
def _(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(residual)


@torch.library.custom_op("muse::draft_int4_aux5_linear", mutates_args=())
def draft_int4_aux5_linear(
    h0: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
    h3: torch.Tensor,
    h4: torch.Tensor,
    blob: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    import muse_bfx_ext as _bfx

    return _bfx.int4_gemm_aux5(h0, h1, h2, h3, h4, blob, n, k)


@draft_int4_aux5_linear.register_fake
def _(h0, h1, h2, h3, h4, blob, n, k):
    return h0.new_empty((h0.shape[0], n))


@torch.library.custom_op("muse::draft_int3_swiglu_linear", mutates_args=())
def draft_int3_swiglu_linear(x: torch.Tensor, blob: torch.Tensor, inter: int, k: int) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    import muse_bfx_ext as _bfx

    i4_bytes = (inter // 4) * (k // 64) * 288
    if blob.numel() == i4_bytes:
        out = _run_chunked(x2, MAX_M_INT4, lambda xc: _bfx.int4_gemm_swiglu(xc, blob, inter, k))
    else:
        out = _run_chunked(x2, MAX_M_INT3, lambda xc: _bfx.int3_gemm_swiglu(xc, blob, inter, k))
    return out.reshape(*shape[:-1], inter)


@draft_int3_swiglu_linear.register_fake
def _(x, blob, inter, k):
    return x.new_empty((*x.shape[:-1], inter))


@torch.library.custom_op("muse::dflash_qkv_norm_rope", mutates_args=())
def dflash_qkv_norm_rope_op(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    nq: int,
    nkv: int,
    eps: float,
    is_neox: bool,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    import muse_bfx_ext as _bfx

    qkv_c = qkv if qkv.is_contiguous() else qkv.contiguous()
    pos_c = positions if positions.is_contiguous() else positions.contiguous()
    if layer_name:
        res = _dflash_qkv_norm_rope_kv_scatter(
            _bfx, qkv_c, pos_c, q_norm_w, k_norm_w, cos_sin_cache, nq, nkv, eps, is_neox, layer_name
        )
        if res is not None:
            return res
    return _bfx.dflash_qkv_norm_rope(
        qkv_c, pos_c, q_norm_w, k_norm_w, cos_sin_cache, nq, nkv, eps, is_neox
    )


def _dflash_qkv_norm_rope_kv_scatter(
    _bfx, qkv_c, pos_c, q_norm_w, k_norm_w, cos_sin_cache, nq, nkv, eps, is_neox, layer_name
):
    """Fused qkv_norm_rope + paged K/V scatter; returns None to use the unfused path.

    On success marks the attention layer so unified_kv_cache_update skips the
    (bit-identical) reshape_and_cache_flash for exactly these rows.
    """
    m = int(qkv_c.size(0))
    if m <= 0 or os.environ.get("MUSE_DFLASH_QKV_CACHE", "1") == "0":
        return None
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        fwd_ctx = get_forward_context()
        attn_layer = fwd_ctx.no_compile_layers.get(layer_name)
        if attn_layer is None:
            return None
        attn_layer._qk_kv_cache_written = False
        attn_layer._qk_kv_cache_written_rows = 0
        if (
            fwd_ctx.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            or attn_layer.calculate_kv_scales
            or attn_layer.kv_sharing_target_layer_name is not None
            or attn_layer.attn_backend.forward_includes_kv_cache_update
            or attn_layer.kv_cache_dtype not in ("auto", "bfloat16")
            or getattr(attn_layer.impl, "dcp_world_size", 1) != 1
            or not isinstance(fwd_ctx.slot_mapping, dict)
        ):
            return None
        cand_slot = fwd_ctx.slot_mapping.get(layer_name)
        kvc = attn_layer.kv_cache
        if not (
            cand_slot is not None
            and cand_slot.is_cuda
            and cand_slot.dtype == torch.int64
            and cand_slot.is_contiguous()
            and cand_slot.numel() >= m
            and isinstance(kvc, torch.Tensor)
            and kvc.is_cuda
            and kvc.numel() > 0
            and kvc.dtype == torch.bfloat16
        ):
            return None
        kc = vc = None
        if kvc.dim() == 4 and kvc.shape[1] == nkv and kvc.shape[3] == 256 and kvc.stride(3) == 1:
            kc, vc = kvc.transpose(1, 2).split(128, dim=-1)
        elif kvc.dim() == 5 and kvc.shape[0] == 2 and kvc.shape[3] == nkv and kvc.shape[4] == 128 and kvc.stride(4) == 1:
            kc, vc = kvc[0], kvc[1]
        if kc is None or vc is None or kc.stride(3) != 1 or vc.stride(3) != 1:
            return None
        res = _bfx.dflash_qkv_norm_rope(
            qkv_c, pos_c, q_norm_w, k_norm_w, cos_sin_cache,
            nq, nkv, eps, is_neox, cand_slot, kc, vc, int(kc.shape[1]),
        )
        attn_layer._qk_kv_cache_written_rows = m
        return res
    except Exception:  # noqa: BLE001
        return None


@dflash_qkv_norm_rope_op.register_fake
def _(qkv, positions, q_norm_w, k_norm_w, cos_sin_cache, nq, nkv, eps, is_neox, layer_name=""):
    m = qkv.shape[0]
    return qkv.new_empty((m, nq * 128)), qkv.new_empty((m, nkv * 128))


def mlp_swiglu_apply(mlp: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    inter, k = mlp._mi3_swiglu
    return torch.ops.muse.draft_int3_swiglu_linear(x, mlp._mi3_swiglu_blob, inter, k)


@torch.library.custom_op("muse::draft_int3_linear", mutates_args=())
def draft_int3_linear(x: torch.Tensor, w: torch.Tensor, blob: torch.Tensor, n: int, k: int) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.dtype == torch.bfloat16 and (1 <= x2.size(0) <= MAX_M_INT3 or w.numel() == 0):
        import muse_bfx_ext as _bfx

        out = _run_chunked(x2, MAX_M_INT3, lambda xc: _bfx.int3_gemm(xc, blob, n, k))
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_int3_linear.register_fake
def _(x, w, blob, n, k):
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("muse::draft_fp8_linear", mutates_args=())
def draft_fp8_linear(
    x: torch.Tensor, w: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.dtype == torch.bfloat16 and (1 <= x2.size(0) <= MAX_M_FP8 or w.numel() == 0):
        out = _run_chunked(x2, MAX_M_FP8, lambda xc: _marlin_fp8(xc, qweight, scales, ws, n, k))
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_fp8_linear.register_fake
def _(x, w, qweight, scales, ws, n, k):
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
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.dtype == torch.bfloat16 and (1 <= x2.size(0) <= MAX_M_NVFP4 or w.numel() == 0):
        out = _run_chunked(x2, MAX_M_NVFP4, lambda xc: _run_nvfp4_tma4(xc, w_ilv, coded, pal, mgs, ws, n, k))
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_nvfp4_linear.register_fake
def _(x, w, w_ilv, coded, pal, mgs, ws, n, k):
    return x.new_empty((*x.shape[:-1], n))


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "_mi4", None) is not None:
        n, k = layer._mi4
        return torch.ops.muse.draft_int4_linear(
            x,
            layer.weight,
            layer._mi4_blob,
            n,
            k,
            bool(getattr(layer, "_mi4_query_streamk", False)),
        )
    if getattr(layer, "_mi3", None) is not None:
        n, k = layer._mi3
        return torch.ops.muse.draft_int3_linear(x, layer.weight, layer._mi3_blob, n, k)
    if getattr(layer, "_mnv4", None) is not None:
        n, k = layer._mnv4
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            layer.weight,
            layer._mnv4_w_ilv,
            layer._mnv4_coded,
            layer._mnv4_pal,
            layer._mnv4_gs,
            layer._mnv4_ws,
            n,
            k,
        )
    n, k = layer._mfp8
    return torch.ops.muse.draft_fp8_linear(
        x, layer.weight, layer._mfp8_qweight, layer._mfp8_scales, layer._mfp8_ws, n, k
    )


def tensor_apply(mod: torch.nn.Module, prefix: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    if getattr(mod, f"_{prefix}_mi4", None) is not None:
        n, k = getattr(mod, f"_{prefix}_mi4")
        return torch.ops.muse.draft_int4_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mi4_blob"),
            n,
            k,
            bool(getattr(mod, f"_{prefix}_mi4_query_streamk", False)),
        )
    if getattr(mod, f"_{prefix}_mi3", None) is not None:
        n, k = getattr(mod, f"_{prefix}_mi3")
        return torch.ops.muse.draft_int3_linear(x, w, getattr(mod, f"_{prefix}_mi3_blob"), n, k)
    if getattr(mod, f"_{prefix}_mnv4", None) is not None:
        n, k = getattr(mod, f"_{prefix}_mnv4")
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mnv4_w_ilv"),
            getattr(mod, f"_{prefix}_mnv4_coded"),
            getattr(mod, f"_{prefix}_mnv4_pal"),
            getattr(mod, f"_{prefix}_mnv4_gs"),
            getattr(mod, f"_{prefix}_mnv4_ws"),
            n,
            k,
        )
    if getattr(mod, f"_{prefix}_mfp8", None) is not None:
        n, k = getattr(mod, f"_{prefix}_mfp8")
        return torch.ops.muse.draft_fp8_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mfp8_qweight"),
            getattr(mod, f"_{prefix}_mfp8_scales"),
            getattr(mod, f"_{prefix}_mfp8_ws"),
            n,
            k,
        )
    return None


# ---------------------------------------------------------------------------
# GB10 UMA DRAM placement (read-only L2 flush + round-robin + slow quarantine)
# ---------------------------------------------------------------------------

def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _spares_allowed(wanted: int, payload_bytes: int) -> int:
    avail = _mem_available()
    if avail is None:
        return wanted
    cached_free = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    effective = avail + (cached_free if payload_bytes <= (128 << 20) else 0)
    return max(0, min(wanted, (effective - _HOST_HEADROOM_BYTES) // max(1, payload_bytes)))


def place_drafter_quant(model: torch.nn.Module) -> int:
    """GB10 UMA DRAM placement for quantized drafter weights (read-only L2 flush + slow quarantine)."""
    if os.environ.get("MUSE_DRAFT_PLACEMENT", "1") == "0":
        return 0
    groups: dict[tuple[str, int, int], list[tuple[torch.nn.Module, str, tuple]]] = {}
    for _, mod in model.named_modules():
        if getattr(mod, "_mi4_swiglu", None) is not None:
            inter, k = mod._mi4_swiglu
            groups.setdefault(("int4_swiglu", inter, k), []).append((mod, "_mi3_swiglu_blob", ()))
        elif getattr(mod, "_mi3_swiglu", None) is not None:
            inter, k = mod._mi3_swiglu
            groups.setdefault(("int3_swiglu", inter, k), []).append((mod, "_mi3_swiglu_blob", ()))
        if getattr(mod, "_mi4", None) is not None:
            n, k = mod._mi4
            groups.setdefault(("int4", n, k), []).append(
                (mod, "_mi4_blob", (bool(getattr(mod, "_mi4_query_streamk", False)),))
            )
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
        groups.setdefault(("int4", n, k), []).append(
            (model, "_fkv_mi4_blob", (bool(getattr(model, "_fkv_mi4_query_streamk", False)),))
        )
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
    first_item = next(iter(groups.values()))[0]
    dev = getattr(first_item[0], first_item[1]).device
    evict_buf = torch.ones((48 << 20) // 4, dtype=torch.float32, device=dev)
    slow_hold: list[torch.Tensor] = []
    total_moved = 0
    total_saved = 0.0

    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: getattr(kv[1][0][0], kv[1][0][1]).numel() * getattr(kv[1][0][0], kv[1][0][1]).element_size(),
        reverse=True,
    )

    for (kind, n, k), items in ordered_groups:
        homes = [getattr(mod, attr) for mod, attr, _ in items]
        payload_bytes = homes[0].numel() * homes[0].element_size()
        wanted = min(max(len(items) * 3, 6), 24)
        n_spares = _spares_allowed(wanted, payload_bytes)
        if n_spares <= 0:
            continue
        spares: list[torch.Tensor] = []
        try:
            for _ in range(n_spares):
                sp = torch.empty_like(homes[0])
                sp.copy_(homes[0])
                spares.append(sp)
        except torch.cuda.OutOfMemoryError:
            pass
        if not spares:
            continue

        m_place = 7 if (kind in ("nvfp4", "fp8") or k >= 20000 or n == 10240) else 16
        x = ((torch.arange(m_place * k, device=dev) % 7).float() - 3.0).to(torch.bfloat16).reshape(m_place, k)
        meta0 = items[0][2]

        def _make_runner(meta: tuple):
            if kind == "int4_swiglu":
                import muse_bfx_ext as _bfx

                return lambda buf: _bfx.int4_gemm_swiglu(x, buf, n, k)
            if kind == "int3_swiglu":
                import muse_bfx_ext as _bfx

                return lambda buf: _bfx.int3_gemm_swiglu(x, buf, n, k)
            if kind == "int4":
                import muse_bfx_ext as _bfx

                qsk = bool(meta[0]) if meta else False
                return lambda buf: _bfx.int4_gemm(x, buf, n, k, qsk)
            if kind == "int3":
                import muse_bfx_ext as _bfx

                return lambda buf: _bfx.int3_gemm(x, buf, n, k)
            if kind == "nvfp4":
                coded, pal, mgs, ws = meta
                return lambda buf: _run_nvfp4_tma4(x, buf, coded, pal, mgs, ws, n, k)
            scales, ws = meta
            return lambda buf: _marlin_fp8(x, buf, scales, ws, n, k)

        run_one = _make_runner(meta0)
        candidates = homes + spares
        _ = evict_buf.sum()
        run_one(candidates[0])
        torch.cuda.synchronize(dev)

        rounds = 5
        samples: list[list[float]] = [[] for _ in candidates]
        for _ in range(rounds):
            for idx, buf in enumerate(candidates):
                _ = evict_buf.sum()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                run_one(buf)
                e_ev.record()
                torch.cuda.synchronize(dev)
                samples[idx].append(s_ev.elapsed_time(e_ev))
        times = [median(s) for s in samples]

        count = len(homes)
        ranked = sorted(range(len(candidates)), key=times.__getitem__)
        kept = set(ranked[:count])
        slow_homes = sorted((i for i in range(count) if i not in kept), key=times.__getitem__, reverse=True)
        fast_spares = [i for i in ranked[:count] if i >= count]

        displaced: list[tuple[float, torch.Tensor]] = []
        used_targets: set[int] = set()
        for home_idx, target_idx in zip(slow_homes, fast_spares):
            if times[home_idx] - times[target_idx] < 0.003:
                continue
            mod, attr, _ = items[home_idx]
            old_buf = homes[home_idx]
            dst_buf = candidates[target_idx]
            dst_buf.copy_(old_buf)
            mod.register_buffer(attr, dst_buf, persistent=False)
            displaced.append((times[home_idx], old_buf))
            used_targets.add(target_idx)
            total_moved += 1
            total_saved += times[home_idx] - times[target_idx]

        unused = [(times[count + i], sp) for i, sp in enumerate(spares) if (count + i) not in used_targets]
        unused.extend(displaced)
        unused.sort(key=lambda pair: pair[0])
        keep_slow = unused[len(unused) // 2 :]
        for _, sb in keep_slow:
            slow_hold.append(sb)
        del spares, candidates, unused, displaced, keep_slow

    del slow_hold, evict_buf
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
    sens_fp8 = _parse_layer_selectors("MUSE_DRAFT_SENSITIVE_FP8")
    sens_nvfp4 = _parse_layer_selectors("MUSE_DRAFT_SENSITIVE_NVFP4")
    sens_int4 = _parse_layer_selectors("MUSE_DRAFT_SENSITIVE_INT4")
    _layer_metrics.clear()
    n = 0
    if _int3_swiglu_enabled():
        for i, layer in enumerate(getattr(model, "layers", ())):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            gu_name = f"layers.{i}.mlp.gate_up_proj"
            gu_mode = _resolve_layer_mode(gu_name, "gate_up_proj", mode, sens_fp8, sens_nvfp4, sens_int4)
            if gu_mode in ("int3", "int4"):
                n += attach_mlp_swiglu(mlp, name=gu_name, mode=gu_mode)
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in _NAMES:
            lm = _resolve_layer_mode(name, leaf, mode, sens_fp8, sens_nvfp4, sens_int4)
            n += attach_linear(mod, mode=lm, name=name, leaf=leaf)
    if getattr(model, "_fused_kv_weight", None) is not None and getattr(model, "_fused_kv_bias", None) is None:
        fkv_mode = _resolve_layer_mode("fkv", "fkv", mode, sens_fp8, sens_nvfp4, sens_int4)
        n += attach_tensor(model, "fkv", model._fused_kv_weight, mode=fkv_mode)
    pack_s = time.perf_counter() - t0
    if _stats["bytes_freed_bf16"] > 0:
        torch.cuda.empty_cache()
    place_drafter_quant(model)
    _verify_dflash_kv_post(model)
    _verify_dflash_qkv_norm_rope(model)
    _verify_dflash_rms_norm(model)
    aux5_admitted = bool(getattr(getattr(model, "fc", None), "_mi4_aux5_ok", False))
    qsk_admitted = sum(
        1 for _, mod in model.named_modules() if getattr(mod, "_mi4_query_streamk", False)
    ) + int(bool(getattr(model, "_fkv_mi4_query_streamk", False)))
    snr_summary = ""
    if _layer_metrics:
        ranked = sorted(_layer_metrics, key=lambda x: x[3])
        mean_snr = sum(x[3] for x in ranked) / len(ranked)
        mean_cos = sum(x[4] for x in ranked) / len(ranked)
        worst = ", ".join(f"{x[0]}({x[1]})={x[3]:.2f}dB/{x[4]:.4f}" for x in ranked[:4])
        snr_summary = f" | M=7 SNR mean={mean_snr:.2f}dB cos={mean_cos:.4f} worst=[{worst}]"
        if os.environ.get("MUSE_DRAFT_LOG_LAYERS", "0") != "0":
            for lname, lkind, lshape, lsnr, lcos, lbytes in ranked:
                logger.info(
                    "  muse_draft_layer: %-32s %-5s %5dx%-5d %6.1f MB  SNR=%5.2f dB  cos=%.5f",
                    lname,
                    lkind,
                    lshape[0],
                    lshape[1],
                    lbytes / 1e6,
                    lsnr,
                    lcos,
                )
    logger.info(
        "muse_draft_quant (%s, nf4=%s, nf3=%d, i3_mse=%d, i4_mse=%d, aux5=%d, qsk=%d, pack=%.2fs): %d INT4-g64 + %d INT4-SwiGLU + %d INT3-SwiGLU + %d INT3-g64 + %d NVFP4-TMA4 + %d FP8-Marlin drafter linears (%d rejected), %.2f GB -> %.2f GB per step (freed %.2f GB dead BF16)%s",
        mode,
        os.environ.get("MUSE_INT4_NF4", "0"),
        int(os.environ.get("MUSE_INT3_NF3", "1") != "0"),
        int(_int3_mse_enabled()),
        int(_int4_mse_enabled()),
        int(aux5_admitted),
        qsk_admitted,
        pack_s,
        _stats["admitted_int4"],
        _stats["admitted_int4_swiglu"],
        _stats["admitted_int3_swiglu"],
        _stats["admitted_int3"],
        _stats["admitted_nvfp4"],
        _stats["admitted_fp8"],
        _stats["rejected"],
        _stats["bytes_bf16"] / 1e9,
        _stats["bytes_quant"] / 1e9,
        _stats["bytes_freed_bf16"] / 1e9,
        snr_summary,
    )
    return n
