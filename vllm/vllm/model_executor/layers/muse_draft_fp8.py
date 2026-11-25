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
import math
import os
import time

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M_FP8 = 16
MAX_M_NVFP4 = 8
_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")
_MLP_FC_NAMES = ("gate_up_proj", "down_proj", "fc")
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
    "admitted_nvfp4": 0,
    "admitted_fp8": 0,
    "rejected": 0,
    "bytes_bf16": 0,
    "bytes_quant": 0,
}
_layer_metrics: list[tuple[str, str, tuple[int, int], float, float, int]] = []

_SHARED_PARTIALS: dict[int, torch.Tensor] = {}


def enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_FP8", "1") != "0"


def _quant_mode() -> str:
    return os.environ.get("MUSE_DRAFT_QUANT", "int3").strip().lower()


def _int3_mse_enabled() -> bool:
    return os.environ.get("MUSE_INT3_MSE_PACK", "0") != "0"


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
) -> str:
    if base_mode == "int3":
        lm = "int3"
    elif base_mode == "nvfp4":
        lm = "nvfp4"
    elif base_mode == "hybrid":
        lm = "nvfp4" if leaf in _MLP_FC_NAMES else "fp8"
    else:
        lm = "fp8"
    if sens_fp8 and _matches_selectors(name, leaf, sens_fp8):
        return "fp8"
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
# INT3 W3A16 (asymmetric group-64 via muse_bfx_ext.int3_gemm)
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
    for m in (1, 7, 8):
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


# ---------------------------------------------------------------------------
# Attachment (modules + fused KV tensor) and custom ops
# ---------------------------------------------------------------------------

def attach_linear(layer: torch.nn.Module, mode: str = "int3", name: str = "") -> bool:
    w = getattr(layer, "weight", None)
    if (
        w is None
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "_mfp8", None) is not None
        or getattr(layer, "_mnv4", None) is not None
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

    if mode == "int3":
        try:
            m: dict[str, float] = {}
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
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_int3: attach failed for %s (%s); trying NVFP4", tuple(w.shape), e)

    if mode in ("int3", "nvfp4"):
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
    return True


def attach_tensor(mod: torch.nn.Module, prefix: str, w: torch.Tensor, mode: str = "int3") -> bool:
    if not enabled() or w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_cuda:
        return False
    n, k = w.shape
    if n % 64 or k % 128:
        return False
    if mode == "int3":
        try:
            m: dict[str, float] = {}
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

    if mode in ("int3", "nvfp4"):
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


@torch.library.custom_op("muse::draft_int3_linear", mutates_args=())
def draft_int3_linear(x: torch.Tensor, w: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M_NVFP4 and x2.dtype == torch.bfloat16:
        import muse_bfx_ext as _bfx

        out = _bfx.int3_gemm(x2.contiguous(), blob, n, k)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_int3_linear.register_fake
def _(x, w, blob):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@torch.library.custom_op("muse::draft_fp8_linear", mutates_args=())
def draft_fp8_linear(
    x: torch.Tensor, w: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M_FP8 and x2.dtype == torch.bfloat16:
        out = _marlin_fp8(x2.contiguous(), qweight, scales, ws, n, k)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_fp8_linear.register_fake
def _(x, w, qweight, scales, ws):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@torch.library.custom_op("muse::draft_nvfp4_linear", mutates_args=())
def draft_nvfp4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_ilv: torch.Tensor,
    coded: torch.Tensor,
    pal: torch.Tensor,
    mgs: torch.Tensor,
    ws: torch.Tensor,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M_NVFP4 and x2.dtype == torch.bfloat16:
        out = _run_nvfp4_tma4(x2, w_ilv, coded, pal, mgs, ws, n, k)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_nvfp4_linear.register_fake
def _(x, w, w_ilv, coded, pal, mgs, ws):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "_mi3", None) is not None:
        return torch.ops.muse.draft_int3_linear(x, layer.weight, layer._mi3_blob)
    if getattr(layer, "_mnv4", None) is not None:
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            layer.weight,
            layer._mnv4_w_ilv,
            layer._mnv4_coded,
            layer._mnv4_pal,
            layer._mnv4_gs,
            layer._mnv4_ws,
        )
    return torch.ops.muse.draft_fp8_linear(
        x, layer.weight, layer._mfp8_qweight, layer._mfp8_scales, layer._mfp8_ws
    )


def tensor_apply(mod: torch.nn.Module, prefix: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    if getattr(mod, f"_{prefix}_mi3", None) is not None:
        return torch.ops.muse.draft_int3_linear(x, w, getattr(mod, f"_{prefix}_mi3_blob"))
    if getattr(mod, f"_{prefix}_mnv4", None) is not None:
        return torch.ops.muse.draft_nvfp4_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mnv4_w_ilv"),
            getattr(mod, f"_{prefix}_mnv4_coded"),
            getattr(mod, f"_{prefix}_mnv4_pal"),
            getattr(mod, f"_{prefix}_mnv4_gs"),
            getattr(mod, f"_{prefix}_mnv4_ws"),
        )
    if getattr(mod, f"_{prefix}_mfp8", None) is not None:
        return torch.ops.muse.draft_fp8_linear(
            x,
            w,
            getattr(mod, f"_{prefix}_mfp8_qweight"),
            getattr(mod, f"_{prefix}_mfp8_scales"),
            getattr(mod, f"_{prefix}_mfp8_ws"),
        )
    return None


def place_drafter_quant(model: torch.nn.Module) -> int:
    """GB10 UMA DRAM placement for quantized drafter weights (INT3, NVFP4 w_ilv, and FP8 qweight)."""
    if os.environ.get("MUSE_DRAFT_PLACEMENT", "1") == "0":
        return 0
    groups: dict[tuple[str, int, int], list[tuple[torch.nn.Module, str, tuple]]] = {}
    for _, mod in model.named_modules():
        if getattr(mod, "_mi3", None) is not None:
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
    if getattr(model, "_fkv_mi3", None) is not None:
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
            if kind == "int3":
                import muse_bfx_ext as _bfx

                run = lambda: _bfx.int3_gemm(x, buf, n, k)
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
    sens_fp8 = _parse_layer_selectors("MUSE_DRAFT_SENSITIVE_FP8")
    sens_nvfp4 = _parse_layer_selectors("MUSE_DRAFT_SENSITIVE_NVFP4")
    _layer_metrics.clear()
    n = 0
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in _NAMES:
            lm = _resolve_layer_mode(name, leaf, mode, sens_fp8, sens_nvfp4)
            n += attach_linear(mod, mode=lm, name=name)
    if getattr(model, "_fused_kv_weight", None) is not None and getattr(model, "_fused_kv_bias", None) is None:
        fkv_mode = _resolve_layer_mode("fkv", "fkv", mode, sens_fp8, sens_nvfp4)
        n += attach_tensor(model, "fkv", model._fused_kv_weight, mode=fkv_mode)
    pack_s = time.perf_counter() - t0
    place_drafter_quant(model)
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
        "muse_draft_quant (%s, mse=%d, pack=%.2fs): %d INT3-g64 + %d NVFP4-TMA4 + %d FP8-Marlin drafter linears (%d rejected), %.2f GB -> %.2f GB per step%s",
        mode,
        int(_int3_mse_enabled()),
        pack_s,
        _stats["admitted_int3"],
        _stats["admitted_nvfp4"],
        _stats["admitted_fp8"],
        _stats["rejected"],
        _stats["bytes_bf16"] / 1e9,
        _stats["bytes_quant"] / 1e9,
        snr_summary,
    )
    return n
