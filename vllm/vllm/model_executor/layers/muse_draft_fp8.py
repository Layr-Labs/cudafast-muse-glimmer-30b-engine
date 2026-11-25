# SPDX-License-Identifier: Apache-2.0
"""FP8 (e4m3, per-output-channel scale) weight-only copies of the pinned DFlash drafter's linears.

The drafter only PROPOSES tokens: every committed token is the target's own greedy token from the
verify step, and the per-depth oracles equal the serial tape. The drafter's arithmetic can therefore
change acceptance, never the emitted stream. A depth-6 speculative step reads the whole 5.1 GB BF16
drafter once; at verify width (M <= 16 rows) these GEMMs are weight-bandwidth bound on GB10, so a
W8A16 copy halves the bytes the drafter streams per step.

At load, each bias-free BF16 drafter linear (qkv/o/gate_up/down of the 5 layers, and `fc`) gets an
FP8 e4m3 copy with an fp32-computed per-row scale (amax / 448), repacked for vLLM's stock Marlin
W8A16 kernel (`ops.marlin_gemm`, b_q_type float8_e4m3fn, the same kernel family the target's NVFP4
path already runs on this box). Activations stay BF16; accumulation is fp32 in the kernel.

Calls with 1..16 rows (the draft block and the accepted-context rows) use the FP8 copy; anything
else (the 512-row prompt context after prefill) uses the original BF16 weight unchanged, so prefill
numerics are untouched. Every layer is admitted only after a load-time check of the Marlin output
against F.linear on the dequantized FP8 weight (finite, tight tolerance) and against the BF16 weight
(cosine); any failure or exception keeps that layer on its previous path. The target is never
touched. MUSE_DRAFT_FP8=0 disables it.
"""
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M = 16
_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")
_stats = {"admitted": 0, "rejected": 0, "bytes_bf16": 0, "bytes_fp8": 0}


def enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_FP8", "1") != "0"


# W4A16 mode: symmetric int4 (uint4b8), group 128, per-group fp-scale, run through stock Marlin
# (ops.marlin_gemm, b_q_type uint4b8, the GPTQ symmetric path). Tried first; a layer that fails its
# check stays on the FP8 path above. MUSE_DRAFT_W4=0 keeps every layer on FP8.
W4_GROUP = 128


def w4_enabled() -> bool:
    return os.environ.get("MUSE_DRAFT_W4", "1") != "0"


def _quantize_w4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] -> (codes uint8 [N, K] in 0..15 with w ~ (code - 8) * scale, fp32 scale [N, K / 128])."""
    n, k = w.shape
    g = k // W4_GROUP
    q = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    s = torch.empty((n, g), dtype=torch.float32, device=w.device)
    step = max(1, (32 << 20) // k)
    for r0 in range(0, n, step):
        wf = w[r0:r0 + step].float().view(-1, g, W4_GROUP)
        sc = (wf.abs().amax(dim=2) / 7.0).clamp_min(1e-30)
        codes = torch.round(wf / sc[:, :, None]).clamp_(-8, 7) + 8
        q[r0:r0 + step] = codes.view(-1, k).to(torch.uint8)
        s[r0:r0 + step] = sc
        del wf, sc, codes
    return q, s


def _pack_w4(w: torch.Tensor):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )

    n, k = w.shape
    q, s = _quantize_w4(w)
    # GPTQ layout: [K / 8, N] int32, 8 consecutive k nibbles per word, lowest k in the lowest nibble.
    qt = q.t().contiguous().view(k // 8, 8, n).to(torch.int32)
    packed = torch.zeros((k // 8, n), dtype=torch.int32, device=w.device)
    for j in range(8):
        packed |= qt[:, j, :] << (4 * j)
    del qt
    perm = torch.empty(0, dtype=torch.int, device=w.device)
    qweight = ops.gptq_marlin_repack(b_q_weight=packed, perm=perm, size_k=k, size_n=n, num_bits=4)
    del packed
    scales = marlin_permute_scales(s=s.t().contiguous().to(w.dtype), size_k=k, size_n=n, group_size=W4_GROUP)
    ws = marlin_make_workspace_new(w.device)
    return q, s, qweight, scales, ws


def _quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _pack(w: torch.Tensor):
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
    q, s = _quantize_rows(w)
    packed = pack_fp8_to_int32(q, size_k_first=False).T.contiguous()  # [K / 4, N] int32 (GPTQ layout)
    perm = torch.empty(0, dtype=torch.int, device=w.device)
    qweight = ops.gptq_marlin_repack(b_q_weight=packed, perm=perm, size_k=k, size_n=n, num_bits=8)
    del packed
    scales = marlin_permute_scales(s=s.to(w.dtype).view(1, n), size_k=k, size_n=n, group_size=-1)
    scales = fp8_fused_exponent_bias_into_scales(scales)
    ws = marlin_make_workspace_new(w.device)
    return q, s, qweight, scales, ws


def _marlin(x2: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor, n: int, k: int,
            w4: bool = False):
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
        b_q_type=scalar_types.uint4b8 if w4 else scalar_types.float8_e4m3fn,
        size_m=x2.size(0),
        size_n=n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )


def _check(w: torch.Tensor, q: torch.Tensor, s: torch.Tensor, qweight, scales, ws, w4: bool = False) -> bool:
    n, k = w.shape
    gen = torch.Generator(device=w.device)
    gen.manual_seed(1009 + n + 7 * k)
    wd = None
    try:
        for m in (1, 7, 8, 16):
            x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
            got = _marlin(x, qweight, scales, ws, n, k, w4)
            if got.shape != (m, n) or not bool(torch.isfinite(got).all()):
                return False
            if wd is None:
                if w4:
                    wd = ((q.float() - 8.0).view(n, -1, W4_GROUP) * s[:, :, None]).view(n, k).to(w.dtype)
                else:
                    wd = (q.float() * s[:, None]).to(w.dtype)
            ref = F.linear(x, wd).float()
            err = (got.float() - ref).abs().max().item()
            tol = 2e-2 * ref.abs().max().item() + 1e-3
            if not err <= tol:
                return False
            dense = F.linear(x, w).float()
            cos = F.cosine_similarity(got.float().flatten(), dense.flatten(), dim=0).item()
            if not cos >= (0.99 if w4 else 0.995):
                return False
        torch.cuda.synchronize(w.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_fp8: Marlin FP8 check failed (%s); layer keeps its BF16 path", e)
        return False
    return True


def attach_linear(layer: torch.nn.Module) -> bool:
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
    if n % 64 or k % 128:
        return False
    if w4_enabled():
        try:
            q, s, qweight, scales, ws = _pack_w4(w.data.contiguous())
            ok = _check(w.data, q, s, qweight, scales, ws, True)
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_fp8: W4 pack failed for %s (%s); trying FP8", tuple(w.shape), e)
            ok = False
        if ok:
            del q, s
            layer.register_buffer("_mfp8_qweight", qweight, persistent=False)
            layer.register_buffer("_mfp8_scales", scales, persistent=False)
            layer.register_buffer("_mfp8_ws", ws, persistent=False)
            layer._mfp8 = (n, k)
            layer._mfp8_w4 = True
            _stats["admitted"] += 1
            _stats["w4"] = _stats.get("w4", 0) + 1
            _stats["bytes_bf16"] += n * k * 2
            _stats["bytes_fp8"] += qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size()
            return True
    try:
        q, s, qweight, scales, ws = _pack(w.data.contiguous())
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_draft_fp8: pack failed for %s (%s)", tuple(w.shape), e)
        _stats["rejected"] += 1
        return False
    if not _check(w.data, q, s, qweight, scales, ws):
        _stats["rejected"] += 1
        return False
    del q, s
    layer.register_buffer("_mfp8_qweight", qweight, persistent=False)
    layer.register_buffer("_mfp8_scales", scales, persistent=False)
    layer.register_buffer("_mfp8_ws", ws, persistent=False)
    layer._mfp8 = (n, k)
    _stats["admitted"] += 1
    _stats["bytes_bf16"] += n * k * 2
    _stats["bytes_fp8"] += qweight.numel() * qweight.element_size()
    return True


@torch.library.custom_op("muse::draft_fp8_linear", mutates_args=())
def draft_fp8_linear(
    x: torch.Tensor, w: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _marlin(x2.contiguous(), qweight, scales, ws, n, k)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_fp8_linear.register_fake
def _(x, w, qweight, scales, ws):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@torch.library.custom_op("muse::draft_w4_linear", mutates_args=())
def draft_w4_linear(
    x: torch.Tensor, w: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _marlin(x2.contiguous(), qweight, scales, ws, n, k, True)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_w4_linear.register_fake
def _(x, w, qweight, scales, ws):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    op = torch.ops.muse.draft_w4_linear if getattr(layer, "_mfp8_w4", False) else torch.ops.muse.draft_fp8_linear
    return op(x, layer.weight, layer._mfp8_qweight, layer._mfp8_scales, layer._mfp8_ws)


def attach_drafter(model: torch.nn.Module) -> int:
    if not enabled():
        return 0
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in _NAMES:
            n += attach_linear(mod)
    logger.info(
        "muse_draft_fp8: %d drafter linears on Marlin (%d of them W4A16 group-%d, rest W8A16; %d rejected), "
        "%.2f GB -> %.2f GB per step",
        _stats["admitted"], _stats.get("w4", 0), W4_GROUP, _stats["rejected"], _stats["bytes_bf16"] / 1e9,
        _stats["bytes_fp8"] / 1e9,
    )
    return n
