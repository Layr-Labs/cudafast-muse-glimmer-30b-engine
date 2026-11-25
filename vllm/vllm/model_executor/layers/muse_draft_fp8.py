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


def _marlin(x2: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor, n: int, k: int):
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


def _check(w: torch.Tensor, q: torch.Tensor, s: torch.Tensor, qweight, scales, ws) -> bool:
    n, k = w.shape
    gen = torch.Generator(device=w.device)
    gen.manual_seed(1009 + n + 7 * k)
    wd = None
    try:
        for m in (1, 7, 8, 16):
            x = (torch.randn((m, k), generator=gen, device=w.device) * 0.5).to(w.dtype)
            got = _marlin(x, qweight, scales, ws, n, k)
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


_I4_GROUP = 32


def _fit_i4_groups(v: torch.Tensor):
    # v: FP32 [out, groups, 32]. Two starts and two alternating least
    # squares updates. Use the actual BF16 scale when choosing the codes.
    bound = torch.maximum(v.amax(-1) / 7, -v.amin(-1) / 8).clamp_min_(1e-12)
    best_err = torch.full_like(bound, float("inf"))
    best_s = bound
    for init in (1.0, 0.84):
        sc = bound * init
        for _ in range(2):
            q = torch.round(v / sc.unsqueeze(-1)).clamp_(-8, 7)
            sc = ((v * q).sum(-1) / q.square().sum(-1).clamp_min_(1)).clamp_min_(1e-12)
        sc = sc.to(torch.bfloat16).float()
        q = torch.round(v / sc.unsqueeze(-1)).clamp_(-8, 7)
        err = (v - q * sc.unsqueeze(-1)).square_().sum(-1)
        best_s = torch.where(err < best_err, sc, best_s)
        best_err = torch.minimum(err, best_err)
    return torch.round(v / best_s.unsqueeze(-1)).clamp_(-8, 7), best_s


def _i4_gemm(x, qweight, scales, ws, n, k):
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    return ops.marlin_gemm(a=x, c=None, b_q_weight=qweight, b_bias=None, b_scales=scales,
                           a_scales=None, global_scale=None, b_zeros=None, g_idx=None, perm=None,
                           workspace=ws, b_q_type=scalar_types.uint4b8,
                           size_m=x.shape[0], size_n=n, size_k=k,
                           use_atomic_add=False, use_fp32_reduce=True)


@torch.no_grad()
def _try_mlp_i4(layer) -> bool:
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new, marlin_permute_scales,
    )

    w = layer.weight.detach()
    if (not w.is_cuda or w.dtype != torch.bfloat16 or getattr(layer, "bias", None) is not None
            or w.shape not in ((39936, 6656), (6656, 19968))):
        return False
    n, k = w.shape
    g = _I4_GROUP
    packed = torch.empty((k // 8, n), dtype=torch.int32, device=w.device)
    scales = torch.empty((k // g, n), dtype=w.dtype, device=w.device)
    gen = torch.Generator(device=w.device).manual_seed(107 * n + k)
    x = torch.randn((16, k), dtype=w.dtype, device=w.device, generator=gen)
    ref = torch.empty((16, n), dtype=w.dtype, device=w.device)
    shift = torch.arange(8, dtype=torch.int32, device=w.device) * 4
    for i in range(0, n, 256):
        wf = w[i:i+256].reshape(-1, k // g, g).float()
        q, sc = _fit_i4_groups(wf)
        cols = sc.shape[0]
        decoded = (q * sc.unsqueeze(-1)).reshape(cols, k).to(w.dtype)
        ref[:, i:i+cols] = F.linear(x, decoded)
        codes = (q.reshape(cols, -1, 8).to(torch.int32) + 8) << shift
        packed[:, i:i+cols] = codes.sum(-1, dtype=torch.int32).T
        scales[:, i:i+cols] = sc.T
    perm = torch.empty(0, dtype=torch.int, device=w.device)
    qw = ops.gptq_marlin_repack(b_q_weight=packed, perm=perm, size_k=k, size_n=n, num_bits=4)
    del packed
    sc = marlin_permute_scales(scales, k, n, g)
    ws = marlin_make_workspace_new(w.device)
    for m in (1, 7, 16):
        got = _i4_gemm(x[:m], qw, sc, ws, n, k).float()
        refm = ref[:m].float()
        if not ((got - refm).abs().max() <= 0.02 * refm.abs().max() + 0.001).item():
            return False
        dense = F.linear(x[:m], w).float()
        if not (F.cosine_similarity(got.flatten(), dense.flatten(), dim=0).item() >= 0.99):
            return False
    layer.register_buffer("_mi4_qw", qw, persistent=False)
    layer.register_buffer("_mi4_scales", sc, persistent=False)
    layer.register_buffer("_mi4_ws", ws, persistent=False)
    layer._mi4 = True
    logger.info("DFlash MLP i4/g32 admitted %dx%d: %d packed + scale bytes", n, k, qw.nbytes + sc.nbytes)
    return True


@torch.library.custom_op("muse::draft_i4_mlp", mutates_args=())
def draft_i4_mlp(x: torch.Tensor, w: torch.Tensor, qw: torch.Tensor, sc: torch.Tensor,
                 ws: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    x2 = x.reshape(-1, k)
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        z = _i4_gemm(x2.contiguous(), qw, sc, ws, n, k)
    else:
        z = F.linear(x2, w)
    return z.reshape(*x.shape[:-1], n)


@draft_i4_mlp.register_fake
def _(x, w, qw, sc, ws):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


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


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.muse.draft_fp8_linear(x, layer.weight, layer._mfp8_qweight, layer._mfp8_scales, layer._mfp8_ws)


def attach_drafter(model: torch.nn.Module) -> int:
    if not enabled():
        return 0
    n = 0
    for name, mod in model.named_modules():
        if (name.split(".")[-1] in ("gate_up_proj", "down_proj")
                and os.environ.get("MUSE_DRAFT_MLP_I4", "1") != "0"):
            try:
                if _try_mlp_i4(mod):
                    n += 1
                    continue
            except (RuntimeError, ImportError, AttributeError, ValueError) as err:
                logger.warning("DFlash MLP i4 not admitted on %s: %s", name, err)
        if name.split(".")[-1] in _NAMES:
            n += attach_linear(mod)
    logger.info(
        "muse_draft_fp8: %d drafter linears on FP8 Marlin W8A16 (%d rejected), %.2f GB -> %.2f GB per step",
        _stats["admitted"], _stats["rejected"], _stats["bytes_bf16"] / 1e9, _stats["bytes_fp8"] / 1e9,
    )
    return n
