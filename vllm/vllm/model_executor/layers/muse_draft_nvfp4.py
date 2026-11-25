# SPDX-License-Identifier: Apache-2.0
"""Load-time NVFP4 copies of the pinned DFlash drafter's small-row linears.

The copy uses the target checkpoint's W4A16 format: packed E2M1 values,
E4M3 group-16 scales, and one FP32 global scale.  It is then passed through
the same Marlin and decode6 preparation used by target NVFP4 layers, including
TMA4-ILV and coded-scale admission.  The original BF16 weight remains the
prompt-sized fallback; only M=1..8 drafter calls use this lossy proposal path.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_M = 8
_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")
_stats = {
    "admitted": 0,
    "rejected": 0,
    "tma4": 0,
    "bytes_bf16": 0,
    "bytes_nvfp4": 0,
}


def _mode() -> str:
    mode = os.environ.get("MUSE_DRAFT_QUANT", "nvfp4-mse").lower()
    if mode in ("nvfp4", "nvfp4-mse", "mse"):
        return "mse"
    if mode in ("nvfp4-maxabs", "maxabs", "amax"):
        return "maxabs"
    return "off"


def enabled() -> bool:
    return _mode() != "off"


def _fp4_magnitude_code(v: torch.Tensor) -> torch.Tensor:
    """Nearest positive E2M1 code, matching vLLM's NVFP4 tie rules."""
    q = (v > 0.25).to(torch.uint8)
    q.add_((v >= 0.75).to(torch.uint8))
    q.add_((v > 1.25).to(torch.uint8))
    q.add_((v >= 1.75).to(torch.uint8))
    q.add_((v > 2.5).to(torch.uint8))
    q.add_((v >= 3.5).to(torch.uint8))
    q.add_((v > 5.0).to(torch.uint8))
    return q


def _mse_quantize(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Groupwise scale search followed by exact NVFP4 nibble packing.

    Candidate group scales bracket the max-abs scale.  Each candidate is
    rounded to E4M3 before its E2M1 reconstruction error is evaluated, so the
    search optimizes the values actually consumed by decode6.
    """
    n, k = w.shape
    amax = w.abs().amax().float().clamp_min(1e-30)
    global_dequant = amax / (6.0 * 448.0)
    qweight = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn, device=w.device)
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=w.device,
    )
    # Wider-than-usual low-side search deliberately permits clipping a rare
    # group outlier when that reduces all-group squared error.
    ratios = (0.50, 0.625, 0.75, 0.875, 1.0, 1.125)
    step = max(1, (32 << 20) // k)
    for r0 in range(0, n, step):
        x = w[r0 : r0 + step].float().reshape(-1, k // 16, 16)
        base = x.abs().amax(dim=-1) / (6.0 * global_dequant)
        best_err = torch.full_like(base, float("inf"))
        best_scale = torch.zeros_like(base)
        for ratio in ratios:
            sf8 = (base * ratio).clamp_(0.0, 448.0).to(torch.float8_e4m3fn)
            sf = sf8.float()
            local = sf * global_dequant
            safe = local.clamp_min(1e-30)
            code = _fp4_magnitude_code(x.abs() / safe[..., None])
            recon = levels[code.long()] * local[..., None]
            err = ((x.abs() - recon) ** 2).sum(dim=-1)
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, sf, best_scale)
        best_scale8 = best_scale.to(torch.float8_e4m3fn)
        scales[r0 : r0 + x.shape[0]] = best_scale8
        local = best_scale8.float() * global_dequant
        code = _fp4_magnitude_code(x.abs() / local.clamp_min(1e-30)[..., None])
        code.bitwise_or_((x < 0).to(torch.uint8) << 3)
        code = code.reshape(-1, k)
        qweight[r0 : r0 + code.shape[0]] = code[:, 0::2] | (code[:, 1::2] << 4)
        del x, base, best_err, best_scale, best_scale8, local, code
    return qweight, scales, global_dequant.reshape(())


def _maxabs_quantize(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stock NVIDIA/vLLM max-abs NVFP4 quantizer, unswizzled."""
    from vllm._custom_ops import scaled_fp4_quant

    amax = w.abs().amax().float().clamp_min(1e-30)
    global_dequant = amax / (6.0 * 448.0)
    global_quant = global_dequant.reciprocal()
    qweight, scales = scaled_fp4_quant(
        w.contiguous(), global_quant, is_sf_swizzled_layout=False
    )
    return qweight, scales, global_dequant.reshape(())


def _prepare(
    w: torch.Tensor, mode: str
) -> tuple[torch.nn.Module, int]:
    from muse_marlin_gemv import decode_linear, prepare_decode_layout
    from muse_marlin6_gemv import decode_linear6, is_tma4, prepare_decode_layout6
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )

    n, k = w.shape
    qweight, scales, global_scale = (
        _mse_quantize(w) if mode == "mse" else _maxabs_quantize(w)
    )
    raw_bytes = qweight.nbytes + scales.nbytes + global_scale.nbytes
    holder = torch.nn.Module()
    holder.weight = torch.nn.Parameter(qweight, requires_grad=False)
    holder.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    holder.weight_global_scale = torch.nn.Parameter(
        global_scale, requires_grad=False
    )
    holder.params_dtype = w.dtype
    holder.input_size_per_partition = k
    holder.output_size_per_partition = n
    holder.bias = None
    prepare_fp4_layer_for_marlin(holder)
    prepare_decode_layout(holder)
    if getattr(holder, "_marlin_decode_weight", None) is None:
        raise RuntimeError(f"decode layout rejected shape {(n, k)}")
    if not prepare_decode_layout6(holder, decode_linear):
        raise RuntimeError(f"decode6 palette rejected shape {(n, k)}")
    if getattr(holder, "_marlin_decode_scales6", None) is None:
        raise RuntimeError(f"decode6 coded scales unavailable for shape {(n, k)}")
    holder._mnv4_is_tma4 = is_tma4(holder._marlin_decode_scales6)
    if not holder._mnv4_is_tma4:
        raise RuntimeError(f"TMA4 coded-scale admission rejected shape {(n, k)}")
    # Multi-row decode6 was admitted against the one-row reference by the
    # target loader.  Drafter attachment additionally proves every M=1..8
    # against the stock Marlin computation of these exact quantized tensors.
    gen = torch.Generator(device=w.device).manual_seed(4001 + n + 3 * k)
    for m in range(1, MAX_M + 1):
        x = torch.randn((m, k), generator=gen, device=w.device).to(w.dtype)
        got = decode_linear6(
            x,
            holder._marlin_decode_weight,
            holder._marlin_decode_scales6,
            holder._marlin_decode_palette,
            holder.weight_global_scale,
            holder.workspace,
            n,
            k,
            bool(holder.marlin_decode6_fixed_grid),
        )
        ref = apply_fp4_marlin_linear(
            x,
            holder.weight,
            holder.weight_scale,
            holder.weight_global_scale,
            holder.workspace,
            n,
            k,
        )
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            raise RuntimeError(f"decode6 M={m} differs from Marlin for shape {(n, k)}")
    return holder, raw_bytes


def _register(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    layer.register_buffer(name, value, persistent=False)


def attach_linear(layer: torch.nn.Module, mode: str) -> bool:
    w = getattr(layer, "weight", None)
    if (
        w is None
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "_mnv4", None) is not None
        or getattr(layer, "_mfp8", None) is not None
        or not isinstance(w, torch.Tensor)
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_cuda
    ):
        return False
    n, k = w.shape
    if n % 64 or k % 128 or n < 2048:
        return False
    try:
        holder, raw_bytes = _prepare(w.data.contiguous(), mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("muse_draft_nvfp4: rejected %s (%s)", tuple(w.shape), exc)
        _stats["rejected"] += 1
        return False
    _register(layer, "_mnv4_weight", holder._marlin_decode_weight)
    _register(layer, "_mnv4_scales", holder._marlin_decode_scales6)
    _register(layer, "_mnv4_palette", holder._marlin_decode_palette)
    _register(layer, "_mnv4_global_scale", holder.weight_global_scale)
    _register(layer, "_mnv4_workspace", holder.workspace)
    layer._mnv4 = (n, k, bool(holder.marlin_decode6_fixed_grid))
    _stats["admitted"] += 1
    _stats["tma4"] += int(holder._mnv4_is_tma4)
    _stats["bytes_bf16"] += w.nbytes
    _stats["bytes_nvfp4"] += raw_bytes
    return True


@torch.library.custom_op("muse::draft_nvfp4_linear", mutates_args=())
def draft_nvfp4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    fixed_grid: bool,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = w.shape
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        from muse_marlin6_gemv import decode_linear6

        out = decode_linear6(
            x2.contiguous(),
            tiled_weight,
            scales,
            palette,
            global_scale,
            workspace,
            n,
            k,
            fixed_grid,
        )
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], n)


@draft_nvfp4_linear.register_fake
def _(
    x, w, tiled_weight, scales, palette, global_scale, workspace, fixed_grid
):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.muse.draft_nvfp4_linear(
        x,
        layer.weight,
        layer._mnv4_weight,
        layer._mnv4_scales,
        layer._mnv4_palette,
        layer._mnv4_global_scale,
        layer._mnv4_workspace,
        layer._mnv4[2],
    )


def attach_drafter(model: torch.nn.Module) -> int:
    mode = _mode()
    if mode == "off":
        return 0
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in _NAMES:
            n += attach_linear(mod, mode)
    logger.info(
        "muse_draft_nvfp4: mode=%s admitted=%d rejected=%d TMA4=%d, %.2f GB BF16 -> %.2f GB NVFP4 per round",
        mode,
        _stats["admitted"],
        _stats["rejected"],
        _stats["tma4"],
        _stats["bytes_bf16"] / 1e9,
        _stats["bytes_nvfp4"] / 1e9,
    )
    return n
