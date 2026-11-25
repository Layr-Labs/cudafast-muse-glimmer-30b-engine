# SPDX-License-Identifier: Apache-2.0
"""Decode (M <= 16) FP8 w8a8 GEMMs through glimmer_fp8_ext (harness/kernels/glimmer_fp8): CUTLASS sm120 FP8 kernels whose
epilogue is vLLM's ScaledEpilogue arithmetic, D = bf16(scale_a * fp32(scale_b * acc)).
The one-row measurements select a 16x32x512 tile. M=2..16 uses the 16x32x128 tile.
The per-element k order is the same for both tiles (no split-K). Only
per-tensor scales (both scalars), no bias, bf16 output and the shapes listed in _CFG take this path. Prompt-sized calls
(M > 16) on weights admitted at load by a bitwise comparison with ops.cutlass_scaled_mm run through muse_fp8_prefill
(harness/kernels/muse_fp8_prefill: the same CUTLASS SM120 mainloop with tiles / pipeline depth chosen for 512 rows);
512-row qkv (4608x6656) and attention output-gate (4096x6656) GEMMs run on fp8ws (SP5D's kernel, Yukon submission
98c1361e) once their shape passed its bitwise admission; everything else calls ops.cutlass_scaled_mm unchanged.
The shape test is inside an opaque custom op."""
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import glimmer_fp8_ext as _ext  # built into KERNELS_OUT
except Exception:
    _ext = None

# (N, K) -> fp8x config id (0 = vLLM's own M16 config; 1..4 alternatives), chosen by quiet-GPU timing on GB10.
_CFG: dict[tuple[int, int], int] = {
    (4608, 6656): 2,   # qkv
    (4096, 6656): 2,   # attention output gate
    (6656, 4096): 2,   # o_proj
    (39936, 6656): 2,  # gate_up
    (6656, 19968): 2,  # down
}


try:
    # 512-row qkv and attention output-gate projections through fp8ws (SP5D, Yukon submission 98c1361e), admitted per
    # shape by a bitwise comparison with ops.cutlass_scaled_mm (vllm/model_executor/kernels/linear/scaled_mm/fp8ws.py).
    from vllm.model_executor.kernels.linear.scaled_mm.fp8ws import maybe_gemm as _fp8ws_maybe_gemm
except Exception:  # noqa: BLE001 - module or package absent: the routes below
    _fp8ws_maybe_gemm = None


try:
    # Prompt-sized FP8 GEMMs (M > 16) through CUTLASS SM120 kernels with deeper pipelines / other tiles than the stock
    # vLLM sm120 kernel (harness/kernels/muse_fp8_prefill); same mainloop instruction chain and k order, same epilogue.
    import muse_fp8_prefill as _prefill  # built into KERNELS_OUT

    if _prefill.lib() is None:
        _prefill = None
except Exception:  # noqa: BLE001 - package absent: stock cutlass for every prompt-sized FP8 GEMM
    _prefill = None

try:
    # 512-row attention output-gate GEMM of a folded FP8-attention layer reads the FP8 operand the native
    # sandwich wrote (harness/kernels/sandwich/muse_sandwich.py, gate-quant fold) and optionally fuses qkv+gate
    # into one persistent fp8ws dual-activation launch.
    from muse_sandwich import gemm_hook as _sandwich_gemm_hook
except Exception:  # noqa: BLE001 - package absent or older build: no fold
    _sandwich_gemm_hook = None

_decode_gemm_operand = None

# Weights (data_ptr of the finished [K, N] column-major e4m3 tensor) admitted at load by a bitwise comparison with
# ops.cutlass_scaled_mm on the layer's own weight; any other weight keeps the stock op.
_prefill_admitted: set[int] = set()
_prefill_real_checked: set[int] = set()


def prefill_admitted(weight: torch.Tensor) -> bool:
    return _prefill is not None and weight.data_ptr() in _prefill_admitted


def prefill_moved(old_ptr: int, new_ptr: int) -> None:
    if old_ptr in _prefill_admitted:
        _prefill_admitted.discard(old_ptr)
        _prefill_admitted.add(new_ptr)
    if old_ptr in _prefill_real_checked:
        _prefill_real_checked.discard(old_ptr)
        _prefill_real_checked.add(new_ptr)
_prefill_stats = {"admitted": 0, "rejected": 0, "served": 0, "fallback": 0, "real_equal": 0, "real_differ": 0,
                  "logged": False}


def _stock_scaled_mm(a, b, scale_a, scale_b):
    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None)


# DFlash verify rows (M = 2..8): the 16x32x512 tile measured fastest for one row also serves the verify block. Both
# tiles are one 16-row M tile over the same SM120 mainloop with the same per-element k order, so each output row is
# the same mma chain; a shape is admitted only after the 512 tile reproduces today's M = 2..16 tile bit for bit on a
# real weight of that shape at M = 2, 7 and 8. Unadmitted shapes keep their _CFG tile.
_ROWS512_MAX_M = 8
_rows512_ok: dict[tuple[int, int], bool] = {}


def admit_rows512(weight: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor) -> bool:
    """Load time, once per (N, K): admit the 512-deep tile for 2..8 rows if bitwise equal to the _CFG tile."""
    if (_ext is None or weight is None or weight_scale is None or input_scale is None or weight.dim() != 2
            or weight.dtype != torch.float8_e4m3fn or not weight.is_cuda
            or weight.stride(0) != 1 or weight.stride(1) != weight.size(0)
            or weight_scale.numel() != 1 or input_scale.numel() != 1
            or weight_scale.dtype != torch.float32 or input_scale.dtype != torch.float32):
        return False
    key = (weight.size(1), weight.size(0))
    cfg = _CFG.get(key)
    if cfg is None:
        return False
    if key in _rows512_ok:
        return _rows512_ok[key]
    ok = True
    try:
        sa, sb = input_scale.reshape(1), weight_scale.reshape(1)
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(key[0] * 31 + key[1])
        for m in (2, 7, 8):
            for mag in (0.05, 1.0, 40.0):
                x = torch.randn(m, key[1], generator=gen, device=weight.device) * mag
                x[:, ::3] *= 16
                x[:, 1::2] = -x[:, ::2]
                xq = (x / sa.float()).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
                ref = _ext.gemm(xq, weight, sa, sb, cfg)
                got = _ext.gemm(xq, weight, sa, sb, 512)
                if got.shape != ref.shape or not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    ok = False
                    break
            if not ok:
                break
    except Exception as e:  # noqa: BLE001
        logger.warning("glimmer_fp8: 512-tile verify-rows check failed (%s); keeping the M=2..16 tile", e)
        ok = False
    _rows512_ok[key] = ok
    logger.info("glimmer_fp8: %dx%d FP8 GEMM %s the 16x32x512 tile for 2..%d rows", key[0], key[1],
                "admits" if ok else "keeps its tile instead of", _ROWS512_MAX_M)
    return ok


def admit_prefill(weight: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor) -> bool:
    """Load time: admit this FP8 layer's prompt-sized GEMMs to muse_fp8_prefill if bitwise equal to the stock op."""
    if (_prefill is None or weight is None or weight_scale is None or input_scale is None
            or weight.dim() != 2 or _prefill.config(weight.size(1), weight.size(0)) is None):
        return False
    try:
        ok = _prefill.admit(weight, weight_scale.reshape(1), input_scale.reshape(1), _stock_scaled_mm)
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_fp8_prefill: admission check failed (%s); keeping cutlass_scaled_mm", e)
        ok = False
    _prefill_stats["admitted" if ok else "rejected"] += 1
    if ok:
        _prefill_admitted.add(weight.data_ptr())
    return ok


def _prefill_gemm(a, b, scale_a, scale_b):
    """M > 16: muse_fp8_prefill for admitted weights, else stock. Outside graph capture, the first call per weight is
    also run through the stock op on the same (real) activation and compared bitwise; the counts are logged once."""
    if (_prefill is not None and b.data_ptr() in _prefill_admitted
            and _prefill.supported(a, b, scale_a, scale_b)):
        cfg = _prefill.config(b.size(1), b.size(0))
        out = _prefill.gemm(a, b, scale_a.reshape(1), scale_b.reshape(1), cfg)
        _prefill_stats["served"] += 1
        if b.data_ptr() not in _prefill_real_checked and not torch.cuda.is_current_stream_capturing():
            _prefill_real_checked.add(b.data_ptr())
            ref = _stock_scaled_mm(a, b, scale_a, scale_b)
            if torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                _prefill_stats["real_equal"] += 1
            else:
                _prefill_stats["real_differ"] += 1
                _prefill_admitted.discard(b.data_ptr())
                logger.error("muse_fp8_prefill: real-activation output differs from cutlass_scaled_mm for a %dx%dx%d "
                             "GEMM; this weight now uses cutlass_scaled_mm", a.size(0), b.size(1), b.size(0))
                out = ref
            if len(_prefill_real_checked) == _prefill_stats["admitted"]:
                logger.info("muse_fp8_prefill: real-activation bitwise check %d/%d equal (%d differ); calls so far: "
                            "%d served, %d stock", _prefill_stats["real_equal"], len(_prefill_real_checked),
                            _prefill_stats["real_differ"], _prefill_stats["served"], _prefill_stats["fallback"])
    else:
        out = _stock_scaled_mm(a, b, scale_a, scale_b)
        _prefill_stats["fallback"] += 1
    if not _prefill_stats["logged"]:
        _prefill_stats["logged"] = True
        logger.info("muse_fp8_prefill: prompt FP8 GEMM layers admitted %d, rejected %d; first call %s",
                    _prefill_stats["admitted"], _prefill_stats["rejected"],
                    "served" if _prefill_stats["served"] else "stock")
    return out


def _prompt_route(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if a.size(0) == 512 and _fp8ws_maybe_gemm is not None:
        out = _fp8ws_maybe_gemm(a, b, scale_a, scale_b)
        if out is not None:
            return out
    if _prefill is not None and b.data_ptr() in _prefill_admitted and _prefill.supported(a, b, scale_a, scale_b):
        return _prefill_gemm(a, b, scale_a, scale_b)
    return _prefill_gemm(a, b, scale_a, scale_b)


@torch.library.custom_op("glimmer::fp8_scaled_mm", mutates_args=())
def fp8_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if a.size(0) == 512 and _sandwich_gemm_hook is not None:
        return _sandwich_gemm_hook(a, b, scale_a, scale_b, lambda x: _prompt_route(x, b, scale_a, scale_b))
    if a.size(0) > 16:
        return _prompt_route(a, b, scale_a, scale_b)
    if _decode_gemm_operand is not None:
        a = _decode_gemm_operand(a)
    if (_ext is not None and 0 < a.size(0) <= 16
            and a.is_cuda and a.device == b.device == scale_a.device == scale_b.device
            and scale_a.numel() == scale_b.numel() == 1
            and scale_a.dtype == scale_b.dtype == torch.float32
            and a.dtype == b.dtype == torch.float8_e4m3fn
            and a.is_contiguous() and b.stride(0) == 1 and b.stride(1) == b.size(0)):
        cfg = _CFG.get((b.size(1), b.size(0)))
        if cfg is not None:
            if a.size(0) == 1 or (a.size(0) <= _ROWS512_MAX_M and _rows512_ok.get((b.size(1), b.size(0)), False)):
                cfg = 512
            return _ext.gemm(a, b, scale_a.reshape(1), scale_b.reshape(1), cfg)
    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None)


@fp8_scaled_mm.register_fake
def _(a, b, scale_a, scale_b):
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.bfloat16)

try:
    import muse_fp8_swiglu as _swiglu
except ImportError:
    _swiglu = None


def prepare_fp8_swiglu_weight(weight: torch.Tensor) -> torch.Tensor | None:
    return _swiglu.prepare(weight) if _swiglu is not None else None


@torch.library.custom_op("glimmer::fp8_gate_swiglu", mutates_args=())
def fp8_gate_swiglu(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    return _swiglu.apply(a, b, scale_a, scale_b, down_scale)


@fp8_gate_swiglu.register_fake
def _(a, b, scale_a, scale_b, down_scale):
    return a.new_empty((a.size(0), b.size(1) // 2))
