"""Use the native row staging for the two verified Inductor sandwich kernels.

Gate-quant fold (prefill3): in an FP8-attention layer the 512-row sandwich writes the bf16 normed row `hidden`, and
the compiled requant kernel of the attention output-gate projection (triton_poi_fused__to_copy_clamp_mul_reciprocal_0)
reads it back from DRAM after the qkv GEMM to quantize it with the gate's own input scale. For a layer registered
with register_gate() the native sandwich also writes that FP8 operand (same formula, from the same bf16-rounded value)
into a per-layer buffer, the requant launch is skipped, and the gate GEMM (glimmer_fp8.fp8_scaled_mm) reads the
buffer instead of the requant's output. A layer is folded only after (1) the compiled requant kernel itself, run on
all 65,536 bf16 bit patterns with the layer's real gate scale, equals the native formula byte for byte, (2) on a
real eager 512-row launch outside graph capture the folding sandwich wrote the same rows / hidden / qkv FP8 bytes as
the plain native sandwich and the same gate FP8 bytes as the compiled requant, and (3) that requant's output was the
operand of the very next 512-row FP8 GEMM. MUSE_SANDWICH_GATE_FOLD=0 switches it off.
"""

import ast
from collections import Counter
import ctypes
from functools import cache
from hashlib import sha256
import os
from pathlib import Path


_library = ctypes.CDLL(str(Path(__file__).with_name("libmuse_sandwich.so")))
_launch = _library.muse_sandwich
_pointer = ctypes.c_void_p
_launch.argtypes = [ctypes.POINTER(_pointer), ctypes.c_int, ctypes.c_bool, ctypes.c_bool, _pointer]
_launch.restype = ctypes.c_int
_launch_counts = Counter()
_gate_launch = getattr(_library, "muse_sandwich_gate", None)
_gate_formula = getattr(_library, "muse_sandwich_gate_formula", None)
if _gate_launch is not None and _gate_formula is not None:
    _gate_launch.argtypes = [ctypes.POINTER(_pointer), ctypes.c_int, ctypes.c_bool, ctypes.c_bool, _pointer]
    _gate_launch.restype = ctypes.c_int
    _gate_formula.argtypes = [_pointer, _pointer, _pointer, ctypes.c_int, _pointer]
    _gate_formula.restype = ctypes.c_int
_hint_launch = getattr(_library, "muse_sandwich_hint", None)
if _hint_launch is not None:
    _hint_launch.argtypes = [ctypes.POINTER(_pointer), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _pointer]
    _hint_launch.restype = ctypes.c_int
PREFILL_CS = os.environ.get("MUSE_SANDWICH_PREFILL_CS", "1") != "0"
_set_prefill_cs = getattr(_library, "muse_sandwich_set_prefill_cs", None)
if _set_prefill_cs is not None:
    _set_prefill_cs.argtypes = [ctypes.c_int]
    _set_prefill_cs.restype = None
    _set_prefill_cs(1 if PREFILL_CS else 0)
_set_pdl = getattr(_library, "muse_sandwich_set_pdl", None)
if _set_pdl is not None:
    _set_pdl.argtypes = [ctypes.c_int]
    _set_pdl.restype = ctypes.c_int


def set_pdl(on: bool) -> bool | None:
    if _set_pdl is None:
        return None
    return bool(_set_pdl(1 if on else 0))
FOLD = os.environ.get("MUSE_SANDWICH_GATE_FOLD", "1") == "1" and _gate_launch is not None
# 512-row prefill only, cache hints only (no value changes): store the folded gate operand / the new residual row at
# L2 evict_last priority, so that they are still in L2 when the gate GEMM / the post-attention norm read them after
# the qkv (and attention) weight streams. Paired with MUSE_SANDWICH_PREFILL_CS ld.global.cs streaming on dead down/res.
EVICT_LAST = os.environ.get("MUSE_SANDWICH_GATE_EVICT_LAST", "1") != "0"
RESIDUAL_EVICT_LAST = os.environ.get("MUSE_SANDWICH_RESIDUAL_EVICT_LAST", "0") == "1"
# Not storing `hidden` in folded layers: its only reader there is the skipped requant (in-boot A/B ab-p3c-1: logits
# and final hidden rows bitwise equal with it dropped; writing it costs the folding sandwich ~27 us per layer).
DROP_HIDDEN = os.environ.get("MUSE_SANDWICH_DROP_HIDDEN", "1") == "1"
_REQUANT = "triton_poi_fused__to_copy_clamp_mul_reciprocal_0"
_REQUANT_BODY = "f7a07e4acde4818e0fe4bcb2f62a211fa163724414f5efcce3d93ee433b8300f"
_ROWS, _WIDTH = 512, 6656


class _Layer:
    __slots__ = ("qkv_scale", "gate_scale", "buffer", "state", "formula", "checked_hidden", "checked_out",
                 "dual", "qkv_call", "gate_out")

    def __init__(self, qkv_scale, gate_scale, buffer):
        self.qkv_scale, self.gate_scale, self.buffer = qkv_scale, gate_scale, buffer
        self.state = "unchecked"  # -> "sandwich_ok" -> "requant_ok" -> "admitted" | "rejected"
        self.formula = None
        self.checked_hidden = self.checked_out = None
        self.dual = "unchecked"  # qkv + gate GEMMs as one dual-activation launch: -> "admitted" | "rejected"
        self.qkv_call = None  # (a, b, sa, sb, out) of the layer's normal qkv GEMM, kept for the dual check
        self.gate_out = None  # gate GEMM output the dual launch computed with the qkv GEMM


_layers = {}  # qkv input-scale data_ptr -> _Layer
_pending = None  # (data_ptr of the hidden rows, _Layer) of the last folding / checked sandwich launch
_expect = None  # (requant output data_ptr, _Layer, folded) awaiting the next 512-row FP8 GEMM
active = FOLD  # runtime switch (diagnostics: a graph captured with active False runs the plain kernels)
_fold_counts = Counter()


def register_gate(qkv_scale, gate_scale) -> bool:
    """An FP8-attention layer: its qkv input scale (the folding sandwich's `scale`) and its output-gate input scale."""
    import torch

    if not FOLD:
        return False
    for t in (qkv_scale, gate_scale):
        if not (isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32 and t.numel() == 1):
            return False
    key = qkv_scale.data_ptr()
    if key not in _layers:
        buffer = torch.empty((_ROWS, _WIDTH), dtype=torch.float8_e4m3fn, device=qkv_scale.device)
        _layers[key] = _Layer(qkv_scale, gate_scale, buffer)
    return True


def _formula_ok(kernel, original, layer, stream) -> bool:
    """The compiled requant on all 65,536 bf16 bit patterns vs the native formula, with the layer's gate scale."""
    import torch

    dev = layer.gate_scale.device
    bits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16)
    theirs = torch.empty(65536, dtype=torch.float8_e4m3fn, device=dev)
    mine = torch.empty(65536, dtype=torch.uint8, device=dev)
    original(kernel, bits.view(torch.bfloat16), layer.gate_scale, theirs, 65536, stream=stream)
    status = _gate_formula(bits.data_ptr(), layer.gate_scale.data_ptr(), mine.data_ptr(), 65536, stream)
    torch.cuda.synchronize()
    return status == 0 and torch.equal(theirs.view(torch.uint8), mine)

_symbols = {
    "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3",
    "triton_red_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_3",
}
_verified_bodies = {
    "096b4dbdbe218cba4368a8a1de38f0bab983e69643a25662d9d9832d8b4bf1a2": False,
    "6611b90dd2aa0ecd3ed3f75b03e5c73895b89554458babc3ab7088c92130e290": True,
}
_inputs = ("*bf16", "*bf16", "*bf16", "*fp32", "*bf16", "*bf16", "*bf16")
_dimensions = ("i32", "i32", "constexpr", "constexpr")


@cache
def _mode(source):
    definitions = ast.parse(source).body
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return None
    kernel = definitions[0]
    kernel.name = "sandwich"
    kernel.decorator_list = []
    body = ast.dump(kernel, include_attributes=False)
    return _verified_bodies.get(sha256(body.encode()).hexdigest())


def _match(autotuner, args):
    function = autotuner.fn
    if getattr(function, "__name__", None) not in _symbols:
        return None
    props = autotuner.device_props
    if props.type != "cuda" or props.cc != 121 or props.multi_processor_count != 48:
        return None
    if len(autotuner.launchers) != 1 or not autotuner._cached_launcher:
        return None
    config = autotuner.launchers[0].config
    if (config.num_warps != 8 or config.num_stages != 1 or config.num_ctas != 1
            or config.kwargs != {"XBLOCK": 1, "R0_BLOCK": 1024}):
        return None
    quantize = _mode(function.src)
    if quantize is None:
        return None
    signature = tuple(autotuner.triton_meta["signature"].values())
    quantization_and_outputs = ("*fp32", "*bf16", "*fp8e4nv") if quantize else ("*bf16",)
    if signature != _inputs + quantization_and_outputs + _dimensions or len(args) != len(signature) - 2:
        return None
    rows, width = args[-2:]
    if type(rows) is not int or type(width) is not int or rows not in (1, 512) or width != 6656:
        return None
    addresses = [tensor.data_ptr() for tensor in args[:-2]]
    if any(address % 8 for address in addresses):
        return None
    return quantize, rows, addresses


@cache
def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, get_active_debug_mode

    original = CachingAutotuner.run

    def run(autotuner, *args, stream, benchmark_run=False, **kwargs):
        if not benchmark_run and not kwargs and not autotuner.triton_interpret:
            match = _match(autotuner, args)
            if match is not None and not get_active_debug_mode():
                quantize, rows, addresses = match
                if quantize and rows == _ROWS and _layers and _fold_sandwich(args, addresses, stream):
                    return None
                pointers = (_pointer * len(addresses))(*addresses)
                if not quantize and rows == _ROWS and RESIDUAL_EVICT_LAST and active and _hint_launch is not None:
                    status = _hint_launch(pointers, rows, 0, 0, 1, stream)
                else:
                    status = _launch(pointers, rows, quantize, rows == 512, stream)
                if status:
                    raise RuntimeError(f"CUDA sandwich launch failed with error {status}")
                _launch_counts[rows, quantize] += 1
                return None
            if _layers and getattr(autotuner.fn, "__name__", None) == _REQUANT:
                if _fold_requant(autotuner, original, args, stream):
                    return None
        return original(autotuner, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)

    CachingAutotuner.run = run


def launch_summary():
    return ", ".join(
        f"{count} {'FP8-output' if quantize else 'BF16-output'} at {rows} rows"
        for (rows, quantize), count in sorted(_launch_counts.items())
    )


# --------------------------------------------------------------------------- gate-quant fold (prefill3)

def _gate_pointers(addresses, layer):
    return (_pointer * 12)(*addresses, layer.gate_scale.data_ptr(), layer.buffer.data_ptr())


def _fold_sandwich(args, addresses, stream) -> bool:
    """A 512-row quantizing sandwich launch of a registered layer. True when this function made the launch."""
    import torch

    global _expect, _pending
    _pending = None
    layer = _layers.get(addresses[7])
    if layer is None or layer.state == "rejected":
        return False
    if _expect is not None:
        if _expect[2]:
            raise RuntimeError("muse_sandwich gate fold: a folded requant output was never consumed by its GEMM")
        _expect[1].state = "rejected"  # the checked requant's output did not reach a 512-row FP8 GEMM
        _fold_counts["rejected_order"] += 1
        _expect = None
    if not active:
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    hidden = args[8]
    if layer.state == "admitted":
        if (EVICT_LAST or RESIDUAL_EVICT_LAST) and DROP_HIDDEN and _hint_launch is not None:
            status = _hint_launch(_gate_pointers(addresses, layer), _ROWS, 1, int(EVICT_LAST), int(RESIDUAL_EVICT_LAST),
                                  stream)
        else:
            status = _gate_launch(_gate_pointers(addresses, layer), _ROWS, True, not DROP_HIDDEN, stream)
        if status:
            raise RuntimeError(f"CUDA folding sandwich launch failed with error {status}")
        _pending = (hidden.data_ptr(), layer, args[9].data_ptr())
        _launch_counts[_ROWS, True] += 1
        _fold_counts["sandwich_folded"] += 1
        return True
    if capturing or layer.state != "unchecked":
        return False  # checks run only on eager launches; until then the plain sandwich runs
    # Check: the plain native sandwich writes the real outputs; the folding one writes copies from the same inputs.
    snapshot = [t.clone() for t in args[:10]]
    pointers = (_pointer * 10)(*addresses)
    status = _launch(pointers, _ROWS, True, True, stream)
    if status:
        raise RuntimeError(f"CUDA sandwich launch failed with error {status}")
    _launch_counts[_ROWS, True] += 1
    copy_addresses = [t.data_ptr() for t in snapshot]
    status = _gate_launch(_gate_pointers(copy_addresses, layer), _ROWS, True, True, stream)
    torch.cuda.synchronize()
    same = status == 0 and all(
        torch.equal(args[i].reshape(-1).view(torch.uint8), snapshot[i].reshape(-1).view(torch.uint8)) for i in (0, 8, 9))
    if same:
        layer.state = "sandwich_ok"
        layer.checked_hidden = hidden.data_ptr()
        _pending = (hidden.data_ptr(), layer, args[9].data_ptr())
    else:
        layer.state = "rejected"
        _fold_counts["rejected_sandwich"] += 1
    return True


def _fold_requant(autotuner, original, args, stream) -> bool:
    """The compiled gate requant. True when this function handled the launch (ran it, or skipped it as folded)."""
    import torch

    global _expect, _pending
    if _pending is None or len(args) != 4 or not all(isinstance(t, torch.Tensor) for t in args[:3]):
        return False
    hidden, scale, out, numel = args
    if hidden.data_ptr() != _pending[0] or scale.data_ptr() != _pending[1].gate_scale.data_ptr():
        if _pending[1].state == "admitted":
            raise RuntimeError("muse_sandwich gate fold: a requant ran between a folded sandwich and its own requant")
        return False
    layer = _pending[1]
    _pending = None
    if numel != _ROWS * _WIDTH or out.numel() != numel or out.dtype != torch.float8_e4m3fn or not out.is_contiguous():
        if layer.state == "admitted":
            raise RuntimeError("muse_sandwich gate fold: the folded layer's requant launch changed shape")
        layer.state = "rejected"
        _fold_counts["rejected_requant_args"] += 1
        return False
    if layer.state == "admitted":
        _expect = (out.data_ptr(), layer, True)
        _fold_counts["requant_skipped"] += 1
        return True
    if layer.state != "sandwich_ok" or torch.cuda.is_current_stream_capturing():
        return False
    if getattr(autotuner, "fn", None) is None or _body_hash(autotuner.fn.src) != _REQUANT_BODY:
        layer.state = "rejected"
        _fold_counts["rejected_requant_source"] += 1
        return False
    original(autotuner, *args, stream=stream)
    torch.cuda.synchronize()
    if layer.formula is None:
        layer.formula = _formula_ok(autotuner, original, layer, stream)
    if layer.formula and torch.equal(out.reshape(-1).view(torch.uint8), layer.buffer.reshape(-1).view(torch.uint8)):
        layer.state = "requant_ok"
        _expect = (out.data_ptr(), layer, False)
    else:
        layer.state = "rejected"
        _fold_counts["rejected_requant_bytes"] += 1
    return True


@cache
def _body_hash(source):
    definitions = ast.parse(source).body
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return None
    kernel = definitions[0]
    kernel.name = "requant"
    kernel.decorator_list = []
    return sha256(ast.dump(kernel, include_attributes=False).encode()).hexdigest()


# prefill3: dual-activation launch. With the fold admitted, both FP8 operands of a layer exist when its qkv GEMM is
# called; one fp8ws launch computes the qkv and the gate GEMM (each output element as the separate kernels compute it:
# one ascending-K chain from +0, then bf16_rn(fmul(fmul(acc, sb), sa))) and the gate call returns the stashed output.
# Admitted per layer on an eager forward: the dual launch equals the separate qkv and gate results bit for bit.
DUAL = os.environ.get("MUSE_FP8_QKV_GATE_DUAL", "1") != "0"


def _fp8ws():
    try:
        import fp8ws
    except ImportError:
        return None
    return fp8ws if hasattr(fp8ws, "gemm_dual_a") else None


def gemm_hook(a, b, sa, sb, compute):
    """The 512-row FP8 GEMM a[512, K] @ b. `compute(a)` runs the normal route. Returns the output tensor."""
    import torch

    global _expect
    pending = _pending
    if pending is not None and len(pending) == 3 and a.data_ptr() == pending[2] and active:
        layer = pending[1]
        if layer.state == "admitted" and DUAL and layer.dual == "admitted":
            g = layer.qkv_call
            lib = _fp8ws()
            outs = lib.gemm_dual_a(a, layer.buffer, b, g[1], sa, layer.gate_scale, sb, g[3]) if lib else None
            if outs is None:
                raise RuntimeError("muse_sandwich dual qkv+gate launch failed")
            layer.gate_out = outs[1]
            _fold_counts["dual_launches"] += 1
            return outs[0]
        out = compute(a)
        if layer.state == "admitted" and DUAL and layer.dual == "unchecked" and not torch.cuda.is_current_stream_capturing():
            layer.qkv_call = (a, b, sa, sb, out)
        return out
    if _expect is None:
        return compute(a)
    ptr, layer, folded = _expect
    if a.data_ptr() == ptr and folded and layer.gate_out is not None:
        _expect = None
        out, layer.gate_out = layer.gate_out, None
        _fold_counts["dual_gate_returned"] += 1
        return out
    a2 = gemm_operand(a)
    out = compute(a2)
    if (folded and DUAL and layer.dual == "unchecked" and layer.qkv_call is not None and a.data_ptr() == ptr
            and not torch.cuda.is_current_stream_capturing()):
        q = layer.qkv_call
        lib = _fp8ws()
        outs = lib.gemm_dual_a(q[0], a2, q[1], b, q[2], sa, q[3], sb) if lib else None
        torch.cuda.synchronize()
        same = outs is not None and torch.equal(outs[0].view(torch.int16), q[4].view(torch.int16)) and torch.equal(
            outs[1].view(torch.int16), out.view(torch.int16))
        layer.dual = "admitted" if same else "rejected"
        layer.qkv_call = (None, b, None, sb, None) if same else None  # keep only the gate weight / scale
        _fold_counts["dual_admitted" if same else "dual_rejected"] += 1
    return out


def gemm_operand(a):
    """Called by the 512-row FP8 GEMM with its activation operand: the folded layer's buffer when `a` is the output
    of a skipped requant (a no-op otherwise). The requant output must be the next 512-row FP8 GEMM's operand."""
    global _expect
    if _expect is None:
        return a
    ptr, layer, folded = _expect
    _expect = None
    if a.data_ptr() != ptr:
        if folded:
            raise RuntimeError("muse_sandwich gate fold: the skipped requant's output is not the next GEMM operand")
        layer.state = "rejected"
        _fold_counts["rejected_order"] += 1
        return a
    if not folded:
        layer.state = "admitted"
        _fold_counts["admitted"] += 1
        return a
    if a.shape != layer.buffer.shape or a.dtype != layer.buffer.dtype or not a.is_contiguous():
        raise RuntimeError("muse_sandwich gate fold: GEMM operand shape changed")
    _fold_counts["gemm_redirected"] += 1
    return layer.buffer


def fold_summary():
    states = Counter(layer.state for layer in _layers.values())
    return (f"{len(_layers)} FP8-attention layers registered; " + ", ".join(f"{v} {k}" for k, v in sorted(states.items()))
            + "; " + ", ".join(f"{k} {v}" for k, v in sorted(_fold_counts.items())))
