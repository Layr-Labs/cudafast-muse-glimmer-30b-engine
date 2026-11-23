"""Complete-K paired gate/up and exact compiled SwiGLU for the SM121 prefill.

Three kernels with the same C ABI and the same bits: ``libmarlinw_swiglu`` (64-row
tiles, the reference), ``libmarlinw_swiglu2`` (128-row tiles, each weight
fragment dequantized once for twice the rows) and ``libmarlinw_swiglu3`` (the
128-row tiles with their operands copied to shared memory by TMA instead of by
the MMA warps). A layer uses the second or third one only after a bitwise
comparison with the first on its own weights.
"""
import ctypes
import os
from pathlib import Path

import torch

_ready = set()
_lib = None
_lib2 = None
_lib3 = None
_fast = set()  # weight data pointers whose layer passed the comparison (128-row kernel)
_fast3 = set()  # the subset that also passed it with the TMA-fed kernel


def _load(name):
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name(name)))
    except OSError:
        return None
    lib.swiglu_prepare.argtypes = []
    lib.swiglu_prepare.restype = ctypes.c_int
    lib.swiglu_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
    lib.swiglu_gemm.restype = ctypes.c_int
    return lib


def _run(lib, x, weight, scales, global_scale):
    out = x.new_empty((512, 19968))
    with torch.cuda.device(x.device):
        status = lib.swiglu_gemm(x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                                 global_scale.data_ptr(), out.data_ptr(), x.stride(0),
                                 torch.cuda.current_stream(x.device).cuda_stream)
    if status:
        raise RuntimeError(f"Paired SwiGLU launch returned CUDA status {status}")
    return out


def _check_input(gen, device):
    x = torch.randn(512, 6656, generator=gen, device=device)
    # Large and cancelling terms make a change of summation order show.
    x[:, ::3] *= 64
    x[1::2, 1::2] = -x[1::2, ::2]
    return x.to(torch.bfloat16)


def prepare(layer):
    global _lib, _lib2, _lib3
    device = layer.weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.get_device_properties(device).multi_processor_count != 48):
        return False
    if _lib is None:
        _lib = _load("libmarlinw_swiglu.so")
        if _lib is None:
            return False
        if os.environ.get("MARLINW_SWIGLU2", "1") != "0":
            _lib2 = _load("libmarlinw_swiglu2.so")
            if os.environ.get("MARLINW_SWIGLU3", "1") != "0":
                _lib3 = _load("libmarlinw_swiglu3.so")
    if device not in _ready:
        with torch.cuda.device(device):
            if _lib.swiglu_prepare() != 0:
                return False
            if _lib2 is not None and _lib2.swiglu_prepare() != 0:
                _lib2 = None
            if _lib3 is not None and (_lib2 is None or _lib3.swiglu_prepare() != 0):
                _lib3 = None
        _ready.add(device)
    if _lib2 is not None and _check(layer, _lib2):
        _fast.add(layer.weight.data_ptr())
        if _lib3 is not None and _check(layer, _lib3):
            _fast3.add(layer.weight.data_ptr())
    return True


def _check(layer, lib):
    """Bitwise comparison of a 128-row kernel with the 64-row one on this layer."""
    try:
        weight, scales, gs = layer.weight, layer.weight_scale, layer.weight_global_scale
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(6656 * 31 + 39936)
        for _ in range(2):
            x = _check_input(gen, weight.device)
            a = _run(_lib, x, weight, scales, gs)
            b = _run(lib, x, weight, scales, gs)
            if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def uses_128(weight) -> bool:
    """True if this weight's layer passed the 128-row comparison (its calls run libmarlinw_swiglu2)."""
    return _lib2 is not None and weight.data_ptr() in _fast


def uses_tma(weight) -> bool:
    """True if this weight's layer also passed the comparison with the TMA-fed kernel (libmarlinw_swiglu3)."""
    return _lib3 is not None and weight.data_ptr() in _fast3


def counts() -> tuple[int, int]:
    """Layers on the 128-row kernels: (cp.async-fed or TMA-fed, of which TMA-fed)."""
    return len(_fast), len(_fast3)


def choose_feed(layers, x, timer, margin=0.997):
    """Keep the TMA-fed 128-row kernel only if it is not slower than the cp.async-fed one (Muse Glimmer).

    Both 128-row kernels passed the bitwise comparison with the 64-row reference on every layer of
    ``layers`` (prepare), so the choice cannot change any output. In the engine the TMA-fed gate/up
    kernel has timed about 25 us per call slower than the cp.async-fed one on some boots, so this
    times both on every layer's own buffers with ``timer`` (the prefill placement's median-of-rounds
    timer), in the order cp.async, TMA, TMA, cp.async to cancel drift, and drops the TMA-fed kernel
    for all layers unless its summed time is within ``margin`` of the cp.async one. Called before the
    placement times the gate/up buffers, so placement then times the kernel every layer will run.
    Returns (cp.async sum, TMA sum, kept TMA) in ms per prefill, or None if there was no choice.
    """
    if _lib2 is None or _lib3 is None or not _fast3 or _fast3 != _fast:
        return None
    if any(layer.weight.data_ptr() not in _fast3 for layer in layers):
        return None
    own = {layer.weight.data_ptr(): layer for layer in layers}

    def runner(lib):
        def run(weight):
            layer = own[weight.data_ptr()]
            _run(lib, x, weight, layer.weight_scale, layer.weight_global_scale)
        return run

    weights = [layer.weight for layer in layers]
    sums = {2: 0.0, 3: 0.0}
    for which in (2, 3, 3, 2):
        sums[which] += sum(timer(runner(_lib2 if which == 2 else _lib3), weights))
    async_ms, tma_ms = sums[2] / 2, sums[3] / 2
    keep = tma_ms <= async_ms / margin
    if not keep:
        _fast3.clear()
    return async_ms, tma_ms, keep


def gemm_with(x, weight, scales, global_scale, fast: bool):
    """Load-time prefill weight placement: time the kernel the group's layers will run (Muse Glimmer)."""
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    lib = _lib
    if fast and _lib2 is not None:
        # `fast`: every layer of the group is on a 128-row kernel; the TMA-fed one if every such layer passed it.
        lib = _lib3 if (_lib3 is not None and _fast3 and _fast3 == _fast) else _lib2
    return _run(lib, x, weight, scales, global_scale)


def moved(old_ptr: int, new_ptr: int) -> None:
    """A placed weight moved to another buffer with the same bytes: it keeps its kernel (Muse Glimmer)."""
    if old_ptr in _fast:
        _fast.discard(old_ptr)
        _fast.add(new_ptr)
    if old_ptr in _fast3:
        _fast3.discard(old_ptr)
        _fast3.add(new_ptr)
    try:
        from . import swiglu_fp8
        swiglu_fp8.moved(old_ptr, new_ptr)
    except Exception:
        pass


def gemm(x, weight, scales, global_scale):
    if (x.ndim != 2 or x.shape != (512, 6656) or x.dtype != torch.bfloat16
            or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
        raise ValueError("Unsupported paired SwiGLU input")
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    ptr = weight.data_ptr()
    lib = _lib3 if ptr in _fast3 else (_lib2 if ptr in _fast else _lib)
    return _run(lib, x, weight, scales, global_scale)
