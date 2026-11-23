# SPDX-License-Identifier: Apache-2.0
"""Exact few-row replacements for the compiled Muse Glimmer sandwich-norm reductions (see norm_glue.cu)."""

import ctypes
from functools import cache
from pathlib import Path


@cache
def lib():
    try:
        native = ctypes.CDLL(str(Path(__file__).with_name("muse_norm_glue.so")))
    except OSError:
        return None
    p = ctypes.c_void_p
    native.muse_norm_glue_post_attn.argtypes = [ctypes.c_int, ctypes.c_int] + [p] * 8 + [p]
    native.muse_norm_glue_post_attn.restype = ctypes.c_int
    native.muse_norm_glue_post_ff.argtypes = [ctypes.c_int, ctypes.c_int] + [p] * 10 + [p]
    native.muse_norm_glue_post_ff.restype = ctypes.c_int
    ff_fin = getattr(native, "muse_norm_glue_post_ff_final", None)
    if ff_fin is not None:
        ff_fin.argtypes = [ctypes.c_int] + [p] * 9 + [ctypes.c_int, p]
        ff_fin.restype = ctypes.c_int
    in_f8 = getattr(native, "muse_norm_glue_input_fp8", None)
    if in_f8 is not None:
        in_f8.argtypes = [ctypes.c_int, ctypes.c_int] + [p] * 5 + [p]
        in_f8.restype = ctypes.c_int
    in_f8_g = getattr(native, "muse_norm_glue_input_fp8_gate", None)
    if in_f8_g is not None:
        in_f8_g.argtypes = [ctypes.c_int, ctypes.c_int] + [p] * 7 + [ctypes.c_int, p]
        in_f8_g.restype = ctypes.c_int
    fg = getattr(native, "muse_norm_glue_post_ff_gate", None)
    if fg is not None:
        fg.argtypes = [ctypes.c_int] + [p] * 12 + [ctypes.c_int, p]
        fg.restype = ctypes.c_int
    gf = getattr(native, "muse_norm_glue_gate_formula", None)
    if gf is not None:
        gf.argtypes = [p, p, p, ctypes.c_int, p]
        gf.restype = ctypes.c_int
    ms = getattr(native, "muse_norm_glue_mul_sigmoid_bf16", None)
    if ms is not None:
        ms.argtypes = [p, p, p, ctypes.c_int, p]
        ms.restype = ctypes.c_int
    ss = getattr(native, "muse_norm_glue_mul_silu_slice_bf16", None)
    if ss is not None:
        ss.argtypes = [p, p, ctypes.c_int, p]
        ss.restype = ctypes.c_int
    return native


def _ptr(t):
    return None if t is None else t.data_ptr()


def post_attn(kind, rows, x, res, w_post, w_pre, scale, sumsq_out, out_bf16, out_fp8, stream):
    """kind 0: bf16 out (compiled ..._mean_mul_pow_rsqrt_1); kind 1: fp8 out (..._clamp_..._rsqrt_1)."""
    err = lib().muse_norm_glue_post_attn(
        kind, rows, _ptr(x), _ptr(res), _ptr(w_post), _ptr(w_pre), _ptr(scale), _ptr(sumsq_out),
        _ptr(out_bf16), _ptr(out_fp8), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue post_attn launch returned CUDA status {err}")


def post_ff(kind, rows, x, d, res, sumsq_x, w_post, w_ff, w_in, scale, out_bf16, out_fp8, stream):
    """kind 0: bf16 out (compiled ..._mean_mul_pow_rsqrt_3); kind 1: bf16 + fp8 out (..._clamp_..._rsqrt_3)."""
    err = lib().muse_norm_glue_post_ff(
        kind, rows, _ptr(x), _ptr(d), _ptr(res), _ptr(sumsq_x), _ptr(w_post), _ptr(w_ff), _ptr(w_in),
        _ptr(scale), _ptr(out_bf16), _ptr(out_fp8), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue post_ff launch returned CUDA status {err}")


def post_ff_final(rows, d, res, x, sumsq_x, w_post, w_ff, w_in, out_h2, out_bf16, write_h2, stream):
    fn = getattr(lib(), "muse_norm_glue_post_ff_final", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_post_ff_final unavailable")
    err = fn(rows, _ptr(d), _ptr(res), _ptr(x), _ptr(sumsq_x), _ptr(w_post), _ptr(w_ff), _ptr(w_in),
             _ptr(out_h2), _ptr(out_bf16), int(bool(write_h2)), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue post_ff_final launch returned CUDA status {err}")


def input_fp8(warps, rows, x, w_in, scale, out_bf16, out_fp8, stream):
    fn = getattr(lib(), "muse_norm_glue_input_fp8", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_input_fp8 unavailable")
    err = fn(int(warps), rows, _ptr(x), _ptr(w_in), _ptr(scale), _ptr(out_bf16), _ptr(out_fp8), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue input_fp8 launch returned CUDA status {err}")


def input_fp8_gate(warps, rows, x, w_in, scale, out_bf16, out_fp8, gate_scale, out_gate, write_bf16, stream):
    fn = getattr(lib(), "muse_norm_glue_input_fp8_gate", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_input_fp8_gate unavailable")
    err = fn(int(warps), rows, _ptr(x), _ptr(w_in), _ptr(scale), _ptr(out_bf16), _ptr(out_fp8),
             _ptr(gate_scale), _ptr(out_gate), int(bool(write_bf16)), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue input_fp8_gate launch returned CUDA status {err}")


def post_ff_gate(rows, x, d, res, sumsq_x, w_post, w_ff, w_in, scale, out_bf16, out_fp8,
                 gate_scale, out_gate, write_bf16, stream):
    fn = getattr(lib(), "muse_norm_glue_post_ff_gate", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_post_ff_gate unavailable")
    err = fn(rows, _ptr(x), _ptr(d), _ptr(res), _ptr(sumsq_x), _ptr(w_post), _ptr(w_ff), _ptr(w_in),
             _ptr(scale), _ptr(out_bf16), _ptr(out_fp8), _ptr(gate_scale), _ptr(out_gate),
             int(bool(write_bf16)), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue post_ff_gate launch returned CUDA status {err}")


def gate_formula(bits, gate_scale, out, n, stream):
    fn = getattr(lib(), "muse_norm_glue_gate_formula", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_gate_formula unavailable")
    err = fn(_ptr(bits), _ptr(gate_scale), _ptr(out), n, stream)
    if err:
        raise RuntimeError(f"muse_norm_glue gate_formula launch returned CUDA status {err}")


def mul_sigmoid_bf16(gate, attn, out, n, stream):
    fn = getattr(lib(), "muse_norm_glue_mul_sigmoid_bf16", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_mul_sigmoid_bf16 unavailable")
    err = fn(_ptr(gate), _ptr(attn), _ptr(out), int(n), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue mul_sigmoid_bf16 launch returned CUDA status {err}")


def mul_silu_slice_bf16(gu, out, rows, stream):
    fn = getattr(lib(), "muse_norm_glue_mul_silu_slice_bf16", None)
    if fn is None:
        raise RuntimeError("muse_norm_glue_mul_silu_slice_bf16 unavailable")
    err = fn(_ptr(gu), _ptr(out), int(rows), stream)
    if err:
        raise RuntimeError(f"muse_norm_glue mul_silu_slice_bf16 launch returned CUDA status {err}")
