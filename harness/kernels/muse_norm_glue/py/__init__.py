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
