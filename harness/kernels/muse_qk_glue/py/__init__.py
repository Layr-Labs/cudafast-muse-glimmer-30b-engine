# SPDX-License-Identifier: Apache-2.0
"""Exact one-pass query/key RMSNorm (+ NEOX RoPE) for prompt batches of the Muse Glimmer attention (qk_glue.cu)."""

import ctypes
from functools import cache
from pathlib import Path


@cache
def lib():
    try:
        native = ctypes.CDLL(str(Path(__file__).with_name("muse_qk_glue.so")))
    except OSError:
        return None
    p = ctypes.c_void_p
    native.muse_qk_glue_launch.argtypes = [ctypes.c_int, ctypes.c_int, p, p, p, ctypes.c_longlong, ctypes.c_float,
                                           ctypes.c_float, p, p, p, p]
    native.muse_qk_glue_launch.restype = ctypes.c_int
    return native


def _ptr(t):
    return None if t is None else t.data_ptr()


def launch(rope, tokens, qkv, pos, cos_sin, cache_rows, qscale, eps, q_sumsq, k_out, q_out, stream):
    """Writes q_sumsq [tokens*32] fp32, k_out [tokens, 2, 128] and q_out [tokens, 32, 128] bf16 from qkv rows."""
    err = lib().muse_qk_glue_launch(int(bool(rope)), int(tokens), _ptr(qkv), _ptr(pos), _ptr(cos_sin),
                                    int(cache_rows), float(qscale), float(eps), _ptr(q_sumsq), _ptr(k_out),
                                    _ptr(q_out), stream)
    if err:
        raise RuntimeError(f"muse_qk_glue launch returned CUDA status {err}")
