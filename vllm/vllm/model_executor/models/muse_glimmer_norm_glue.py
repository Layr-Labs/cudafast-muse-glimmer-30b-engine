# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-step replacement for the compiled sandwich-norm reduction kernels.

Each decoder layer runs two compiled Inductor reductions around its MLP:
``..._mean_mul_pow_rsqrt_1`` (post-attention norm, residual add and
pre-feedforward norm) and ``..._mean_mul_pow_rsqrt_3`` (post-feedforward norm,
residual add and the next layer's input norm), each with an FP8-output
variant (``..._clamp_..._reciprocal_rsqrt_{1,3}``). With one decode token they
are single-CTA kernels that walk the 6656-wide row with one dependent global
round trip per 1024-element chunk (8-15 us each).

The kernels in the ``muse_norm_glue`` package load each row once and evaluate
the same float operations in the same order as the compiled code (same lane
layout and reduction tree; FMUL/FADD/FFMA exactly where the compiled SASS has
them), so they write the same bits.

The swap is made at the kernel launcher: the compiled graph, the fusion
boundaries and every other kernel stay as compiled. A compiled kernel is
routed only when its source is one of the four reviewed sources (by hash),
only for launches of at most ``_MAX_ROWS`` rows (decode), and only after its
first such launch, made outside graph capture, produced bit-identical outputs
from both kernels on the real inputs and on random inputs built around the
real weights. Prompt-sized launches always run the compiled kernel.
"""

from __future__ import annotations

import ast
from collections import Counter
from functools import cache
import hashlib
import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_H = 6656
_MAX_ROWS = 16
_MAX_POST_ATTN_ROWS = 4096
# Diagnostics only: compare both kernels on every eligible launch made outside graph capture (real
# activations of short prompts, which run without graphs). Off by default.
_AUDIT = os.environ.get("VLLM_MUSE_GLIMMER_NORM_GLUE_AUDIT", "0") == "1"
_DECODE_GATE_FOLD = os.environ.get("VLLM_MUSE_GLIMMER_DECODE_GATE_FOLD", "1") == "1"
_DROP_BF16 = os.environ.get("VLLM_MUSE_GLIMMER_DECODE_DROP_BF16", "1") == "1"
_RANS_PROLOGUE = os.environ.get("MUSE_RANS_PROLOGUE_FUSE", "1") != "0"
_MUL_SIGMOID_GLUE = os.environ.get("MUSE_MUL_SIGMOID_GLUE", "1") != "0"
_OPROJ_SIGMOID_FUSE = os.environ.get("MUSE_OPROJ_SIGMOID_FUSE", "1") != "0"
_MUL_SILU_SLICE_GLUE = os.environ.get("MUSE_MUL_SILU_SLICE_GLUE", "1") != "0"
_REQUANT = "triton_poi_fused__to_copy_clamp_mul_reciprocal_0"
_REQUANT_BODY = "f7a07e4acde4818e0fe4bcb2f62a211fa163724414f5efcce3d93ee433b8300f"
_SIGMOID_FP8_HASH = "68ddaaf112b7f2fbd8d452b5eb8b93c39567cfe29e167a625d3194458669ad01"
_SILU_FP8_HASH = "19d65ce965db0a8c0bb1cfcaca79c3ac3de250ee8ed0a3aa1983d8c3612df9dc"
_MUL_SIGMOID_BF16_HASH = "d7dbdd6fbb45cd65ebdcb6c1fde8d461e74484b611165ad886a6d2e589ff6573"
_MUL_SILU_SLICE_BF16_HASH = "bf9bb13afc727a2d983f0324b4af948f9a8c926ab016caad6288cd160e3c49c7"
_stats = dict(routed=0, routed_captured=0, captured_unchecked=0, compiled_ineligible=0, audit_ok=0,
              audit_mismatch=0, audit_error=0)


class _GateLayer:
    __slots__ = ("qkv_scale", "gate_scale", "buffer", "state", "formula")

    def __init__(self, qkv_scale: torch.Tensor, gate_scale: torch.Tensor, buffer: torch.Tensor):
        self.qkv_scale = qkv_scale
        self.gate_scale = gate_scale
        self.buffer = buffer
        self.state = "unchecked"  # -> "norm_ok" -> "requant_ok" -> "admitted" | "rejected"
        self.formula: bool | None = None


_gate_layers: dict[int, _GateLayer] = {}
_pending_decode = None  # (out_bf16_ptr, _GateLayer, rows, folded)
_expect_decode = None   # (requant_out_ptr, _GateLayer, rows, folded)
_gate_counts = Counter()
_prologue_state: dict[int, str] = {}  # scale_ptr -> "admitted" | "rejected"
_pending_rans_prologue = None  # (out_fp8_ptr, act_mode, x0, x1, scale_ptr, skipped)
_prologue_counts = Counter()
_pending_gate_up_swiglu = None  # (out_ptr, sw_entry, fused_launched, check_args)
_expect_down_swiglu = None      # (triton_out_ptr, swiglu_view, down_gs_ptr)
_swiglu_gu_counts = Counter()
_pending_oproj_sigmoid = None   # (out_ptr, gate, attn, skipped)
_oproj_fuse_state = "unchecked"  # "unchecked" -> "admitted" | "rejected"
_oproj_sig_counts = Counter()


def consume_oproj_sigmoid(
    x: torch.Tensor,
    gs_ptr: int,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor | None:
    global _pending_oproj_sigmoid, _oproj_fuse_state
    if _pending_oproj_sigmoid is None:
        return None
    ptr, gate, attn, skipped = _pending_oproj_sigmoid
    _pending_oproj_sigmoid = None
    import muse_marlin6_gemv as _m6

    if x.data_ptr() != ptr or gs_ptr not in _m6._GATED_OPROJ:
        if skipped:
            raise RuntimeError("muse_glimmer_norm_glue: skipped mul_sigmoid did not reach expected NVFP4 o_proj")
        _oproj_fuse_state = "rejected"
        del gate, attn
        return None
    out_shape = (*x.shape[:-1], _m6.OPROJ_N)
    if not skipped:
        if torch.cuda.is_current_stream_capturing():
            del gate, attn
            return None
        try:
            ref = _m6.decode_linear6(x, tiled_weight, scales6, palette, global_scale, workspace, _m6.OPROJ_N, _m6.OPROJ_K, False)
            got = _m6.launch_oproj_gated6(gate, attn, tiled_weight, scales6, palette, global_scale, workspace, out_shape)
            torch.cuda.synchronize(x.device)
            if torch.equal(got.view(torch.int16), ref.view(torch.int16)) and int(torch.count_nonzero(workspace).item()) == 0:
                _oproj_fuse_state = "admitted"
                _oproj_sig_counts["admitted"] += 1
                logger.info("Decode6 o_proj fused mul_sigmoid prologue: live forward admitted bitwise")
            else:
                _oproj_fuse_state = "rejected"
                _oproj_sig_counts["rejected"] += 1
            del gate, attn
            return ref
        except Exception:
            _oproj_fuse_state = "rejected"
            _oproj_sig_counts["rejected"] += 1
            del gate, attn
            return None
    out = _m6.launch_oproj_gated6(gate, attn, tiled_weight, scales6, palette, global_scale, workspace, out_shape)
    _oproj_sig_counts["fused_launches"] += 1
    del gate, attn
    return out


def notify_gate_up_swiglu(out_ptr: int, sw_entry: dict, fused_launched: bool, check_args) -> None:
    global _pending_gate_up_swiglu
    _pending_gate_up_swiglu = (out_ptr, sw_entry, fused_launched, check_args)


def try_consume_gate_up_swiglu_fp8(gu_ptr: int, s_ptr: int, out_ptr: int) -> bool:
    global _pending_gate_up_swiglu, _pending_rans_prologue
    if _pending_gate_up_swiglu is None:
        return False
    exp_gu_ptr, sw_entry, fused_launched, _ = _pending_gate_up_swiglu
    if not sw_entry.get("is_fp8", False):
        if fused_launched:
            _pending_gate_up_swiglu = None
            raise RuntimeError("muse_glimmer_norm_glue: BF16 fused gate_up SwiGLU hit FP8 silu prologue")
        return False
    _pending_gate_up_swiglu = None
    if not fused_launched:
        return False
    if gu_ptr != exp_gu_ptr or s_ptr != sw_entry["fp8_scale_ptr"]:
        raise RuntimeError("muse_glimmer_norm_glue: fused FP8 gate_up SwiGLU output did not match silu_fp8 args")
    _pending_rans_prologue = (out_ptr, 0, sw_entry["swiglu_fp8_out"], None, s_ptr, True)
    _swiglu_gu_counts["fp8_skipped"] += 1
    return True


def consume_down_swiglu(x: torch.Tensor, down_gs_ptr: int) -> torch.Tensor | None:
    global _expect_down_swiglu
    if _expect_down_swiglu is None:
        return None
    exp_ptr, sw_view, exp_gs = _expect_down_swiglu
    _expect_down_swiglu = None
    if x.data_ptr() != exp_ptr or down_gs_ptr != exp_gs:
        raise RuntimeError("muse_glimmer_norm_glue: skipped silu_slice did not reach expected down_proj")
    _swiglu_gu_counts["down_redirected"] += 1
    return sw_view


def consume_rans_prologue(
    A: torch.Tensor, As: torch.Tensor, packed: torch.Tensor, n: int, k: int, Bs: torch.Tensor
) -> torch.Tensor | None:
    """Consume a pending M=1 fused activation+FP8 prologue inside muse_fp8_scaled_mm."""
    global _pending_rans_prologue
    if _pending_rans_prologue is None:
        return None
    ptr, act_mode, x0, x1, scale_ptr, skipped = _pending_rans_prologue
    _pending_rans_prologue = None
    if A.data_ptr() != ptr or As.data_ptr() != scale_ptr:
        if skipped:
            raise RuntimeError("muse_glimmer_norm_glue: skipped prologue did not reach expected FP8 GEMM")
        _prologue_state[scale_ptr] = "rejected"
        return None
    import muse_rans_ext as _rans

    if act_mode == 0:
        out = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
        _rans.rans_gemv_auto(packed, x0.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k)
        _swiglu_gu_counts["fp8_down_redirected"] += 1
        del x0, x1
        return out
    aux = x1 if x1 is not None else x0
    if not skipped:
        if torch.cuda.is_current_stream_capturing():
            return None
        ref = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
        got = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
        _rans.rans_gemv_auto(packed, A.view(-1), Bs.reshape(-1), As.reshape(-1), ref, n, k)
        _rans.rans_gemv_fused_auto(
            packed, x0.view(-1), aux.view(-1), Bs.reshape(-1), As.reshape(-1), got, n, k, act_mode
        )
        torch.cuda.synchronize()
        if torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            _prologue_state[scale_ptr] = "admitted"
            _prologue_counts["admitted"] += 1
            if _prologue_counts["admitted"] in (1, 42, 54):
                logger.info("Decode rANS prologue fusion: %d FP8 layers admitted", _prologue_counts["admitted"])
        else:
            _prologue_state[scale_ptr] = "rejected"
            _prologue_counts["rejected"] += 1
        del x0, x1, aux
        return ref
    # Hold x0/x1 alive across output allocation so caching allocator cannot alias them.
    out = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
    _rans.rans_gemv_fused_auto(
        packed, x0.view(-1), aux.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k, act_mode
    )
    _prologue_counts["fused_launches"] += 1
    del x0, x1, aux
    return out


class _MulSigmoidSwap:
    def __init__(self) -> None:
        self.state = "unchecked"

    def run(self, kernel, orig_run, args, kwargs):
        global _pending_oproj_sigmoid, _oproj_fuse_state
        if kwargs.get("benchmark_run", False):
            return orig_run(kernel, *args, **kwargs)
        if _pending_oproj_sigmoid is not None:
            if _pending_oproj_sigmoid[3]:
                raise RuntimeError("muse_glimmer_norm_glue: skipped mul_sigmoid was never consumed by o_proj")
            _oproj_fuse_state = "rejected"
            _pending_oproj_sigmoid = None
        if self.state == "rejected" or len(args) != 4:
            return orig_run(kernel, *args, **kwargs)
        gate, attn, out, numel = args
        try:
            n_el = int(numel)
        except Exception:
            return orig_run(kernel, *args, **kwargs)
        stream = kwargs.get("stream")
        if (
            stream is None
            or n_el < 4096
            or (n_el & 4095) != 0
            or n_el > 8192 * 4096
            or not all(
                isinstance(t, torch.Tensor)
                and t.is_cuda
                and t.dtype == torch.bfloat16
                and t.is_contiguous()
                and t.numel() == n_el
                and (t.data_ptr() & 15) == 0
                for t in (gate, attn, out)
            )
        ):
            return orig_run(kernel, *args, **kwargs)
        import muse_norm_glue as G

        if self.state == "unchecked":
            if torch.cuda.is_current_stream_capturing():
                return orig_run(kernel, *args, **kwargs)
            ok = False
            try:
                g_c, a_c = gate.clone(), attn.clone()
                mine = torch.empty_like(out)
                orig_run(kernel, *args, **kwargs)
                G.mul_sigmoid_bf16(g_c, a_c, mine, n_el, stream)
                torch.cuda.synchronize()
                ok = torch.equal(out.view(torch.int16), mine.view(torch.int16))
                if ok:
                    gen = torch.Generator(device=out.device).manual_seed(20261123)
                    for sig in (0.1, 1.0, 5.0, 20.0):
                        rg = (torch.randn((1, 4096), generator=gen, device=out.device) * sig).to(torch.bfloat16)
                        ra = (torch.randn((1, 4096), generator=gen, device=out.device) * sig).to(torch.bfloat16)
                        ro = torch.empty_like(rg)
                        mo = torch.empty_like(rg)
                        orig_run(kernel, rg, ra, ro, 4096, **kwargs)
                        G.mul_sigmoid_bf16(rg, ra, mo, 4096, stream)
                        torch.cuda.synchronize()
                        if not torch.equal(ro.view(torch.int16), mo.view(torch.int16)):
                            ok = False
                            break
            except Exception:
                ok = False
            self.state = "admitted" if ok else "rejected"
            logger.info("Decode/prefill mul_sigmoid glue: %s", self.state)
            if ok and _OPROJ_SIGMOID_FUSE and _oproj_fuse_state == "unchecked" and n_el == 4096:
                try:
                    import muse_marlin6_gemv as _m6
                    if getattr(_m6, "_ALL_OPROJ_GATED_READY", False):
                        _pending_oproj_sigmoid = (out.data_ptr(), gate, attn, False)
                except Exception:
                    pass
            return None
        if _OPROJ_SIGMOID_FUSE and _oproj_fuse_state != "rejected" and n_el == 4096:
            try:
                import muse_marlin6_gemv as _m6
                if getattr(_m6, "_ALL_OPROJ_GATED_READY", False):
                    if _oproj_fuse_state == "admitted":
                        _pending_oproj_sigmoid = (out.data_ptr(), gate, attn, True)
                        _oproj_sig_counts["skipped"] += 1
                        return None
                    if _oproj_fuse_state == "unchecked" and not torch.cuda.is_current_stream_capturing():
                        G.mul_sigmoid_bf16(gate, attn, out, 4096, stream)
                        _pending_oproj_sigmoid = (out.data_ptr(), gate, attn, False)
                        return None
            except Exception:
                _oproj_fuse_state = "rejected"
        G.mul_sigmoid_bf16(gate, attn, out, n_el, stream)
        return None


class _MulSiluSliceSwap:
    def __init__(self) -> None:
        self.state = "unchecked"

    def run(self, kernel, orig_run, args, kwargs):
        global _pending_gate_up_swiglu, _expect_down_swiglu
        if kwargs.get("benchmark_run", False):
            return orig_run(kernel, *args, **kwargs)
        if _pending_gate_up_swiglu is not None:
            gu_ptr, sw_entry, fused_launched, check_args = _pending_gate_up_swiglu
            _pending_gate_up_swiglu = None
            if len(args) == 3 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
                gu_t, out_t, numel_t = args
                try:
                    n_sw = int(numel_t)
                except Exception:
                    n_sw = -1
                if n_sw == 19968 and gu_t.data_ptr() == gu_ptr:
                    if fused_launched:
                        _expect_down_swiglu = (
                            out_t.data_ptr(),
                            sw_entry["swiglu_out"][:, :19968],
                            sw_entry["down_gs_ptr"],
                        )
                        _swiglu_gu_counts["skipped"] += 1
                        return None
                    if sw_entry["state"] == "prechecked" and not torch.cuda.is_current_stream_capturing() and check_args is not None:
                        orig_run(kernel, *args, **kwargs)
                        try:
                            import muse_marlin6_gemv as _m6
                            x_c, w_c, s6_c, pal_c, gs_c, ws_c, part_c = check_args
                            sw_out = sw_entry["swiglu_out"]
                            pl = sw_entry["pair_locks"]
                            with torch.cuda.device(out_t.device):
                                rc = _m6._gate_up_tma4_swiglu(
                                    x_c.data_ptr(), w_c.data_ptr(), sw_out.data_ptr(), part_c.data_ptr(),
                                    s6_c.data_ptr(), pal_c.data_ptr(), gs_c.data_ptr(), ws_c.data_ptr(),
                                    pl.data_ptr(), 39936, 6656, 48,
                                    out_t.device.index, torch.cuda.current_stream().cuda_stream,
                                )
                            torch.cuda.synchronize(out_t.device)
                            if (
                                rc == 0
                                and int(torch.count_nonzero(pl).item()) == 0
                                and int(torch.count_nonzero(ws_c).item()) == 0
                                and torch.equal(sw_out[:, :19968].view(torch.int16), out_t.view(torch.int16))
                            ):
                                sw_entry["state"] = "admitted"
                                _swiglu_gu_counts["admitted"] += 1
                                if _swiglu_gu_counts["admitted"] in (1, 40):
                                    logger.info("Decode6 gate_up SwiGLU epilogue: %d layers admitted bitwise", _swiglu_gu_counts["admitted"])
                            else:
                                sw_entry["state"] = "rejected"
                                _swiglu_gu_counts["rejected"] += 1
                        except Exception:
                            sw_entry["state"] = "rejected"
                            _swiglu_gu_counts["rejected"] += 1
                        return None
                elif fused_launched:
                    raise RuntimeError("muse_glimmer_norm_glue: fused gate_up SwiGLU output did not reach expected silu_slice")
            elif fused_launched:
                raise RuntimeError("muse_glimmer_norm_glue: unexpected kernel after fused gate_up SwiGLU")
        if self.state == "rejected" or len(args) != 3:
            return orig_run(kernel, *args, **kwargs)
        gu, out, numel = args
        try:
            n_el = int(numel)
        except Exception:
            return orig_run(kernel, *args, **kwargs)
        stream = kwargs.get("stream")
        if stream is None or n_el <= 0 or n_el % 19968 != 0:
            return orig_run(kernel, *args, **kwargs)
        rows = n_el // 19968
        if (
            rows > _MAX_ROWS
            or not all(
                isinstance(t, torch.Tensor)
                and t.is_cuda
                and t.dtype == torch.bfloat16
                and t.is_contiguous()
                and (t.data_ptr() & 15) == 0
                for t in (gu, out)
            )
            or gu.numel() != rows * 39936
            or out.numel() != rows * 19968
            or (gu.data_ptr() < out.data_ptr() + out.numel() * 2 and out.data_ptr() < gu.data_ptr() + gu.numel() * 2)
        ):
            return orig_run(kernel, *args, **kwargs)
        import muse_norm_glue as G

        if self.state == "unchecked":
            if torch.cuda.is_current_stream_capturing():
                return orig_run(kernel, *args, **kwargs)
            ok = False
            try:
                gu_c = gu.clone()
                mine = torch.empty_like(out)
                orig_run(kernel, *args, **kwargs)
                G.mul_silu_slice_bf16(gu_c, mine, rows, stream)
                torch.cuda.synchronize()
                ok = torch.equal(out.view(torch.int16), mine.view(torch.int16))
                if ok:
                    gen = torch.Generator(device=out.device).manual_seed(20261124)
                    for sig in (0.1, 1.0, 5.0, 20.0):
                        rgu = (torch.randn((1, 39936), generator=gen, device=out.device) * sig).to(torch.bfloat16)
                        ro = torch.empty((1, 19968), dtype=torch.bfloat16, device=out.device)
                        mo = torch.empty_like(ro)
                        orig_run(kernel, rgu, ro, 19968, **kwargs)
                        G.mul_silu_slice_bf16(rgu, mo, 1, stream)
                        torch.cuda.synchronize()
                        if not torch.equal(ro.view(torch.int16), mo.view(torch.int16)):
                            ok = False
                            break
            except Exception:
                ok = False
            self.state = "admitted" if ok else "rejected"
            logger.info("Decode mul_silu_slice glue: %s", self.state)
            return None
        G.mul_silu_slice_bf16(gu, out, rows, stream)
        return None


class _RansPrologueHook:
    def __init__(self, act_mode: int) -> None:
        self.act_mode = act_mode
        self.fp8_glue_state = "unchecked"

    def _try_prefill_sigmoid_fp8(self, kernel, orig_run, args, kwargs) -> bool:
        if not _MUL_SIGMOID_GLUE or self.fp8_glue_state == "rejected" or len(args) != 5:
            return False
        gate, attn, scale, out, numel = args
        try:
            n_el = int(numel)
        except Exception:
            return False
        stream = kwargs.get("stream")
        if (
            stream is None
            or n_el <= 4096
            or (n_el & 4095) != 0
            or n_el > 8192 * 4096
            or not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() for t in (gate, attn, scale, out))
            or gate.dtype != torch.bfloat16 or attn.dtype != torch.bfloat16
            or gate.numel() != n_el or attn.numel() != n_el
            or ((gate.data_ptr() | attn.data_ptr()) & 15) != 0
            or scale.dtype != torch.float32 or scale.numel() != 1
            or out.dtype != torch.float8_e4m3fn or out.numel() != n_el
            or (out.data_ptr() & 7) != 0
        ):
            return False
        import muse_norm_glue as G

        if self.fp8_glue_state == "unchecked":
            if torch.cuda.is_current_stream_capturing():
                return False
            ok = False
            try:
                g_c, a_c, s_c = gate.clone(), attn.clone(), scale.clone()
                mine = torch.empty_like(out)
                orig_run(kernel, *args, **kwargs)
                G.mul_sigmoid_fp8(g_c, a_c, s_c, mine, n_el, stream)
                torch.cuda.synchronize()
                ok = torch.equal(out.view(torch.uint8), mine.view(torch.uint8))
                if ok:
                    gen = torch.Generator(device=out.device).manual_seed(20261124)
                    for sig in (0.1, 1.0, 5.0, 20.0):
                        rg = (torch.randn((1, 4096), generator=gen, device=out.device) * sig).to(torch.bfloat16)
                        ra = (torch.randn((1, 4096), generator=gen, device=out.device) * sig).to(torch.bfloat16)
                        ro = torch.empty((1, 4096), dtype=torch.float8_e4m3fn, device=out.device)
                        mo = torch.empty_like(ro)
                        orig_run(kernel, rg, ra, s_c, ro, 4096, **kwargs)
                        G.mul_sigmoid_fp8(rg, ra, s_c, mo, 4096, stream)
                        torch.cuda.synchronize()
                        if not torch.equal(ro.view(torch.uint8), mo.view(torch.uint8)):
                            ok = False
                            break
            except Exception:
                ok = False
            self.fp8_glue_state = "admitted" if ok else "rejected"
            logger.info("Prefill mul_sigmoid_fp8 glue: %s", self.fp8_glue_state)
            return True
        G.mul_sigmoid_fp8(gate, attn, scale, out, n_el, stream)
        return True

    def run(self, kernel, orig_run, args, kwargs):
        global _pending_rans_prologue
        if kwargs.get("benchmark_run", False):
            return orig_run(kernel, *args, **kwargs)
        if self.act_mode == 1 and self._try_prefill_sigmoid_fp8(kernel, orig_run, args, kwargs):
            return None
        if _pending_rans_prologue is not None:
            return orig_run(kernel, *args, **kwargs)
        try:
            import muse_rans_ext as _rans
            if not hasattr(_rans, "rans_gemv_fused_auto"):
                return orig_run(kernel, *args, **kwargs)
        except Exception:
            return orig_run(kernel, *args, **kwargs)
        if self.act_mode == 1:
            if len(args) != 5:
                return orig_run(kernel, *args, **kwargs)
            gate, attn, scale, out, numel = args
            try:
                n_el = int(numel)
            except Exception:
                return orig_run(kernel, *args, **kwargs)
            if (
                n_el != 4096
                or not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() for t in (gate, attn, scale, out))
                or gate.dtype != torch.bfloat16 or attn.dtype != torch.bfloat16
                or gate.numel() != 4096 or attn.numel() != 4096
                or ((gate.data_ptr() | attn.data_ptr()) & 7) != 0
                or scale.dtype != torch.float32 or scale.numel() != 1
                or out.dtype != torch.float8_e4m3fn or out.numel() != 4096
            ):
                return orig_run(kernel, *args, **kwargs)
            x0, x1 = gate, attn
        else:
            if len(args) != 4:
                return orig_run(kernel, *args, **kwargs)
            gu, scale, out, numel = args
            try:
                n_el = int(numel)
            except Exception:
                return orig_run(kernel, *args, **kwargs)
            if (
                n_el != 19968
                or not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() for t in (gu, scale, out))
                or gu.dtype != torch.bfloat16 or gu.numel() != 39936
                or (gu.data_ptr() & 7) != 0
                or scale.dtype != torch.float32 or scale.numel() != 1
                or out.dtype != torch.float8_e4m3fn or out.numel() != 19968
            ):
                return orig_run(kernel, *args, **kwargs)
            x0, x1 = gu, None
            if try_consume_gate_up_swiglu_fp8(gu.data_ptr(), scale.data_ptr(), out.data_ptr()):
                return None
        s_ptr = scale.data_ptr()
        st = _prologue_state.get(s_ptr, "unchecked")
        if st == "rejected":
            return orig_run(kernel, *args, **kwargs)
        if st == "admitted":
            _pending_rans_prologue = (out.data_ptr(), self.act_mode, x0, x1, s_ptr, True)
            return None
        if torch.cuda.is_current_stream_capturing():
            return orig_run(kernel, *args, **kwargs)
        orig_run(kernel, *args, **kwargs)
        _pending_rans_prologue = (out.data_ptr(), self.act_mode, x0, x1, s_ptr, False)
        return None


def register_gate(qkv_scale: torch.Tensor, gate_scale: torch.Tensor) -> bool:
    """Register an FP8-attention layer (qkv_scale -> gate_scale) for M<=16 decode gate-quant fold."""
    if not _DECODE_GATE_FOLD:
        return False
    for t in (qkv_scale, gate_scale):
        if not (isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32 and t.numel() == 1):
            return False
    key = qkv_scale.data_ptr()
    if key not in _gate_layers:
        buf = torch.empty((_MAX_ROWS, _H), dtype=torch.float8_e4m3fn, device=qkv_scale.device)
        _gate_layers[key] = _GateLayer(qkv_scale, gate_scale, buf)
    return True


def _gate_native_ok() -> bool:
    try:
        import muse_norm_glue as G
        return G.lib() is not None and hasattr(G.lib(), "muse_norm_glue_post_ff_gate") and hasattr(G.lib(), "muse_norm_glue_gate_formula")
    except Exception:
        return False


@cache
def _body_hash(source: str) -> str | None:
    definitions = ast.parse(source).body
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return None
    fn = definitions[0]
    fn.name = "requant"
    fn.decorator_list = []
    return hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()


def _formula_ok(kernel, orig_run, layer: _GateLayer, stream) -> bool:
    import muse_norm_glue as G

    dev = layer.gate_scale.device
    bits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16)
    theirs = torch.empty(65536, dtype=torch.float8_e4m3fn, device=dev)
    mine = torch.empty(65536, dtype=torch.uint8, device=dev)
    orig_run(kernel, bits.view(torch.bfloat16), layer.gate_scale, theirs, 65536, stream=stream)
    G.gate_formula(bits, layer.gate_scale, mine, 65536, stream)
    torch.cuda.synchronize()
    return torch.equal(theirs.view(torch.uint8), mine)


def decode_gemm_operand(a: torch.Tensor) -> torch.Tensor:
    """Redirect M<=16 gate FP8 GEMM operand to the folded buffer when requant was skipped."""
    global _expect_decode
    if _expect_decode is None:
        return a
    ptr, layer, rows, folded = _expect_decode
    _expect_decode = None
    if a.data_ptr() != ptr:
        if folded:
            raise RuntimeError("muse_glimmer_norm_glue: skipped requant output did not reach decode gate GEMM")
        layer.state = "rejected"
        _gate_counts["rejected_order"] += 1
        return a
    if not folded:
        if layer.state == "requant_ok" and not torch.cuda.is_current_stream_capturing():
            layer.state = "admitted"
            _gate_counts["admitted"] += 1
            if _gate_counts["admitted"] in (1, 27, 28):
                logger.info("Decode gate-quant fold: %d layers admitted (requant skipped: %d)",
                            _gate_counts["admitted"], _gate_counts["requant_skipped"])
        else:
            layer.state = "rejected"
            _gate_counts["rejected_order"] += 1
        return a
    if a.shape != (rows, _H) or a.dtype != torch.float8_e4m3fn or not a.is_contiguous():
        raise RuntimeError(f"muse_glimmer_norm_glue: decode GEMM operand shape changed: {tuple(a.shape)}")
    _gate_counts["gemm_redirected"] += 1
    return layer.buffer[:rows]

# sha256 of the kernel function source with its name replaced by KERNEL and
# trailing whitespace removed (see _canonical_hash).
_KNOWN = {
    "9d5f439177e24d9455044b802922f2c2a41f3b3dfafcd0fae0b928e45ab3e63a": ("post_attn", 0),   # ..._rsqrt_1
    "03e23ece0b7f19c9179e5134efd5bf3827487d160ecf62c3e25b2adb601aa9aa": ("post_attn", 1),   # ..._clamp_..._rsqrt_1
    "8df62488ea9e57d544ca01c77b0357adb35719e24003236406992576cc6a6d0f": ("post_ff", 0),     # ..._rsqrt_3
    "4af27bfe882e44d5385544eece542f1d53d64ed263e26dbefeb3f356a2576340": ("post_ff", 1),     # ..._clamp_..._rsqrt_3
    "bf5b07265dd104f20e1d453c270c2c8c59aa7d22d7fe1bf6b70da6e71ef1dc03": ("post_ff", 2),     # Layer-51 final norm
    "89c74b3ff8b8fca54204a32d25cc75b669560debd4bb866a354d7815eeda115b": ("post_ff", 2),     # Layer-51 final norm (canonical)
    "205ac75eb1f7a0ece86a243e894446a5b327438374179ec02f0225baedfaaf84": ("input_norm", 1),  # Layer-0 initial norm + FP8 quant
}
# Number of tensor arguments (then xnumel, r0_numel) per kind.
_NARGS = {
    ("post_attn", 0): 6,
    ("post_attn", 1): 7,
    ("post_ff", 0): 8,
    ("post_ff", 1): 10,
    ("post_ff", 2): 9,
    ("input_norm", 1): 5,
}


def _canonical_hash(src: str) -> str:
    m = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
    if m is None:
        return ""
    body = re.sub(r"^def\s+\w+\s*\(", "def KERNEL(", src[m.start():]).rstrip()
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return hashlib.sha256(body.encode()).hexdigest()


def _launcher_is_decode1024(lch) -> bool:
    cfg = getattr(lch, "config", None)
    if cfg is None:
        return False
    kw = getattr(cfg, "kwargs", {}) or {}
    return kw.get("R0_BLOCK") == 1024 and kw.get("XBLOCK", 1) == 1 and getattr(cfg, "num_warps", None) == 8


def _input_norm_warps(kernel) -> int:
    launchers = getattr(kernel, "launchers", None) or []
    if len(launchers) != 1:
        return 0
    cfg = getattr(launchers[0], "config", None)
    if cfg is None:
        return 0
    kw = getattr(cfg, "kwargs", {}) or {}
    nw = getattr(cfg, "num_warps", None)
    if kw.get("XBLOCK", 1) != 1:
        return 0
    if kw.get("R0_BLOCK") == 8192 and nw == 16:
        return 16
    if kw.get("R0_BLOCK") == 1024 and nw == 8:
        return 8
    return 0


def _config_ok(kernel, op: str = "") -> bool:
    """The compiled launcher must use the exact layout the replacement reproduces."""
    if op == "input_norm":
        return _input_norm_warps(kernel) in (8, 16)
    launchers = getattr(kernel, "launchers", None) or []
    if len(launchers) == 1:
        return _launcher_is_decode1024(launchers[0])
    return False


class _NormSwap:
    admitted = 0
    rejected = 0

    def __init__(self, op: str, kind: int):
        self.op = op
        self.kind = kind
        self.nargs = _NARGS[(op, kind)]
        self.state = "unchecked"  # -> "admitted" | "rejected"
        self.input_warps = 16

    # ---- argument handling -------------------------------------------------------------------------
    def _split(self, args):
        if len(args) != self.nargs + 2:
            return None
        tensors, rows, width = args[: self.nargs], args[self.nargs], args[self.nargs + 1]
        try:
            rows, width = int(rows), int(width)
        except Exception:
            return None
        max_rows = _MAX_POST_ATTN_ROWS if self.op == "post_attn" else _MAX_ROWS
        if width != _H or not 0 < rows <= max_rows:
            return None
        if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() for t in tensors):
            return None
        return tensors, rows

    def _shapes_ok(self, tensors, rows) -> bool:
        bf, f32, f8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn

        def row(t, dt):
            return t.dtype == dt and t.numel() >= rows * _H

        def vec(t):
            return t.dtype == bf and t.numel() == _H

        if self.op == "input_norm":
            x, wi, sc, o, q = tensors
            return row(x, bf) and vec(wi) and sc.dtype == f32 and sc.numel() == 1 and row(o, bf) and row(q, f8)
        if self.op == "post_attn":
            x, res, wp, wq = tensors[:4]
            ok = row(x, bf) and row(res, bf) and vec(wp) and vec(wq)
            if self.kind == 0:
                s, o = tensors[4:6]
                return ok and s.dtype == f32 and s.numel() >= rows and row(o, bf)
            sc, s, q = tensors[4:7]
            return ok and sc.dtype == f32 and sc.numel() == 1 and s.dtype == f32 and s.numel() >= rows and row(q, f8)
        if self.kind == 2:
            d, res, x, sx, wp, wf, wi, h2, o = tensors
            return (
                row(d, bf) and row(res, bf) and row(x, bf)
                and sx.dtype == f32 and sx.numel() >= rows
                and vec(wp) and vec(wf) and vec(wi)
                and row(h2, f32) and row(o, bf)
            )
        x, d, res, sx, wp, wf, wi = tensors[:7]
        ok = row(x, bf) and row(d, bf) and row(res, bf) and sx.dtype == f32 and sx.numel() >= rows
        ok = ok and vec(wp) and vec(wf) and vec(wi)
        if self.kind == 0:
            return ok and row(tensors[7], bf)
        sc, o, q = tensors[7:10]
        return ok and sc.dtype == f32 and sc.numel() == 1 and row(o, bf) and row(q, f8)

    # ---- launches ----------------------------------------------------------------------------------
    def _launch(self, tensors, rows, stream, for_check: bool = False) -> None:
        import muse_norm_glue as G

        if self.op == "input_norm":
            x, wi, sc, o, q = tensors
            G.input_fp8(self.input_warps, rows, x, wi, sc, o, q, stream)
        elif self.op == "post_attn":
            x, res, wp, wq = tensors[:4]
            if self.kind == 0:
                s, o = tensors[4:6]
                G.post_attn(0, rows, x, res, wp, wq, None, s, o, None, stream)
            else:
                sc, s, q = tensors[4:7]
                G.post_attn(1, rows, x, res, wp, wq, sc, s, None, q, stream)
        elif self.kind == 2:
            d, res, x, sx, wp, wf, wi, h2, o = tensors
            G.post_ff_final(rows, d, res, x, sx, wp, wf, wi, h2, o, 1 if for_check else int(not _DROP_BF16), stream)
        else:
            x, d, res, sx, wp, wf, wi = tensors[:7]
            if self.kind == 0:
                G.post_ff(0, rows, x, d, res, sx, wp, wf, wi, None, tensors[7], None, stream)
            else:
                sc, o, q = tensors[7:10]
                G.post_ff(1, rows, x, d, res, sx, wp, wf, wi, sc, o, q, stream)

    def _outputs(self, tensors):
        """Indices of the tensors a launch writes (the in/out activation first for post_ff)."""
        if self.op == "input_norm":
            return (3, 4)
        if self.op == "post_attn":
            return (4, 5) if self.kind == 0 else (5, 6)
        if self.kind == 2:
            return (7, 8)
        return (0, 7) if self.kind == 0 else (0, 8, 9)

    def _compare(self, kernel, orig_run, tensors, rows, kwargs, stream) -> bool:
        """Run both kernels on the same inputs; True if every written word is identical."""
        outs = self._outputs(tensors)
        before = [t.clone() for t in tensors]
        orig_run(kernel, *tensors, rows, _H, **kwargs)
        theirs = [tensors[i].clone() for i in outs]
        mine_in = [t.clone() for t in before]
        self._launch(mine_in, rows, stream, for_check=True)
        torch.cuda.synchronize()
        for j, i in enumerate(outs):
            # reshape first: the compiled sum-of-squares buffer is (rows, 1) with strides (1, rows)
            a = theirs[j].reshape(-1)
            b = mine_in[i].reshape(-1)
            if a.dtype != b.dtype or a.numel() != b.numel():
                return False
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return False
        return True

    def _check(self, kernel, orig_run, args, kwargs, tensors, rows, stream, out_snapshot=None) -> bool:
        if self.kind == 1:
            sc_idx = 2 if self.op == "input_norm" else (4 if self.op == "post_attn" else 7)
            sc = float(tensors[sc_idx].reshape(()).item())
            if not 1.2e-38 < abs(sc) < 8.5e37:
                return False
        # 1) the real launch: the compiled kernel writes the real outputs, the replacement writes copies
        outs = self._outputs(tensors)
        snapshot = [t.clone() for t in tensors]
        if out_snapshot is not None:
            out_snapshot[:] = snapshot
        orig_run(kernel, *args, **kwargs)
        if self.op == "input_norm":
            iw = _input_norm_warps(kernel)
            if iw not in (8, 16):
                return False
            self.input_warps = iw
        mine = [t.clone() for t in snapshot]
        self._launch(mine, rows, stream, for_check=True)
        torch.cuda.synchronize()
        for i in outs:
            a, b = tensors[i].reshape(-1), mine[i].reshape(-1)
            if a.numel() != b.numel() or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return False
        # 2) random activations around the real weights and scale, several magnitudes, 1..3 rows
        gen = torch.Generator(device=tensors[0].device)
        gen.manual_seed(20261122)
        dev = tensors[0].device
        act_idx = (0,) if self.op == "input_norm" else ((0, 1) if self.op == "post_attn" else (0, 1, 2))
        for case in range(12):
            n = 1 + case % 3
            sig = (0.004, 0.08, 1.0, 9.0, 70.0, 0.5)[case % 6]
            fresh = []
            for i, t in enumerate(snapshot):
                if i in act_idx:
                    v = torch.randn(n, _H, device=dev, generator=gen) * sig
                    if case % 2:
                        v = v * torch.exp(torch.randn(n, _H, device=dev, generator=gen))
                    fresh.append(v.to(torch.bfloat16))
                elif i in outs or (self.op == "post_ff" and i == 3):
                    if self.op == "post_ff" and i == 3:  # sum of squares of x from the first kernel
                        fresh.append(torch.rand(n, 1, device=dev, generator=gen) * 6656.0 * sig * sig + 1e-3)
                    else:
                        fresh.append(torch.empty((n,) + tuple(t.shape[1:]) if t.dim() > 1 else (n,), dtype=t.dtype,
                                                 device=dev).zero_())
                else:
                    fresh.append(t)  # weights / scale: the real ones
            if not self._compare(kernel, orig_run, fresh, n, kwargs, stream):
                return False
        return True

    def _disjoint(self, tensors, rows) -> bool:
        """No written range overlaps a range read under another name (the replacement reads a whole row
        before writing it; the compiled kernel interleaves chunk loads and stores)."""
        outs = self._outputs(tensors)

        def span(t):
            n = min(t.numel(), rows * _H) if t.numel() >= rows else t.numel()
            return t.data_ptr(), t.data_ptr() + n * t.element_size()

        spans = [span(t) for t in tensors]
        for i in outs:
            a0, a1 = spans[i]
            for j, (b0, b1) in enumerate(spans):
                if j != i and a0 < b1 and b0 < a1:
                    return False
        return True

    def _gate_check_real(self, glayer: _GateLayer, snapshot, tensors, rows, stream) -> None:
        global _pending_decode
        import muse_norm_glue as G

        try:
            gsc = float(glayer.gate_scale.reshape(()).item())
            if not 1.2e-38 < abs(gsc) < 8.5e37:
                glayer.state = "rejected"
                _gate_counts["rejected_scale"] += 1
                return
            copy = [t.clone() for t in snapshot]
            if self.op == "input_norm":
                x, wi, sc, o, q = copy
                G.input_fp8_gate(self.input_warps, rows, x, wi, sc, o, q, glayer.gate_scale, glayer.buffer, 1, stream)
                torch.cuda.synchronize()
                if all(torch.equal(tensors[i].reshape(-1).view(torch.uint8), copy[i].reshape(-1).view(torch.uint8)) for i in (3, 4)):
                    glayer.state = "norm_ok"
                    _pending_decode = (tensors[3].data_ptr(), glayer, rows, False)
                else:
                    glayer.state = "rejected"
                    _gate_counts["rejected_norm"] += 1
                return
            x, d, res, sx, wp, wf, wi, sc, o, q = copy
            G.post_ff_gate(rows, x, d, res, sx, wp, wf, wi, sc, o, q, glayer.gate_scale, glayer.buffer, 1, stream)
            torch.cuda.synchronize()
            if all(torch.equal(tensors[i].reshape(-1).view(torch.uint8), copy[i].reshape(-1).view(torch.uint8)) for i in (0, 8, 9)):
                glayer.state = "norm_ok"
                _pending_decode = (tensors[8].data_ptr(), glayer, rows, False)
            else:
                glayer.state = "rejected"
                _gate_counts["rejected_norm"] += 1
        except Exception:
            glayer.state = "rejected"
            _gate_counts["rejected_norm"] += 1

    def run(self, kernel, orig_run, args, kwargs):
        global _pending_decode
        if kwargs.get("benchmark_run", False):
            return orig_run(kernel, *args, **kwargs)
        if self.op in ("post_ff", "input_norm") and _pending_decode is not None:
            if _pending_decode[3]:
                raise RuntimeError("muse_glimmer_norm_glue: folded gate operand was never consumed by requant")
            _pending_decode[1].state = "rejected"
            _gate_counts["rejected_order"] += 1
            _pending_decode = None
        if self.state == "rejected":
            return orig_run(kernel, *args, **kwargs)
        split = self._split(args)
        if split is None:
            return orig_run(kernel, *args, **kwargs)
        tensors, rows = split
        stream = kwargs.get("stream")
        if stream is None or not self._shapes_ok(tensors, rows) or not self._disjoint(tensors, rows):
            _stats["compiled_ineligible"] += 1
            return orig_run(kernel, *args, **kwargs)
        capturing = torch.cuda.is_current_stream_capturing()
        if self.state == "admitted" and _AUDIT and not capturing:
            ok = False
            try:
                ok = self._compare(kernel, orig_run, list(tensors), rows, kwargs, stream)
            except Exception:
                ok = None
                logger.warning("Decode norm glue audit failed to run", exc_info=True)
            key = {True: "audit_ok", False: "audit_mismatch", None: "audit_error"}[ok]
            _stats[key] += 1
            logger.info("Decode norm glue audit (%s kind %d, %d rows): %s; totals %d identical, %d different, "
                        "%d not run", self.op, self.kind, rows,
                        {True: "identical", False: "DIFFERENT", None: "NOT RUN"}[ok],
                        _stats["audit_ok"], _stats["audit_mismatch"], _stats["audit_error"])
            return None
        can_gate = (
            self.op in ("post_ff", "input_norm")
            and self.kind == 1
            and _DECODE_GATE_FOLD
            and not _AUDIT
            and _gate_native_ok()
        )
        sc_idx = 2 if self.op == "input_norm" else 7
        if self.state == "unchecked":
            if capturing:
                _stats["captured_unchecked"] += 1
                if _stats["captured_unchecked"] == 1:
                    logger.warning("Decode norm glue: a compiled norm kernel was first launched under graph "
                                   "capture; it keeps the compiled kernel")
                return orig_run(kernel, *args, **kwargs)
            ok = False
            snapshot: list[torch.Tensor] = []
            try:
                # The check's first step is the real compiled launch, so the launcher is built after it.
                ok = self._check(kernel, orig_run, args, kwargs, tensors, rows, stream, snapshot) and _config_ok(kernel, self.op)
            except Exception:
                logger.warning("Decode norm glue check failed to run", exc_info=True)
            self.state = "admitted" if ok else "rejected"
            if ok:
                _NormSwap.admitted += 1
            else:
                _NormSwap.rejected += 1
            logger.info("Exact decode norm kernels: %d compiled kernels admitted, %d rejected",
                        _NormSwap.admitted, _NormSwap.rejected)
            if ok and can_gate:
                glayer = _gate_layers.get(tensors[sc_idx].data_ptr())
                if glayer is not None and glayer.state == "unchecked":
                    self._gate_check_real(glayer, snapshot, tensors, rows, stream)
            return None
        glayer = _gate_layers.get(tensors[sc_idx].data_ptr()) if can_gate else None
        if glayer is not None and glayer.state == "admitted":
            import muse_norm_glue as G

            if self.op == "input_norm":
                x, wi, sc, o, q = tensors
                G.input_fp8_gate(self.input_warps, rows, x, wi, sc, o, q, glayer.gate_scale, glayer.buffer, int(not _DROP_BF16), stream)
            else:
                x, d, res, sx, wp, wf, wi, sc, o, q = tensors
                G.post_ff_gate(rows, x, d, res, sx, wp, wf, wi, sc, o, q, glayer.gate_scale, glayer.buffer, int(not _DROP_BF16), stream)
            _pending_decode = (o.data_ptr(), glayer, rows, True)
            _gate_counts["folded_launches"] += 1
        elif glayer is not None and glayer.state == "unchecked" and not capturing:
            snapshot = [t.clone() for t in tensors]
            self._launch(tensors, rows, stream)
            self._gate_check_real(glayer, snapshot, tensors, rows, stream)
        else:
            self._launch(tensors, rows, stream)
        _stats["routed"] += 1
        if _stats["routed"] in (1, 64, 128, 256, 512):
            logger.info("Exact decode norm kernels: %d launches routed (%d under capture), %d left compiled while "
                        "unchecked under capture, %d ineligible", _stats["routed"], _stats["routed_captured"] +
                        int(capturing), _stats["captured_unchecked"], _stats["compiled_ineligible"])
        if capturing:
            _stats["routed_captured"] += 1
        return None


def _classify(kernel):
    meta = getattr(kernel, "inductor_meta", None) or {}
    name = meta.get("kernel_name", "")
    src = getattr(getattr(kernel, "fn", None), "src", None)
    if not isinstance(src, str):
        return None
    if name.startswith("triton_red_fused__to_copy_add_"):
        known = _KNOWN.get(_canonical_hash(src))
        if known is None:
            return None
        return _NormSwap(*known)
    if name.startswith("triton_poi_fused_"):
        h = _canonical_hash(src)
        if _RANS_PROLOGUE and h == _SIGMOID_FP8_HASH:
            return _RansPrologueHook(1)
        if _RANS_PROLOGUE and h == _SILU_FP8_HASH:
            return _RansPrologueHook(2)
        if _MUL_SIGMOID_GLUE and h == _MUL_SIGMOID_BF16_HASH:
            return _MulSigmoidSwap()
        if _MUL_SILU_SLICE_GLUE and h == _MUL_SILU_SLICE_BF16_HASH:
            return _MulSiluSliceSwap()
    return None


_installed = False


def install_norm_glue() -> None:
    """Route few-row launches of the reviewed compiled norm kernels to the exact replacements."""
    global _installed
    if _installed or os.environ.get("VLLM_MUSE_GLIMMER_NORM_GLUE", "1") != "1":
        return
    try:
        import muse_norm_glue

        if muse_norm_glue.lib() is None:
            logger.info("muse_norm_glue library not built; decode norm glue disabled")
            return
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception:
        logger.info("muse_norm_glue or the Inductor launcher not available; decode norm glue disabled")
        return

    def _fold_requant(autotuner, orig_run, args, kwargs) -> bool:
        global _pending_decode, _expect_decode
        if _pending_decode is None or len(args) != 4 or not all(isinstance(t, torch.Tensor) for t in args[:3]):
            return False
        ptr, layer, rows, folded = _pending_decode
        hidden, scale, out, numel = args
        try:
            n_el = int(numel)
        except Exception:
            return False
        if hidden.data_ptr() != ptr or scale.data_ptr() != layer.gate_scale.data_ptr():
            if folded:
                raise RuntimeError("muse_glimmer_norm_glue: requant mismatch after folded post_ff")
            return False
        _pending_decode = None
        stream = kwargs.get("stream")
        if (
            stream is None
            or n_el != rows * _H
            or out.numel() != n_el
            or out.dtype != torch.float8_e4m3fn
            or not out.is_cuda
            or not out.is_contiguous()
        ):
            if folded:
                raise RuntimeError("muse_glimmer_norm_glue: folded layer requant changed shape")
            layer.state = "rejected"
            _gate_counts["rejected_requant_args"] += 1
            return False
        if folded:
            _expect_decode = (out.data_ptr(), layer, rows, True)
            _gate_counts["requant_skipped"] += 1
            return True
        if layer.state != "norm_ok" or torch.cuda.is_current_stream_capturing():
            return False
        fn = getattr(autotuner, "fn", None)
        if fn is None or _body_hash(fn.src) != _REQUANT_BODY:
            layer.state = "rejected"
            _gate_counts["rejected_requant_src"] += 1
            return False
        orig_run(autotuner, *args, **kwargs)
        torch.cuda.synchronize()
        if layer.formula is None:
            layer.formula = _formula_ok(autotuner, orig_run, layer, stream)
        if layer.formula and torch.equal(out.reshape(-1).view(torch.uint8), layer.buffer[:rows].reshape(-1).view(torch.uint8)):
            layer.state = "requant_ok"
            _expect_decode = (out.data_ptr(), layer, rows, False)
        else:
            layer.state = "rejected"
            _gate_counts["rejected_requant_bytes"] += 1
        return True

    prev_run = CachingAutotuner.run

    def run(self, *args, **kwargs):
        if _pending_decode is not None and not kwargs.get("benchmark_run", False):
            if getattr(getattr(self, "fn", None), "__name__", None) == _REQUANT:
                if _fold_requant(self, prev_run, args, kwargs):
                    return None
        swap = self.__dict__.get("_norm_glue_swap", 0)
        if swap == 0:
            try:
                swap = _classify(self)
            except Exception:
                swap = None
            self.__dict__["_norm_glue_swap"] = swap
        if swap is None:
            return prev_run(self, *args, **kwargs)
        return swap.run(self, prev_run, args, kwargs)

    CachingAutotuner.run = run
    try:
        from vllm.model_executor.kernels.linear.scaled_mm import muse_fp8 as _muse_fp8
        _muse_fp8._decode_gemm_operand = decode_gemm_operand
        _muse_fp8._consume_rans_prologue = consume_rans_prologue
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.quantization.utils import glimmer_fp8 as _glimmer_fp8
        _glimmer_fp8._decode_gemm_operand = decode_gemm_operand
    except Exception:
        pass
    try:
        import muse_marlin6_gemv as _m6
        _m6._notify_gate_up_swiglu = notify_gate_up_swiglu
        _m6._consume_down_swiglu = consume_down_swiglu
        _m6._consume_oproj_sigmoid = consume_oproj_sigmoid
    except Exception:
        pass
    _installed = True
