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

import hashlib
import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_H = 6656
_MAX_ROWS = 16
# Diagnostics only: compare both kernels on every eligible launch made outside graph capture (real
# activations of short prompts, which run without graphs). Off by default.
_AUDIT = os.environ.get("VLLM_MUSE_GLIMMER_NORM_GLUE_AUDIT", "0") == "1"
_stats = dict(routed=0, routed_captured=0, captured_unchecked=0, compiled_ineligible=0, audit_ok=0,
              audit_mismatch=0, audit_error=0)

# sha256 of the kernel function source with its name replaced by KERNEL and
# trailing whitespace removed (see _canonical_hash).
_KNOWN = {
    "9d5f439177e24d9455044b802922f2c2a41f3b3dfafcd0fae0b928e45ab3e63a": ("post_attn", 0),  # ..._rsqrt_1
    "03e23ece0b7f19c9179e5134efd5bf3827487d160ecf62c3e25b2adb601aa9aa": ("post_attn", 1),  # ..._clamp_..._rsqrt_1
    "8df62488ea9e57d544ca01c77b0357adb35719e24003236406992576cc6a6d0f": ("post_ff", 0),    # ..._rsqrt_3
    "4af27bfe882e44d5385544eece542f1d53d64ed263e26dbefeb3f356a2576340": ("post_ff", 1),    # ..._clamp_..._rsqrt_3
}
# Number of tensor arguments (then xnumel, r0_numel) per kind.
_NARGS = {("post_attn", 0): 6, ("post_attn", 1): 7, ("post_ff", 0): 8, ("post_ff", 1): 10}


def _canonical_hash(src: str) -> str:
    m = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
    if m is None:
        return ""
    body = re.sub(r"^def\s+\w+\s*\(", "def KERNEL(", src[m.start():]).rstrip()
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return hashlib.sha256(body.encode()).hexdigest()


def _config_ok(kernel) -> bool:
    """The compiled launcher must use the layout the replacement reproduces."""
    launchers = getattr(kernel, "launchers", None) or []
    if len(launchers) != 1:
        return False
    cfg = getattr(launchers[0], "config", None)
    if cfg is None:
        return False
    kw = getattr(cfg, "kwargs", {}) or {}
    return kw.get("R0_BLOCK") == 1024 and kw.get("XBLOCK", 1) == 1 and getattr(cfg, "num_warps", None) == 8


class _NormSwap:
    admitted = 0
    rejected = 0

    def __init__(self, op: str, kind: int):
        self.op = op
        self.kind = kind
        self.nargs = _NARGS[(op, kind)]
        self.state = "unchecked"  # -> "admitted" | "rejected"

    # ---- argument handling -------------------------------------------------------------------------
    def _split(self, args):
        if len(args) != self.nargs + 2:
            return None
        tensors, rows, width = args[: self.nargs], args[self.nargs], args[self.nargs + 1]
        try:
            rows, width = int(rows), int(width)
        except Exception:
            return None
        if width != _H or not 0 < rows <= _MAX_ROWS:
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

        if self.op == "post_attn":
            x, res, wp, wq = tensors[:4]
            ok = row(x, bf) and row(res, bf) and vec(wp) and vec(wq)
            if self.kind == 0:
                s, o = tensors[4:6]
                return ok and s.dtype == f32 and s.numel() >= rows and row(o, bf)
            sc, s, q = tensors[4:7]
            return ok and sc.dtype == f32 and sc.numel() == 1 and s.dtype == f32 and s.numel() >= rows and row(q, f8)
        x, d, res, sx, wp, wf, wi = tensors[:7]
        ok = row(x, bf) and row(d, bf) and row(res, bf) and sx.dtype == f32 and sx.numel() >= rows
        ok = ok and vec(wp) and vec(wf) and vec(wi)
        if self.kind == 0:
            return ok and row(tensors[7], bf)
        sc, o, q = tensors[7:10]
        return ok and sc.dtype == f32 and sc.numel() == 1 and row(o, bf) and row(q, f8)

    # ---- launches ----------------------------------------------------------------------------------
    def _launch(self, tensors, rows, stream) -> None:
        import muse_norm_glue as G

        if self.op == "post_attn":
            x, res, wp, wq = tensors[:4]
            if self.kind == 0:
                s, o = tensors[4:6]
                G.post_attn(0, rows, x, res, wp, wq, None, s, o, None, stream)
            else:
                sc, s, q = tensors[4:7]
                G.post_attn(1, rows, x, res, wp, wq, sc, s, None, q, stream)
        else:
            x, d, res, sx, wp, wf, wi = tensors[:7]
            if self.kind == 0:
                G.post_ff(0, rows, x, d, res, sx, wp, wf, wi, None, tensors[7], None, stream)
            else:
                sc, o, q = tensors[7:10]
                G.post_ff(1, rows, x, d, res, sx, wp, wf, wi, sc, o, q, stream)

    def _outputs(self, tensors):
        """Indices of the tensors a launch writes (the in/out activation first for post_ff)."""
        if self.op == "post_attn":
            return (4, 5) if self.kind == 0 else (5, 6)
        return (0, 7) if self.kind == 0 else (0, 8, 9)

    def _compare(self, kernel, orig_run, tensors, rows, kwargs, stream) -> bool:
        """Run both kernels on the same inputs; True if every written word is identical."""
        outs = self._outputs(tensors)
        before = [t.clone() for t in tensors]
        orig_run(kernel, *tensors, rows, _H, **kwargs)
        theirs = [tensors[i].clone() for i in outs]
        mine_in = [t.clone() for t in before]
        self._launch(mine_in, rows, stream)
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

    def _check(self, kernel, orig_run, args, kwargs, tensors, rows, stream) -> bool:
        if self.kind == 1:
            sc = float(tensors[4 if self.op == "post_attn" else 7].reshape(()).item())
            if not 1.2e-38 < abs(sc) < 8.5e37:
                return False
        # 1) the real launch: the compiled kernel writes the real outputs, the replacement writes copies
        outs = self._outputs(tensors)
        snapshot = [t.clone() for t in tensors]
        orig_run(kernel, *args, **kwargs)
        mine = [t.clone() for t in snapshot]
        self._launch(mine, rows, stream)
        torch.cuda.synchronize()
        for i in outs:
            a, b = tensors[i].reshape(-1), mine[i].reshape(-1)
            if a.numel() != b.numel() or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return False
        # 2) random activations around the real weights and scale, several magnitudes, 1..3 rows
        gen = torch.Generator(device=tensors[0].device)
        gen.manual_seed(20261122)
        dev = tensors[0].device
        act_idx = (0, 1) if self.op == "post_attn" else (0, 1, 2)
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

    def run(self, kernel, orig_run, args, kwargs):
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
            # _compare ran the compiled kernel on the real tensors; its outputs stand.
            return None
        if self.state == "unchecked":
            if capturing:
                _stats["captured_unchecked"] += 1
                if _stats["captured_unchecked"] == 1:
                    logger.warning("Decode norm glue: a compiled norm kernel was first launched under graph "
                                   "capture; it keeps the compiled kernel")
                return orig_run(kernel, *args, **kwargs)
            ok = False
            try:
                # The check's first step is the real compiled launch, so the launcher is built after it.
                ok = self._check(kernel, orig_run, args, kwargs, tensors, rows, stream) and _config_ok(kernel)
            except Exception:
                logger.warning("Decode norm glue check failed to run", exc_info=True)
            self.state = "admitted" if ok else "rejected"
            if ok:
                _NormSwap.admitted += 1
            else:
                _NormSwap.rejected += 1
            logger.info("Exact decode norm kernels: %d compiled kernels admitted, %d rejected",
                        _NormSwap.admitted, _NormSwap.rejected)
            # The compiled kernel already wrote this launch's outputs.
            return None
        self._launch(tensors, rows, stream)
        _stats["routed"] += 1
        if _stats["routed"] in (1, 64, 128, 256, 512):
            logger.info("Exact decode norm kernels: %d launches routed (%d under capture), %d left compiled while "
                        "unchecked under capture, %d ineligible", _stats["routed"], _stats["routed_captured"] +
                        int(capturing), _stats["captured_unchecked"], _stats["compiled_ineligible"])
        if capturing:
            _stats["routed_captured"] += 1
        return None


def _classify(kernel) -> _NormSwap | None:
    meta = getattr(kernel, "inductor_meta", None) or {}
    name = meta.get("kernel_name", "")
    if not name.startswith("triton_red_fused__to_copy_add_"):
        return None
    src = getattr(getattr(kernel, "fn", None), "src", None)
    if not isinstance(src, str):
        return None
    known = _KNOWN.get(_canonical_hash(src))
    if known is None:
        return None
    return _NormSwap(*known)


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
    prev_run = CachingAutotuner.run

    def run(self, *args, **kwargs):
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
    _installed = True
