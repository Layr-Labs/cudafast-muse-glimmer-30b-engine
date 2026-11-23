# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-batch fusion of the compiled query/key RMSNorm (+ RoPE) kernel pair.

For every attention layer the compiled graph launches two Inductor kernels
between the qkv projection and attention:

* a combined reduction (``triton_red_fused_N``) that writes the sums of
  squares of the 32 query head rows and the normalized (and, on RoPE layers,
  rotated) 2 key heads, one warp per 128-wide head row;
* a pointwise kernel (``..._split_with_sizes_view_N``) that reads the queries
  and those sums again and writes the normalized, pre-scaled (and rotated)
  queries.

Both kernels are consecutive in the same compiled piece. For prompt batches
the pair is launched as one kernel (harness/kernels/muse_qk_glue) that reads
each head row once and evaluates the same float operations in the same order
as the compiled code (same lane layout and reduction tree; see qk_glue.cu),
and writes the same three tensors, the sums of squares included.

The swap is made at the kernel launcher: the launch of the reduction is held
back and issued together with the pointwise kernel that follows it, and only
when (a) both compiled kernels are the reviewed sources (by hash), (b) the
reduction runs with the reviewed launch configuration, (c) the compiled piece
issues the pointwise kernel right after the reduction on the same inputs
(read from the piece's own code), and (d) the first such pair, launched
outside graph capture, produced bit-identical outputs from both paths on the
real inputs and on random inputs. Any other launch while one is held back
issues the held reduction first. Batches of fewer than
``VLLM_MUSE_GLIMMER_QK_GLUE_MIN_TOKENS`` tokens (default 128), in particular
every decode step, run the compiled kernels.
"""

from __future__ import annotations

import hashlib
import linecache
import os
import re
import sys
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# sha256 of the kernel function source with its name replaced by KERNEL and trailing whitespace removed.
_COMBO = {
    "e559a4dbc9e1c70d302ed77e2bd2ddd358a6a89c3d10e786bc949295cd49a00e": True,  # with RoPE
    "4271f497d76aeb84d365d83f981cb08f4d9e303ac83ae992514099a3693da732": False,  # NoPE
}
_APPLY = {
    "5b03942835dc17af6bdfdae439bb1856f4e9791650a4d151bd107e90f2c4f75b": True,
    "752f9dee60719511f5e39465992e34c348c7b6c67ec9ed313fdd725ca3b17694": False,
}
_COMBO_NAME = re.compile(r"^triton_red_fused_\d+$")
_APPLY_NAME = re.compile(
    r"^triton_poi_fused__to_copy__unsafe_view_add_(cat_clone_index_select_|clone_)mean_mul_pow_rsqrt_"
    r"split_(split_)?with_sizes_(sub_unsqueeze_)?view_\d+$"
)
# Reduction launch configurations whose thread layout is one warp per head row, four elements per lane.
_CONFIGS = {(4, 128, 4), (8, 128, 8)}
_Q_HEADS, _K_HEADS, _HEAD, _STRIDE = 32, 2, 128, 4608

_AUDIT = os.environ.get("VLLM_MUSE_GLIMMER_QK_GLUE_AUDIT", "0") == "1"
_BENCH = int(os.environ.get("VLLM_MUSE_GLIMMER_QK_GLUE_BENCH", "0"))
_stats = dict(fused=0, fused_captured=0, flushed=0, audit_ok=0, audit_mismatch=0, bench=0)


def _min_tokens() -> int:
    return int(os.environ.get("VLLM_MUSE_GLIMMER_QK_GLUE_MIN_TOKENS", "128"))


def _canonical_hash(src: str) -> str:
    m = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
    if m is None:
        return ""
    body = re.sub(r"^def\s+\w+\s*\(", "def KERNEL(", src[m.start():]).rstrip()
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return hashlib.sha256(body.encode()).hexdigest()


def _float_consts(src: str) -> set[float]:
    return {float(v) for v in re.findall(r"tl\.full\(\[[^\]]*\], ([0-9.eE+-]+), tl\.float32\)", src)}


class _Kernel:
    """What is known about one compiled kernel object."""

    def __init__(self, role: str, rope: bool, src: str):
        self.role = role  # "combo" | "apply"
        self.rope = rope
        consts = _float_consts(src)
        self.eps = 1e-05 if 1e-05 in consts else None
        rest = consts - {128.0, 1e-05, 0.0}
        self.qscale = next(iter(rest)) if role == "apply" and len(rest) == 1 else None
        m = re.search(r"tl\.full\(\[[^\]]*\], (\d+), tl\.int32\)", src)
        self.cache_rows = int(m.group(1)) if m else None
        self.next_ok: dict = {}
        # combo only: "unchecked" -> "admitted" | "rejected"
        self.state = "unchecked"


def _classify(kernel) -> _Kernel | None:
    meta = getattr(kernel, "inductor_meta", None) or {}
    name = meta.get("kernel_name", "")
    if _COMBO_NAME.match(name):
        table, role = _COMBO, "combo"
    elif _APPLY_NAME.match(name):
        table, role = _APPLY, "apply"
    else:
        return None
    src = getattr(getattr(kernel, "fn", None), "src", None)
    if not isinstance(src, str):
        return None
    rope = table.get(_canonical_hash(src))
    if rope is None:
        return None
    k = _Kernel(role, rope, src)
    if k.eps is None or (role == "apply" and k.qscale is None) or (rope and role == "combo" and not k.cache_rows):
        return None
    return k


def _config_ok(kernel) -> bool:
    launchers = getattr(kernel, "launchers", None) or []
    if len(launchers) != 1:
        return False
    cfg = getattr(launchers[0], "config", None)
    if cfg is None:
        return False
    kw = getattr(cfg, "kwargs", {}) or {}
    return (kw.get("XBLOCK"), kw.get("R0_BLOCK"), getattr(cfg, "num_warps", None)) in _CONFIGS


_RUN_RE = re.compile(r"^(\w+)\.run\((.*)\)\s*$")
_FILLER = (
    re.compile(r"^buf\d+ = empty_strided_cuda\(.*\)$"),
    re.compile(r"^\w+_xnumel(_\d+)? = [\w*+ ()]+$"),
    re.compile(r"^raw_stream\d+ = get_raw_stream\(\d+\)$"),
)


def _launch_frame(name_re):
    """The compiled piece's frame that is launching a kernel whose name matches name_re (its current line is
    that kernel's ``.run(`` call), found by walking up from the launcher wrappers."""
    f = sys._getframe(1)
    for _ in range(12):
        if f is None:
            return None
        m = _RUN_RE.match(linecache.getline(f.f_code.co_filename, f.f_lineno).strip())
        if m is not None and name_re.match(m.group(1)):
            return f
        f = f.f_back
    return None


def _next_launch_ok(frame, rope: bool) -> bool:
    """The compiled piece issues the matching pointwise kernel right after this reduction (only buffer
    allocations, size expressions and stream lookups in between) on the same qkv and sum-of-squares tensors."""
    if frame is None:
        return False
    fn, ln = frame.f_code.co_filename, frame.f_lineno
    first = _RUN_RE.match(linecache.getline(fn, ln).strip())
    if first is None or not _COMBO_NAME.match(first.group(1)):
        return False
    cargs = [a.strip() for a in first.group(2).split(",")]
    qkv, sumsq = (cargs[0], cargs[3]) if rope else (cargs[0], cargs[2])
    for k in range(1, 16):
        s = linecache.getline(fn, ln + k).strip()
        if not s or s.startswith("#"):
            continue
        m = _RUN_RE.match(s)
        if m is not None:
            aargs = [a.strip() for a in m.group(2).split(",")]
            return bool(_APPLY_NAME.match(m.group(1))) and aargs[:2] == [qkv, sumsq]
        if not any(p.match(s) for p in _FILLER):
            return False
    return False


class _Held:
    __slots__ = ("kernel", "info", "args", "kwargs", "tokens")

    def __init__(self, kernel, info, args, kwargs, tokens):
        self.kernel, self.info, self.args, self.kwargs, self.tokens = kernel, info, args, kwargs, tokens


_held: _Held | None = None


def _combo_parts(info: _Kernel, args):
    """(qkv, pos, cos_sin, q_sumsq, k_out, tokens) of a reduction launch, or None."""
    if info.rope:
        if len(args) != 8:
            return None
        qkv, pos, cache, sumsq, k_out, ks0, x0, x1 = args
    else:
        if len(args) != 5:
            return None
        qkv, k_out, sumsq, x0, x1 = args  # NoPE: the key rows come first
        pos = cache = None
        ks0 = int(x0) // _K_HEADS
        x0, x1 = x1, x0
    try:
        t = int(ks0)
    except Exception:
        return None
    if t <= 0 or int(x0) != _Q_HEADS * t or int(x1) != _K_HEADS * t:
        return None
    return qkv, pos, cache, sumsq, k_out, t


def _tensors_ok(info: _Kernel, parts, q_out) -> bool:
    qkv, pos, cache, sumsq, k_out, t = parts
    bf = torch.bfloat16
    ts = [qkv, sumsq, k_out, q_out] + ([pos, cache] if info.rope else [])
    if not all(isinstance(x, torch.Tensor) and x.is_cuda for x in ts):
        return False
    ok = (qkv.dtype == bf and qkv.dim() == 2 and qkv.shape[0] >= t and qkv.shape[1] == _STRIDE
          and qkv.stride() == (_STRIDE, 1) and qkv.data_ptr() % 16 == 0)
    ok = ok and sumsq.dtype == torch.float32 and sumsq.numel() >= _Q_HEADS * t
    ok = ok and k_out.dtype == bf and k_out.is_contiguous() and k_out.numel() >= _K_HEADS * _HEAD * t
    ok = ok and q_out.dtype == bf and q_out.is_contiguous() and q_out.numel() >= _Q_HEADS * _HEAD * t
    ok = ok and k_out.data_ptr() % 8 == 0 and q_out.data_ptr() % 8 == 0
    if info.rope:
        ok = ok and pos.dtype == torch.int64 and pos.numel() >= t and pos.is_contiguous()
        ok = ok and cache.dtype == bf and cache.is_contiguous() and cache.dim() == 2 and cache.shape[1] == _HEAD
        ok = ok and cache.data_ptr() % 8 == 0
    return ok


def _apply_parts(info: _Kernel, args):
    """(qkv, sumsq, pos, cos_sin, q_out, tokens) of a pointwise launch, or None."""
    if info.rope:
        if len(args) != 7:
            return None
        qkv, sumsq, pos, cache, out, ks0, xn = args
    else:
        if len(args) != 5:
            return None
        qkv, sumsq, out, ks0, xn = args
        pos = cache = None
    try:
        t = int(ks0)
    except Exception:
        return None
    if int(xn) != _Q_HEADS * _HEAD * t:
        return None
    return qkv, sumsq, pos, cache, out, t


def _launch(combo: _Kernel, apply: _Kernel, parts, q_out, stream, sumsq=None, k_out=None) -> None:
    import muse_qk_glue as G

    qkv, pos, cache, s0, k0, t = parts
    G.launch(combo.rope, t, qkv, pos, cache, combo.cache_rows or 0, apply.qscale, combo.eps,
             s0 if sumsq is None else sumsq, k0 if k_out is None else k_out, q_out, stream)


def _same_bits(a: torch.Tensor, b: torch.Tensor, n: int) -> bool:
    a = a.reshape(-1)[:n]
    b = b.reshape(-1)[:n]
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _compare_real(combo, apply, parts, q_out, stream) -> bool:
    """The stock pair has written the real outputs; run the fused kernel into copies and compare."""
    qkv, pos, cache, sumsq, k_out, t = parts
    s2 = torch.full_like(sumsq, float("nan"))
    k2 = torch.full_like(k_out, float("nan"))
    q2 = torch.full_like(q_out, float("nan"))
    _launch(combo, apply, parts, q2, stream, sumsq=s2, k_out=k2)
    torch.cuda.synchronize()
    return (_same_bits(sumsq, s2, _Q_HEADS * t) and _same_bits(k_out, k2, _K_HEADS * _HEAD * t)
            and _same_bits(q_out, q2, _Q_HEADS * _HEAD * t))


def _random_cases(combo_k, combo, apply_k, apply, prev_run, parts, q_out, kwargs, stream) -> bool:
    """Both paths on random activations (normal values and raw bit patterns) and random positions."""
    qkv, pos, cache, _, _, t = parts
    dev = qkv.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261122)
    for n in (t + 3, 200, 128, 100, 33):
        q_r = torch.randint(-32768, 32768, (n, _STRIDE), dtype=torch.int16, device=dev, generator=gen)
        q_r = q_r.view(torch.bfloat16)
        q_r[: n // 2].copy_(torch.randn(n // 2, _STRIDE, device=dev, generator=gen) * 4.0)
        q_r[n // 2: n // 2 + n // 8].mul_(0)
        pos_r = None
        if combo.rope:
            pos_r = torch.randint(-combo.cache_rows, combo.cache_rows, (n,), dtype=torch.int64, device=dev,
                                  generator=gen)
        outs = []
        for _ in range(2):
            outs.append((torch.full((n * _Q_HEADS, 1), float("nan"), dtype=torch.float32, device=dev),
                         torch.full((n, _K_HEADS, _HEAD), float("nan"), dtype=torch.bfloat16, device=dev),
                         torch.full((n, _Q_HEADS, _HEAD), float("nan"), dtype=torch.bfloat16, device=dev)))
        (s_a, k_a, o_a), (s_b, k_b, o_b) = outs
        if combo.rope:
            cargs = (q_r, pos_r, cache, s_a, k_a, n, _Q_HEADS * n, _K_HEADS * n)
            aargs = (q_r, s_a, pos_r, cache, o_a, n, _Q_HEADS * _HEAD * n)
        else:
            cargs = (q_r, k_a.view(n * _K_HEADS, _HEAD), s_a, _K_HEADS * n, _Q_HEADS * n)
            aargs = (q_r, s_a, o_a.view(n, _Q_HEADS * _HEAD), n, _Q_HEADS * _HEAD * n)
        prev_run(combo_k, *cargs, **kwargs)
        prev_run(apply_k, *aargs, **kwargs)
        _launch(combo, apply, (q_r, pos_r, cache, s_b, k_b, n), o_b, stream)
        torch.cuda.synchronize()
        if not (_same_bits(s_a, s_b, n * _Q_HEADS) and _same_bits(k_a, k_b, n * _K_HEADS * _HEAD)
                and _same_bits(o_a, o_b, n * _Q_HEADS * _HEAD)):
            return False
    return True


def _bench(held: _Held, apply_k, apply, prev_run, parts, q_out, kwargs, stream) -> None:
    """Diagnostics (VLLM_MUSE_GLIMMER_QK_GLUE_BENCH=N): time the compiled pair and the fused kernel on this
    launch's real inputs, each after an L2 flush and back to back, alternating; outputs are rewritten with the
    same bits."""
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=parts[0].device)
    reps = 15
    res = {}
    for mode in ("flush", "hot"):
        ts = {"stock": [], "fused": []}
        for i in range(reps):
            for which in (("stock", "fused") if i % 2 == 0 else ("fused", "stock")):
                if mode == "flush":
                    flush.fill_(i & 0xFF)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                if which == "stock":
                    prev_run(held.kernel, *held.args, **held.kwargs)
                    prev_run(apply_k, *_cur_apply_args, **kwargs)
                else:
                    _launch(held.info, apply, parts, q_out, stream)
                e1.record()
                torch.cuda.synchronize()
                ts[which].append(e0.elapsed_time(e1) * 1e3)
        res[mode] = {k: sorted(v)[len(v) // 2] for k, v in ts.items()}
    _stats["bench"] += 1
    logger.info("q/k norm glue bench (%s, %d tokens): stock pair %.1f us, fused %.1f us after an L2 flush; "
                "%.1f us vs %.1f us back to back", "RoPE" if held.info.rope else "NoPE", parts[5],
                res["flush"]["stock"], res["flush"]["fused"], res["hot"]["stock"], res["hot"]["fused"])


_cur_apply_args: tuple = ()


def _flush(prev_run) -> None:
    global _held
    h, _held = _held, None
    if h is not None:
        _stats["flushed"] += 1
        prev_run(h.kernel, *h.args, **h.kwargs)


class _Swap:
    admitted = 0
    rejected = 0


def _on_apply(kernel, info: _Kernel, prev_run, args, kwargs):
    global _held, _cur_apply_args
    h = _held
    stream = kwargs.get("stream")
    if h is None:
        return prev_run(kernel, *args, **kwargs)
    parts = _combo_parts(h.info, h.args)
    ap = _apply_parts(info, args)
    match = (parts is not None and ap is not None and info.rope == h.info.rope and ap[5] == parts[5]
             and ap[0] is parts[0] and ap[1].data_ptr() == parts[3].data_ptr()
             and (not info.rope or (ap[2] is parts[1] and ap[3] is parts[2]))
             and stream is not None and h.kwargs.get("stream") == stream
             and _tensors_ok(h.info, parts, ap[4]))
    if not match:
        _flush(prev_run)
        return prev_run(kernel, *args, **kwargs)
    _held = None
    q_out = ap[4]
    capturing = torch.cuda.is_current_stream_capturing()
    if h.info.state == "unchecked":
        # The first pair outside graph capture: run both compiled kernels (their outputs stand), then compare.
        prev_run(h.kernel, *h.args, **h.kwargs)
        prev_run(kernel, *args, **kwargs)
        ok = False
        try:
            ok = (_compare_real(h.info, info, parts, q_out, stream)
                  and _random_cases(h.kernel, h.info, kernel, info, prev_run, parts, q_out, kwargs, stream))
        except Exception:
            logger.warning("q/k norm glue check failed to run", exc_info=True)
        h.info.state = "admitted" if ok else "rejected"
        if ok:
            _Swap.admitted += 1
        else:
            _Swap.rejected += 1
        logger.info("Exact prompt q/k norm glue: %d compiled kernel pairs admitted, %d rejected (%s, %d tokens)",
                    _Swap.admitted, _Swap.rejected, "RoPE" if h.info.rope else "NoPE", parts[5])
        return None
    if not capturing and (_AUDIT or _stats["bench"] < _BENCH):
        prev_run(h.kernel, *h.args, **h.kwargs)
        prev_run(kernel, *args, **kwargs)
        ok = False
        try:
            ok = _compare_real(h.info, info, parts, q_out, stream)
        except Exception:
            logger.warning("q/k norm glue audit failed to run", exc_info=True)
        _stats["audit_ok" if ok else "audit_mismatch"] += 1
        logger.info("q/k norm glue audit (%s, %d tokens): %s; totals %d identical, %d different",
                    "RoPE" if h.info.rope else "NoPE", parts[5], "identical" if ok else "DIFFERENT",
                    _stats["audit_ok"], _stats["audit_mismatch"])
        if _stats["bench"] < _BENCH:
            _cur_apply_args = args
            try:
                _bench(h, kernel, info, prev_run, parts, q_out, kwargs, stream)
            except Exception:
                logger.warning("q/k norm glue bench failed", exc_info=True)
            _cur_apply_args = ()
        return None
    _launch(h.info, info, parts, q_out, stream)
    _stats["fused"] += 1
    if capturing:
        _stats["fused_captured"] += 1
    if _stats["fused"] == 1 or (capturing and _stats["fused_captured"] % 52 == 0):
        logger.info("Exact prompt q/k norm glue: %d pairs fused (%d under graph capture), %d held launches issued "
                    "alone", _stats["fused"], _stats["fused_captured"], _stats["flushed"])
    return None


def _on_combo(kernel, info: _Kernel, prev_run, args, kwargs):
    global _held
    parts = _combo_parts(info, args)
    if (parts is None or parts[5] < _min_tokens() or info.state == "rejected" or kwargs.get("stream") is None
            or set(kwargs) != {"stream"}):
        return prev_run(kernel, *args, **kwargs)
    if info.state == "unchecked" and torch.cuda.is_current_stream_capturing():
        return prev_run(kernel, *args, **kwargs)
    if not _config_ok(kernel):
        # Not autotuned yet (the first launch picks the configuration) or another layout: compiled kernel.
        return prev_run(kernel, *args, **kwargs)
    frame = _launch_frame(_COMBO_NAME)
    if frame is None:
        return prev_run(kernel, *args, **kwargs)
    key = (frame.f_code.co_filename, frame.f_lineno)
    ok = info.next_ok.get(key)
    if ok is None:
        try:
            ok = _next_launch_ok(frame, info.rope)
        except Exception:
            ok = False
        info.next_ok[key] = ok
        if not ok:
            logger.info("q/k norm glue: the compiled piece does not issue the pointwise kernel next; not fused")
    if not ok:
        return prev_run(kernel, *args, **kwargs)
    _held = _Held(kernel, info, args, kwargs, parts[5])
    return None


_installed = False


def install_qk_glue() -> None:
    """Fuse the prompt-batch launches of the reviewed compiled q/k norm kernel pairs (see the module doc)."""
    global _installed
    if _installed or os.environ.get("VLLM_MUSE_GLIMMER_QK_GLUE", "1") != "1":
        return
    try:
        import muse_qk_glue

        if muse_qk_glue.lib() is None:
            logger.info("muse_qk_glue library not built; prompt q/k norm glue disabled")
            return
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception:
        logger.info("muse_qk_glue or the Inductor launcher not available; prompt q/k norm glue disabled")
        return
    prev_run = CachingAutotuner.run

    def run(self, *args, **kwargs):
        info = self.__dict__.get("_qk_glue_info", 0)
        if info == 0:
            try:
                info = _classify(self)
            except Exception:
                info = None
            self.__dict__["_qk_glue_info"] = info
        if info is not None and info.role == "apply":
            return _on_apply(self, info, prev_run, args, kwargs)
        if _held is not None:
            _flush(prev_run)
        if info is not None:
            return _on_combo(self, info, prev_run, args, kwargs)
        return prev_run(self, *args, **kwargs)

    CachingAutotuner.run = run
    _installed = True
    logger.info("Prompt q/k norm glue installed (min %d tokens)", _min_tokens())
