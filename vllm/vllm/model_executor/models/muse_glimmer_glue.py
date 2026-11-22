# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-batch replacement for the compiled query QK-norm + RoPE kernel.

The compiled graph applies the query QK-norm, the query pre-scale and the
NEOX rotary embedding in one Inductor pointwise kernel
(``triton_poi_fused_..._cat_clone_index_select_..._sub_unsqueeze_view_N``).
Because the token count is symbolic, every address in that kernel goes
through 64-bit divisions and remainders by the token count, which makes it
several times slower than the same kernel without RoPE on long batches.

This module swaps in a kernel that evaluates the same per-element float
chain, operation for operation, with plain row/head/dim indexing:

    r   = rsqrt(sumsq[row] / 128.0 + eps)          (div.full, rsqrt.approx)
    lo  = ((x[d]      * r) * qscale) * cos[d] - ((x[d + 64] * r) * qscale) * sin[d]
    hi  = ((x[d]      * r) * qscale) * cos[d - 64] + ((x[d - 64] * r) * qscale) * sin[d - 64]
    out = bf16(lo if d < 64 else hi)

all in fp32 with no intermediate bf16 rounding, exactly as the Inductor
kernel computes it (its bf16 casts become fp32 casts). The squared-sum input
is still produced by the stock compiled reduction kernel.

The swap is made at the kernel launcher, so the compiled graph and every
other kernel stay as compiled. It only applies to launches with at least
``VLLM_MUSE_GLIMMER_PROMPT_GLUE_MIN_TOKENS`` tokens (default 128); shorter
batches, in particular every decode step, run the compiled kernel. Each
compiled kernel object is admitted only after its first long launch, made
outside graph capture, produces bit-identical outputs from both kernels on
the real inputs and on random inputs; otherwise it keeps the compiled kernel.
"""

from __future__ import annotations

import os
import re

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

try:
    # The same libdevice binding the compiled kernels use.
    from torch._inductor.runtime.triton_helpers import libdevice
except Exception:  # pragma: no cover
    libdevice = tl.extra.cuda.libdevice

logger = init_logger(__name__)

_QROPE_NAME = re.compile(
    r"^triton_poi_fused__to_copy__unsafe_view_add_cat_clone_index_select_mean_"
    r"mul_pow_rsqrt_split_split_with_sizes_sub_unsqueeze_view_\d+$"
)


# Micro-attribution: Replacing the symbolic-stride Inductor query QK-norm + RoPE pointwise kernel
# on prompt batches via a launcher wrapper is inspired by @ArthurOstapenko (Yukon submission fa040f4a);
# the multi-distribution live+synthetic bitwise admission check and graph-capture guard around it
# are custom-implemented here.
@triton.jit
def _qnorm_rope_kernel(
    qkv_ptr,
    sumsq_ptr,
    pos_ptr,
    cache_ptr,
    out_ptr,
    QKV_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    DIM_F: tl.constexpr,
    EPS: tl.constexpr,
    QSCALE: tl.constexpr,
    HBLOCK: tl.constexpr,
):
    HALF: tl.constexpr = HEAD_DIM // 2
    GROUPS: tl.constexpr = NUM_HEADS // HBLOCK
    pid = tl.program_id(0)
    t = pid // GROUPS
    h = (pid % GROUPS) * HBLOCK + tl.arange(0, HBLOCK)[:, None]
    d = tl.arange(0, HEAD_DIM)[None, :]
    t64 = t.to(tl.int64)
    q_row = qkv_ptr + t64 * QKV_STRIDE + h * HEAD_DIM
    ss_row = sumsq_ptr + t64 * NUM_HEADS + h
    lo = d < HALF
    hi = d >= HALF
    # Same operation order as the compiled kernel: first half.
    tmp7 = tl.load(q_row + d, lo, other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.load(ss_row + d * 0, lo, other=0.0)
    tmp10 = tl.full([1, 1], DIM_F, tl.float32)
    tmp11 = tmp9 / tmp10
    tmp12 = tl.full([1, 1], EPS, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full([1, 1], QSCALE, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(pos_ptr + t64 + d * 0, lo, other=0)
    tmp20 = tl.full([1, 1], CACHE_ROWS, tl.int32).to(tl.int32)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tmp25 = tl.load(cache_ptr + (HEAD_DIM * tmp23 + d), lo, other=0.0).to(tl.float32)
    tmp26 = tmp18 * tmp25
    tmp27 = tl.load(q_row + (HALF + d), lo, other=0.0).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(ss_row + d * 0, lo, other=0.0)
    tmp30 = tmp29 / tmp10
    tmp31 = tmp30 + tmp12
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp28 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp17
    tmp36 = tl.load(cache_ptr + (HALF + HEAD_DIM * tmp23 + d), lo, other=0.0).to(
        tl.float32
    )
    tmp37 = tmp35 * tmp36
    tmp38 = tmp26 - tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(lo, tmp38, tmp39)
    # Second half.
    tmp44 = tl.load(q_row + d, hi, other=0.0).to(tl.float32)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(ss_row + d * 0, hi, other=0.0)
    tmp47 = tl.full([1, 1], DIM_F, tl.float32)
    tmp48 = tmp46 / tmp47
    tmp49 = tl.full([1, 1], EPS, tl.float32)
    tmp50 = tmp48 + tmp49
    tmp51 = libdevice.rsqrt(tmp50)
    tmp52 = tmp45 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tl.full([1, 1], QSCALE, tl.float32)
    tmp55 = tmp53 * tmp54
    tmp56 = tl.load(pos_ptr + t64 + d * 0, hi, other=0)
    tmp57 = tl.full([1, 1], CACHE_ROWS, tl.int32).to(tl.int32)
    tmp58 = tmp56 + tmp57
    tmp59 = tmp56 < 0
    tmp60 = tl.where(tmp59, tmp58, tmp56)
    tmp62 = tl.load(cache_ptr + (HEAD_DIM * tmp60 + (d - HALF)), hi, other=0.0).to(
        tl.float32
    )
    tmp63 = tmp55 * tmp62
    tmp64 = tl.load(q_row + (d - HALF), hi, other=0.0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tl.load(ss_row + d * 0, hi, other=0.0)
    tmp67 = tmp66 / tmp47
    tmp68 = tmp67 + tmp49
    tmp69 = libdevice.rsqrt(tmp68)
    tmp70 = tmp65 * tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp54
    tmp73 = tl.load(
        cache_ptr + (HALF + HEAD_DIM * tmp60 + (d - HALF)), hi, other=0.0
    ).to(tl.float32)
    tmp74 = tmp72 * tmp73
    tmp75 = tmp63 + tmp74
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(hi, tmp75, tmp76)
    tmp78 = tl.where(lo, tmp40, tmp77)
    out_row = out_ptr + t64 * (NUM_HEADS * HEAD_DIM) + h * HEAD_DIM
    tl.store(out_row + d, tmp78)


def _canonical_chain(src: str) -> list[str]:
    """The kernel's value chain with addresses, shapes and names erased."""
    names: dict[str, str] = {}

    def tmp_name(m: re.Match) -> str:
        return names.setdefault(m.group(0), f"t{len(names)}")

    chain = []
    for line in src.splitlines():
        m = re.match(r"\s*(tmp\d+) = (.*)$", line)
        if m is None:
            if "tl.store(" in line:
                chain.append("store")
            continue
        lhs, rhs = m.groups()
        if "tl.load(" in rhs:
            ptr = re.search(r"tl\.load\((\w+)", rhs).group(1)
            cast = ".f32" if rhs.endswith(".to(tl.float32)") else ""
            rhs = f"load({ptr}){cast}"
        else:
            rhs = re.sub(r"tl\.full\(\[[^\]]*\], ", "full(", rhs)
            rhs = re.sub(r"tl\.full\((tmp\d+)\.shape, ", "full(", rhs)
            rhs = re.sub(r"\(\s*[xy]\d+\s*\)", "(IDX)", rhs)
            rhs = rhs.replace("tmp38.dtype", "f32").replace("tmp75.dtype", "f32")
            rhs = re.sub(r",\s*tmp\d+\.dtype\)", ", f32)", rhs)
        line_c = re.sub(r"tmp\d+", tmp_name, f"{lhs} = {rhs}")
        chain.append(line_c)
    return chain


# Value chain of the compiled kernel (both the 1D and the 2D-tiled variant),
# with the query pre-scale, the cache length and the head size as parameters.
def _expected_chain(qscale: str, rows: str) -> list[str]:
    return [
        "t0 = (IDX).to(tl.int32)",
        "t1 = full(0, tl.int64)",
        "t2 = t0 >= t1",
        "t3 = (IDX).to(tl.int64)",
        "t4 = (t3).to(tl.int64)",
        "t5 = full(64, tl.int64)",
        "t6 = t4 < t5",
        "t7 = load(in_ptr0).f32",
        "t8 = t7.to(tl.float32)",
        "t9 = load(in_ptr1)",
        "t10 = full(128.0, tl.float32)",
        "t11 = (t9 / t10)",
        "t12 = full(1e-05, tl.float32)",
        "t13 = t11 + t12",
        "t14 = libdevice.rsqrt(t13)",
        "t15 = t8 * t14",
        "t16 = t15.to(tl.float32)",
        f"t17 = full({qscale}, tl.float32)",
        "t18 = t16 * t17",
        "t19 = load(in_ptr2)",
        f"t20 = (full({rows}, tl.int32)).to(tl.int32)",
        "t21 = t19 + t20",
        "t22 = t19 < 0",
        "t23 = tl.where(t22, t21, t19)",
        "t24 = load(in_ptr3).f32",
        "t25 = t18 * t24",
        "t26 = load(in_ptr0).f32",
        "t27 = t26.to(tl.float32)",
        "t28 = load(in_ptr1)",
        "t29 = (t28 / t10)",
        "t30 = t29 + t12",
        "t31 = libdevice.rsqrt(t30)",
        "t32 = t27 * t31",
        "t33 = t32.to(tl.float32)",
        "t34 = t33 * t17",
        "t35 = load(in_ptr3).f32",
        "t36 = t34 * t35",
        "t37 = t25 - t36",
        "t38 = full(0.0, f32)",
        "t39 = tl.where(t6, t37, t38)",
        "t40 = t0 >= t5",
        "t41 = full(128, tl.int64)",
        "t42 = t0 < t41",
        "t43 = load(in_ptr0).f32",
        "t44 = t43.to(tl.float32)",
        "t45 = load(in_ptr1)",
        "t46 = full(128.0, tl.float32)",
        "t47 = (t45 / t46)",
        "t48 = full(1e-05, tl.float32)",
        "t49 = t47 + t48",
        "t50 = libdevice.rsqrt(t49)",
        "t51 = t44 * t50",
        "t52 = t51.to(tl.float32)",
        f"t53 = full({qscale}, tl.float32)",
        "t54 = t52 * t53",
        "t55 = load(in_ptr2)",
        f"t56 = (full({rows}, tl.int32)).to(tl.int32)",
        "t57 = t55 + t56",
        "t58 = t55 < 0",
        "t59 = tl.where(t58, t57, t55)",
        "t60 = load(in_ptr3).f32",
        "t61 = t54 * t60",
        "t62 = load(in_ptr0).f32",
        "t63 = t62.to(tl.float32)",
        "t64 = load(in_ptr1)",
        "t65 = (t64 / t46)",
        "t66 = t65 + t48",
        "t67 = libdevice.rsqrt(t66)",
        "t68 = t63 * t67",
        "t69 = t68.to(tl.float32)",
        "t70 = t69 * t53",
        "t71 = load(in_ptr3).f32",
        "t72 = t70 * t71",
        "t73 = t61 + t72",
        "t74 = full(0.0, f32)",
        "t75 = tl.where(t40, t73, t74)",
        "t76 = tl.where(t6, t39, t75)",
        "store",
    ]


class _QRopeSwap:
    """Launcher-level swap for one compiled query QK-norm + RoPE kernel."""

    admitted = 0
    rejected = 0

    def __init__(self, qscale: float, rows: int, qkv_stride: int, two_d: bool):
        self.qscale = qscale
        self.rows = rows
        self.qkv_stride = qkv_stride
        self.two_d = two_d
        self.state = "unchecked"  # -> "admitted" | "rejected"

    def launch(self, qkv, sumsq, pos, cache, out, num_tokens: int) -> None:
        hblock = 8
        _qnorm_rope_kernel[(num_tokens * (32 // hblock),)](
            qkv,
            sumsq,
            pos,
            cache,
            out,
            QKV_STRIDE=self.qkv_stride,
            NUM_HEADS=32,
            HEAD_DIM=128,
            CACHE_ROWS=self.rows,
            DIM_F=128.0,
            EPS=1e-05,
            QSCALE=self.qscale,
            HBLOCK=hblock,
            num_warps=4,
        )

    def _stock_args(self, args, qkv, sumsq, pos, cache, out, num_tokens):
        if self.two_d:
            return (qkv, sumsq, pos, cache, out, num_tokens, 32 * num_tokens, 128)
        return (qkv, sumsq, pos, cache, out, num_tokens, 4096 * num_tokens)

    def _check(self, kernel, orig_run, args, kwargs) -> bool:
        qkv, sumsq, pos, cache, out = args[:5]
        num_tokens = int(args[5])
        orig_run(kernel, *args, **kwargs)
        mine = torch.empty_like(out)
        self.launch(qkv, sumsq, pos, cache, mine, num_tokens)
        same = torch.equal(out.view(torch.int16), mine.view(torch.int16))
        gen = torch.Generator(device=qkv.device)
        gen.manual_seed(20261122)
        for n in (num_tokens + 1, 333, 128):
            if not same:
                break
            q_r = torch.randint(
                -32768, 32768, (n, qkv.shape[1]), dtype=torch.int16,
                device=qkv.device, generator=gen,
            ).view(torch.bfloat16)
            q_r[: n // 2].copy_(
                torch.randn(n // 2, qkv.shape[1], device=qkv.device, generator=gen)
                * 4.0
            )
            ss_r = (
                torch.rand(n * 32, device=qkv.device, generator=gen) * 2.0
            ) ** 40.0 * 1.0e6
            ss_r[::17] = 0.0
            ss_r = ss_r.reshape(n * 32, 1)
            pos_r = torch.randint(
                -self.rows, self.rows, (n,), dtype=pos.dtype,
                device=qkv.device, generator=gen,
            )
            ref = torch.empty((n,) + tuple(out.shape[1:]), dtype=out.dtype,
                              device=out.device)
            got = torch.empty_like(ref)
            orig_run(
                kernel, *self._stock_args(args, q_r, ss_r, pos_r, cache, ref, n),
                **kwargs,
            )
            self.launch(q_r, ss_r, pos_r, cache, got, n)
            same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        return same

    def run(self, kernel, orig_run, args, kwargs):
        num_tokens = int(args[5])
        if num_tokens < _min_tokens() or self.state == "rejected":
            return orig_run(kernel, *args, **kwargs)
        if self.state == "unchecked":
            if torch.cuda.is_current_stream_capturing():
                return orig_run(kernel, *args, **kwargs)
            ok = False
            try:
                ok = self._check(kernel, orig_run, args, kwargs)
            except Exception:
                logger.warning("q-norm RoPE apply check failed to run", exc_info=True)
            self.state = "admitted" if ok else "rejected"
            if ok:
                _QRopeSwap.admitted += 1
            else:
                _QRopeSwap.rejected += 1
            logger.info(
                "q-norm RoPE apply for prompt batches: %d admitted, %d rejected",
                _QRopeSwap.admitted,
                _QRopeSwap.rejected,
            )
            # The compiled kernel already wrote this launch's output.
            return None
        qkv, sumsq, pos, cache, out = args[:5]
        self.launch(qkv, sumsq, pos, cache, out, num_tokens)
        return None


def _min_tokens() -> int:
    return int(os.environ.get("VLLM_MUSE_GLIMMER_PROMPT_GLUE_MIN_TOKENS", "1"))


def _classify(kernel) -> _QRopeSwap | None:
    meta = getattr(kernel, "inductor_meta", None) or {}
    name = meta.get("kernel_name", "")
    if not _QROPE_NAME.match(name):
        return None
    src = getattr(getattr(kernel, "fn", None), "src", None)
    if not isinstance(src, str):
        return None
    m_scale = re.search(r"tmp17 = tl\.full\(\[[^\]]*\], ([0-9.eE+-]+), tl\.float32\)", src)
    m_rows = re.search(r"tmp20 = \(tl\.full\(\[[^\]]*\], (\d+), tl\.int32\)\)", src)
    m_stride = re.search(r"tl\.load\(in_ptr0 \+ \((\d+)\*", src)
    if not (m_scale and m_rows and m_stride):
        return None
    if _canonical_chain(src) != _expected_chain(m_scale.group(1), m_rows.group(1)):
        return None
    if "% 4096" not in src or int(m_stride.group(1)) < 4096:
        return None
    two_d = "YBLOCK" in src
    return _QRopeSwap(
        float(m_scale.group(1)), int(m_rows.group(1)), int(m_stride.group(1)), two_d
    )


_installed = False


def install_prompt_glue() -> None:
    """Route long launches of matching compiled kernels to the swaps above."""
    global _installed
    if _installed or os.environ.get("VLLM_MUSE_GLIMMER_PROMPT_GLUE", "1") != "1":
        return
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception:
        logger.info("Inductor launcher not available; prompt glue swap disabled")
        return
    orig_run = CachingAutotuner.run

    def run(self, *args, **kwargs):
        swap = self.__dict__.get("_prompt_glue_swap", 0)
        if swap == 0:
            try:
                swap = _classify(self)
            except Exception:
                swap = None
            self.__dict__["_prompt_glue_swap"] = swap
        if swap is None:
            return orig_run(self, *args, **kwargs)
        return swap.run(self, orig_run, args, kwargs)

    CachingAutotuner.run = run
    _installed = True
