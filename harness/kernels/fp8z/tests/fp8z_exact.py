"""Bit-exactness of fp8z (lossless 7-bit FP8 weight code, 1..8-row GEMM) against stock ops.cutlass_scaled_mm on
the real FP8 linears of the checkpoint (merged and requantized exactly as vLLM's ModelOpt FP8 path does).

Per layer: load-time encode + full decode check, then M=1 (decode) on >= 7 activation kinds, M=2..8 on two kinds,
every ring depth, NaN-prefilled outputs, three repeats; then an adversarial pass (poisoned allocator, CUDA graph
replay with changed inputs) on the first layer of every shape. No timing.

Run: bin/atom glim exec SHA harness/kernels/fp8z/tests/fp8z_exact.py [--layers all|N] [--kinds qkv,o_proj,...]
Exit code 0 only if every comparison is bit-identical and every layer was admitted."""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale


def log(*a):
    print(*a, flush=True)


class Snap:
    def __init__(self, snap: str):
        self.snap = snap
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self._files = {}

    def f(self, name):
        fn = os.path.join(self.snap, self.wmap[name])
        if fn not in self._files:
            self._files[fn] = safe_open(fn, "pt", device="cpu")
        return self._files[fn]

    def get(self, name):
        return self.f(name).get_tensor(name)

    def meta(self, name):
        sl = self.f(name).get_slice(name)
        return sl.get_dtype(), tuple(sl.get_shape())


GROUPS = [("qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
          ("attn_gate", ["self_attn.gate_proj"]), ("o_proj", ["self_attn.o_proj"]),
          ("gate_up", ["mlp.gate_proj", "mlp.up_proj"]), ("down", ["mlp.down_proj"])]


def fp8_linears(snap: Snap):
    """[(kind, layer, [hf prefixes])] of the merged FP8 linears of the language model."""
    pat = re.compile(r"^(.*language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+))\.weight$")
    per = collections.defaultdict(dict)
    for name in sorted(snap.wmap):
        m = pat.match(name)
        if m and snap.meta(name)[0] == "F8_E4M3":
            per[int(m.group(2))][m.group(3) + "." + m.group(4)] = m.group(1)
    out = []
    for li in sorted(per):
        for kind, ps in GROUPS:
            if all(p in per[li] for p in ps):
                out.append((kind, li, [per[li][p] for p in ps]))
    return out


class Fp8Linear:
    """W [N, K] e4m3 (B = W.t() column-major), sb / sa fp32 scalars, as ModelOptFp8LinearMethod +
    CutlassFP8ScaledMMLinearKernel leave them (max shard scale, requantize shards whose scale differs)."""

    def __init__(self, snap: Snap, prefixes):
        ws = [snap.get(p + ".weight") for p in prefixes]
        w = torch.cat(ws, 0).cuda()
        wsc = torch.stack([snap.get(p + ".weight_scale").reshape(()).float() for p in prefixes]).cuda()
        isc = torch.stack([snap.get(p + ".input_scale").reshape(()).float() for p in prefixes]).cuda()
        max_w = wsc.max()
        self.requantized = not bool((wsc == wsc[0]).all())
        if self.requantized:
            max_w, w = requantize_with_max_scale(w, wsc, [x.shape[0] for x in ws])
        self.w = w.contiguous()
        self.b = self.w.t()
        self.sb = max_w.reshape(())
        self.sa = isc.max().reshape(())
        self.n, self.k = self.w.shape

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        q, _ = ops.scaled_fp8_quant(x.contiguous(), self.sa.reshape(1))
        return q

    def stock(self, a: torch.Tensor) -> torch.Tensor:
        return ops.cutlass_scaled_mm(a, self.b, out_dtype=torch.bfloat16, scale_a=self.sa, scale_b=self.sb, bias=None)


def gen_x(m: int, k: int, kind: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float32)
    if kind == 1:  # heavy outliers, exact zeros and negative zeros
        x = x * 4.0
        idx = torch.randint(0, k, (m, 16), generator=gen, device="cuda")
        x.scatter_(1, idx, torch.randn(m, 16, generator=gen, device="cuda") * 300.0)
        x = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.05, torch.zeros_like(x), x)
        x = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.02, torch.full_like(x, -0.0), x)
    elif kind == 2:
        x = x * 0.05
    elif kind == 3:  # per-column scales over 2^-6 .. 2^6 (RMSNorm-weight-like)
        x = x * torch.exp2(torch.randint(-6, 7, (k,), generator=gen, device="cuda").float())
    elif kind == 4:  # zeros and negative zeros only
        x = torch.where(x > 0, torch.zeros_like(x), torch.full_like(x, -0.0))
    elif kind == 5:  # saturating
        x = x * 1e4
    return x.to(torch.bfloat16)


def raw_fp8(m: int, k: int, gen: torch.Generator, nan: bool = False) -> torch.Tensor:
    """Every e4m3 bit pattern (NaN 0x7f / 0xff excluded unless nan)."""
    b = torch.randint(0, 256, (m, k), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    if not nan:
        b = torch.where((b & 0x7f) == 0x7f, b ^ 1, b)
    return b.view(torch.float8_e4m3fn)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


class Tally:
    def __init__(self):
        self.cases = 0
        self.elements = 0
        self.fails = []

    def check(self, tag, ref, new):
        self.cases += 1
        self.elements += ref.numel()
        if not bits_equal(ref, new):
            bad = ref.view(torch.int16) != new.view(torch.int16)
            info = {"elements": int(bad.sum()), "first": bad.nonzero()[:4].tolist()}
            self.fails.append((tag, info))
            log(f"   MISMATCH {tag}: {info}")
            return False
        return True


def main():
    import fp8z
    from vllm.model_executor.layers.quantization.utils import glimmer_fp8 as g8
    from vllm.model_executor.kernels.linear.scaled_mm import fp8z as fzmod
    ext = g8._ext
    log("glimmer_fp8_ext available:", ext is not None, "routing table:", g8._CFG)
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    lins = fp8_linears(snap)
    want = None
    if "--kinds" in sys.argv:
        want = set(sys.argv[sys.argv.index("--kinds") + 1].split(","))
    nl = sys.argv[sys.argv.index("--layers") + 1] if "--layers" in sys.argv else "all"
    by = collections.OrderedDict()
    for kind, li, prefs in lins:
        if want is None or kind in want:
            by.setdefault(kind, []).append((li, prefs))
    log("FP8 linears:", {k: [li for li, _ in v] for k, v in by.items()})
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261122)
    t = Tally()
    adv = Tally()
    rejected = []
    cls = collections.defaultdict(lambda: [0, 0, 0])
    t0 = time.time()
    first = {}
    for kind, lst in by.items():
        picks = lst if nl == "all" else lst[: int(nl)]
        for li, prefs in picks:
            L = Fp8Linear(snap, prefs)
            res = fp8z.prepare(L.w, return_info=True)
            if res is None:
                rejected.append((kind, li, fp8z.stats["last_reason"]))
                log(f"[{kind} L{li} {L.n}x{L.k}] REJECTED: {fp8z.stats['last_reason']}")
                continue
            buf, info = res
            c = cls[kind]
            c[0] += L.n * L.k
            c[1] += info["read_bytes"]
            c[2] += 1
            n0 = len(t.fails)
            xs = [(f"gen{kd}", L.quant(gen_x(8, L.k, kd, gen))) for kd in range(6)]
            xs.append(("raw", raw_fp8(8, L.k, gen)))
            for nm, a8 in xs:
                for m in (1,) if nm not in ("gen0", "raw") else (1, 2, 3, 4, 5, 8):
                    a = a8[:m].contiguous()
                    ref = L.stock(a)
                    cfg = g8._CFG.get((L.n, L.k))
                    if ext is not None and cfg is not None:
                        # the record's decode path (G2 tile) against stock: fp8z must equal both
                        t.check(f"{kind} L{li} {nm} M{m} record-G2 vs stock", ref,
                                ext.gemm(a, L.b, L.sa.reshape(1), L.sb.reshape(1), cfg))
                    for ring in (2, 3, 33, 4, 6, 32, 31, 21):
                        out = torch.full((m, L.n), float("nan"), dtype=torch.bfloat16, device="cuda")
                        fp8z.gemm(a, buf, L.sa, L.sb, L.n, L.k, ring=ring, out=out)
                        t.check(f"{kind} L{li} {nm} M{m} ring{ring}", ref, out)
                    for rep in range(2):
                        t.check(f"{kind} L{li} {nm} M{m} repeat{rep}", ref, fp8z.gemm(a, buf, L.sa, L.sb, L.n, L.k))
            # production op: glimmer::fp8_scaled_mm with this layer's code registered
            fzmod.REGISTRY[L.b.data_ptr()] = buf
            for nm, a8 in xs[:3]:
                for m in (1, 2):
                    a = a8[:m].contiguous()
                    out = torch.ops.glimmer.fp8_scaled_mm(a, L.b, L.sa, L.sb)
                    t.check(f"{kind} L{li} {nm} M{m} production op", L.stock(a), out)
            del fzmod.REGISTRY[L.b.data_ptr()]
            log(f"[{time.time() - t0:6.1f}s] [{kind} L{li} N={L.n} K={L.k}] requant={L.requantized} "
                f"{info['bits_per_weight']:.4f} bits/w esc {info['escape_rate']:.5f} maxwc {info['max_warp_chunk']} "
                f"tile {info['tile_bytes']} | {len(t.fails) - n0} mismatches")
            if (L.n, L.k) not in first:
                first[(L.n, L.k)] = True
                adversarial(L, buf, gen, adv, kind, li)
            del L, buf
            torch.cuda.empty_cache()
    log("\n==== fp8z exactness ====")
    log(f"direct: {t.cases} comparisons, {t.elements} output elements, {len(t.fails)} mismatches")
    log(f"adversarial/graph: {adv.cases} comparisons, {adv.elements} output elements, {len(adv.fails)} mismatches")
    log(f"rejected layers: {rejected}")
    tot_raw = sum(c[0] for c in cls.values())
    tot_read = sum(c[1] for c in cls.values())
    for kind, c in cls.items():
        log(f"{kind}: {c[2]} layers {c[0] / 1e9:.4f} GB -> {c[1] / 1e9:.4f} GB ({8 * c[1] / c[0]:.4f} bits/w), "
            f"saves {(c[0] - c[1]) / 1e9:.4f} GB = {(c[0] - c[1]) / 240e9 * 1e3:.3f} ms at 240 GB/s")
    if tot_raw:
        log(f"all: {tot_raw / 1e9:.4f} GB -> {tot_read / 1e9:.4f} GB ({8 * tot_read / tot_raw:.4f} bits/w), "
            f"saves {(tot_raw - tot_read) / 240e9 * 1e3:.3f} ms at 240 GB/s")
    ok = not t.fails and not adv.fails and not rejected
    log("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def adversarial(L, buf, gen, adv, kind, li):
    import fp8z
    tag = f"{kind} L{li}"
    # 1) allocator pool pre-filled with NaN / garbage bytes, then freed: the op's output comes from that memory
    for _ in range(3):
        junk = [torch.full((1, L.n), float("nan"), dtype=torch.bfloat16, device="cuda") for _ in range(8)]
        junk += [torch.randint(0, 255, (L.n * 2,), dtype=torch.uint8, device="cuda") for _ in range(8)]
        del junk
        a = L.quant(gen_x(1, L.k, 1, gen))
        adv.check(f"{tag} poisoned-pool", L.stock(a), fp8z.gemm(a, buf, L.sa, L.sb, L.n, L.k))
    # 2) CUDA graph: capture once, replay with changed inputs in the same static buffers
    a_st = torch.empty((1, L.k), dtype=torch.float8_e4m3fn, device="cuda")
    a_st.copy_(L.quant(gen_x(1, L.k, 0, gen)))
    out_st = torch.empty((1, L.n), dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fp8z.gemm(a_st, buf, L.sa, L.sb, L.n, L.k, out=out_st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fp8z.gemm(a_st, buf, L.sa, L.sb, L.n, L.k, out=out_st)
    for i in range(6):
        x = L.quant(gen_x(1, L.k, i % 6, gen)) if i < 5 else raw_fp8(1, L.k, gen)
        a_st.copy_(x)
        out_st.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        adv.check(f"{tag} graph-replay{i}", L.stock(a_st), out_st)
    del g


if __name__ == "__main__":
    main()
