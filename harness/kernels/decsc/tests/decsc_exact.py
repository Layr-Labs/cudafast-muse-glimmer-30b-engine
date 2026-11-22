"""Bit-exactness of decsc (the record's one-row tile-major Marlin NVFP4 GEMV reading 6-bit block-scale codes)
against the record's decode GEMV (muse_marlin_gemv.decode_linear) and stock ops.marlin_gemm, on every NVFP4 linear
of the checkpoint (merged and prepared as vLLM's Marlin NVFP4 path and prepare_decode_layout do), lm_head included.
Per layer: prepare() (codes decode to every scale byte + kernel check), >= 7 activation kinds, NaN-filled outputs,
repeats, and the production path (MarlinNvFp4LinearKernel.apply_weights at one row). Adversarial pass (poisoned
allocator, CUDA graph replay with changed inputs) on the first layer of every shape. No timing.

Run: bin/atom glim exec SHA harness/kernels/decsc/tests/decsc_exact.py [--layers all|N]"""

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
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
from vllm.scalar_type import scalar_types


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


def nvfp4_linears(snap: Snap):
    pat = re.compile(r"^(.*language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+))\.weight$")
    per = collections.defaultdict(dict)
    for name in sorted(snap.wmap):
        m = pat.match(name)
        if m and snap.meta(name)[0] == "U8" and (m.group(1) + ".weight_scale_2") in snap.wmap:
            per[int(m.group(2))][m.group(3) + "." + m.group(4)] = m.group(1)
    out = []
    for li in sorted(per):
        for kind, ps in GROUPS:
            if all(p in per[li] for p in ps):
                out.append((kind, li, [per[li][p] for p in ps]))
    lm = [n[: -len(".weight")] for n in snap.wmap if re.match(r"^(.*\.)?lm_head\.weight$", n)]
    if lm and snap.meta(lm[0] + ".weight")[0] == "U8":
        out.append(("lm_head", -1, lm[:1]))
    return out


def load_merged(snap: Snap, prefixes):
    ws, ss, s2 = [], [], []
    for p in prefixes:
        ws.append(snap.get(p + ".weight"))
        ss.append(snap.get(p + ".weight_scale"))
        s2.append(snap.get(p + ".weight_scale_2").reshape(-1))
    layer = torch.nn.Module()
    w = torch.cat(ws, 0)
    n, k2 = w.shape
    layer.output_size_per_partition = n
    layer.input_size_per_partition = 2 * k2
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.cat(ss, 0).cuda(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.cat(s2).cuda().max().to(torch.float32),
                                                   requires_grad=False)
    prepare_fp4_layer_for_marlin(layer)
    return layer


def stock(layer, x):
    return ops.marlin_gemm(
        a=x, c=None, b_q_weight=layer.weight, b_bias=None, b_scales=layer.weight_scale,
        a_scales=None, global_scale=layer.weight_global_scale, b_zeros=None, g_idx=None,
        perm=None, workspace=layer.workspace, b_q_type=scalar_types.float4_e2m1f,
        size_m=x.size(0), size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition, use_atomic_add=False, use_fp32_reduce=True)


def gen_x(m: int, k: int, kind: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float32)
    if kind == 1:
        x = x * 4.0
        idx = torch.randint(0, k, (m, 16), generator=gen, device="cuda")
        x.scatter_(1, idx, torch.randn(m, 16, generator=gen, device="cuda") * 300.0)
        x = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.05, torch.zeros_like(x), x)
        x = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.02, torch.full_like(x, -0.0), x)
    elif kind == 2:
        x = x * 0.05
    elif kind == 3:
        x = x * torch.exp2(torch.randint(-6, 7, (k,), generator=gen, device="cuda").float())
    elif kind == 4:
        x = torch.where(x > 0, torch.zeros_like(x), torch.full_like(x, -0.0))
    elif kind == 5:
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2][:, : x[:, 1::2].shape[1]]
    elif kind == 6:
        b = torch.randint(-32768, 32768, (m, k), generator=gen, device="cuda", dtype=torch.int32).to(torch.int16)
        b = torch.where((b & 0x7F80) == 0x7F80, b & ~0x4000, b)
        return b.view(torch.bfloat16)
    return x.to(torch.bfloat16)


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


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
            self.fails.append(tag)
            log(f"   MISMATCH {tag}: {int(bad.sum())} elements, first {bad.nonzero()[:4].tolist()}")
            return False
        return True


def main():
    import decsc
    from muse_marlin_gemv import decode_linear, prepare_decode_layout
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as mk
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    nl = sys.argv[sys.argv.index("--layers") + 1] if "--layers" in sys.argv else "all"
    by = collections.OrderedDict()
    for kind, li, prefs in nvfp4_linears(snap):
        by.setdefault(kind, []).append((li, prefs))
    log("NVFP4 linears:", {k: len(v) for k, v in by.items()})
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261122)
    t, t5, rec, prod, adv = Tally(), Tally(), Tally(), Tally(), Tally()
    saved5 = collections.defaultdict(lambda: [0, 0, 0])
    rejected, untiled = [], []
    saved = collections.defaultdict(lambda: [0, 0, 0])
    seen = set()
    t0 = time.time()
    for kind, lst in by.items():
        picks = lst if nl == "all" else lst[: int(nl)]
        for li, prefs in picks:
            L = load_merged(snap, prefs)
            prepare_decode_layout(L)
            n, k = L.output_size_per_partition, L.input_size_per_partition
            tw = getattr(L, "_marlin_decode_weight", None)
            ts = getattr(L, "_marlin_decode_scales", None)
            if tw is None:
                untiled.append((kind, li))
                log(f"[{kind} L{li} {n}x{k}] no tile-major decode layout (record keeps ops.marlin_gemm)")
                continue
            res = decsc.prepare(L)
            if res is None:
                rejected.append((kind, li, decsc.stats["last_reason"]))
                log(f"[{kind} L{li} {n}x{k}] REJECTED: {decsc.stats['last_reason']}")
                del L
                torch.cuda.empty_cache()
                continue
            codes, tab = decsc.encode(ts)
            s5 = decsc.prepare5(L)
            sv = saved[kind]
            sv[0] += ts.numel()
            sv[1] += codes.numel()
            sv[2] += 1
            sv5 = saved5[kind]
            sv5[0] += ts.numel()
            sv5[1] += (s5[0] if s5 is not None else codes).numel()
            sv5[2] += s5 is not None
            n0 = len(t.fails) + len(t5.fails) + len(rec.fails) + len(prod.fails)
            gs, ws = L.weight_global_scale, L.workspace
            for kd in range(7):
                x = gen_x(1, k, kd, gen)
                ref = stock(L, x)
                theirs = decode_linear(x, tw, ts, gs, ws, n, k)
                rec.check(f"{kind} L{li} kind{kd} record GEMV vs stock", ref, theirs)
                for rep in range(2):
                    got = decsc.decode_linear_sc6(x, tw, codes, tab, gs, ws, n, k)
                    t.check(f"{kind} L{li} kind{kd} sc6 rep{rep}", theirs, got)
                    if s5 is not None:
                        got = decsc.decode_linear_sc5(x, tw, s5[0], s5[1], gs, ws, n, k, s5[2])
                        t5.check(f"{kind} L{li} kind{kd} sc5 rep{rep}", theirs, got)
                # production: the kernel class at one row (routes through _marlin_fp4_single -> sc6)
                if kd < 3:
                    prod.check(f"{kind} L{li} kind{kd} apply_weights", theirs,
                               mk.MarlinNvFp4LinearKernel.apply_weights(None, L, x))
            log(f"[{time.time() - t0:6.1f}s] [{kind} L{li} N={n} K={k}] state {res[0]}-bit, 5-bit "
                f"{'esc/stage ' + str(s5[2]) if s5 is not None else 'not admitted'}, tile {ts.shape[2]}, "
                f"distinct {torch.unique(ts.view(torch.uint8)).numel()} | "
                f"{len(t.fails) + len(t5.fails) + len(rec.fails) + len(prod.fails) - n0} mismatches")
            if (n, k) not in seen:
                seen.add((n, k))
                adversarial(L, tw, ts, codes, tab, gen, adv, f"{kind} L{li}", s5)
            decsc.STATES.pop(ts.data_ptr(), None)
            del L, codes, tab
            torch.cuda.empty_cache()
    log("\n==== decsc exactness ====")
    log(f"decsc 6-bit vs record GEMV: {t.cases} comparisons, {t.elements} output elements, {len(t.fails)} mismatches")
    log(f"decsc 5-bit vs record GEMV: {t5.cases} comparisons, {t5.elements} output elements, {len(t5.fails)} "
        f"mismatches")
    log(f"record GEMV vs stock: {rec.cases} comparisons, {len(rec.fails)} mismatches")
    log(f"production apply_weights: {prod.cases} comparisons, {len(prod.fails)} mismatches")
    log(f"adversarial/graph: {adv.cases} comparisons, {adv.elements} output elements, {len(adv.fails)} mismatches")
    log(f"rejected: {rejected}; without tile-major layout: {untiled}")
    tot_s = sum(v[0] for v in saved.values())
    tot_c = sum(v[1] for v in saved.values())
    for kind, v in saved.items():
        log(f"{kind}: {v[2]} layers, scales {v[0] / 1e9:.4f} GB -> {v[1] / 1e9:.4f} GB, saves "
            f"{(v[0] - v[1]) / 240e9 * 1e3:.3f} ms at 240 GB/s")
    log(f"all 6-bit: saves {(tot_s - tot_c) / 1e9:.4f} GB = {(tot_s - tot_c) / 240e9 * 1e3:.3f} ms at 240 GB/s")
    for kind, v in saved5.items():
        log(f"{kind} 5-bit where admitted ({v[2]} layers), else 6-bit: scales {v[0] / 1e9:.4f} GB -> "
            f"{v[1] / 1e9:.4f} GB, saves {(v[0] - v[1]) / 240e9 * 1e3:.3f} ms")
    t5s = sum(v[0] for v in saved5.values())
    t5c = sum(v[1] for v in saved5.values())
    log(f"all 5/6-bit: saves {(t5s - t5c) / 1e9:.4f} GB = {(t5s - t5c) / 240e9 * 1e3:.3f} ms at 240 GB/s")
    ok = not t.fails and not t5.fails and not rec.fails and not prod.fails and not adv.fails
    log("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def adversarial(L, tw, ts, codes, tab, gen, adv, tag, s5=None):
    import decsc
    from muse_marlin_gemv import decode_linear
    n, k = L.output_size_per_partition, L.input_size_per_partition
    gs, ws = L.weight_global_scale, L.workspace
    for i in range(3):
        junk = [torch.full((1, n), float("nan"), dtype=torch.bfloat16, device="cuda") for _ in range(8)]
        junk += [torch.full((48, 16, 128), float("nan"), dtype=torch.float32, device="cuda") for _ in range(4)]
        del junk
        x = gen_x(1, k, i, gen)
        ref = decode_linear(x, tw, ts, gs, ws, n, k)
        adv.check(f"{tag} poisoned-pool{i}", ref, decsc.decode_linear_sc6(x, tw, codes, tab, gs, ws, n, k))
        if s5 is not None:
            adv.check(f"{tag} sc5 poisoned-pool{i}", ref,
                      decsc.decode_linear_sc5(x, tw, s5[0], s5[1], gs, ws, n, k, s5[2]))
    x_st = gen_x(1, k, 0, gen)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.ops.vllm.marlin_tiled_decode_sc6.default(x_st, tw, codes, tab, gs, ws, n, k)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_st = torch.ops.vllm.marlin_tiled_decode_sc6.default(x_st, tw, codes, tab, gs, ws, n, k)
    g5 = None
    if s5 is not None:
        g5 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g5):
            o5 = torch.ops.vllm.marlin_tiled_decode_sc5.default(x_st, tw, s5[0], s5[1], gs, ws, n, k, s5[2])
    for i in range(7):
        x_st.copy_(gen_x(1, k, i, gen))
        o_st.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        ref = decode_linear(x_st, tw, ts, gs, ws, n, k)
        adv.check(f"{tag} graph-replay{i}", ref, o_st)
        if g5 is not None:
            o5.fill_(float("nan"))
            g5.replay()
            torch.cuda.synchronize()
            adv.check(f"{tag} sc5 graph-replay{i}", ref, o5)
    del g, g5


if __name__ == "__main__":
    main()
