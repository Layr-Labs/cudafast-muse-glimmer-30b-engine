"""Timing of decsc (6-bit scale codes) against the record's one-row tile-major GEMV (muse_marlin_gemv.decode_linear)
on the real NVFP4 linears (lm_head included). For each shape a CUDA graph runs the GEMVs of every admitted layer of that
shape back to back (distinct weights, cold as in decode); the median replay is divided by the number of layers; then the
whole set in decode order. Outputs are checked bitwise on the way. The Marlin-layout copy of each weight is freed after
the check, so only the tile-major tensors stay resident. Needs a free box.

Run: bin/atom glim exec SHA harness/kernels/decsc/tests/decsc_time.py [--reps 20]"""

from __future__ import annotations

import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decsc_exact import Snap, bits_equal, gen_x, load_merged, log, nvfp4_linears  # noqa: E402


def smi() -> str:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e!r}"


def graph_ms(calls, reps: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    del g
    return ts[len(ts) // 2]


class Item:
    pass


def main():
    import decsc
    from muse_marlin_gemv import decode_linear, prepare_decode_layout
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
    log("nvidia-smi compute apps at start:", repr(smi()))
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    items = []
    t0 = time.time()
    bad = 0
    for kind, li, prefs in nvfp4_linears(snap):
        L = load_merged(snap, prefs)
        prepare_decode_layout(L)
        if getattr(L, "_marlin_decode_weight", None) is None:
            continue
        res = decsc.prepare(L)
        if res is None:
            log(f"{kind} L{li}: rejected {decsc.stats['last_reason']}")
            continue
        it = Item()
        it.kind, it.li = kind, li
        it.n, it.k = L.output_size_per_partition, L.input_size_per_partition
        it.tw, it.ts, it.gs, it.ws = L._marlin_decode_weight, L._marlin_decode_scales, L.weight_global_scale, L.workspace
        it.codes, it.tab = decsc.encode(L._marlin_decode_scales)
        it.s5 = decsc.prepare5(L)
        it.x = gen_x(1, it.k, 0, gen)
        ref = decode_linear(it.x, it.tw, it.ts, it.gs, it.ws, it.n, it.k)
        bad += not bits_equal(ref, decsc.decode_linear_sc6(it.x, it.tw, it.codes, it.tab, it.gs, it.ws, it.n, it.k))
        if it.s5 is not None:
            bad += not bits_equal(ref, decsc.decode_linear_sc5(it.x, it.tw, it.s5[0], it.s5[1], it.gs, it.ws, it.n,
                                                               it.k, it.s5[2]))
        del L
        torch.cuda.empty_cache()
        items.append(it)
    log(f"loaded {len(items)} NVFP4 linears in {time.time() - t0:.1f}s; bitwise check: {bad} mismatches")

    def rec(it):
        return lambda: torch.ops.vllm.marlin_tiled_decode.default(it.x, it.tw, it.ts, it.gs, it.ws, it.n, it.k)

    def sc6(it):
        return lambda: torch.ops.vllm.marlin_tiled_decode_sc6.default(it.x, it.tw, it.codes, it.tab, it.gs, it.ws,
                                                                      it.n, it.k)

    def sc5(it):
        if it.s5 is None:
            return sc6(it)
        return lambda: torch.ops.vllm.marlin_tiled_decode_sc5.default(it.x, it.tw, it.s5[0], it.s5[1], it.gs, it.ws,
                                                                      it.n, it.k, it.s5[2])

    by = collections.OrderedDict()
    for it in items:
        by.setdefault((it.kind, it.n, it.k), []).append(it)
    log(f"\n{'shape':28s} {'layers':>6s} {'5b':>3s} {'record ms':>10s} {'GB/s':>6s} {'sc6 ms':>9s} {'gain':>7s} "
        f"{'ms/token':>9s} {'sc5 ms':>9s} {'gain':>7s} {'ms/token':>9s}")
    tot = [0.0, 0.0, 0.0]
    for (kind, n, k), lst in by.items():
        b = n * k // 2 + n * k // 16
        ms = [graph_ms([f(it) for it in lst], reps) / len(lst) for f in (rec, sc6, sc5)]
        for i in range(3):
            tot[i] += ms[i] * len(lst)
        n5 = sum(it.s5 is not None for it in lst)
        log(f"{kind + f' {n}x{k}':28s} {len(lst):6d} {n5:3d} {ms[0]:10.4f} {b / ms[0] / 1e6:6.1f} {ms[1]:9.4f} "
            f"{100 * (ms[0] / ms[1] - 1):+6.2f}% {(ms[0] - ms[1]) * len(lst):+9.4f} {ms[2]:9.4f} "
            f"{100 * (ms[0] / ms[2] - 1):+6.2f}% {(ms[0] - ms[2]) * len(lst):+9.4f}")
    log(f"sum over shapes: record {tot[0]:.3f} ms, sc6 {tot[1]:.3f} ms ({tot[0] - tot[1]:+.3f}), "
        f"sc5 {tot[2]:.3f} ms ({tot[0] - tot[2]:+.3f}) per token")
    order = {"qkv": 0, "attn_gate": 1, "o_proj": 2, "gate_up": 3, "down": 4, "lm_head": 5}
    seq = sorted(items, key=lambda it: (it.li if it.li >= 0 else 999, order[it.kind]))
    ms = [graph_ms([f(it) for it in seq], reps) for f in (rec, sc6, sc5)]
    log(f"decode-order sequence of {len(seq)} GEMVs: record {ms[0]:.3f} ms, sc6 {ms[1]:.3f} ms "
        f"({ms[0] - ms[1]:+.3f}), sc5 {ms[2]:.3f} ms ({ms[0] - ms[2]:+.3f})")
    log("nvidia-smi compute apps at end:", repr(smi()))


if __name__ == "__main__":
    main()
