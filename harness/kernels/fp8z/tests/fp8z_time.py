"""Timing of fp8z against the record's G2 tile and stock ops.cutlass_scaled_mm at one row (decode), on the real FP8 linears.

Every FP8 linear of the checkpoint is loaded (as vLLM merges/requantizes it) and encoded. For each shape, a CUDA
graph runs the GEMMs of every layer of that shape back to back (distinct weights, each larger than L2 except where
noted, so the stream is cold as in decode); the graph is replayed REPS times and the median replay is divided by
the number of layers. The same is done for the whole set in decode order (layer by layer). Outputs are checked
bitwise on the way. Needs a free box.

Run: bin/atom glim exec SHA harness/kernels/fp8z/tests/fp8z_time.py [--reps 20] [--rings 3,4,6]"""

from __future__ import annotations

import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8z_exact import Fp8Linear, Snap, bits_equal, fp8_linears, gen_x, log  # noqa: E402


def smi() -> str:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e!r}"


def graph_ms(calls, reps: int) -> float:
    """Median ms of one replay of a graph that issues every call once."""
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


def main():
    import fp8z
    import vllm._custom_ops as ops
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
    rings = [int(v) for v in sys.argv[sys.argv.index("--rings") + 1].split(",")] if "--rings" in sys.argv else [2, 3, 33, 4, 6, 32, 31, 21]
    log("nvidia-smi compute apps at start:", repr(smi()))
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    layers = []
    t0 = time.time()
    for kind, li, prefs in fp8_linears(snap):
        L = Fp8Linear(snap, prefs)
        res = fp8z.prepare(L.w, return_info=True)
        if res is None:
            log(f"{kind} L{li}: rejected {fp8z.stats['last_reason']}")
            continue
        L.buf, L.info = res
        L.kind, L.li = kind, li
        L.a = L.quant(gen_x(1, L.k, 0, gen))
        L.out_s = torch.empty((1, L.n), dtype=torch.bfloat16, device="cuda")
        L.out_z = torch.empty((1, L.n), dtype=torch.bfloat16, device="cuda")
        del L.w  # B (= W^T view) keeps the storage alive
        layers.append(L)
    log(f"loaded and encoded {len(layers)} FP8 linears in {time.time() - t0:.1f}s; "
        f"{sum(L.n * L.k for L in layers) / 1e9:.3f} GB -> {sum(L.info['read_bytes'] for L in layers) / 1e9:.3f} GB")

    from vllm.model_executor.layers.quantization.utils import glimmer_fp8 as g8
    ext = g8._ext
    log("record G2 tile available:", ext is not None, g8._CFG)
    sa1 = {id(L): L.sa.reshape(1) for L in layers}
    sb1 = {id(L): L.sb.reshape(1) for L in layers}

    def cutlass(L):
        return lambda: torch.ops._C.cutlass_scaled_mm(L.out_s, L.a, L.b, L.sa, L.sb, None)

    def stock(L):
        # the record's decode path for these shapes (glimmer_fp8 G2 tile), else CUTLASS
        cfg = g8._CFG.get((L.n, L.k))
        if ext is None or cfg is None:
            return cutlass(L)
        return lambda: ext.gemm(L.a, L.b, sa1[id(L)], sb1[id(L)], cfg)

    def comp(L, ring):
        return lambda: fp8z.gemm(L.a, L.buf, L.sa, L.sb, L.n, L.k, ring=ring, out=L.out_z)

    # bitwise check of the timed calls
    bad = 0
    for L in layers:
        ref = ops.cutlass_scaled_mm(L.a, L.b, out_dtype=torch.bfloat16, scale_a=L.sa, scale_b=L.sb)
        for r in rings:
            bad += not bits_equal(ref, fp8z.gemm(L.a, L.buf, L.sa, L.sb, L.n, L.k, ring=r))
    log(f"bitwise check of the timed calls: {bad} mismatches")
    by = collections.OrderedDict()
    for L in layers:
        by.setdefault((L.kind, L.n, L.k), []).append(L)
    log(f"\n{'shape':28s} {'layers':>6s} {'MB/layer':>9s} {'G2 ms':>9s} {'GB/s':>6s} " +
        " ".join(f"{'ring' + str(r) + ' ms':>10s} {'eqGB/s':>7s} {'gain':>6s}" for r in rings))
    tot_s = 0.0
    tot_z = {r: 0.0 for r in rings}
    for (kind, n, k), lst in by.items():
        ms_c = graph_ms([cutlass(L) for L in lst], reps) / len(lst)
        ms_s = graph_ms([stock(L) for L in lst], reps) / len(lst)
        log(f"   {kind}: CUTLASS {ms_c:.4f} ms/call ({n * k / ms_c / 1e6:.1f} GB/s), record G2 {ms_s:.4f} ms/call")
        row = f"{kind + f' {n}x{k}':28s} {len(lst):6d} {n * k / 1e6:9.1f} {ms_s:9.4f} {n * k / ms_s / 1e6:6.1f} "
        tot_s += ms_s * len(lst)
        for r in rings:
            ms_z = graph_ms([comp(L, r) for L in lst], reps) / len(lst)
            tot_z[r] += ms_z * len(lst)
            row += f"{ms_z:10.4f} {n * k / ms_z / 1e6:7.1f} {100 * (ms_s / ms_z - 1):+5.1f}% "
        log(row)
    log(f"sum over shapes: stock {tot_s:.3f} ms; " + ", ".join(f"ring{r} {tot_z[r]:.3f} ms" for r in rings))
    # whole set in decode order (layer, then qkv, gate, o_proj, gate_up, down), one graph each
    order = {"qkv": 0, "attn_gate": 1, "o_proj": 2, "gate_up": 3, "down": 4}
    seq = sorted(layers, key=lambda L: (L.li, order[L.kind]))
    ms_c = graph_ms([cutlass(L) for L in seq], reps)
    ms_s = graph_ms([stock(L) for L in seq], reps)
    log(f"decode-order sequence of {len(seq)} GEMMs: CUTLASS {ms_c:.3f} ms, record G2 {ms_s:.3f} ms")
    for r in rings:
        ms_z = graph_ms([comp(L, r) for L in seq], reps)
        log(f"   ring{r}: {ms_z:.3f} ms ({ms_s - ms_z:+.3f} ms, {100 * (ms_s / ms_z - 1):+.1f}%)")
    log("nvidia-smi compute apps at end:", repr(smi()))


if __name__ == "__main__":
    main()
