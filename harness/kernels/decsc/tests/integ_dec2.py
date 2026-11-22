"""Integration check of the dec2 engine hooks without a model server, on real layers: the record's FP8 op
(glimmer::fp8_scaled_mm, now launching fp8z_gemm at one row) and the Marlin NVFP4 kernel class (one row now launching
the decsc 5/6-bit GEMV), eager, under torch.compile with a dynamic row count, and inside a CUDA graph whose capture
runs the l2pf prefetch session (record on an eager pass, apply during capture) - against the stock ops bit for bit.

Run: bin/atom glim exec SHA harness/kernels/decsc/tests/integ_dec2.py"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "fp8z", "tests"))
import decsc_exact as de  # noqa: E402
import fp8z_exact as fe  # noqa: E402

fails = []


def check(tag, ref, got):
    ok = de.bits_equal(ref, got)
    de.log(f"  {tag}: {'ok' if ok else 'MISMATCH'}")
    if not ok:
        fails.append(tag)


def capture(fn, x_static):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x_static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(x_static)
    return g, out


def main():
    import decsc
    from muse_marlin_gemv import prepare_decode_layout
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as mk
    from vllm.model_executor.kernels.linear.scaled_mm import fp8z as fzmod
    from vllm.model_executor.layers.quantization.utils import glimmer_fp8  # noqa: F401 (registers the op)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(3)
    holder = torch.nn.Module()
    # ---------------------------------------------------------------- FP8
    fsnap = fe.Snap(os.environ["GLIM_SNAPSHOT"])
    kind, li, prefs = [x for x in fe.fp8_linears(fsnap) if x[0] == "o_proj"][0]
    L = fe.Fp8Linear(fsnap, prefs)
    comp = fzmod.prepare_weight(L.b)
    assert comp is not None, "fp8z rejected the layer"
    holder.register_buffer("fp8_b", L.w, persistent=False)
    holder.register_buffer("fp8_comp", comp, persistent=False)
    de.log(f"FP8 {kind} L{li} {L.n}x{L.k}")

    def f(a):
        return torch.ops.glimmer.fp8_scaled_mm(a, L.b, L.sa, L.sb)

    for m in (1, 2, 16, 64):
        a = L.quant(fe.gen_x(m, L.k, 1, gen))
        check(f"fp8 eager M={m}", L.stock(a), f(a))
    cf = torch.compile(f, dynamic=True)
    for m in (1, 3, 1):
        a = L.quant(fe.gen_x(m, L.k, 0, gen))
        check(f"fp8 compiled M={m}", L.stock(a), cf(a))
    # ---------------------------------------------------------------- NVFP4
    nsnap = de.Snap(os.environ["GLIM_SNAPSHOT"])
    layers = {}
    for k2, li2, prefs2 in de.nvfp4_linears(nsnap):
        if k2 in ("gate_up", "o_proj") and k2 not in layers:
            layer = de.load_merged(nsnap, prefs2)
            mk._prepare_prompt_gemm(layer)
            prepare_decode_layout(layer)
            mk._prepare_sc6(layer)
            st = decsc.STATES.get(layer._marlin_decode_scales.data_ptr())
            de.log(f"NVFP4 {k2} L{li2}: state {None if st is None else st[0]}-bit")
            if st is None:
                fails.append(f"{k2} not admitted")
            layers[k2] = layer
            holder.add_module(k2, layer)
    apply = mk.MarlinNvFp4LinearKernel.apply_weights
    for k2, layer in layers.items():
        kk = layer.input_size_per_partition
        for m in (1, 2, 512):
            x = de.gen_x(m, kk, 5, gen)
            got = apply(None, layer, x)
            if m < 128:
                ref = de.stock(layer, x)
            else:
                keep = decsc.STATES.pop(layer._marlin_decode_scales.data_ptr())
                ref = apply(None, layer, x)
                decsc.STATES[layer._marlin_decode_scales.data_ptr()] = keep
            check(f"{k2} eager M={m}", ref, got)
        ch = torch.compile(lambda x, layer=layer: apply(None, layer, x), dynamic=True)
        for m in (1, 4, 1):
            x = de.gen_x(m, kk, 0, gen)
            check(f"{k2} compiled M={m}", de.stock(layer, x), ch(x))
    # ---------------------------------------------------------------- one graph, l2pf session
    from l2pf.sites import Session, weight_ranges
    gu, op = layers["gate_up"], layers["o_proj"]
    a_st = L.quant(fe.gen_x(1, L.k, 0, gen))
    xg_st = de.gen_x(1, gu.input_size_per_partition, 0, gen)
    xo_st = de.gen_x(1, op.input_size_per_partition, 0, gen)

    def step(_):
        return f(a_st), apply(None, gu, xg_st), apply(None, op, xo_st)

    sess = Session(weight_ranges(holder))
    with sess.record():
        step(None)
    kinds = [s.kind for s in sess.sites]
    de.log(f"l2pf recorded sites: {kinds}, with prefetch ranges: {[bool(s.descs) for s in sess.sites]}")
    if not {"fp8z", "marlin_tiled_sc6"} <= set(kinds):
        fails.append(f"l2pf sites {kinds}")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step(None)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with sess.apply():
            outs = step(None)
    de.log(f"l2pf session: {sess.summary()}")
    if sess.stopped:
        fails.append("l2pf session stopped")
    for i in range(3):
        a_st.copy_(L.quant(fe.gen_x(1, L.k, i, gen)))
        xg_st.copy_(de.gen_x(1, gu.input_size_per_partition, i, gen))
        xo_st.copy_(de.gen_x(1, op.input_size_per_partition, i, gen))
        g.replay()
        torch.cuda.synchronize()
        check(f"graph replay {i} fp8", L.stock(a_st), outs[0])
        check(f"graph replay {i} gate_up", de.stock(gu, xg_st), outs[1])
        check(f"graph replay {i} o_proj", de.stock(op, xo_st), outs[2])
    de.log("RESULT:", "PASS" if not fails else f"FAIL {fails}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
