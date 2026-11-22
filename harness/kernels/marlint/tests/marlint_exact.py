#!/usr/bin/env python3
"""Bitwise test of the tile-major M <= 8 Marlin NVFP4 path against ops.marlin_gemm.

Runs in the pinned vLLM image with the checkout's vLLM overlay and built kernels
on PYTHONPATH. Every NVFP4 projection of the checkpoint and lm_head (weights read
from the snapshot and prepared by MarlinNvFp4LinearKernel.process_weights_after_loading,
which also builds the tile-major copy and runs its load-time check) is called
through the dispatch the engine uses (marlin._marlin_fp4_any) and through
apply_fp4_marlin_linear, and the bf16 outputs are compared as int16.

  --layers all|first   every NVFP4 layer, or the first of each shape
  --m 1,...,8          row counts for every selected layer
  --m-extra ...        row counts for the first layer of each shape (other paths)
  --inputs ...         input families (see make_inputs)
  --acts DIR           also every captured activation file DIR/<proj>.<layer>.pt
  --bench              time both paths per shape at M=1 (CUDA graphs)

The "teeth" line per shape is the fraction of outputs of the stock call at M=1
that differ from the same row inside a 512-row stock call (another summation
tree): a test that cannot see that could not see a wrong kernel either.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import torch

DT = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32,
      "BF16": torch.bfloat16}
PFX = "model.language_model.layers"
PROJS = {
    "qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")),
    "attn_gate": ("self_attn", ("gate_proj",)),
    "o_proj": ("self_attn", ("o_proj",)),
    "gate_up": ("mlp", ("gate_proj", "up_proj")),
    "down": ("mlp", ("down_proj",)),
}


class Snapshot:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self.hdr = {}

    def _header(self, fn):
        if fn not in self.hdr:
            with open(os.path.join(self.root, fn), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
        return self.hdr[fn]

    def has(self, name):
        return name in self.wmap

    def meta(self, name):
        return self._header(self.wmap[name])[0][name]

    def get(self, name):
        fn = self.wmap[name]
        h, base = self._header(fn)
        m = h[name]
        a, b = m["data_offsets"]
        buf = bytearray(b - a)
        with open(os.path.join(self.root, fn), "rb", buffering=0) as f:
            mv, got = memoryview(buf), 0
            while got < len(buf):
                k = os.preadv(f.fileno(), [mv[got:]], base + a + got)
                if k <= 0:
                    raise IOError(name)
                got += k
        t = torch.frombuffer(buf, dtype=torch.uint8).view(DT[m["dtype"]])
        return t.reshape(m["shape"]) if m["shape"] else t.reshape(())


def nvfp4_layers(snap, n_layers):
    out = []
    for L in range(n_layers):
        for proj, (blk, parts) in PROJS.items():
            name = f"{PFX}.{L}.{blk}.{parts[0]}.weight"
            if snap.meta(name)["dtype"] == "U8":
                out.append((proj, L))
    return out


def load(snap, proj, L):
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    if proj == "lm_head":
        pre = ["lm_head"]
    else:
        blk, parts = PROJS[proj]
        pre = [f"{PFX}.{L}.{blk}.{p}" for p in parts]
    w = torch.cat([snap.get(p + ".weight") for p in pre], 0)
    s = torch.cat([snap.get(p + ".weight_scale") for p in pre], 0)
    s2 = torch.stack([snap.get(p + ".weight_scale_2").reshape(()) for p in pre])
    layer = torch.nn.Module()
    n, k2 = w.shape
    layer.output_size_per_partition = n
    layer.input_size_per_partition = 2 * k2
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s.cuda(), requires_grad=False)
    # ModelOptNvFp4W4A16LinearMethod.process_weights_after_loading
    layer.weight_global_scale = torch.nn.Parameter(s2.cuda().max().to(torch.float32),
                                                   requires_grad=False)
    MarlinNvFp4LinearKernel.__new__(MarlinNvFp4LinearKernel).process_weights_after_loading(layer)
    return layer


def stock(layer, x):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )
    return apply_fp4_marlin_linear(
        input=x, weight=layer.weight, weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,
        size_n=layer.output_size_per_partition, size_k=layer.input_size_per_partition)


def engine(layer, x):
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as hook
    tm = layer.marlin_tile_major
    path = layer.marlin_prompt_path
    mode, sl, ss, sms = 0, None, None, 0
    if path is not None and path[0] == "wide":
        mode = 1
    elif path is not None:
        mode, sl, ss, sms = 2, path[1], path[2], path[3]
    return hook._marlin_fp4_any(
        x, layer.weight, layer.weight_scale, layer.weight_global_scale, layer.workspace,
        layer.output_size_per_partition, layer.input_size_per_partition,
        tm[0] if tm else None, tm[1] if tm else None, mode, sl, ss, sms)


def make_inputs(kind, m, k, gen):
    def rn(*s):
        return torch.randn(*s, generator=gen, device="cuda")
    if kind == "randn":
        x = rn(m, k)
    elif kind == "cancel":
        # large equal-and-opposite pairs plus small terms: the low bits of every
        # dot product depend on the order in which partial sums meet
        base = rn(m, k // 2) * 512
        x = torch.stack([base, -base], -1).reshape(m, k) + rn(m, k)
        x = x[:, torch.randperm(k, generator=gen, device="cuda")]
    elif kind == "small":
        x = rn(m, k) * 1e-30
    elif kind == "heavy":
        x = rn(m, k)
        x = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.01, x * 3000, x)
    elif kind == "chan":
        x = rn(m, k) * torch.exp(rn(k) * 2)
    elif kind == "sparse":
        x = torch.zeros(m, k, device="cuda")
        x.scatter_(1, torch.randint(0, k, (m, 32), generator=gen, device="cuda"), rn(m, 32) * 30)
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def neq(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def graph_us(fn, reps=9):
    g, _ = capture(fn)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    del g
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("KBENCH_SNAPSHOT"))
    ap.add_argument("--out", default=os.environ.get("KBENCH_OUT", "/out"))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--m", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--m-extra", default="9,16,64,512")
    ap.add_argument("--inputs", default="randn,cancel,small,heavy,chan,sparse")
    ap.add_argument("--acts", default="")
    ap.add_argument("--no-lm-head", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-layers", type=int, default=4)
    a = ap.parse_args()
    import marlint
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as hook
    assert hook.marlint is marlint and marlint.available(), (marlint.LIB_PATH, marlint._load_error)
    snap = Snapshot(a.snapshot)
    with open(os.path.join(a.snapshot, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    todo = nvfp4_layers(snap, n_layers)
    if (not a.no_lm_head and snap.has("lm_head.weight")
            and snap.meta("lm_head.weight")["dtype"] == "U8"):
        todo.append(("lm_head", -1))
    firsts = {}
    for proj, L in todo:
        firsts.setdefault(proj, L)
    if a.layers == "first":
        todo = [(p, L) for p, L in todo if firsts[p] == L]
    ms = [int(v) for v in a.m.split(",") if v]
    extra = [int(v) for v in a.m_extra.split(",") if v]
    kinds = [v for v in a.inputs.split(",") if v]
    gen = torch.Generator(device="cuda")
    res = {"cases": 0, "mismatch_cases": 0, "mismatch_elems": 0, "fails": [], "teeth": {},
           "admitted": 0, "layers": len(todo), "graph_cases": 0, "bench": {}, "copy_bytes": 0}
    t0 = time.time()
    for proj, L in todo:
        layer = load(snap, proj, L)
        tm = layer.marlin_tile_major
        res["admitted"] += int(tm is not None)
        if tm is not None:
            res["copy_bytes"] += tm[0].numel() * 4 + tm[1].numel()
        else:
            res["fails"].append((proj, L, 0, "-", "not admitted"))
        n, k = layer.output_size_per_partition, layer.input_size_per_partition
        cases = [(m, kind) for m in ms for kind in kinds]
        if firsts[proj] == L:
            cases += [(m, "randn") for m in extra]
        if a.acts:
            path = os.path.join(a.acts, f"{proj}.{L}.pt")
            if os.path.exists(path):
                cases.append((None, path))
        for (m, kind) in cases:
            gen.manual_seed(zlib.crc32(f"{proj}.{L}.{m}.{kind}".encode()))
            if m is None:
                x = torch.load(kind, map_location="cuda").to(torch.bfloat16)
                x = x.reshape(-1, x.shape[-1])[:8].contiguous()
                m = x.shape[0]
                kind = "captured"
            else:
                x = make_inputs(kind, m, k, gen)
            ref = stock(layer, x)
            got = engine(layer, x)
            res["cases"] += 1
            d = neq(got, ref)
            if d:
                res["mismatch_cases"] += 1
                res["mismatch_elems"] += d
                res["fails"].append((proj, L, m, kind, d))
                print(f"MISMATCH {proj} L{L} M={m} {kind}: {d} of {ref.numel()}", flush=True)
        # M = 1 and M = 8 captured in a CUDA graph, replayed on new inputs
        for m in (1, 8):
            xin = make_inputs("randn", m, k, gen)
            g, out = capture(lambda: engine(layer, xin))
            for _ in range(2):
                xin.copy_(make_inputs("heavy", m, k, gen))
                g.replay()
                torch.cuda.synchronize()
                d = neq(out, stock(layer, xin))
                res["graph_cases"] += 1
                if d:
                    res["mismatch_cases"] += 1
                    res["fails"].append((proj, L, m, "graph", d))
                    print(f"MISMATCH graph {proj} L{L} M={m}: {d}", flush=True)
            del g, out
        if firsts[proj] == L:
            x = make_inputs("randn", 512, k, gen)
            res["teeth"][proj] = neq(stock(layer, x[:1].contiguous()), stock(layer, x)[:1]) / n
        print(f"{proj:9s} L{L:2d} N={n} K={k} admitted={tm is not None} cases so far "
              f"{res['cases']} mismatches {res['mismatch_cases']} ({time.time()-t0:.0f}s)",
              flush=True)
        del layer
        torch.cuda.empty_cache()
    print("teeth (fraction of M=1 outputs that the 512-row summation tree changes):",
          json.dumps(res["teeth"]), flush=True)

    if a.bench:
        allp = nvfp4_layers(snap, n_layers)
        for proj in firsts:
            if proj == "lm_head":
                Ls = [-1]
            else:
                Ls = [L for p, L in allp if p == proj][-a.bench_layers:]
            layers = [load(snap, proj, L) for L in Ls]
            k = layers[0].input_size_per_partition
            xs = [make_inputs("randn", 1, k, gen) for _ in layers]
            us_s = graph_us(lambda: [stock(l, x) for l, x in zip(layers, xs)])
            us_t = graph_us(lambda: [engine(l, x) for l, x in zip(layers, xs)])
            us_s2 = graph_us(lambda: [stock(l, x) for l, x in zip(layers, xs)])
            b = {"stock_us": us_s / len(layers), "stock_us_again": us_s2 / len(layers),
                 "tilemajor_us": us_t / len(layers), "layers": Ls}
            res["bench"][proj] = b
            print(f"BENCH {proj:9s} M=1: stock {b['stock_us']:8.1f} us (again "
                  f"{b['stock_us_again']:.1f}) tile-major {b['tilemajor_us']:8.1f} us "
                  f"(engine allocation order: placement not controlled)", flush=True)
            del layers, xs
            torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "marlint_exact.json"), "w") as f:
        json.dump(res, f, indent=1)
    ok = res["mismatch_cases"] == 0 and not res["fails"] and res["admitted"] == res["layers"]
    print(f"RESULT {'PASS' if ok else 'FAIL'}: {res['cases']} cases + {res['graph_cases']} graph "
          f"replays on {res['layers']} layers, {res['mismatch_cases']} mismatching, admitted "
          f"{res['admitted']}/{res['layers']}, copies {res['copy_bytes']/1e9:.2f} GB", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
