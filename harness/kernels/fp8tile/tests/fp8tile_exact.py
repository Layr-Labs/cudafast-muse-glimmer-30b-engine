# SPDX-License-Identifier: Apache-2.0
"""Bit-identity and speed of fp8tile against vLLM v0.27.1 ops.cutlass_scaled_mm.

Runs inside the pinned vLLM image with the built kernels on PYTHONPATH and the
checkpoint snapshot at --snapshot. Every FP8 linear of the text decoder is
rebuilt exactly as ModelOptFp8LinearMethod.process_weights_after_loading does
(merged shards re-quantized with the max shard scale when the shard scales
differ, weight transposed to the [K, N] column-major operand, max input scale).

Modes (any combination):
  exact  every FP8 module x M in --ms x {random, small, extreme, spikes, all
         e4m3 bit patterns (NaN included)} x every built configuration:
         integer comparison of the bf16 bits, plus the load-time prepare();
  bench  M = 1 timing in CUDA graphs over distinct real layers (cold weights),
         stock vs each built configuration;
  op     the vLLM custom op: dispatch inside a dynamic torch.compile graph for
         small and large M, and CUDA graph replay with changed inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import fp8tile
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

PFX = "model.language_model.layers"
MODULES = {
    "qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")),
    "attn_gate": ("self_attn", ("gate_proj",)),
    "o_proj": ("self_attn", ("o_proj",)),
    "gate_up": ("mlp", ("gate_proj", "up_proj")),
    "down": ("mlp", ("down_proj",)),
}
DT = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16, "U8": torch.uint8}


def log(*a):
    print(*a, flush=True)


class Snapshot:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self.hdr = {}

    def meta(self, name):
        fn = self.wmap[name]
        if fn not in self.hdr:
            with open(os.path.join(self.root, fn), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
        h, base = self.hdr[fn]
        return h[name], base, fn

    def get(self, name, device="cuda"):
        m, base, fn = self.meta(name)
        a, b = m["data_offsets"]
        buf = bytearray(b - a)
        with open(os.path.join(self.root, fn), "rb", buffering=0) as f:
            mv, got = memoryview(buf), 0
            while got < len(buf):
                got += os.preadv(f.fileno(), [mv[got:]], base + a + got)
        t = torch.frombuffer(buf, dtype=torch.uint8).view(DT[m["dtype"]])
        t = t.reshape(m["shape"]) if m["shape"] else t.reshape(())
        return t.to(device)


def fp8_modules(snap, n_layers):
    out = []
    for L in range(n_layers):
        for mod, (blk, parts) in MODULES.items():
            pre = [f"{PFX}.{L}.{blk}.{p}" for p in parts]
            if snap.meta(pre[0] + ".weight")[0]["dtype"] == "F8_E4M3":
                out.append((L, mod, pre))
    return out


def load_module(snap, pre):
    """ModelOptFp8LinearMethod.process_weights_after_loading, then the operand."""
    w = torch.cat([snap.get(p + ".weight") for p in pre], 0) if len(pre) > 1 else snap.get(pre[0] + ".weight")
    ws = torch.stack([snap.get(p + ".weight_scale").reshape(()) for p in pre])
    xs = torch.stack([snap.get(p + ".input_scale").reshape(()) for p in pre])
    widths = [snap.meta(p + ".weight")[0]["shape"][0] for p in pre]
    requant = bool(not (ws == ws[0]).all())
    sb = ws.max()
    if requant:
        sb, w = requantize_with_max_scale(w, ws, widths)
    return w.t(), xs.max(), sb, requant


def stock(a, b, sa, sb):
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def make_x(kind, m, k, gen):
    if kind == "all_bits":
        base = torch.arange(m * k, device="cuda", dtype=torch.int64)
        base = base + 37 * torch.randint(0, 256, (1,), generator=gen, device="cuda")
        return base.remainder(256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(m, k)
    if kind == "spikes":
        x = torch.randn((m, k), generator=gen, device="cuda") * 0.05
        idx = torch.randint(0, k, (m, 16), generator=gen, device="cuda")
        x.scatter_(1, idx, torch.randn((m, 16), generator=gen, device="cuda").sign() * 448.0)
        return x.to(torch.float8_e4m3fn)
    scale = {"random": 20.0, "small": 0.02, "extreme": 300.0}[kind]
    x = torch.randn((m, k), generator=gen, device="cuda") * scale
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


KINDS = ("random", "small", "extreme", "spikes", "all_bits")


def nbad(ref, got):
    return int((ref.view(torch.int16) != got.view(torch.int16)).sum())


def mode_exact(snap, mods, res, cfgs, ms):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    cases = bad_cases = elems = 0
    per_cfg = {c: 0 for c in cfgs}
    fails = []
    t0 = time.time()
    for L, mod, pre in mods:
        b, sa, sb, rq = load_module(snap, pre)
        k, n = b.shape
        if fp8tile.prepare(b, sa, sb, stock) is None:
            fails.append({"layer": L, "module": mod, "prepare": False})
            bad_cases += 1
        for m in ms:
            for kind in KINDS:
                x = make_x(kind, m, k, gen)
                ref = stock(x, b, sa, sb)
                for c in cfgs:
                    bad = nbad(ref, fp8tile.gemm(x, b, sa, sb, c))
                    cases += 1
                    elems += ref.numel()
                    if bad:
                        bad_cases += 1
                        per_cfg[c] += 1
                        if len(fails) < 20:
                            fails.append({"layer": L, "module": mod, "M": m, "kind": kind, "cfg": c,
                                          "bad": bad})
        del b
    res["exact"] = {"modules": len(mods), "configs": cfgs, "M": ms, "cases": cases, "elements": elems,
                    "bad_cases": bad_cases, "bad_cases_per_config": per_cfg, "fails": fails,
                    "seconds": round(time.time() - t0, 1)}
    log(f"exact: {cases - bad_cases}/{cases} cases bit-identical over {len(mods)} FP8 modules, "
        f"{elems} outputs, configurations {cfgs}, M {ms}; bad per configuration {per_cfg}; "
        f"fails {fails[:5]}")
    return bad_cases == 0


def graph_ms(fn, replays=15, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    del g
    ts.sort()
    return ts[len(ts) // 2]


def mode_bench(snap, mods, res, max_layers, cfgs, ms):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    by = {}
    for L, mod, pre in mods:
        by.setdefault(mod, []).append((L, pre))
    out = {}
    total = {c: 0.0 for c in cfgs}
    for mod, lst in by.items():
        step = max(1, len(lst) // max_layers)
        sel = lst[::step][:max_layers]
        layers = [load_module(snap, pre) for _, pre in sel]
        k, n = layers[0][0].shape
        for m in ms:
            xs = [make_x("random", m, k, gen) for _ in layers]
            row = {"N": n, "K": k, "M": m, "rot": len(layers), "calls_per_token": len(lst)}
            ms_ = graph_ms(lambda: [stock(x, b, sa, sb) for x, (b, sa, sb, _) in zip(xs, layers)])
            row["stock_us"] = round(ms_ * 1e3 / len(layers), 2)
            for c in cfgs:
                ms_ = graph_ms(lambda c=c: [fp8tile.gemm(x, b, sa, sb, c) for x, (b, sa, sb, _) in zip(xs, layers)])
                us = ms_ * 1e3 / len(layers)
                row[f"cfg{c}_us"] = round(us, 2)
                if m == 1:
                    total[c] += (row["stock_us"] - us) * len(lst) / 1e3
            out[f"{mod}_M{m}"] = row
            log(f"bench {mod:9s} M={m} N={n} K={k} rot={len(layers)}: stock {row['stock_us']:.1f} us; "
                + " ".join(f"{fp8tile.TILES[c]}={row[f'cfg{c}_us']:.1f}"
                           f"({100 * (row[f'cfg{c}_us'] / row['stock_us'] - 1):+.1f}%)" for c in cfgs))
        del layers
        torch.cuda.empty_cache()
    res["bench"] = out
    res["bench_saving_ms_per_token_M1"] = {c: round(v, 3) for c, v in total.items()}
    log("bench: M=1 saving per token (ms, isolated kernels, cold weights) per configuration: "
        + ", ".join(f"{fp8tile.TILES[c]}: {v:.3f}" for c, v in total.items()))
    return True


def mode_op(snap, mods, res):
    from vllm.model_executor.kernels.linear.scaled_mm import fp8tile as hook

    L, mod, pre = next(t for t in mods if t[1] == "qkv")
    b, sa, sb, _ = load_module(snap, pre)
    k = b.shape[0]
    cfg = hook.prepare(b, sa, sb)
    assert cfg is not None
    gen = torch.Generator(device="cuda")
    gen.manual_seed(99)

    def projected(x):
        return torch.ops.vllm.fp8tile_scaled_mm(x, b, sa, sb, cfg)

    compiled = torch.compile(projected, dynamic=True, fullgraph=True)
    ok = {}
    for m in (1, 2, 4, 8, 9, 16, 17, 512):
        x = make_x("random", m, k, gen)
        ok[f"compiled_M{m}"] = nbad(stock(x, b, sa, sb), compiled(x)) == 0
    x = make_x("random", 1, k, gen)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            projected(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = projected(x)
    for i in range(3):
        x.copy_(make_x("extreme" if i % 2 else "random", 1, k, gen))
        g.replay()
        torch.cuda.synchronize()
        ok[f"graph_replay_{i}"] = nbad(stock(x, b, sa, sb), y) == 0
    res["op"] = ok
    log("op:", ok)
    return all(ok.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("KBENCH_SNAPSHOT"))
    ap.add_argument("--out", default=os.environ.get("KBENCH_OUT", "/out"))
    ap.add_argument("--modes", default="exact,bench")
    ap.add_argument("--configs", default="", help="comma-separated; default: every built one")
    ap.add_argument("--ms", default="1,2,3,5,8,9,16")
    ap.add_argument("--bench-ms", default="1")
    ap.add_argument("--max-layers", type=int, default=8)
    args, _ = ap.parse_known_args()
    log("torch", torch.__version__, torch.cuda.get_device_name(), "fp8tile available", fp8tile.available(),
        "built", sorted(fp8tile.built_configs()))
    cfgs = [int(c) for c in args.configs.split(",")] if args.configs else sorted(fp8tile.built_configs())
    snap = Snapshot(args.snapshot)
    with open(os.path.join(args.snapshot, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    mods = fp8_modules(snap, n_layers)
    log(f"{len(mods)} FP8 modules")
    res = {"argv": sys.argv[1:]}
    ok = True
    for mode in args.modes.split(","):
        if mode == "exact":
            ok = mode_exact(snap, mods, res, cfgs, [int(m) for m in args.ms.split(",")]) and ok
        elif mode == "bench":
            ok = mode_bench(snap, mods, res, args.max_layers, cfgs, [int(m) for m in args.bench_ms.split(",")]) and ok
        elif mode == "op":
            ok = mode_op(snap, mods, res) and ok
    res["ok"] = ok
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fp8tile_exact.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("RESULT", "EXACT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
