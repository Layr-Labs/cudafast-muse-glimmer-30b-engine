"""Bit-identity and speed of pfx against stock Marlin (vLLM v0.27.1 ops.marlin_gemm) at M = 512.

Runs inside the pinned image (bin/atom glim exec SHA harness/kernels/pfx/tests/pfx_exact.py) with
the staged overlay and $KERNELS_OUT on PYTHONPATH and the snapshot at $GLIM_SNAPSHOT. For every
NVFP4 linear shape of the snapshot (merged as vLLM merges them) it loads several real layers
through prepare_fp4_layer_for_marlin, feeds M = 512 bf16 inputs of several kinds (and, for down,
SiluAndMul of the stock gate_up output of the same layer), compares pfx with stock Marlin bit for
bit (torch.equal on the int16 views) and times both with CUDA events.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import torch

import pfx
from pfx_common import (Snap, bits_equal, diff_info, gen_x, load_merged, log, nvfp4_linears, silu_and_mul, smi,
                        stock, timed)


def pick(items, n):
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="gate_up,down,qkv,attn_gate,o_proj")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--inputs", default="0,1,2,3")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    out_dir = os.environ.get("GLIM_OUT", "/tmp")
    res: dict = {"argv": sys.argv[1:], "smi_start": smi()}
    log("nvidia-smi compute apps at start:", repr(res["smi_start"]))
    log("pfx available", pfx.available(), pfx.load_error())
    if not pfx.available():
        return 2
    log("pfx attrs", pfx.attrs())
    res["attrs"] = pfx.attrs()
    lin = nvfp4_linears(snap)
    kinds = args.kinds.split(",")
    inputs = [int(v) for v in args.inputs.split(",")]
    m = args.m
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261121)
    total_cases = bad_cases = 0
    total_elems = 0
    per = {}
    examples = []
    timing = {}
    gate_up_by_layer = {li: prefs for li, prefs in lin.get(next((s for s in lin if s[0] == "gate_up"), None), [])}
    for shape, layers in lin.items():
        kd, n, k = shape
        if kd not in kinds:
            continue
        plan = pfx.Plan(m, n, k, dev)
        nseg = collections.Counter(len(u[2]) for u in plan.units_py)
        log(f"[{kd} N={n} K={k}] {len(layers)} layers; units {len(plan.units_py)} segs {dict(nseg)}")
        for li, prefs in pick(layers, args.layers):
            t0 = time.time()
            layer = load_merged(snap, prefs)
            xs = [(f"in{v}", gen_x(m, k, v, gen)) for v in inputs]
            if kd == "down" and li in gate_up_by_layer:
                gu = load_merged(snap, gate_up_by_layer[li])
                h = stock(gu, gen_x(m, gu.input_size_per_partition, 0, gen))
                xs.append(("silu_gate_up", silu_and_mul(h)))
                h = stock(gu, gen_x(m, gu.input_size_per_partition, 1, gen))
                xs.append(("silu_gate_up_outl", silu_and_mul(h)))
                del gu, h
            fails = 0
            for name, x in xs:
                ref = stock(layer, x)
                new = pfx.gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, plan)
                ok = bits_equal(ref, new)
                total_cases += 1
                total_elems += ref.numel()
                if not ok:
                    fails += 1
                    bad_cases += 1
                    info = {"case": f"{kd}.L{li}.{name}", **diff_info(ref, new)}
                    if len(examples) < 16:
                        examples.append(info)
                    log("  MISMATCH", json.dumps(info))
            # strided input (lda = K + 64) and determinism, on the first input kind
            xw = gen_x(m, k + 64, 0, gen)
            xs_ = xw[:, :k]
            ok_str = bits_equal(stock(layer, xs_.contiguous()), pfx.gemm(xs_, layer.weight, layer.weight_scale,
                                                                       layer.weight_global_scale, plan))
            x0 = xs[0][1]
            r1 = pfx.gemm(x0, layer.weight, layer.weight_scale, layer.weight_global_scale, plan)
            ok_det = all(bits_equal(r1, pfx.gemm(x0, layer.weight, layer.weight_scale, layer.weight_global_scale, plan))
                         for _ in range(2))
            total_cases += 2
            total_elems += 2 * r1.numel()
            if not ok_str or not ok_det:
                bad_cases += 1
                log("  MISMATCH strided", ok_str, "deterministic", ok_det)
            per[f"{kd}.L{li}"] = {"cases": len(xs) + 2, "fails": fails + (not ok_str) + (not ok_det)}
            log(f"  L{li}: {len(xs) - fails}/{len(xs)} inputs bit-identical, strided {ok_str}, deterministic {ok_det}"
                f" ({time.time() - t0:.1f}s)")
            if not args.skip_timing and li == pick(layers, args.layers)[0][0]:
                x = gen_x(m, k, 0, gen)
                ts = timed(lambda: stock(layer, x), reps=args.reps)
                tp = timed(lambda: pfx.gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, plan),
                           reps=args.reps)
                fl = 2 * m * n * k
                timing[f"{kd}:{n}x{k}"] = {"stock_ms": ts, "pfx_ms": tp, "stock_tflops": fl / ts / 1e9,
                                            "pfx_tflops": fl / tp / 1e9, "calls_per_pass": len(layers)}
                log(f"  TIME {kd}: stock {ts:.3f} ms ({fl / ts / 1e9:.1f} TFLOP/s)  pfx {tp:.3f} ms "
                    f"({fl / tp / 1e9:.1f} TFLOP/s)  x{len(layers)} per pass: {ts * len(layers):.1f} -> "
                    f"{tp * len(layers):.1f} ms")
            del layer
            torch.cuda.empty_cache()
    if timing:
        s = sum(v["stock_ms"] * v["calls_per_pass"] for v in timing.values())
        p = sum(v["pfx_ms"] * v["calls_per_pass"] for v in timing.values())
        log(f"TIME per 512-row pass over the timed shapes: stock {s:.1f} ms, pfx {p:.1f} ms")
    res.update({"cases": total_cases, "failing": bad_cases, "elements_compared": total_elems, "per_layer": per,
                "examples": examples, "timing": timing, "smi_end": smi()})
    with open(os.path.join(out_dir, "pfx_exact.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"RESULT {'EXACT' if bad_cases == 0 else 'MISMATCH'}: {total_cases - bad_cases}/{total_cases} cases, "
        f"{total_elems} output elements compared")
    return 0 if bad_cases == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
