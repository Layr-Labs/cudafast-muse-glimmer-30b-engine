"""Probe: which linears of the snapshot are NVFP4, what stock Marlin runs for them at M = 512, and how long.

Runs in the pinned image (bin/atom glim exec SHA harness/kernels/pfx/tests/probe_marlin.py) with the
staged overlay on PYTHONPATH and the snapshot at $GLIM_SNAPSHOT. Loads single layers from the
safetensors shards the way vLLM does (merged gate_up / qkv, ModelOptNvFp4W4A16LinearMethod global
scale = max of the shard scales, prepare_fp4_layer_for_marlin), so weight packing is production.
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import time

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
from vllm.scalar_type import scalar_types


def log(*a):
    print(*a, flush=True)


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e!r}"


class Snap:
    def __init__(self, snap):
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


def load_merged(snap: Snap, prefixes):
    ws, ss, s2 = [], [], []
    for p in prefixes:
        ws.append(snap.get(p + ".weight"))
        ss.append(snap.get(p + ".weight_scale"))
        s2.append(snap.get(p + ".weight_scale_2").reshape(-1))
    w = torch.cat(ws, 0)
    s = torch.cat(ss, 0)
    layer = torch.nn.Module()
    n, k2 = w.shape
    layer.output_size_per_partition = n
    layer.input_size_per_partition = 2 * k2
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s.cuda(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.cat(s2).cuda().max().to(torch.float32), requires_grad=False)
    prepare_fp4_layer_for_marlin(layer)
    return layer


def stock(layer, x):
    return ops.marlin_gemm(
        a=x, c=None, b_q_weight=layer.weight, b_bias=None, b_scales=layer.weight_scale,
        a_scales=None, global_scale=layer.weight_global_scale, b_zeros=None, g_idx=None,
        perm=None, workspace=layer.workspace, b_q_type=scalar_types.float4_e2m1f,
        size_m=x.size(0), size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition, use_atomic_add=False, use_fp32_reduce=True)


def timed(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def main():
    snap_dir = os.environ["GLIM_SNAPSHOT"]
    out_dir = os.environ.get("GLIM_OUT", "/tmp")
    res = {"smi_start": smi()}
    log("nvidia-smi compute apps at start:", repr(res["smi_start"]))
    p = torch.cuda.get_device_properties(0)
    log("torch", torch.__version__, p.name, "sms", p.multi_processor_count, "cc", p.major, p.minor,
        "smem_optin", getattr(p, "shared_memory_per_block_optin", None), "L2", getattr(p, "L2_cache_size", None))
    import vllm
    log("vllm", vllm.__version__, "VLLM_MARLIN_USE_ATOMIC_ADD", os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD"))
    with open(os.path.join(snap_dir, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    log("quantization_config keys:", sorted(qc.keys()))
    for k, v in qc.items():
        if k != "quantized_layers":
            log("  qc", k, json.dumps(v)[:300])
    ql = qc.get("quantized_layers", {})
    log("quantized_layers type", type(ql).__name__, "n", len(ql))
    if isinstance(ql, dict):
        algos = collections.Counter(json.dumps(v, sort_keys=True) for v in ql.values())
        log("  algos:", algos.most_common(10))
        items = list(ql.items())
        for k, v in items[:12]:
            log("   ", k, v)
    tc = cfg.get("text_config", cfg)
    for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "num_hidden_layers", "vocab_size"):
        log("  text_config", key, tc.get(key))

    snap = Snap(snap_dir)
    names = sorted(snap.wmap)
    l0 = [n for n in names if re.search(r"layers\.0\.", n) and "vision" not in n]
    for n in l0:
        log("  L0", n, snap.meta(n))
    # classify every text linear, keyed by "self_attn.q_proj", "mlp.gate_proj", ...
    pat = re.compile(r"^(.*layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+))\.weight$")
    kinds = collections.defaultdict(dict)  # layer -> "block.proj" -> (fmt, shape)
    pref = {}
    for n in names:
        if "vision" in n:
            continue
        m = pat.match(n)
        if not m:
            continue
        prefix, li, blk, proj = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        dt, shp = snap.meta(n)
        fmt = "nvfp4" if (prefix + ".weight_scale_2") in snap.wmap and dt in ("U8", "uint8") else str(dt)
        kinds[li][blk + "." + proj] = (fmt, shp)
        pref[(li, blk + "." + proj)] = prefix
    per = collections.Counter()
    for li in sorted(kinds):
        per[json.dumps({k: v[0] for k, v in sorted(kinds[li].items())}, sort_keys=True)] += 1
    for k, v in per.most_common():
        log("  layer pattern x%d: %s" % (v, k))
    fixed = []  # (layer, kind, [hf prefixes]) of the merged NVFP4 linears as vLLM builds them
    for li in sorted(kinds):
        k = kinds[li]

        def nv(p):
            return p in k and k[p][0] == "nvfp4"

        groups = [("qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
                  ("o_proj", ["self_attn.o_proj"]), ("attn_gate", ["self_attn.gate_proj"]),
                  ("gate_up", ["mlp.gate_proj", "mlp.up_proj"]), ("down", ["mlp.down_proj"])]
        for kind, ps in groups:
            if all(nv(p) for p in ps):
                fixed.append((li, kind, [pref[(li, p)] for p in ps]))
            elif any(nv(p) for p in ps):
                log("  MIXED group", li, kind, [k.get(p) for p in ps])
    shapes = collections.OrderedDict()
    for li, kind, prefs in fixed:
        n = 0
        for p in prefs:
            dt, shp = snap.meta(p + ".weight")
            n += shp[0]
            kk = 2 * shp[1]
        shapes.setdefault((kind, n, kk), []).append(li)
    res["nvfp4_shapes"] = {f"{kd}:{n}x{k}": lis for (kd, n, k), lis in shapes.items()}
    tot_calls = sum(len(v) for v in shapes.values())
    tot_flop = sum(2 * 512 * n * k * len(v) for (kd, n, k), v in shapes.items())
    log(f"NVFP4 merged linears: {tot_calls} calls per M=512 pass, {tot_flop / 1e12:.2f} TFLOP")
    for (kd, n, k), lis in shapes.items():
        log(f"  {kd} N={n} K={k} x{len(lis)} layers {lis[:6]}...")


    # bandwidth roofline
    big = torch.empty(1 << 31, dtype=torch.uint8, device="cuda")
    big.fill_(1)
    bv = big.view(torch.float32)
    t = timed(lambda: bv.sum(), reps=5, warm=1)
    res["read_GBps"] = big.numel() / (t * 1e-3) / 1e9
    log(f"stream read (torch sum of 2 GiB): {res['read_GBps']:.1f} GB/s")
    del big, bv
    torch.cuda.empty_cache()

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    timing = {}
    from torch.profiler import ProfilerActivity, profile
    for (kd, n, k), lis in shapes.items():
        li = lis[0]
        prefs = [pp for (l2, kd2, pp) in fixed if l2 == li and kd2 == kd][0]
        t0 = time.time()
        layer = load_merged(snap, prefs)
        log(f"[{kd} N={n} K={k}] loaded layer {li} in {time.time() - t0:.1f}s; gs={float(layer.weight_global_scale):.4e}"
            f" w {tuple(layer.weight.shape)} s {tuple(layer.weight_scale.shape)} {layer.weight_scale.dtype}")
        row = {}
        for m in (512, 64, 128, 4096):
            x = (torch.randn(m, k, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            ms = timed(lambda: stock(layer, x))
            tf = 2 * m * n * k / (ms * 1e-3) / 1e12
            row[m] = {"ms": ms, "tflops": tf}
            log(f"   M={m}: {ms:.3f} ms  {tf:.1f} TFLOP/s")
        x = (torch.randn(512, k, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            stock(layer, x)
            torch.cuda.synchronize()
        knames = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        row["kernels"] = knames
        for kn in knames:
            log("   kernel:", kn[:400])
        timing[f"{kd}:{n}x{k}"] = row
        del layer
        torch.cuda.empty_cache()
    res["timing"] = timing
    res["smi_end"] = smi()
    with open(os.path.join(out_dir, "probe_marlin.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("DONE")


if __name__ == "__main__":
    sys.exit(main())
