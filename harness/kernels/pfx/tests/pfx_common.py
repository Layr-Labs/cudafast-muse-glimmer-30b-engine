"""Shared helpers of the pfx tests: load single NVFP4 linears from the snapshot the way vLLM does,
run stock Marlin, generate inputs, compare bits. Imported by the scripts in this directory."""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
from vllm.scalar_type import scalar_types


def log(*a):
    print(*a, flush=True)


def smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e!r}"


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
          ("o_proj", ["self_attn.o_proj"]), ("attn_gate", ["self_attn.gate_proj"]),
          ("gate_up", ["mlp.gate_proj", "mlp.up_proj"]), ("down", ["mlp.down_proj"])]


def nvfp4_linears(snap: Snap):
    """{(kind, N, K): [(layer, [hf prefixes])]} of the merged NVFP4 linears as vLLM builds them."""
    pat = re.compile(r"^(.*layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+))\.weight$")
    kinds = collections.defaultdict(dict)
    pref = {}
    for n in sorted(snap.wmap):
        if "vision" in n:
            continue
        m = pat.match(n)
        if not m:
            continue
        prefix, li, blk, proj = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        dt, _ = snap.meta(n)
        fmt = "nvfp4" if (prefix + ".weight_scale_2") in snap.wmap and dt == "U8" else dt
        kinds[li][blk + "." + proj] = fmt
        pref[(li, blk + "." + proj)] = prefix
    out = collections.OrderedDict()
    for li in sorted(kinds):
        for kind, ps in GROUPS:
            if all(kinds[li].get(p) == "nvfp4" for p in ps):
                prefs = [pref[(li, p)] for p in ps]
                n = sum(snap.meta(p + ".weight")[1][0] for p in prefs)
                k = 2 * snap.meta(prefs[0] + ".weight")[1][1]
                out.setdefault((kind, n, k), []).append((li, prefs))
    return out


def load_merged(snap: Snap, prefixes):
    """A Marlin NVFP4 linear exactly as ModelOptNvFp4W4A16LinearMethod + MarlinNvFp4LinearKernel
    prepare it (merged shards along N, global scale = max of the shard scales)."""
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


def silu_and_mul(h: torch.Tensor) -> torch.Tensor:
    """vLLM's SiluAndMul kernel (the engine's op between gate_up and down)."""
    d = h.shape[-1] // 2
    out = torch.empty(h.shape[:-1] + (d,), dtype=h.dtype, device=h.device)
    torch.ops._C.silu_and_mul(out, h)
    return out


def gen_x(m: int, k: int, kind: int, gen: torch.Generator) -> torch.Tensor:
    """0: N(0, 1); 1: N(0, 4) with heavy outliers, exact zeros and negative zeros; 2: N(0, 0.05);
    3: N(0, 1) times a per-column scale spread over 2^-6 .. 2^6 (RMSNorm-weight-like)."""
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float32)
    if kind == 1:
        x = x * 4.0
        idx = torch.randint(0, k, (m, 16), generator=gen, device="cuda")
        x.scatter_(1, idx, torch.randn(m, 16, generator=gen, device="cuda") * 300.0)
        z = torch.rand(m, k, generator=gen, device="cuda") < 0.05
        x = torch.where(z, torch.zeros_like(x), x)
        nz = torch.rand(m, k, generator=gen, device="cuda") < 0.02
        x = torch.where(nz, torch.full_like(x, -0.0), x)
    elif kind == 2:
        x = x * 0.05
    elif kind == 3:
        col = torch.exp2(torch.randint(-6, 7, (k,), generator=gen, device="cuda").float())
        x = x * col
    return x.to(torch.bfloat16)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def diff_info(ref: torch.Tensor, new: torch.Tensor) -> dict:
    bad = ref.view(torch.int16) != new.view(torch.int16)
    rows = bad.any(1).nonzero().flatten()
    cols = bad.any(0).nonzero().flatten()
    rb = sorted(set((rows // 64).tolist()))
    ct = sorted(set((cols // 256).tolist()))
    d = (ref.float() - new.float()).abs()
    return {"elements": int(bad.sum()), "row_blocks": rb[:16], "col_tiles": ct[:40], "n_col_tiles": len(ct),
            "max_abs": float(d[torch.isfinite(d)].max()) if bool(torch.isfinite(d).any()) else None,
            "first": [(int(r), int(c)) for r, c in bad.nonzero()[:8].tolist()]}


def timed(fn, reps: int = 20, warm: int = 3) -> float:
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
