# SPDX-License-Identifier: Apache-2.0
"""Tile/depth sweep for the 512-row FP8 gate/up + SiLU + FP4-quantize kernel. Not part of the scored path.

    flock /home/zereraz/muse/gpu.lock python3 sweep_swiglu.py

Shape: N = 2 x 19968 interleaved gate/up columns, K = 6656, M = 512, output FP4 (e2m1) packed two per byte. Seven
layers per prefill. Config 0 is the shipping kernel; every other config must reproduce it bit for bit to be
admissible, since they differ only in tile shape and pipeline depth.
"""

from __future__ import annotations

import os
import sys

import torch

import importlib.util  # noqa: E402

_PKG = os.environ.get("SWIGLU_PKG", "/home/zereraz/muse/attn_out/muse_fp8_swiglu")
_spec = importlib.util.spec_from_file_location("muse_fp8_swiglu", os.path.join(_PKG, "__init__.py"))
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

SMEM_PER_SM = 228 * 1024
CFGS = [0, 1, 2, 3]


def main() -> int:
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device {torch.cuda.get_device_name(0)}  SMs {sms}  gate/up M=512 N={2*sw._INNER} K={sw._HIDDEN}")
    gen = torch.Generator(device=dev).manual_seed(39936 * 7919 + 6656)
    a = (torch.randn(sw._ROWS, sw._HIDDEN, generator=gen, device=dev) * 40.0).clamp(-448, 448) \
        .to(torch.float8_e4m3fn).contiguous()
    b = torch.empty_strided((sw._HIDDEN, 2 * sw._INNER), (1, sw._HIDDEN), dtype=torch.float8_e4m3fn, device=dev)
    b.copy_(torch.randn(2 * sw._INNER, sw._HIDDEN, generator=gen, device=dev).clamp(-448, 448)
            .to(torch.float8_e4m3fn).t())
    sa = torch.tensor([0.0031], dtype=torch.float32, device=dev)
    sb = torch.tensor([0.0022], dtype=torch.float32, device=dev)
    down = torch.tensor([0.021], dtype=torch.float32, device=dev)
    # prepare() is what loads the SiLU table and records the SM count; it also re-interleaves the weight, which the
    # benchmark does not need, so its return value is discarded and the fabricated b is used instead.
    sw.prepare(b)
    flop = 2.0 * sw._ROWS * (2 * sw._INNER) * sw._HIDDEN
    print(f"{'cfg':>4} {'KiB':>7} {'CTA/SM':>6} {'ms':>8} {'TFLOPS':>7} {'x7':>7}  bits-vs-cfg0")
    res = sw.bench_cfg(a, b, sa, sb, down, CFGS)
    base = None
    for cfg in CFGS:
        r = res.get(cfg, {})
        smem = sw.shared_bytes_cfg(cfg)
        ctas = SMEM_PER_SM // smem if smem else 0
        if "error" in r:
            print(f"{cfg:>4} {smem/1024:>7.1f} {ctas:>6} {'-':>8} {'-':>7} {'-':>7}  {r['error']}")
            continue
        ms = r["ms"]
        if cfg == 0:
            base = ms
        tag = "ref" if cfg == 0 else ("EXACT" if r["same_bits"] else "DIFF")
        print(f"{cfg:>4} {smem/1024:>7.1f} {ctas:>6} {ms:>8.4f} {flop/(ms*1e-3)/1e12:>7.1f} {ms*7:>7.2f}  {tag}")
    if base:
        print(f"  shipping config 0: {base*7:.2f} ms of prefill for this shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
