# SPDX-License-Identifier: Apache-2.0
"""Tile sweep for the two FP8 prefill shapes that `muse_fp8_prefill` serves at M=512. Not part of the scored path.

    flock /home/zereraz/muse/gpu.lock python3 sweep_shapes.py

Shapes, from the tree's own `_CFG` table and the model's dimensions:
  down    (N=6656, K=19968) -- 12 FP8 down layers per prefill, currently config 2
  o_proj  (N=6656, K=4096)  -- 52 layers per prefill, currently config 3
o_proj is the larger half of the block by call count (52 vs 12) and is the one nobody has swept.

Per config it prints the shared-memory footprint, how many CTAs per SM that allows, median ms/call, effective TFLOPS,
and whether the output is bit-identical to the shape's shipping config. The shipping config is the reference because
it is the one the tree admits; a config that reproduces it bit for bit is admissible by construction.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import sys

import torch

LIB = os.environ.get("PREFILL_LIB", "/home/zereraz/muse/attn_out/muse_fp8_prefill/libmuse_fp8_prefill.so")
SMEM_PER_SM = 228 * 1024
M = 512
CFGS = [1, 2, 3, 4]
SHAPES = [("down", 6656, 19968, 2, 12), ("o_proj", 6656, 4096, 3, 52), ("qkv", 4608, 6656, 3, 52)]


def load():
    h = ctypes.CDLL(LIB)
    vp = ctypes.c_void_p
    h.fp8_prefill_gemm.argtypes = [ctypes.c_int] * 4 + [vp] * 6 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, vp]
    h.fp8_prefill_gemm.restype = ctypes.c_int
    h.fp8_prefill_shared_bytes.argtypes = [ctypes.c_int]
    h.fp8_prefill_shared_bytes.restype = ctypes.c_size_t
    h.fp8_prefill_status_str.argtypes = [ctypes.c_int]
    h.fp8_prefill_status_str.restype = ctypes.c_char_p
    return h


def run(h, a, b, sa, sb, cfg, out, scratch, sms):
    st = h.fp8_prefill_gemm(cfg, a.size(0), b.size(1), a.size(1), a.data_ptr(), b.data_ptr(), sa.data_ptr(),
                            sb.data_ptr(), out.data_ptr(), scratch.data_ptr(), scratch.numel(), a.device.index, sms,
                            torch.cuda.current_stream(a.device).cuda_stream)
    return st


def main() -> int:
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device {torch.cuda.get_device_name(0)}  SMs {sms}  M={M}")
    h = load()
    sa = torch.tensor([0.0031], dtype=torch.float32, device=dev)
    sb = torch.tensor([0.0022], dtype=torch.float32, device=dev)
    scratch = torch.empty(1 << 20, dtype=torch.uint8, device=dev)

    for name, N, K, ship, layers in SHAPES:
        gen = torch.Generator(device=dev).manual_seed(N * 7919 + K)
        a = (torch.randn(M, K, generator=gen, device=dev) * 40.0).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        b = torch.empty_strided((K, N), (1, K), dtype=torch.float8_e4m3fn, device=dev)
        b.copy_(torch.randn(N, K, generator=gen, device=dev).clamp(-448, 448).to(torch.float8_e4m3fn).t())
        out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        flop = 2.0 * M * N * K
        print(f"\n=== {name}: N={N} K={K}, {layers} layers per prefill, shipping config {ship} ===")
        print(f"{'cfg':>4} {'KiB':>7} {'CTA/SM':>6} {'ms':>8} {'TFLOPS':>7} {'x'+str(layers):>9}  bits-vs-cfg{ship}")
        ref = None
        base = None
        for cfg in [ship] + [c for c in CFGS if c != ship]:
            smem = int(h.fp8_prefill_shared_bytes(cfg))
            ctas = SMEM_PER_SM // smem if smem else 0
            st = run(h, a, b, sa, sb, cfg, out, scratch, sms)
            if st:
                why = h.fp8_prefill_status_str(st - 100 if st >= 100 else st).decode()
                print(f"{cfg:>4} {smem/1024:>7.1f} {ctas:>6} {'-':>8} {'-':>7} {'-':>9}  status {st} ({why})")
                continue
            torch.cuda.synchronize()
            bits = out.view(torch.int16).clone()
            if cfg == ship:
                ref = bits
            for _ in range(3):
                run(h, a, b, sa, sb, cfg, out, scratch, sms)
            torch.cuda.synchronize()
            ts = []
            for _ in range(20):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                run(h, a, b, sa, sb, cfg, out, scratch, sms)
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e))
            med = statistics.median(ts)
            if cfg == ship:
                base = med
            run(h, a, b, sa, sb, cfg, out, scratch, sms)
            torch.cuda.synchronize()
            if cfg == ship:
                tag = "ref"
            elif ref is not None and torch.equal(out.view(torch.int16), ref):
                tag = "EXACT"
            else:
                n_bad = int((out.view(torch.int16) != ref).sum().item()) if ref is not None else -1
                tag = f"DIFF {n_bad} elems"
            print(f"{cfg:>4} {smem/1024:>7.1f} {ctas:>6} {med:>8.4f} {flop/(med*1e-3)/1e12:>7.1f} "
                  f"{med*layers:>9.2f}  {tag}")
        if base is not None:
            print(f"  shipping config {ship}: {base*layers:.2f} ms of prefill for this shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
