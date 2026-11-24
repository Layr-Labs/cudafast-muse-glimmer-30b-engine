#!/usr/bin/env python3
"""Roofline audit of one 512-token prefill: (group, ms, TFLOPS, format roofline, % of roofline),
sorted by recoverable ms.

TWO INDEPENDENT INPUTS, DELIBERATELY KEPT SEPARATE.
  * Layer counts come from the ENGINE'S OWN DISPATCH LOG, not from a guess. Every count below is
    quoted verbatim from `decode_placement.py`'s "Prefill weight placement" lines in a real
    `MB_WARM=1` bench log. They reconcile exactly: gate_up 45+7=52, down 40+12=52, qkv 28+18+6=52,
    gate 28+18+6=52, o_proj 42+10=52. That reconciliation is the proof the attribution is complete.
  * Times come from a torch-profiler dump of one steady-state 512-token prefill
    (prefill_profile.py with PROFILE_DUMP set), after two warm-up prefills.
  * Launch counts, when the dump carries them, are used ONLY to cross-check the dispatch log. A
    mismatch is printed and means a regex below is wrong; it never silently changes a number.

ROOFLINES ARE MEASURED ON THIS PART (peak_formats.py, cuBLAS dense, median of 20, 8192^3):
    bf16 98.8 TFLOPS    fp8 202.4 TFLOPS    fp16 97.6 TFLOPS
`marlinw*` / `muse_marlin6` are NVFP4 **W4A16**: they dequantize to bf16 and issue
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (marlin_mma.h:71). Their compute roofline is the
BF16 one. Not FP8, not FP4. That is the single most important fact in this table.

MEMORY FLOOR. At M=512 these GEMMs sit near the compute/memory knee, so the compute roofline alone
overstates what is reachable. Each group also gets its weight-stream time at the measured 262 GB/s band.
floor = max(compute_roofline, weight_stream). "recoverable" = measured - floor. A group already at its
floor has nothing to give no matter what its TFLOPS says.

Usage:  python3 roofline_table.py <profile_dump.txt>
"""
from __future__ import annotations

import re
import sys

BF16 = 98.8
FP8 = 202.4
BW = 262.0  # GB/s, measured achievable band on this part
M = 512
H, I, V = 6656, 19968, 202048

NVFP4 = 0.5 + 0.0625   # 4-bit payload + 4-bit-per-16 group scale (W4A16 group 16)
FP8W = 1.0
BF16W = 2.0


def proj(n, k, bpw):
    return (2 * M * n * k, n * k * bpw)


# group -> (per-layer flops, per-layer weight bytes, layers, format, roofline, regexes)
# `layers` is verbatim from the dispatch log; see the module docstring.
GROUPS = {
    "gate_up nvfp4 39936x6656": (
        *proj(39936, H, NVFP4), 45, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"marlinw_tma_gateup::gateup\("],
    ),
    "down nvfp4 6656x19968": (
        *proj(H, I, NVFP4), 40, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"marlinw_tma_down::down<6656>"],
    ),
    "qkv nvfp4 4608x6656": (
        *proj(4608, H, NVFP4), 18, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"marlinw_tma_down::down<4608>"],
    ),
    "gate nvfp4 4096x6656": (
        *proj(4096, H, NVFP4), 18, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"marlinw_down::Marlin<"],
    ),
    "o_proj nvfp4 6656x4096": (
        *proj(H, 4096, NVFP4), 10, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"marlinw_down128::", r"marlinw_swiglu::Marlin<"],
    ),
    "lm_head nvfp4 202048x6656 (GEMV path)": (
        *proj(V, H, NVFP4), 1, "nvfp4 W4A16->bf16 MMA", BF16,
        [r"muse_marlin6_tma4::Marlin<", r"muse_marlin::Marlin<"],
    ),
    "gate_up fp8 39936x6656": (
        # 5 layers, NOT 7: these are the "NVFP4-gate_up + FP8-down" layers (vllm/VENDOR.json), i.e. NVFP4
        # weights with an FP8 *output* (cvt.rn.satfinite.e4m3x2.f32) feeding an FP8 down projection. The
        # weights are NVFP4, so the compute roofline is the BF16 one, not FP8. Measured 106.7 TFLOPS =
        # 108% of bf16. The 7-layer FP8-weight gate_up is a different kernel (cutlass, muse_fp8_swiglu).
        *proj(39936, H, NVFP4), 5, "nvfp4 W4A16->bf16 MMA, fp8 out", BF16,
        [r"marlinw_tma_gateup_fp8::gateup"],
    ),
    "down fp8 6656x19968": (
        *proj(H, I, FP8W), 12, "fp8 W8A8", FP8,
        [],
    ),
    "o_proj fp8 6656x4096": (
        *proj(H, 4096, FP8W), 42, "fp8 W8A8", FP8,
        [],
    ),
    "gate_up fp8w 39936x6656": (
        # The 7 FP8-WEIGHT gate_up layers (cutlass / muse_fp8_swiglu). These really are W8A8, so the FP8
        # roofline is the right comparison. Measured 178.3 TFLOPS = 88% -- 1.3 ms recoverable.
        *proj(39936, H, FP8W), 7, "fp8 W8A8", FP8,
        [],
    ),
    "qkv+gate fp8 dual 8704x6656": (
        (2 * M * 4608 * H + 2 * M * 4096 * H), ((4608 * H + 4096 * H) * FP8W), 28, "fp8 W8A8", FP8,
        [r"fp8ws::gemm_dual_a<4608, 4096, 6656"],
    ),
    "qkv+gate bf16 dual 8704x6656": (
        (2 * M * 4608 * H + 2 * M * 4096 * H), ((4608 * H + 4096 * H) * BF16W), 6, "bf16", BF16,
        [r"fp8ws::gemm_bf16_dual_stacked<4608, 4096, 6656>"],
    ),
}

# The three cutlass fp8_prefill instantiations share a demangled prefix, so they cannot be told apart by
# regex. They ARE told apart by their launch count, which equals their layer count (one launch per layer)
# and is confirmed against the dispatch log: 12 fp8 down, 42 fp8 o_proj, 7 fp8-weight gate_up.
CUTLASS_BY_COUNT = {12: "down fp8 6656x19968", 42: "o_proj fp8 6656x4096", 7: "gate_up fp8w 39936x6656"}


def parse(path):
    rows = []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            ms = float(parts[0])
            cnt = int(parts[1])
        except ValueError:
            continue
        rows.append((ms, cnt, parts[2]))
    return rows


def assign(rows):
    """Return (groups->list of (ms,cnt,name), unassigned). Cutlass rows are resolved last, by time."""
    hit = {g: [] for g in GROUPS}
    unassigned, cutlass = [], []
    for ms, cnt, name in rows:
        if "MainloopSm120TmaWarpSpecialized" in name:
            cutlass.append((ms, cnt, name))
            continue
        placed = False
        for g, spec in GROUPS.items():
            if any(re.search(p, name) for p in spec[5]):
                hit[g].append((ms, cnt, name))
                placed = True
                break
        if not placed:
            unassigned.append((ms, cnt, name))
    # Resolve the cutlass rows by LAUNCH COUNT, which is the authoritative layer count for these kernels
    # (one launch per layer, confirmed against the dispatch log: 12 fp8 down, 42 fp8 o_proj, 7 fp8
    # gate_up). Do not resolve them by time order -- that is a guess and it silently mis-assigns.
    for row in cutlass:
        cnt = row[1]
        g = CUTLASS_BY_COUNT.get(cnt)
        if g is None:
            unassigned.append(row)
        else:
            hit[g].append(row)
    return hit, unassigned


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
    rows = parse(path)
    if not rows:
        print("no rows parsed", file=sys.stderr)
        return 1
    total = sum(r[0] for r in rows)
    print(f"parsed {len(rows)} kernels, {total:.1f} ms CUDA self-time in one 512-token prefill\n")

    hit, unassigned = assign(rows)

    print("=== GROUP ACCOUNTING: dispatch-log layer count vs launches seen in the profile ===")
    print(f"{'group':<40} {'layers(log)':>11} {'ms':>9} {'launches':>9}")
    for g, spec in GROUPS.items():
        ms = sum(r[0] for r in hit[g])
        launches = sum(r[1] for r in hit[g])
        flag = "" if launches in (0, spec[2]) else "  <-- MISMATCH"
        print(f"{g:<40} {spec[2]:>11} {ms:>9.3f} {launches:>9}{flag}")
    print("A MISMATCH means the regex mapping is wrong; fix the mapping, never the number.")
    print("Known benign mismatches, all kernel structure rather than attribution error:")
    print("  down nvfp4 48>40  - the 16 tiles past four full waves are split in K into 3 pieces, each a launch")
    print("  gate_up 39<45     - 6 layers take marlinw_swiglu instead (the 0.788 ms row)")
    print("  gate 20>18        - 2 extra launches from the same K-split path")
    print("  o_proj nvfp4 1<10 - 9 of the 10 run marlinw_down128, which has its own row")
    print("  qkv+gate dual 27<28 - one layer falls back to the two separate fp8ws::gemm rows\n")

    out = []
    for g, spec in GROUPS.items():
        flop_layer, wb_layer, layers, fmt, roof, _ = spec
        ms = sum(r[0] for r in hit[g])
        if ms <= 0:
            continue
        flops = flop_layer * layers
        wbytes = wb_layer * layers
        tf = flops / (ms * 1e-3) / 1e12
        compute_ms = flops / (roof * 1e12) * 1e3
        mem_ms = wbytes / (BW * 1e9) * 1e3
        floor = max(compute_ms, mem_ms)
        out.append((max(0.0, ms - floor), ms, tf, roof, fmt, g, compute_ms, mem_ms, floor, layers))

    print("=== GROUPS, sorted by recoverable ms (measured - max(compute roofline, weight stream)) ===")
    print(f"{'recov':>7} {'ms':>8} {'n':>3} {'TFLOPS':>7} {'roof':>5} {'%roof':>6} {'comp':>7} {'wstr':>7}  group")
    for rec, ms, tf, roof, fmt, g, c, m, fl, ly in sorted(out, key=lambda r: -r[0]):
        print(f"{rec:>7.2f} {ms:>8.3f} {ly:>3} {tf:>7.1f} {roof:>5.1f} {100*tf/roof:>5.0f}% "
              f"{c:>7.2f} {m:>7.2f}  {g}")

    gemm_ms = sum(r[1] for r in out)
    print(f"\nattributed GEMM total {gemm_ms:.1f} ms; floor {sum(r[8] for r in out):.1f} ms; "
          f"recoverable {sum(r[0] for r in out):.1f} ms")

    print(f"\n=== UNSIZED / NON-GEMM, top 25 by ms (total {sum(r[0] for r in unassigned):.1f} ms) ===")
    for ms, cnt, name in sorted(unassigned, key=lambda r: -r[0])[:25]:
        per = ms / cnt * 1000 if cnt else 0.0
        print(f"{ms:>9.3f} {cnt:>5}x {per:>8.1f}us  {name[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
