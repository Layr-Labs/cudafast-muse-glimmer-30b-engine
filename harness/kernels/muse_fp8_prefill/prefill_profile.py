#!/usr/bin/env python3
"""Per-kernel PREFILL profile. Runs INSIDE the pinned container, same env as bench/run_bench.sh.

WHY THERE ARE TWO WINDOWS AND A SUBTRACTION.
`VllmBackend.start()` (harness/vllm_resident/backends.py:361) loops `_collect()` until a token is
emitted, and `_collect()` calls `engine.step()`. So `start()` returns after **prefill plus at least one
decode step**, and a profile of `start()` alone is NOT a prefill profile. Profiling it as one is what
produced the "prefill lm_head at 30.5 TFLOPS" reading, which was really the decode GEMV (board #143).

So this tool takes two windows:
    A = start(prompt)   -> prefill + decode@512   (first decode after a full prefill: cold L2/TLB)
    B = step()          -> decode@513             (same request, immediately after, same cold state)
and reports A - B as the prefill estimate. The subtraction is exact to the extent decode@512 equals
decode@513, which is a one-token difference in an otherwise identical engine state; B is taken
immediately after A on the same live request so that both share the cold-cache condition. A kernel that
appears only in B comes out negative and is clamped to 0 and flagged, never silently dropped.

Launch counts are carried through: a per-layer kernel's call count is the only honest way to attribute
it to a number of layers. Do not guess layer counts.

Read-only: it changes nothing and is not part of the scored path.
    PROFILE_DUMP=<path> writes "<ms>\\t<count>\\t<name>" for the A-B difference.
    PROFILE_DUMP_A / PROFILE_DUMP_B write the raw windows for anyone who wants to redo the subtraction.
"""
import json
import os
import subprocess
import sys
import time

import torch

ENG = os.environ["ENG"]
sys.path.insert(0, os.path.join(ENG, "harness"))
from vllm_resident.backends import VllmBackend, VllmConfig  # noqa: E402


def demangle(names):
    try:
        p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, timeout=120)
        return dict(zip(names, p.stdout.splitlines()))
    except Exception:  # noqa: BLE001
        return {n: n for n in names}


def capture(fn):
    """Run fn() under the CUDA profiler and return {key: (self_us, count)}."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0)
        if us and us > 0:
            out[e.key] = (us, int(getattr(e, "count", 0) or 0))
    return out


def main() -> int:
    cfg = VllmConfig.from_env()
    t0 = time.perf_counter()
    b = VllmBackend(cfg)
    print(f"BOOT {time.perf_counter() - t0:.1f}s", flush=True)
    gold = json.load(open(os.path.join(ENG, "correctness_prompts/public-longcopy-gate-english-512.golden.json")))
    prompt = gold["cases"][0]["prompt_tokens"]

    # Warm up so the windows contain only steady-state kernels (no allocator growth, no one-time setup).
    for _ in range(2):
        b.start(prompt, top_logits=False)
        torch.cuda.synchronize()
        b.abort()
    torch.cuda.synchronize()
    time.sleep(2)

    t = time.perf_counter()
    b.start(prompt, top_logits=False)
    torch.cuda.synchronize()
    wall_a = (time.perf_counter() - t) * 1000
    b.abort()
    print(f"WALL start() {wall_a:.1f} ms  (prefill + 1 decode step)", flush=True)
    time.sleep(2)

    # Window A: start() = prefill + decode@512.
    a = capture(lambda: b.start(prompt, top_logits=False))
    # Window B: step() on the SAME live request = decode@513, same cold state.
    bb = capture(lambda: b.step())
    b.abort()
    time.sleep(1)

    sum_a = sum(v[0] for v in a.values())
    sum_b = sum(v[0] for v in bb.values())
    print(f"\nwindow A (start) {sum_a/1000:.1f} ms CUDA across {len(a)} kernels")
    print(f"window B (step)  {sum_b/1000:.1f} ms CUDA across {len(bb)} kernels")
    print(f"A - B            {(sum_a-sum_b)/1000:.1f} ms  <- the prefill estimate\n")

    keys = set(a) | set(bb)
    dm = demangle(list(keys))
    diff = {}
    for k in keys:
        ua, ca = a.get(k, (0.0, 0))
        ub, cb = bb.get(k, (0.0, 0))
        diff[k] = (ua - ub, ca - cb, dm.get(k, k))

    neg = [(k, v) for k, v in diff.items() if v[0] < 0]
    if neg:
        print(f"{len(neg)} kernels are larger in B than in A (decode-only); clamped to 0:")
        for k, v in sorted(neg, key=lambda kv: kv[1][0])[:8]:
            print(f"  {v[0]/1000:>9.3f} ms  {v[2][:100]}")
        print()

    prefill = {k: (max(0.0, v[0]), max(0, v[1]), v[2]) for k, v in diff.items() if v[0] > 0}
    total = sum(v[0] for v in prefill.values())
    print(f"=== PREFILL-ONLY kernels, A - B: {len(prefill)} distinct, {total/1000:.1f} ms ===")
    print(f"{'ms':>9} {'%':>6} {'n':>5}  kernel")
    for k, (us, cnt, name) in sorted(prefill.items(), key=lambda kv: -kv[1][0])[:70]:
        print(f"{us/1000:>9.3f} {100*us/total:>6.2f} {cnt:>5}  {name[:190]}")

    print("\n=== every row whose name contains 'Marlin<' but NOT 'muse_marlin6_tma4' (ms, count) ===")
    for k, (us, cnt, name) in sorted(prefill.items(), key=lambda kv: -kv[1][0]):
        if "Marlin<" in name and "muse_marlin6_tma4" not in name:
            print(f"{us/1000:>9.3f} {cnt:>5}x  {name[:170]}")

    for env, rows in (("PROFILE_DUMP", prefill), ("PROFILE_DUMP_A", a), ("PROFILE_DUMP_B", bb)):
        p = os.environ.get(env, "")
        if p:
            with open(p, "w") as fh:
                for key, v in sorted(rows.items(), key=lambda kv: -kv[1][0]):
                    name = v[2] if len(v) > 2 else dm.get(key, key)
                    fh.write(f"{v[0]/1000:.6f}\t{v[1]}\t{name}\n")
            print(f"\n{env}: {len(rows)} rows -> {p}", flush=True)
    b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
