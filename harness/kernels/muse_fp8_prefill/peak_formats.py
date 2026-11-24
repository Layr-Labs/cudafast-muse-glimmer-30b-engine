#!/usr/bin/env python3
"""Dense GEMM peak on this part, per numeric format. Runs INSIDE the pinned container. Read-only, ~20 s of GPU.

The question it settles: `marlinw` is NVFP4 **W4A16** -- it dequantizes to bf16 and issues `mma.sync.m16n8k16`
(bf16 x bf16 -> fp32), per its own README. So its compute roofline is the BF16 tensor-core roofline, not the FP4 one.
If bf16 peak here is about half the fp8 peak, then an NVFP4 kernel at 104 TFLOPS next to an FP8 kernel at 203 TFLOPS
is at the SAME fraction of its own roofline, and the "2x gap" is the format, not a tuning bug.
"""
import statistics
import sys

import torch


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main() -> int:
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)}  SMs {torch.cuda.get_device_properties(0).multi_processor_count}  "
          f"torch {torch.__version__}")
    # Large square shapes: the point is the sustained MMA rate, not any particular model shape.
    for n in (4096, 8192):
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        ms = bench(lambda: torch.matmul(a, b))
        print(f"bf16  {n}x{n}x{n}: {ms:8.3f} ms  {2*n**3/(ms*1e-3)/1e12:7.1f} TFLOPS")

    # fp8 e4m3 x e4m3 -> bf16 via the scaled-mm path (same instruction class the FP8 prefill kernels use).
    for n in (4096, 8192):
        a = (torch.randn(n, n, device=dev) * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(n, n, device=dev) * 0.1).to(torch.float8_e4m3fn)
        sa = torch.tensor([1.0], device=dev)
        sb = torch.tensor([1.0], device=dev)
        try:
            ms = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            print(f"fp8   {n}x{n}x{n}: {ms:8.3f} ms  {2*n**3/(ms*1e-3)/1e12:7.1f} TFLOPS  (_scaled_mm)")
        except Exception as e:  # noqa: BLE001
            print(f"fp8   {n}x{n}x{n}: unavailable ({type(e).__name__}: {str(e)[:120]})")

    # fp16 as a cross-check on the bf16 number (same tensor-core class, so it should match).
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    ms = bench(lambda: torch.matmul(a, b))
    print(f"fp16  8192sq: {ms:8.3f} ms  {2*8192**3/(ms*1e-3)/1e12:7.1f} TFLOPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
