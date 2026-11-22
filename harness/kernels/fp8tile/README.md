# fp8tile

The FP8 w8a8 GEMM that vLLM v0.27.1's `ops.cutlass_scaled_mm` runs on SM120 for
M <= 16 (`sm120_fp8_config_M16`), instantiated with another tile shape:
16x32x512 instead of 16x64x128. Twice as many output tiles spread the weight
stream more evenly over the SMs, and the longer K tile issues fewer, larger
loads per pipeline stage.

## Why the output is the same

Everything but the tile is vLLM's: e4m3 A row-major, e4m3 B column-major (the
layer's own `[K, N]` operand, nothing is copied), fp32 accumulation,
`KernelTmaWarpSpecializedPingpong`, cluster 1x1x1, `EpilogueScheduleAuto`, the
default persistent tile scheduler (no split-K), and the `ScaledEpilogue` tree
`out = bf16_rn(scale_a * fp32_rn(scale_b * acc))`.

- The SM120 pingpong collective builder uses a 2x2x1 warp layout (K is never
  split across warps) and a permutation tile of `min(M,128) x min(N,32) x 32`,
  so a 32-wide tile has exactly the tiled MMA of the 64-wide one; every shape
  runs `mma.sync ... kind::f8f6f4.m16n8k32 ... e4m3.e4m3.f32`.
- The mainloop clears the accumulators, then for every k tile in ascending
  order and every k32 block of it in ascending order issues one MMA into each
  output element's accumulator. The N tile width only decides which CTA owns an
  output column; a 512-deep K tile is the same ascending sequence of k32 blocks.
- The epilogue is element-wise.

So each output element sees the same instructions on the same operands in the
same order. `prepare` still compares the kernel with the stock one bit for bit
on every layer's own weight at load time (M = 1, 2 and 16; random, clipped,
spiked and all-bit-pattern activations, NaN included) and keeps the stock path
for any layer that differs or whose shape is not a whole number of tiles. The
vLLM hook logs `fp8tile: <n> of <m> FP8 linear layers admitted` once.

## Files

- `csrc/fp8tile_gemm.cuh`: the GEMM template and the tile configurations;
  `csrc/fp8tile_cfg.cu`: one configuration per object; `csrc/fp8tile.cu`: the C
  API (`fp8tile_gemm`, `fp8tile_workspace`, `fp8tile_configs`).
- `py/__init__.py`: ctypes loader, `gemm`, `prepare`. `FP8TILE=0` disables it.
- `tests/fp8tile_exact.py`: `exact` (every FP8 module of the checkpoint, several
  M, five input kinds, every built configuration, integer comparison of the
  bf16 bits), `bench` (CUDA-graph timing with cold weights) and `op` (the custom
  op under a dynamic `torch.compile` and CUDA graph replay).

## Build

`build.sh` compiles the configurations in `FP8TILE_CONFIGS` (default: the
shipped 16x32x512) for `sm_121a` with `-O3` and no fast-math, against the
CUTLASS headers bundled in the pinned image, and copies the Python package next
to `libfp8tile.so`.
