# marlinw

`ops.marlin_gemm` for NVFP4 W4A16 linears (bf16 activations, fe2m1f weights,
fe4m3fn group-16 scales, fp32 global scale) at 128 rows and more, with the same
output bits and a weight-friendly schedule.

## What stock Marlin does at prefill widths (vLLM v0.27.1)

`marlin_mm` (marlin.cu) splits M into launches of up to `max_par` 64-row chunks
(`max_par` = 16, or 128 when N <= 4096) and picks the first valid large-batch
thread configuration, `{thread_k 64, thread_n 256, 256 threads}` for every
shape here (4 stages, `thread_m_blocks` 4). One launch has one block per SM. In
the kernel (`marlin_template.h`, `init_slice`) the work units are
(chunk, 256-column tile) pairs in chunk-major order `t = chunk * n_tiles + col`;
block b computes units b, b + #SMs, ... over the full K, and the last `part2`
units are cut along K into stripes of `iters` 64-deep k-tiles, whose partial
sums meet in an fp32 scratch buffer under a lock, in `slice_idx` order.

A weight tile is therefore fetched once per chunk, and for a wide matrix the
fetches of the same tile are a whole matrix apart in time: at 512 rows the
39936 x 6656 matrix (150 MB of codes and scales) is read from DRAM about eight
times per call.

## What every output element is

For row r and column n, chunk `p = r / 64`, column tile `c = n / 256`:

1. weights: the fe2m1 code placed in a bf16 (`dequant<.., kFE2M1f, true>`)
   times the S0E5M3 group scale converted to bf16 (`__hmul2`; exact);
2. in a k-tile, warp row 0 issues `mma.sync.m16n8k16` (bf16 x bf16 -> fp32) for
   k16 blocks 0 and 1, warp row 1 for blocks 2 and 3, so the element has two
   accumulators, `A0` over blocks {4t, 4t+1} and `A1` over {4t+2, 4t+3}, t
   ascending over the slice, each from +0;
3. `thread_block_reduce`: `P = A0 + A1`;
4. if the tile is cut along K: the slices' `P` meet in the fp32 scratch in
   `slice_idx` order (highest-K slice first, lowest-K slice last);
5. `write_result`: `bf16_rn(R * global_scale)`.

Which slices a (chunk, column tile) has, and their order, is a function of
(rows of the launch, N, K, #SMs, configuration) only.

## What marlinw changes

Only the assignment of slices to blocks. The host (`py/__init__.py`, `plan`)
replays the stock host loop and `init_slice` for the call and lists every
slice with its stock chunk, column tile, k range, `slice_idx` and
`slice_count`. The kernel (`csrc/marlinw_template.h`) is `marlin_template.h`
with the partition code replaced by a per-block read of that list; the code
that loads, dequantizes, multiplies, reduces inside the block, reduces through
the scratch under the lock protocol and writes the result is unchanged and is
instantiated with the stock template arguments. Unreduced (full-K) tiles are
handed out column-major within groups of 8 chunks, so the chunks that use a
weight tile run at the same time and share its L2 lines; cut tiles keep their
stock slices, and every block runs its cut slices last, by increasing
`slice_idx`, so each lock wait ends (all blocks are resident, as stock
assumes). Rows of a last partial chunk, and any launch with another
configuration, go through `ops.marlin_gemm` on those rows.

Same instructions on the same operands in the same order per element, same
reduction tree, same epilogue: the same bits. Built without `--use_fast_math`,
as the stock kernels are.

## Gated-MLP epilogue (`silu_mul`)

For the merged MLP projection (weight `[gate | up]`, N = 2D) the model's
compiled graph follows `ops.marlin_gemm` with one Inductor (Triton) kernel for
`SiluAndMul`: it reads the bf16 gate and up values, computes in fp32
`g / (exp(-g) + 1) * u` with libdevice's `expf` (flush-to-zero variant, its
final scaling contracted with the `+ 1` into one fma), `div.full.f32` and one
round to bf16 at the end (the bf16 cast of `silu(g)` in the graph is not
materialised). `csrc/silu_mul.cuh` issues that instruction sequence verbatim.

`marlinw_gemm(..., silu_mul = 1)` instantiates the same kernel with
`silu_mul = true`: the work list holds whole-K tiles only, and each block runs
the gate tile (chunk, c) and then the up tile (chunk, c + N / 512). The gate
tile's epilogue is unchanged and stores its bf16 values at the output position;
the up tile's epilogue, after the unchanged rounding of its own values, makes
every thread read back exactly the elements it stored there and write
`silu_mul(gate, up)` instead. So the output equals the stock GEMM followed by
the Inductor kernel, bit for bit, and the [M, 2D] intermediate never reaches
DRAM. Row counts whose stock partition cuts a gate_up tile along K (M not a
multiple of 256) use the unfused schedule followed by `marlinw_silu_mul`, the
standalone kernel with the same arithmetic.

## Where it is used

`vllm/vllm/model_executor/kernels/linear/nvfp4/marlin.py` routes the NVFP4
layers whose codes and scales exceed twice the L2 size (the MLP gate_up and
down projections of this model) through it for calls of 128 rows or more.
Smaller layers fit in L2 already and keep the stock call.

## Other CTA shapes (`csrc/marlinw_sched.h`)

`MarlinS` is the same main loop restricted to this type combination, with the
CTA shape and pipeline as template parameters: CTA tile 64 rows x
16 * `thread_n_blocks` columns, warps of 64 x 64 (or 64 x 32) outputs, the
cp.async depth, and the CTAs per SM its register budget is set for. Per
element nothing changes: the k-tile is 64 deep, warp row r issues the mma for
k16 blocks 2r and 2r + 1 of every k-tile into its own fp32 accumulator from +0,
the B operand is the same `dequant` of the same code word times the same
`dequant_fp8_scales` scale, the two rows' accumulators are added once in fp32,
k-split slices meet in the fp32 scratch in `slice_idx` order, and the epilogue
(and the fused SiLU-and-mul) is Marlin's. A 128-column CTA runs the column
halves of the stock 256-column slices (`plan(..., tile_n=128)`): the same chunk,
k range, `slice_idx` and `slice_count`, one scratch slot per half. Marlin's
code and scale layouts are local to 64-column groups, so a 128-column tile reads
them unchanged.

The configurations built in are `MARLINW_SCHED_LIST` in `csrc/marlinw.cu`, and
`SCHED_SHAPES` assigns them per (N, K). The gate_up projection runs 64 x 128
tiles (4 warps of 64 x 64, 3 stages, two CTAs per SM): each CTA's barriers,
waits and tile changes overlap the other CTA's mma. The down projection runs
64 x 256 tiles (the stock slices unchanged) with 5 stages and the next k16
step's codes dequantized during the current step's mma. A configuration is used
only where the occupancy query reports as many resident CTAs per SM as its plan
assumes (k-split slices wait on each other) and only for layers admitted by the
checks below. `MARLINW_SCHED=0` disables them.

## Checks

At load, `MarlinNvFp4LinearKernel.process_weights_after_loading` compares
marlinw with `ops.marlin_gemm` bitwise on the layer's own weights at 512 and
4096 rows (random input with large and cancelling columns). A layer that
differs, or any error, falls back to the column-sliced schedule of stock
calls in `marlin.py` where that is exact, else to the single stock call.
The engine logs one line with the number of layers on each path.

The fused SiLU-and-mul path is admitted per merged layer by comparing it, at
512, 4096 and 640 rows, with the single stock call followed by
`marlinw_silu_mul`, after `marlinw_silu_mul` itself was compared once with a
Triton copy of the Inductor kernel on every bf16 gate value.

A layer listed in `SCHED_SHAPES` is checked on the MarlinS kernel first, at
512, 4096 and 640 rows (640 rows give the gate_up shape k-split tiles), and its
fused op the same way; a layer that differs keeps the marlinw kernel and its
checks. The log line `Marlin NVFP4 prompt GEMM sched kernel: ...` counts them.

`MARLINW=0` disables it; `MARLINW_SILU=0` disables only the fused epilogue;
`MARLINW_SCHED=0` only the MarlinS kernel.
