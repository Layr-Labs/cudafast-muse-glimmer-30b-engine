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

## Where it is used

`vllm/vllm/model_executor/kernels/linear/nvfp4/marlin.py` routes the NVFP4
layers whose codes and scales exceed twice the L2 size (the MLP gate_up and
down projections of this model) through it for calls of 128 rows or more.
Smaller layers fit in L2 already and keep the stock call.

## Checks

At load, `MarlinNvFp4LinearKernel.process_weights_after_loading` compares
marlinw with `ops.marlin_gemm` bitwise on the layer's own weights at 512 and
4096 rows (random input with large and cancelling columns). A layer that
differs, or any error, falls back to the column-sliced schedule of stock
calls in `marlin.py` where that is exact, else to the single stock call.
The engine logs one line with the number of layers on each path.

`MARLINW=0` disables it.
