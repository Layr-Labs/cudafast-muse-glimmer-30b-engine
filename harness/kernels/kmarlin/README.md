# kmarlin: Marlin-exact NVFP4 W4A16 GEMM for M = 1 (decode)

Replaces vLLM v0.27.1's dense Marlin kernel (`ops.marlin_gemm`, reached through
`MarlinNvFp4LinearKernel.apply_weights`) for the Muse-Glimmer-30B NVFP4 W4A16 linears when a
call has one row: MLP gate_up (6656 -> 39936), down (19968 -> 6656) and lm_head
(6656 -> 202048). Every bf16 output element is meant to be bit-identical to stock Marlin; it
reads the Marlin-format tensors the layer already holds (no repack, no extra memory).
Other calls (prefill M = 512, other shapes, bias, non-bf16) run stock Marlin through the same
custom op. `KMARLIN=0` disables it.

Credit: a port of "marlinx" by dadon (Nemotron CUDA track, submission 100549d4): its
schedule, dequant, segment folds and register ring. New here: Marlin's 64-wide
{128, 64, 128} configuration (lm_head, N % 128 != 0), a torch custom op so the dispatch
survives torch.compile tracing of the dense layers, and switchable value-neutral hints
(`KMARLIN_VARIANT`, default 0; bit 0 = L2 evict_first policy on the weight and scale stream, bit 1 =
prefetch.global.L2 look-ahead, bit 2 = programmatic dependent launch).

Stock Marlin at M = 1 on 48 SMs: gate_up and down run {thread_k 128, thread_n 128, 256 thr},
lm_head runs {128, 64, 128}; fp32 global reduce, no atomics. Per output element: 4 warp-row
fp32 mma chains over k16 blocks 8*tau + 2w, 8*tau + 2w + 1; P = acc0 + (acc1 + (acc3 + acc2));
stripe segments of each column folded from the highest-K segment up (R = P + R);
out = bf16_rn(R * global_scale). Partition per shape: see tests/partition_check.py.

Files: `csrc/kmarlin.cu` (kernel, C ABI), `py/__init__.py` (ctypes loader, custom op,
prepare/apply), `build.sh` (libkmarlin.so into $KERNELS_OUT/kmarlin).
