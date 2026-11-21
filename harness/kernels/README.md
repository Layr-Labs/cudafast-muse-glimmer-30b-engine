The Marlin headers are from vLLM commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, under the original Apache-2.0
license. `marlin_template.h` changes only the assignment of whole-K output
slices to CTAs and adds the original logical grid size as an argument. The
dequantization, tensor-core instructions, local reduction and split-K tail
reduction are unchanged.

The extension covers single-row BF16 x NVFP4 group-16 decode on SM121.
It uses the stock tile selector and keeps all other shapes on stock Marlin.
The 128-column tile uses 45,056 bytes of shared memory; the 64-column tile
uses 26,624 bytes. Independent whole-K slices receive separate CTAs. Tail
CTAs retain the stock logical grid and ordered FP32 reduction.

`build.sh` writes its compiler output and importable module only to
`KERNELS_OUT`. It can compile in the pinned image without GPU access.

Validation on 2026-11-21: compiled in the pinned v0.27.1 ARM64 image without
`--gpus`, against Torch 2.13.0+cu130 for SM121a. Both tile specializations use
118 registers per thread and have no register spills. Library loading and
Meta registration passed without a GPU. Exact GPU equality and performance
measurements are pending an exclusive GPU slot.
