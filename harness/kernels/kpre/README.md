kpre: vLLM v0.27.1 Marlin NVFP4 W4A16 kernel (vendored from csrc/libtorch_stable/quantization/marlin,
Apache-2.0, Copyright Marlin.2024 Elias Frantar, modified by Neural Magic / vLLM) with one patch under
KPRE_REMAP in marlin/marlin_template.h: part-1 tiles are visited column-major over the 64-row M slabs.
Used for prefill calls (M multiple of 64, single stock launch) of the MLP gate_up and down shapes.

Wiring: hook.diff (on top of kmarlin's hook) calls kpre.prepare at weight load and kpre.apply first in
MarlinNvFp4LinearKernel.apply_weights; the torch custom op kpre::nvfp4_gemm keeps the M branch opaque to
torch.compile and routes ineligible M to kmarlin's op (layers kmarlin prepared) or stock Marlin.
build: harness/kernels/build.sh runs kmarlin (if present) then kpre. KPRE_BUILD_REF=1 also builds the
test-only libkpre_ref.so (stock slab-major order). Tests: tests/kpre_exact.py (bit-identity vs
ops.marlin_gemm at the device SM count and remap vs stock order at grid 48), tests/kpre_op.py
(eager, torch.compile dynamic fullgraph, CUDA graph replay vs stock).
