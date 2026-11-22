// SPDX-License-Identifier: Apache-2.0
// decsc 5-bit variant: the one-row tile-major Marlin NVFP4 GEMV of
// harness/kernels/decode reading 5-bit block-scale codes (a per-layer table of
// the 31 most frequent scale bytes, code 31 = escape) with per-stage escape
// entries, for the 128-column configuration. The template is
// decode/marlin/marlin_template.h compiled with MARLIN_SC5: the stage's scale
// bytes are rebuilt in shared memory before any use, exactly as the stock
// cp.async of the scale tile would leave them; everything else is the decode
// kernel's, so the output bits are the same.
#define MARLIN_NAMESPACE_NAME muse_marlin_sc5
#define MARLIN_SC5 1
#include "marlin/marlin_template.h"

extern "C" int muse_marlin_gemv_sc5(const int4* input, const int4* weight, int4* output,
                                    int4* partials, const int4* codes, const uint32_t* table,
                                    const float* global_scale, int* locks, int n, int k,
                                    int esc, int blocks, int device, cudaStream_t stream) {
    if (n % 128 || k % 128 || esc < 0 || esc > 8) return (int)cudaErrorInvalidValue;
    int shared = 0;
    auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    auto kernel = muse_marlin_sc5::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    kernel<<<blocks, 256, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, nullptr, global_scale,
        nullptr, nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared,
        codes, table, esc);
    return cudaGetLastError();
}
