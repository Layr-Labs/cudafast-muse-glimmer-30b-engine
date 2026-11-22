// SPDX-License-Identifier: Apache-2.0
// decsc: the one-row tile-major Marlin NVFP4 GEMV of harness/kernels/decode
// (gemv.cu: stock Marlin body and partition, weights and scales stored
// tile-major) reading the block scales from a lossless 6-bit code: per layer a
// table of at most 64 distinct scale bytes and one 6-bit index per scale, in
// the same tile-major order. The template is decode/marlin/marlin_template.h
// compiled with MARLIN_SC6: a stage's codes are cp.async'd to shared memory
// and, right after the stage lands, rebuilt by table lookup into exactly the
// bytes the stock cp.async of the scale tile would have put there. Everything
// else (dequantization, MMA chain, partition, fp32 reduction, epilogue) is the
// decode kernel's, so the output bits are the same.
#define MARLIN_NAMESPACE_NAME muse_marlin_sc6
#define MARLIN_SC6 1
#include "marlin/marlin_template.h"

namespace {
template <int TileN, int Threads>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* codes, const uint32_t* table,
                   const float* global_scale, int* locks, int n, int k,
                   int blocks, int shared, cudaStream_t stream) {
    auto kernel = muse_marlin_sc6::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false>;
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    kernel<<<blocks, Threads, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, nullptr, global_scale,
        nullptr, nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared,
        codes, table);
    return cudaGetLastError();
}
}  // namespace

extern "C" int muse_marlin_gemv_sc6(const int4* input, const int4* weight, int4* output,
                                    int4* partials, const int4* codes, const uint32_t* table,
                                    const float* global_scale, int* locks, int n, int k,
                                    int blocks, int device, cudaStream_t stream) {
    int shared = 0;
    auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    if (n % 128 == 0)
        return launch<128, 256>(input, weight, output, partials, codes, table, global_scale,
                                locks, n, k, blocks, shared, stream);
    return launch<64, 128>(input, weight, output, partials, codes, table, global_scale,
                           locks, n, k, blocks, shared, stream);
}
