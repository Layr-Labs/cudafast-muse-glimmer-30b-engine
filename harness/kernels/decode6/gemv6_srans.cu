// SPDX-License-Identifier: Apache-2.0
// rANS-coded block-scale M = 1 NVFP4 decode GEMV for both the 336-CTA fixed gate/up grid
// (RANS_HDR = 2688 B, RANS_CAP = 704 B) and 48-CTA stream-K down_proj (6656x19968, RANS_HDR = 3136 B).
// Arithmetic is untouched; layers are admitted only after bitwise comparison with the six-bit kernel.
#define MARLIN_NAMESPACE_NAME muse_marlin6_srans
#define MUSE_SCALE_RANS
#include "marlin/marlin_template.h"

namespace {
__global__ void srans_hdr_pass1_kernel(
    const uint8_t* __restrict__ tiled_scales, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    uint32_t* __restrict__ stage_rows16, uint64_t* __restrict__ stage_cw) {
  __shared__ uint32_t sf[64], sc[64];
  __shared__ uint8_t cw_sh[8];
  if (threadIdx.x < 64) { sf[threadIdx.x] = enc_f[threadIdx.x]; sc[threadIdx.x] = enc_c[threadIdx.x]; }
  if (threadIdx.x < 8) cw_sh[threadIdx.x] = 0;
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t x = 65536u;

  for (int s = s1 - 1; s >= s0; --s) {
    const int col = stage_col_k[2 * s];
    const int kt = stage_col_k[2 * s + 1];
    const uint8_t* raw = tiled_scales + (static_cast<size_t>(col) * k_tiles + kt) * 1024 + 128 * w + lane;
    uint32_t wcnt = 0;
#pragma unroll
    for (int m = 3; m >= 0; --m) {
      const uint32_t sy = raw[32 * m];
      const uint32_t fs = sf[sy], cs = sc[sy];
      const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
      wcnt += __popc(__ballot_sync(0xffffffffu, emit));
      if (emit) x >>= 16;
      x = ((x / fs) << 10) + (x % fs) + cs;
    }
    if (lane == 0) cw_sh[w] = static_cast<uint8_t>(wcnt);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t tot = 0;
      uint64_t packed_cw = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tot += cw_sh[i];
        packed_cw |= static_cast<uint64_t>(cw_sh[i]) << (8 * i);
      }
      stage_cw[s] = packed_cw;
      stage_rows16[s] = (4u + tot + 7u) / 8u;
    }
    __syncthreads();
  }
}

__global__ void srans_hdr_pass2_kernel(
    const uint8_t* __restrict__ tiled_scales, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    const uint64_t* __restrict__ stage_cw,
    int hdr_bytes, int b0_off, int cap_bytes,
    uint8_t* __restrict__ hdr_out, uint8_t* __restrict__ var_out) {
  __shared__ uint32_t sf[64], sc[64];
  if (threadIdx.x < 64) { sf[threadIdx.x] = enc_f[threadIdx.x]; sc[threadIdx.x] = enc_c[threadIdx.x]; }
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t lt = (1u << lane) - 1u;
  uint8_t* my_hdr = hdr_out + static_cast<size_t>(b) * hdr_bytes;
  const uint32_t* my_offs = reinterpret_cast<const uint32_t*>(my_hdr + 1024);
  uint32_t x = 65536u;

  for (int s = s1 - 1; s >= s0; --s) {
    const int local_s = s - s0;
    const int col = stage_col_k[2 * s];
    const int kt = stage_col_k[2 * s + 1];
    const uint8_t* raw = tiled_scales + (static_cast<size_t>(col) * k_tiles + kt) * 1024 + 128 * w + lane;
    const uint64_t packed_cw = stage_cw[s];
    uint8_t* blk = (local_s < 2)
                       ? (my_hdr + b0_off + local_s * cap_bytes)
                       : (var_out + static_cast<size_t>(my_offs[local_s]) * 16);
    if (threadIdx.x == 0) *reinterpret_cast<uint64_t*>(blk) = packed_cw;
    uint32_t start = 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i <= w) start += static_cast<uint32_t>((packed_cw >> (8 * i)) & 255u);
    }
    uint16_t* words = reinterpret_cast<uint16_t*>(blk);
    uint32_t wpos = start;
#pragma unroll
    for (int m = 3; m >= 0; --m) {
      const uint32_t sy = raw[32 * m];
      const uint32_t fs = sf[sy], cs = sc[sy];
      const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
      const uint32_t msk = __ballot_sync(0xffffffffu, emit);
      wpos -= __popc(msk);
      if (emit) {
        words[wpos + __popc(msk & lt)] = static_cast<uint16_t>(x & 0xFFFFu);
        x >>= 16;
      }
      x = ((x / fs) << 10) + (x % fs) + cs;
    }
  }
  reinterpret_cast<uint32_t*>(my_hdr)[threadIdx.x] = x;
}
}  // namespace

extern "C" int muse_marlin6_gate_up_srans(const int4* input, const int4* weight, int4* output,
                                          int4* partials, const int4* coded, const int4* palette,
                                          const float* global_scale, int* locks, int n, int k, int blocks,
                                          int device, cudaStream_t stream) {
  int shared = 0;
  auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut = reinterpret_cast<const int*>(base);
  const auto* headers = reinterpret_cast<const int4*>(base + 4096);
  const auto* var = reinterpret_cast<const int4*>(base + 4096 + 336 * 2688);
  auto kernel = muse_marlin6_srans::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false, true, true>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<336, 256, shared, stream>>>(input, weight, output, partials, nullptr, nullptr, headers, global_scale,
                                       var, lut, k / 16, 1, n, k, k, locks, false, false, true, shared, palette);
  return cudaGetLastError();
}

extern "C" int muse_marlin6_down_srans(const int4* input, const int4* weight, int4* output,
                                       int4* partials, const int4* coded, const int4* palette,
                                       const float* global_scale, int* locks, int n, int k, int blocks,
                                       int device, cudaStream_t stream) {
  int shared = 0;
  auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (n != 6656 || k != 19968 || blocks != 48) return cudaErrorInvalidValue;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut = reinterpret_cast<const int*>(base);
  const auto* headers = reinterpret_cast<const int4*>(base + 4096);
  const auto* var = reinterpret_cast<const int4*>(base + 4096 + 48 * 3136);
  auto kernel = muse_marlin6_srans::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false, true, false>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<48, 256, shared, stream>>>(input, weight, output, partials, nullptr, nullptr, headers, global_scale,
                                      var, lut, k / 16, 1, n, k, k, locks, false, false, true, shared, palette);
  return cudaGetLastError();
}

extern "C" int muse_srans_hdr_pass1(const uint8_t* sym_tiled, int k_tiles,
                                    const int* stage_col_k, const int* blk_stage_off,
                                    const uint32_t* enc_f, const uint32_t* enc_c,
                                    uint32_t* stage_rows16, uint64_t* stage_cw,
                                    int num_blks, cudaStream_t stream) {
  srans_hdr_pass1_kernel<<<num_blks, 256, 0, stream>>>(
      sym_tiled, k_tiles, stage_col_k, blk_stage_off, enc_f, enc_c, stage_rows16, stage_cw);
  return cudaGetLastError();
}

extern "C" int muse_srans_hdr_pass2(const uint8_t* sym_tiled, int k_tiles,
                                    const int* stage_col_k, const int* blk_stage_off,
                                    const uint32_t* enc_f, const uint32_t* enc_c,
                                    const uint64_t* stage_cw,
                                    int hdr_bytes, int b0_off, int cap_bytes,
                                    uint8_t* hdr_out, uint8_t* var_out,
                                    int num_blks, cudaStream_t stream) {
  srans_hdr_pass2_kernel<<<num_blks, 256, 0, stream>>>(
      sym_tiled, k_tiles, stage_col_k, blk_stage_off, enc_f, enc_c, stage_cw,
      hdr_bytes, b0_off, cap_bytes, hdr_out, var_out);
  return cudaGetLastError();
}
