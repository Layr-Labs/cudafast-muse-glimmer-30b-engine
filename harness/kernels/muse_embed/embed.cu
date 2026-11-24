// SPDX-License-Identifier: Apache-2.0
// prefill3: the token embedding + weightless embedding RMSNorm of MuseGlimmerModel.embed_input_ids in two passes
// instead of eight eager kernels (index copy, gather, to-fp32 copy, pow, mean, add, rsqrt, mul, to-bf16 copy, and
// the copy into inputs_embeds). The reduction stays ATen's own mean kernel on the same fp32 tensor:
//   square: x2[r, c] = x * x with x = float(table[ids[r], c])        (ATen pow(x, 2) on float is base * base)
//   [ATen]  inv = rsqrt(mean(x2, -1, keepdim=True) + eps)              (unchanged ops)
//   scale : out[r, c] = bf16_rn(x * inv[r])                            (ATen mul on float, then type_as bf16)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 128;

template <typename Id>
__global__ void __launch_bounds__(kThreads) embed_square(const Id* __restrict__ ids, const uint4* __restrict__ table,
                                                         float4* __restrict__ x2, int vec_per_row) {
  const int row = blockIdx.y;
  const int v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= vec_per_row) return;
  const long long id = static_cast<long long>(ids[row]);
  const uint4 raw = table[id * vec_per_row + v];
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  float f[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
  float4* dst = x2 + (static_cast<size_t>(row) * vec_per_row + v) * 2;
  dst[0] = make_float4(__fmul_rn(f[0], f[0]), __fmul_rn(f[1], f[1]), __fmul_rn(f[2], f[2]), __fmul_rn(f[3], f[3]));
  dst[1] = make_float4(__fmul_rn(f[4], f[4]), __fmul_rn(f[5], f[5]), __fmul_rn(f[6], f[6]), __fmul_rn(f[7], f[7]));
}

__device__ __forceinline__ unsigned bf16_rne(float a) {
  // c10::BFloat16 round_to_nearest_even (finite inputs; NaN -> 0x7fc0 as c10)
  const unsigned u = __float_as_uint(a);
  if (a != a) return 0x7fc0u;
  return (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

template <typename Id>
__global__ void __launch_bounds__(kThreads) embed_scale(const Id* __restrict__ ids, const uint4* __restrict__ table,
                                                        const float* __restrict__ inv, uint4* __restrict__ out,
                                                        int vec_per_row) {
  const int row = blockIdx.y;
  const int v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= vec_per_row) return;
  const long long id = static_cast<long long>(ids[row]);
  const uint4 raw = table[id * vec_per_row + v];
  const float r = inv[row];
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  unsigned o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned lo = bf16_rne(__fmul_rn(__uint_as_float(w[i] << 16), r));
    const unsigned hi = bf16_rne(__fmul_rn(__uint_as_float(w[i] & 0xffff0000u), r));
    o[i] = lo | (hi << 16);
  }
  out[static_cast<size_t>(row) * vec_per_row + v] = make_uint4(o[0], o[1], o[2], o[3]);
}

// Single-CTA exact M=1 decode embedding + weightless RMSNorm (width <= 8192, width % 8 == 0).
// Matches ATen's gpu_reduce_kernel<float, MeanOps, vt0=4, input_vec_size=4> for rows == 1:
// block_width = 512, step_input = 512, 4 vectorized accumulators per thread combined in order
// (((a0 + a1) + a2) + a3), shared tree reduction offset 256 -> 32, warp shfl_down 16 -> 1,
// mean = sum * float(1.0 / width), inv = rsqrtf(mean + eps), then bf16_rne(x * inv).
template <typename Id>
__global__ void __launch_bounds__(512) embed_norm_m1(
    const Id* __restrict__ ids, const uint2* __restrict__ table, float eps,
    uint2* __restrict__ out, int width) {
  asm volatile("griddepcontrol.launch_dependents;");
  __shared__ float smem[512];
  const int t = threadIdx.x;
  const int vec4 = width >> 2;
  const long long id = static_cast<long long>(ids[0]);
  const uint2* row_ptr = table + id * static_cast<size_t>(vec4);

  float f[4][4];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const uint2 raw = row_ptr[idx];
      const float x0 = __uint_as_float(raw.x << 16);
      const float x1 = __uint_as_float(raw.x & 0xffff0000u);
      const float x2 = __uint_as_float(raw.y << 16);
      const float x3 = __uint_as_float(raw.y & 0xffff0000u);
      f[it][0] = x0;
      f[it][1] = x1;
      f[it][2] = x2;
      f[it][3] = x3;
      acc0 = __fadd_rn(acc0, __fmul_rn(x0, x0));
      acc1 = __fadd_rn(acc1, __fmul_rn(x1, x1));
      acc2 = __fadd_rn(acc2, __fmul_rn(x2, x2));
      acc3 = __fadd_rn(acc3, __fmul_rn(x3, x3));
    }
  }
  float v = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
  smem[t] = v;
#pragma unroll
  for (int offset = 256; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (t < offset) {
      v = __fadd_rn(v, smem[t + offset]);
      smem[t] = v;
    }
  }
  __syncthreads();
  if (t < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (t == 0) {
      const float factor = static_cast<float>(1.0 / static_cast<double>(width));
      const float mean = __fmul_rn(v, factor);
      smem[0] = rsqrtf(__fadd_rn(mean, eps));
    }
  }
  __syncthreads();
  const float r = smem[0];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const unsigned lo0 = bf16_rne(__fmul_rn(f[it][0], r));
      const unsigned hi0 = bf16_rne(__fmul_rn(f[it][1], r));
      const unsigned lo1 = bf16_rne(__fmul_rn(f[it][2], r));
      const unsigned hi1 = bf16_rne(__fmul_rn(f[it][3], r));
      out[idx] = make_uint2(lo0 | (hi0 << 16), lo1 | (hi1 << 16));
    }
  }
}
}  // namespace

extern "C" int muse_embed_norm_m1(const void* ids, int ids_64, const void* table, float eps,
                                  void* out, int width, cudaStream_t stream) {
  if (width <= 0 || (width & 7) != 0 || width > 8192) return cudaErrorInvalidValue;
  if (ids_64)
    embed_norm_m1<long long><<<1, 512, 0, stream>>>(
        static_cast<const long long*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  else
    embed_norm_m1<int><<<1, 512, 0, stream>>>(
        static_cast<const int*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_embed_square(const void* ids, int ids_64, const void* table, void* x2, int rows, int width,
                                 cudaStream_t stream) {
  if (rows <= 0 || width % 8) return cudaErrorInvalidValue;
  const int vec = width / 8;
  dim3 grid((vec + kThreads - 1) / kThreads, rows);
  if (ids_64)
    embed_square<long long><<<grid, kThreads, 0, stream>>>(static_cast<const long long*>(ids),
                                                            static_cast<const uint4*>(table),
                                                            static_cast<float4*>(x2), vec);
  else
    embed_square<int><<<grid, kThreads, 0, stream>>>(static_cast<const int*>(ids), static_cast<const uint4*>(table),
                                                      static_cast<float4*>(x2), vec);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_embed_scale(const void* ids, int ids_64, const void* table, const void* inv, void* out, int rows,
                                int width, cudaStream_t stream) {
  if (rows <= 0 || width % 8) return cudaErrorInvalidValue;
  const int vec = width / 8;
  dim3 grid((vec + kThreads - 1) / kThreads, rows);
  if (ids_64)
    embed_scale<long long><<<grid, kThreads, 0, stream>>>(static_cast<const long long*>(ids),
                                                           static_cast<const uint4*>(table),
                                                           static_cast<const float*>(inv), static_cast<uint4*>(out),
                                                           vec);
  else
    embed_scale<int><<<grid, kThreads, 0, stream>>>(static_cast<const int*>(ids), static_cast<const uint4*>(table),
                                                     static_cast<const float*>(inv), static_cast<uint4*>(out), vec);
  return static_cast<int>(cudaGetLastError());
}
