// SPDX-License-Identifier: Apache-2.0
// muse_draft_rows: BF16 GEMM for 1..8 activation rows (the DFlash drafter's 1 + depth query rows), out = x . W^T.
//
// The drafter only proposes tokens; the target verifies every one, so the drafter's summation order changes which
// drafts are accepted, never which tokens are emitted. This kernel therefore does not try to match cuBLAS bit for
// bit; it streams the weight once with wide loads:
//   one warp owns 16 output columns; per 32-wide k block each lane (g, t) loads one uint4 of weight row g, one of row
//   g + 8 and one of activation row g (k = 8t .. 8t + 7), and feeds them to two mma.sync m16n8k16 bf16 (weight as the
//   16-row A operand, x^T as the 8-column B operand). Weight and activation use the same k permutation inside the
//   block, so each output is the full dot product (fp32 accumulate, one bf16 rounding at the end).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p, uint64_t pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}

__device__ __forceinline__ void step(float* d, const uint4& a, const uint4& b, const uint4& c) {
  // a: weight row g, b: weight row g + 8, c: activation row g; k = 8t + {0..7} of this 32-wide block.
  mma_bf16(d, a.x, b.x, a.y, b.y, c.x, c.y);
  mma_bf16(d, a.z, b.z, a.w, b.w, c.z, c.w);
}

template <int U>
__global__ void __launch_bounds__(256) rows_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
                                                   __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  const int NT = N >> 4, KB = K >> 5;
  const bool xv = g < M;
  const uint64_t pol = evict_first_policy();
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  for (int tile = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; tile < NT; tile += warps) {
    const int n0 = tile << 4;
    const uint4* w0 = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * K) + t;
    const uint4* w1 = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g + 8) * K) + t;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(xv ? g : 0) * K) + t;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    int j = 0;
    for (; j + U <= KB; j += U) {
      uint4 a[U], b[U], c[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        a[u] = ld_stream(w0 + 4 * (j + u), pol);
        b[u] = ld_stream(w1 + 4 * (j + u), pol);
        c[u] = xv ? __ldg(xr + 4 * (j + u)) : zero;
      }
#pragma unroll
      for (int u = 0; u < U; u++) step(d, a[u], b[u], c[u]);
    }
    for (; j < KB; j++) {
      const uint4 a = ld_stream(w0 + 4 * j, pol), b = ld_stream(w1 + 4 * j, pol);
      const uint4 c = xv ? __ldg(xr + 4 * j) : zero;
      step(d, a, b, c);
    }
    // d[0], d[1]: column n0 + g, tokens 2t, 2t + 1; d[2], d[3]: column n0 + g + 8, same tokens.
    const int tok0 = 2 * t, tok1 = 2 * t + 1;
    if (tok0 < M) {
      out[(size_t)tok0 * N + n0 + g] = __float2bfloat16_rn(d[0]);
      out[(size_t)tok0 * N + n0 + g + 8] = __float2bfloat16_rn(d[2]);
    }
    if (tok1 < M) {
      out[(size_t)tok1 * N + n0 + g] = __float2bfloat16_rn(d[1]);
      out[(size_t)tok1 * N + n0 + g + 8] = __float2bfloat16_rn(d[3]);
    }
  }
}

}  // namespace

extern "C" int muse_draft_rows(const void* x, const void* w, void* out, int m, int n, int k, int blocks,
                               cudaStream_t st) {
  if (m < 1 || m > 8 || n % 16 || k % 32 || blocks < 1) return 1;
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 15u) return 2;
  rows_kernel<4><<<blocks, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w),
                                         static_cast<__nv_bfloat16*>(out), m, n, k);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
