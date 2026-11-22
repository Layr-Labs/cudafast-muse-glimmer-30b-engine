// musefp8: bit-exact M == 1 FP8 (e4m3) GEMV for GB10 (sm_121a), the decode replacement of vLLM's cutlass_scaled_mm.
//
// What cutlass_scaled_mm computes for one token (SM120 "M16" config, 16x64x128 tiles, no split-K): for every output
// column n, acc(n) = the chain of mma.sync m16n8k32 e4m3 instructions over k = 0, 32, 64, ... (one fp32 accumulator,
// from +0), then D(n) = bf16_rn(scale_a * fp32(scale_b * acc(n))) (ScaledEpilogue: Compute0 = scale_b * acc, Compute1 =
// scale_a * that, both fp32 round-to-nearest, then RN to bf16). The tile shape changes neither the chain nor the
// epilogue, so this kernel, which issues the SAME instruction on the SAME operands in the SAME k order and applies the
// same two multiplies, produces the same bits (checked bitwise against cutlass_scaled_mm on the real checkpoint and at
// load time on every layer, see py/__init__.py).
//
// What differs is only how the weight is read: one warp owns 8 output columns (one mma n block); the weight is re-packed
// once at load into fragment order, [n block][k pair][lane][16 B], so each warp streams its 8 columns as one contiguous
// run with coalesced 16 B loads, several loads in flight, instead of cutlass' 64-column TMA tiles.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void mma_e4m3(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a2, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// The weight is read exactly once per step: stream it through L2 marked evict-first, so it does not push out what other
// kernels keep there (the L2 prefetch of the next GEMM's weights, activations, the KV cache).
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

// W [N][K] row-major e4m3 -> P [N/8][K/64][32 lanes][16 B]: lane (g, t) of n block nb holds, for k steps s = 2*s2 and
// s + 1, its mma B fragments {b0, b1}: b0 = W[8nb+g][32s+4t .. +3], b1 = W[8nb+g][32s+16+4t .. +3].
__global__ void pack_kernel(const uint8_t* __restrict__ W, uint4* __restrict__ P, int N, int K) {
  const size_t total = (size_t)(N / 8) * (K / 64) * 32;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = i & 31;
    const size_t r = i >> 5;
    const int S2 = K / 64;
    const int s2 = r % S2;
    const size_t nb = r / S2;
    const int g = lane >> 2, t = lane & 3;
    const uint8_t* row = W + (nb * 8 + g) * (size_t)K;
    uint32_t v[4];
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int s = 2 * s2 + q;
      v[2 * q + 0] = *(const uint32_t*)(row + 32 * s + 4 * t);
      v[2 * q + 1] = *(const uint32_t*)(row + 32 * s + 16 + 4 * t);
    }
    P[i] = make_uint4(v[0], v[1], v[2], v[3]);
  }
}

template <int U>
__global__ void __launch_bounds__(576, 1)
    gemv_kernel(const uint4* __restrict__ P, const uint8_t* __restrict__ x, const float* __restrict__ sb, int sb_scalar,
                const float* __restrict__ sa, __nv_bfloat16* __restrict__ out, int N, int K) {
  extern __shared__ uint2 xs[];  // x fragments: xs[s*4 + t] = {x[32s+4t .. +3], x[32s+16+4t .. +3]}
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int n4 = K / 32 * 4;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    const int s = i >> 2, tt = i & 3;
    xs[i] = make_uint2(*(const uint32_t*)(x + 32 * s + 4 * tt), *(const uint32_t*)(x + 32 * s + 16 + 4 * tt));
  }
  __syncthreads();
  const int NB = N / 8, S2 = K / 64, total_warps = gridDim.x * wpb;
  const uint64_t pol = evict_first_policy();
  for (int nb = blockIdx.x * wpb + wib; nb < NB; nb += total_warps) {
    const uint4* base = P + ((size_t)nb * S2) * 32 + lane;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    uint4 cur[U], nxt[U];
#pragma unroll
    for (int u = 0; u < U; u++) cur[u] = (u < S2) ? ld_stream(base + (size_t)u * 32, pol) : make_uint4(0, 0, 0, 0);
    for (int s2 = 0; s2 < S2; s2 += U) {
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int j = s2 + U + u;
        nxt[u] = (j < S2) ? ld_stream(base + (size_t)j * 32, pol) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int j = s2 + u;
        if (j < S2) {
          uint2 a0 = make_uint2(0, 0), a1 = make_uint2(0, 0);
          if (g == 0) {
            a0 = xs[(2 * j) * 4 + t];
            a1 = xs[(2 * j + 1) * 4 + t];
          }
          mma_e4m3(c0, c1, c2, c3, a0.x, a0.y, cur[u].x, cur[u].y);
          mma_e4m3(c0, c1, c2, c3, a1.x, a1.y, cur[u].z, cur[u].w);
        }
      }
#pragma unroll
      for (int u = 0; u < U; u++) cur[u] = nxt[u];
    }
    if (lane < 4) {
      const float sa_v = __ldg(sa);
      const int n = nb * 8 + 2 * t;
      const float s0 = sb_scalar ? __ldg(sb) : __ldg(sb + n), s1 = sb_scalar ? __ldg(sb) : __ldg(sb + n + 1);
      __nv_bfloat162 r;
      r.x = __float2bfloat16_rn(sa_v * (s0 * c0));
      r.y = __float2bfloat16_rn(sa_v * (s1 * c1));
      *(__nv_bfloat162*)(out + n) = r;
    }
  }
}

template <int U>
int launch(const void* P, const void* x, const float* sb, int sb_scalar, const float* sa, void* out, int N, int K, int blocks,
           int wpb, cudaStream_t st) {
  const size_t smem = (size_t)(K / 32) * 4 * sizeof(uint2);
  if (smem > 48 * 1024) return 2;
  gemv_kernel<U><<<blocks, wpb * 32, smem, st>>>((const uint4*)P, (const uint8_t*)x, sb, sb_scalar, sa, (__nv_bfloat16*)out,
                                                  N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

}  // namespace

extern "C" {

int musefp8_pack(const void* W, void* P, int N, int K, cudaStream_t st) {
  if (N % 8 || K % 64) return 1;
  pack_kernel<<<1024, 256, 0, st>>>((const uint8_t*)W, (uint4*)P, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int musefp8_gemv(const void* P, const void* x, const float* sb, int sb_scalar, const float* sa, void* out, int N, int K,
                 int blocks, int wpb, int U, cudaStream_t st) {
  if (N % 8 || K % 64 || blocks < 1 || wpb < 1 || wpb > 18) return 1;
  switch (U) {
    case 4: return launch<4>(P, x, sb, sb_scalar, sa, out, N, K, blocks, wpb, st);
    case 6: return launch<6>(P, x, sb, sb_scalar, sa, out, N, K, blocks, wpb, st);
    default: return launch<8>(P, x, sb, sb_scalar, sa, out, N, K, blocks, wpb, st);
  }
}

// Default launch: 48 blocks (one per GB10 SM), about one warp per 8 columns (at most 16 warps a block), 8 loads in flight.
int musefp8_gemv_auto(const void* P, const void* x, const float* sb, int sb_scalar, const float* sa, void* out, int N, int K,
                      cudaStream_t st) {
  int wpb = (N / 8 + 47) / 48;
  wpb = wpb < 4 ? 4 : (wpb > 16 ? 16 : wpb);
  return musefp8_gemv(P, x, sb, sb_scalar, sa, out, N, K, 48, wpb, 8, st);
}

}  // extern "C"
