// SPDX-License-Identifier: Apache-2.0
// muse_bf16_rans (v2): Lossless 12-bit exponent-pair rANS + inline escape + raw (sign, mantissa)
// BF16 weights (~10.6 bits/weight, 100% bit-exact) with K32 (2x mma.sync.aligned.m16n8k16) Tensor-Core
// small-M (1..8) GEMM, persistent 48-CTA cp.async (L2::evict_first), 2 MiB TLB prefetch, and SM121 PDL.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace {

constexpr uint32_t kRansL = 65536u;
constexpr int kNS = 4;
constexpr uint32_t kHdrBytes = 256u;
constexpr uint32_t kLutBytes = 16384u;  // 4096 x u32 (12-bit exponent-pair LUT)
constexpr uint32_t kMagic = 0x52363142u;  // "B16R"
constexpr uint32_t kVersion = 2u;
constexpr uint32_t kGB = 512u;
constexpr uint32_t kRG = 4u;
constexpr uint32_t kRB = kGB * kRG;  // 2048 B per warp
constexpr uint32_t kMaxSmem = 99u * 1024u;
constexpr uint32_t kBrXStrideU4 = 68u;                       // 16 rows * 4 uint4 + 4 pad -> 0 bank conflicts on write & read
constexpr uint32_t kBrXsmemBytes = 2u * 16u * kBrXStrideU4 * 16u;  // 34,816 B (2 buffers x 16 K-steps)

__device__ __forceinline__ void bf16_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void bf16_mma_m16n8k16_m16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void br_cp_async16_ef(uint32_t sd, const void* gs) {
  // Materialize L2::evict_first policy locally per copy to avoid sm_121a ptxas odd-UR descriptor bug;
  // .L2::256B requests full 256B DRAM sectors for contiguous 512B-aligned warp ring refills.
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.cg.shared.global.L2::256B.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "l"(pol));
}

__device__ __forceinline__ void br_cp_async16_ca(uint32_t sd, const void* gs) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ uint32_t br_lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ uint32_t br_lds16(uint32_t a) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return static_cast<uint32_t>(v);
}

__device__ __forceinline__ void br_xpose4(
    uint4 v, int t, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
  const bool t1 = (t & 1) != 0;
  const bool t2 = (t & 2) != 0;
  const uint32_t s0 = __shfl_xor_sync(0xffffffffu, t1 ? v.x : v.y, 1);
  const uint32_t s1 = __shfl_xor_sync(0xffffffffu, t1 ? v.z : v.w, 1);
  const uint32_t p0 = t1 ? s0 : v.x;
  const uint32_t p1 = t1 ? v.y : s0;
  const uint32_t p2 = t1 ? s1 : v.z;
  const uint32_t p3 = t1 ? v.w : s1;
  const uint32_t r0 = __shfl_xor_sync(0xffffffffu, t2 ? p0 : p2, 2);
  const uint32_t r1 = __shfl_xor_sync(0xffffffffu, t2 ? p1 : p3, 2);
  o0 = t2 ? r0 : p0;
  o1 = t2 ? r1 : p1;
  o2 = t2 ? p2 : r0;
  o3 = t2 ? p3 : r1;
}

__device__ __forceinline__ size_t br_phys_off(const int* __restrict__ woff16, int tw, int gmin, int w, uint32_t lb) {
  const uint32_t g = lb >> 9;
  const size_t gi = (g < static_cast<uint32_t>(gmin))
                        ? (static_cast<size_t>(g) * static_cast<size_t>(tw) + static_cast<size_t>(w))
                        : (static_cast<size_t>(woff16[tw + 1 + w]) + static_cast<size_t>(g - static_cast<uint32_t>(gmin)));
  return (gi << 9) | static_cast<size_t>(lb & 511u);
}

__device__ __forceinline__ uint32_t br_exp_to_code(uint32_t e, uint32_t base) {
  if (e == 0u) return 0u;
  const uint32_t d = e - base;
  return (d < 14u) ? (d + 1u) : 15u;
}

// ---------------------------------------------------------------- histograms
__global__ void br_exp_hist_kernel(const uint16_t* __restrict__ W, size_t n, unsigned long long* __restrict__ counts) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const uint4* W4 = reinterpret_cast<const uint4*>(W);
  const size_t n8 = n / 8;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n8;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = W4[i];
    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      atomicAdd(&h[(a[q] >> 7) & 255u], 1u);
      atomicAdd(&h[(a[q] >> 23) & 255u], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    atomicAdd(&counts[i], static_cast<unsigned long long>(h[i]));
  }
}

__global__ void br_pair_hist_kernel(const uint16_t* __restrict__ W, size_t n, uint32_t base,
                                    unsigned long long* __restrict__ counts) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const uint4* W4 = reinterpret_cast<const uint4*>(W);
  const size_t n8 = n / 8;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n8;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = W4[i];
    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t c0 = br_exp_to_code((a[q] >> 7) & 255u, base);
      const uint32_t c1 = br_exp_to_code((a[q] >> 23) & 255u, base);
      atomicAdd(&h[(c0 << 4) | c1], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    atomicAdd(&counts[i], static_cast<unsigned long long>(h[i]));
  }
}

// ---------------------------------------------------------------- continuous-warp pair encoder (12-bit rANS)
struct BrPairEncSym {
  unsigned long long rcp;
  uint32_t xmax_m1;
  uint32_t cum_nf;
};

__device__ __forceinline__ uint32_t br_pair_enc_step(uint32_t x, const BrPairEncSym& e, bool emit) {
  if (emit) x >>= 16;
  const uint32_t q = e.rcp ? static_cast<uint32_t>(__umul64hi(static_cast<unsigned long long>(x), e.rcp)) : x;
  return x + q * (e.cum_nf >> 16) + (e.cum_nf & 0xffffu);
}

__global__ void __launch_bounds__(128) br_pair_enc_pass1_kernel(
    const uint16_t* __restrict__ W, int NB, int K, int tw, uint32_t base,
    const BrPairEncSym* __restrict__ gtab, uint32_t* __restrict__ wwords) {
  __shared__ BrPairEncSym tab[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = gtab[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  int nblast = w;
  while (nblast + tw < NB) nblast += tw;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  uint32_t total_u16 = 0;
  for (int nb = nblast; nb >= w; nb -= tw) {
    const uint16_t* row = W + static_cast<size_t>(nb * 8 + g) * K;
    for (int s = S - 1; s >= 0; --s) {
      const uint32_t p[4] = {
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 8 + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 16 + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 24 + 2 * t),
      };
      uint32_t ws = 0;
      // Count inline escape u16s (emitted after rANS renorms in forward order, so counted first in reverse)
      uint32_t c_arr[8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        c_arr[2 * q] = br_exp_to_code((p[q] >> 7) & 255u, base);
        c_arr[2 * q + 1] = br_exp_to_code((p[q] >> 23) & 255u, base);
      }
      uint32_t any_esc = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) any_esc |= (c_arr[i] == 15u);
      if (__any_sync(0xffffffffu, any_esc != 0u)) {
#pragma unroll
        for (int i = 7; i >= 0; --i) {
          ws += __popc(__ballot_sync(0xffffffffu, c_arr[i] == 15u));
        }
      }
#pragma unroll
      for (int q = 3; q >= 0; --q) {
        uint32_t& x = st[q];
        const BrPairEncSym e = tab[(c_arr[2 * q] << 4) | c_arr[2 * q + 1]];
        const bool emit = x > e.xmax_m1;
        ws += __popc(__ballot_sync(0xffffffffu, emit));
        x = br_pair_enc_step(x, e, emit);
      }
      total_u16 += 128u + ws;  // 256 B sm_words = 128 u16 + renorms + escapes
    }
  }
  if (lane == 0) wwords[w] = total_u16;
}

__global__ void __launch_bounds__(128) br_pair_enc_pass2_kernel(
    const uint16_t* __restrict__ W, int NB, int K, int tw, int gmin, uint32_t base,
    const BrPairEncSym* __restrict__ gtab, const uint32_t* __restrict__ wwords,
    const int* __restrict__ woff16, uint8_t* __restrict__ stream) {
  __shared__ BrPairEncSym tab[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = gtab[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  const uint32_t lt = (1u << lane) - 1u;
  int nblast = w;
  while (nblast + tw < NB) nblast += tw;
  const uint32_t lbase = 128u * kNS;
  uint32_t posw = wwords[w];
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  for (int nb = nblast; nb >= w; nb -= tw) {
    const uint16_t* row = W + static_cast<size_t>(nb * 8 + g) * K;
    for (int s = S - 1; s >= 0; --s) {
      const uint32_t p[4] = {
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 8 + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 16 + 2 * t),
          *reinterpret_cast<const uint32_t*>(row + 32 * s + 24 + 2 * t),
      };
      uint16_t wv[8];
      uint32_t ex[8];
      uint32_t c_arr[8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        wv[2 * q] = static_cast<uint16_t>(p[q] & 0xFFFFu);
        wv[2 * q + 1] = static_cast<uint16_t>(p[q] >> 16);
        ex[2 * q] = (wv[2 * q] >> 7) & 255u;
        ex[2 * q + 1] = (wv[2 * q + 1] >> 7) & 255u;
        c_arr[2 * q] = br_exp_to_code(ex[2 * q], base);
        c_arr[2 * q + 1] = br_exp_to_code(ex[2 * q + 1], base);
      }
      // 1. Write inline escape exponents (last in forward order -> written first in reverse)
      uint32_t any_esc = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) any_esc |= (c_arr[i] == 15u);
      if (__any_sync(0xffffffffu, any_esc != 0u)) {
#pragma unroll
        for (int i = 7; i >= 0; --i) {
          const bool is_esc = (c_arr[i] == 15u);
          const uint32_t m = __ballot_sync(0xffffffffu, is_esc);
          if (m) {
            posw -= __popc(m);
            if (is_esc) {
              *reinterpret_cast<uint16_t*>(
                  stream + br_phys_off(woff16, tw, gmin, w, lbase + 2u * (posw + __popc(m & lt)))) =
                  static_cast<uint16_t>(ex[i]);
            }
          }
        }
      }
      // 2. Write 4 pair rANS renorms
#pragma unroll
      for (int q = 3; q >= 0; --q) {
        uint32_t& x = st[q];
        const BrPairEncSym e = tab[(c_arr[2 * q] << 4) | c_arr[2 * q + 1]];
        const bool emit = x > e.xmax_m1;
        const uint32_t m = __ballot_sync(0xffffffffu, emit);
        posw -= __popc(m);
        if (emit) {
          *reinterpret_cast<uint16_t*>(
              stream + br_phys_off(woff16, tw, gmin, w, lbase + 2u * (posw + __popc(m & lt)))) =
              static_cast<uint16_t>(x & 0xFFFFu);
        }
        x = br_pair_enc_step(x, e, emit);
      }
      // 3. Write 256 B raw (sign, 7-bit mantissa) bytes (8 bytes/lane)
      posw -= 128u;
      const uint32_t nboff = lbase + 2u * posw + 8u * lane;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t sm_a = ((wv[2 * q] >> 8) & 128u) | (wv[2 * q] & 127u);
        const uint32_t sm_b = ((wv[2 * q + 1] >> 8) & 128u) | (wv[2 * q + 1] & 127u);
        *reinterpret_cast<uint16_t*>(stream + br_phys_off(woff16, tw, gmin, w, nboff + 2u * q)) =
            static_cast<uint16_t>(sm_a | (sm_b << 8));
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    *reinterpret_cast<uint32_t*>(stream + br_phys_off(woff16, tw, gmin, w, 128u * s + 4u * lane)) = st[s];
  }
}

// ---------------------------------------------------------------- load-time exact unpack verification
__global__ void __launch_bounds__(128) br_pair_unpack_kernel(
    const uint8_t* __restrict__ stream, const int* __restrict__ woff16,
    const uint32_t* __restrict__ lut, uint16_t* __restrict__ out,
    int NB, int K, int tw, int gmin, uint32_t base, int* __restrict__ status) {
  __shared__ uint32_t lut_s[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) lut_s[i] = lut[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  const uint32_t lt = (1u << lane) - 1u;
  const uint32_t base_m1 = base - 1u;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    st[s] = *reinterpret_cast<const uint32_t*>(stream + br_phys_off(woff16, tw, gmin, w, 128u * s + 4u * lane));
  }
  uint32_t pos = 128u * kNS;
  for (int nb = w; nb < NB; nb += tw) {
    uint16_t* orow = out + static_cast<size_t>(nb * 8 + g) * K;
    for (int s = 0; s < S; ++s) {
      const uint32_t nboff = pos + 8u * lane;
      const uint32_t sm_w0 =
          static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(stream + br_phys_off(woff16, tw, gmin, w, nboff))) |
          (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(stream + br_phys_off(woff16, tw, gmin, w, nboff + 2u))) << 16);
      const uint32_t sm_w1 =
          static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(stream + br_phys_off(woff16, tw, gmin, w, nboff + 4u))) |
          (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(stream + br_phys_off(woff16, tw, gmin, w, nboff + 6u))) << 16);
      pos += 256u;
      uint32_t c_arr[8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t& x = st[q];
        const uint32_t e = lut_s[x & 4095u];
        const uint32_t xh = x >> 12;
        x = ((e >> 8) & 4095u) * xh + xh + (e >> 20);
        c_arr[2 * q] = (e >> 4) & 15u;
        c_arr[2 * q + 1] = e & 15u;
        const bool need = x < kRansL;
        const uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) {
          x = (x << 16) | *reinterpret_cast<const uint16_t*>(
                              stream + br_phys_off(woff16, tw, gmin, w, pos + 2u * __popc(m & lt)));
        }
        pos += 2u * __popc(m);
      }
      uint32_t ex[8];
      uint32_t any_esc = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        ex[i] = c_arr[i] ? (base_m1 + c_arr[i]) : 0u;
        any_esc |= (c_arr[i] == 15u);
      }
      if (__any_sync(0xffffffffu, any_esc != 0u)) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const bool is_esc = (c_arr[i] == 15u);
          const uint32_t m = __ballot_sync(0xffffffffu, is_esc);
          if (m) {
            if (is_esc) {
              ex[i] = *reinterpret_cast<const uint16_t*>(
                  stream + br_phys_off(woff16, tw, gmin, w, pos + 2u * __popc(m & lt)));
            }
            pos += 2u * __popc(m);
          }
        }
      }
      const uint32_t sm01 = __byte_perm(sm_w0, 0u, 0x4140);
      const uint32_t sm23 = __byte_perm(sm_w0, 0u, 0x4342);
      const uint32_t sm45 = __byte_perm(sm_w1, 0u, 0x4140);
      const uint32_t sm67 = __byte_perm(sm_w1, 0u, 0x4342);
      *reinterpret_cast<uint32_t*>(orow + 32 * s + 2 * t) =
          ((sm01 & 0x00800080u) << 8) | (sm01 & 0x007F007Fu) | (ex[0] << 7) | (ex[1] << 23);
      *reinterpret_cast<uint32_t*>(orow + 32 * s + 8 + 2 * t) =
          ((sm23 & 0x00800080u) << 8) | (sm23 & 0x007F007Fu) | (ex[2] << 7) | (ex[3] << 23);
      *reinterpret_cast<uint32_t*>(orow + 32 * s + 16 + 2 * t) =
          ((sm45 & 0x00800080u) << 8) | (sm45 & 0x007F007Fu) | (ex[4] << 7) | (ex[5] << 23);
      *reinterpret_cast<uint32_t*>(orow + 32 * s + 24 + 2 * t) =
          ((sm67 & 0x00800080u) << 8) | (sm67 & 0x007F007Fu) | (ex[6] << 7) | (ex[7] << 23);
    }
  }
  bool bad = false;
#pragma unroll
  for (int s = 0; s < kNS; ++s) bad |= (st[s] != kRansL);
  if (__any_sync(0xffffffffu, bad) && lane == 0) atomicAdd(status, 1);
}

// ---------------------------------------------------------------- K32 pair-rANS + 2x mma.sync.m16n8k16 GEMM
struct BrProb {
  const uint8_t* hdr;
  const uint8_t* stream;
  const int* woff16;
  const uint32_t* lut;
  const uint16_t* x;
  __nv_bfloat16* out;
  __nv_bfloat16* out2;
  int M;
  int N;
  int K;
  int N1;
  int N2;
  int nb_split;
  int tw;
  int pdl_mode;
  int xsmem_async;
  uint32_t base_m1;
  size_t stream_bytes;
  size_t tpf_stride;
};

static size_t br_tpf_stride() {
  static size_t stride = []() -> size_t {
    const char* env = std::getenv("MUSE_BF16_RANS_TLB_PREFETCH");
    if (!env || !*env) return 2048u << 10;
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return stride;
}

static int br_pdl_cfg() {
  static const int mode = []() -> int {
    const char* env = std::getenv("MUSE_BF16_RANS_PDL");
    if (!env || !*env) return 2;
    long v = std::strtol(env, nullptr, 10);
    return (v == 0 || v == 1 || v == 2) ? static_cast<int>(v) : 2;
  }();
  return mode;
}

template <bool kPar4, bool kM16 = false, bool kXsmem = false>
__global__ void __launch_bounds__(640, 1) br_pair_mma_gemm_kernel(BrProb pb) {
  __shared__ __align__(16) uint32_t sh_lut[4096];
  extern __shared__ __align__(16) unsigned char smem[];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wib = tid >> 5;
  const int K = pb.K;
  const int S = K >> 5;

  uint4* const x_smem_u4 = reinterpret_cast<uint4*>(smem);
  const uint32_t sbase0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + (kXsmem ? kBrXsmemBytes : 0u);
  const uint32_t ring_s = ((sbase0 + kRB - 1u) & ~(kRB - 1u)) + static_cast<uint32_t>(wib) * kRB;

  const int q_w = pb.tw / gridDim.x;
  const int rem_w = pb.tw % gridDim.x;
  const int w_cnt = q_w + (blockIdx.x < rem_w ? 1 : 0);
  const int w = blockIdx.x * q_w + (blockIdx.x < rem_w ? blockIdx.x : rem_w) + wib;
  const bool active = (wib < w_cnt) && (w < pb.tw);

  uint32_t wbytes = 0;
  int gmin = 0, toffw = 0, issued = 0;

#define BR_ISSUE()                                                                                                    \
  {                                                                                                                   \
    const uint32_t boff = static_cast<uint32_t>(issued) * kGB + static_cast<uint32_t>(lane) * 16u;                    \
    if (boff < wbytes) {                                                                                              \
      const size_t gi = (issued < gmin)                                                                               \
                            ? (static_cast<size_t>(issued) * pb.tw + static_cast<size_t>(w))                          \
                            : (static_cast<size_t>(toffw) + static_cast<size_t>(issued - gmin));                      \
      br_cp_async16_ef(ring_s + (boff & (kRB - 1u)), pb.stream + (gi << 9) + static_cast<uint32_t>(lane) * 16u);      \
    }                                                                                                                 \
    asm volatile("cp.async.commit_group;" ::: "memory");                                                              \
    issued++;                                                                                                         \
  }

  // 1. Preload 16 KiB pair-exponent LUT via cp.async BEFORE weight ring pre-issue
  const uint32_t lut_sa = static_cast<uint32_t>(__cvta_generic_to_shared(sh_lut));
  for (int i = tid; i < 1024; i += blockDim.x) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
                 ::"r"(lut_sa + 16u * static_cast<uint32_t>(i)),
                   "l"(reinterpret_cast<const uint4*>(pb.lut) + i));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");

  const bool fast_pre = (K >= 2048);
  int pre_wo0 = 0, pre_wo1 = 0, pre_tof = 0, pre_gmin = 0;
  if (active && fast_pre) {
    pre_wo0 = __ldg(pb.woff16 + w);
    pre_wo1 = __ldg(pb.woff16 + w + 1);
    pre_tof = __ldg(pb.woff16 + pb.tw + 1 + w);
    pre_gmin = static_cast<int>(__ldg(reinterpret_cast<const uint32_t*>(pb.hdr) + 14));
  }
  if (active) {
    if (fast_pre) {
#pragma unroll
      for (int i = 0; i < static_cast<int>(kRG) - 1; ++i) {
        const uint32_t boff = static_cast<uint32_t>(i) * kGB + static_cast<uint32_t>(lane) * 16u;
        br_cp_async16_ef(
            ring_s + (boff & (kRB - 1u)),
            pb.stream + (((static_cast<size_t>(i) * pb.tw + static_cast<size_t>(w)) << 9) +
                         static_cast<uint32_t>(lane) * 16u));
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
      issued = static_cast<int>(kRG) - 1;
    } else {
      wbytes = static_cast<uint32_t>(pb.woff16[w + 1] - pb.woff16[w]) * 16u;
      gmin = static_cast<int>(reinterpret_cast<const uint32_t*>(pb.hdr)[14]);
      toffw = pb.woff16[pb.tw + 1 + w];
#pragma unroll
      for (int i = 0; i < static_cast<int>(kRG) - 1; ++i) BR_ISSUE();
    }
  }

  if (blockIdx.x == 0 && pb.tpf_stride > 0) {
    for (size_t off = static_cast<size_t>(tid) * pb.tpf_stride; off < pb.stream_bytes;
         off += static_cast<size_t>(blockDim.x) * pb.tpf_stride) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(pb.stream + off));
    }
  }

  if (active && fast_pre) {
    wbytes = static_cast<uint32_t>(pre_wo1 - pre_wo0) * 16u;
    gmin = pre_gmin;
    toffw = pre_tof;
  }

  if (pb.pdl_mode & 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const uint32_t x_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const bool xsmem_async = (pb.xsmem_async != 0);
  auto load_x_chunk_sync = [&](int buf, int s_base) {
    const int total_u4 = 64 * pb.M;
    const uint32_t buf_u4 = static_cast<uint32_t>(buf * 16) * kBrXStrideU4;
    for (int base_i = tid; base_i < total_u4; base_i += 3 * static_cast<int>(blockDim.x)) {
      const int i0 = base_i;
      const int i1 = base_i + static_cast<int>(blockDim.x);
      const int i2 = base_i + 2 * static_cast<int>(blockDim.x);
      const int m0 = i0 >> 6, sl0 = (i0 >> 2) & 15, u0 = i0 & 3;
      const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(m0) * K) + 4 * (s_base + sl0) + u0);
      uint4 v1 = make_uint4(0u, 0u, 0u, 0u), v2 = make_uint4(0u, 0u, 0u, 0u);
      int m1 = 0, sl1 = 0, u1 = 0, m2 = 0, sl2 = 0, u2 = 0;
      if (i1 < total_u4) {
        m1 = i1 >> 6; sl1 = (i1 >> 2) & 15; u1 = i1 & 3;
        v1 = __ldg(reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(m1) * K) + 4 * (s_base + sl1) + u1);
      }
      if (i2 < total_u4) {
        m2 = i2 >> 6; sl2 = (i2 >> 2) & 15; u2 = i2 & 3;
        v2 = __ldg(reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(m2) * K) + 4 * (s_base + sl2) + u2);
      }
      x_smem_u4[buf_u4 + static_cast<uint32_t>(sl0) * kBrXStrideU4 + static_cast<uint32_t>(m0 * 4 + u0)] = v0;
      if (i1 < total_u4) x_smem_u4[buf_u4 + static_cast<uint32_t>(sl1) * kBrXStrideU4 + static_cast<uint32_t>(m1 * 4 + u1)] = v1;
      if (i2 < total_u4) x_smem_u4[buf_u4 + static_cast<uint32_t>(sl2) * kBrXStrideU4 + static_cast<uint32_t>(m2 * 4 + u2)] = v2;
    }
  };
  auto load_x_chunk_async = [&](int buf, int s_base) {
    const int total_u4 = 64 * pb.M;
    const uint32_t buf_b = x_smem_base + static_cast<uint32_t>(buf * 16) * (kBrXStrideU4 * 16u);
    for (int idx = tid; idx < total_u4; idx += blockDim.x) {
      const int m = idx >> 6;
      const int sl = (idx >> 2) & 15;
      const int u = idx & 3;
      const uint32_t dst = buf_b + (static_cast<uint32_t>(sl) * kBrXStrideU4 + static_cast<uint32_t>(m * 4 + u)) * 16u;
      const uint4* src = reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(m) * K) + 4 * (s_base + sl) + u;
      br_cp_async16_ca(dst, src);
    }
  };
  if constexpr (kXsmem && kPar4) {
    // Overlap chunk-0 activation load with in-flight sh_lut & kRG-1 weight cp.asyncs;
    // synced together by the single __syncthreads() below.
    if (xsmem_async) {
      load_x_chunk_sync(0, 0);
    }
  }

  if (active) {
    asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 1) : "memory");
  } else {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  __syncthreads();

  if constexpr (!kXsmem) {
    if (!active) return;
  }

  const int g = lane >> 2;
  const int t = lane & 3;
  const uint32_t lt = (1u << lane) - 1u;
  const int NB = pb.N >> 3;
  const bool valid_m = (g < pb.M);
  const bool valid_m_hi = kM16 && ((g + 8) < pb.M);
  const uint32_t* xrow32 =
      valid_m ? reinterpret_cast<const uint32_t*>(pb.x + static_cast<size_t>(g) * K + 2 * t) : nullptr;
  const uint32_t* xrow32_hi =
      valid_m_hi ? reinterpret_cast<const uint32_t*>(pb.x + static_cast<size_t>(g + 8) * K + 2 * t) : nullptr;
  // Vectorized 1-step (64 B = 4 uint4 across t=0..3) pointers for kPar4 (K%32==0 -> 2*K%64==0, 16B aligned).
  const uint4* xrow_u4 =
      valid_m ? reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(g) * K + 8 * t) : nullptr;
  const uint4* xrow_u4_hi =
      valid_m_hi ? reinterpret_cast<const uint4*>(pb.x + static_cast<size_t>(g + 8) * K + 8 * t) : nullptr;
  const uint32_t base_m1 = pb.base_m1;

  uint32_t st[kNS];
  if (active) {
    asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 2) : "memory");
    __syncwarp();
#pragma unroll
    for (int s = 0; s < kNS; ++s) {
      st[s] = br_lds32(ring_s + ((128u * s + 4u * lane) & (kRB - 1u)));
    }
  }
  uint32_t pwb = 128u * kNS;

  if constexpr (kPar4) {
    const uint32_t lane16 = static_cast<uint32_t>(lane) * 16u;
    const uint32_t ring_lane = ring_s + lane16;
    const size_t stride_g = static_cast<size_t>(pb.tw) << 9;
    const uint8_t* const tail_w = pb.stream + (static_cast<size_t>(toffw) << 9) + lane16;
    const uint8_t* ilv_ptr =
        pb.stream + ((static_cast<size_t>(issued) * static_cast<size_t>(pb.tw) + static_cast<size_t>(w)) << 9) + lane16;
#define BR_ISSUE_FAST()                                                                                       \
  {                                                                                                           \
    const uint32_t boff = (static_cast<uint32_t>(issued) << 9) + lane16;                                      \
    if (boff < wbytes) {                                                                                      \
      const uint8_t* src = (issued < gmin) ? ilv_ptr : (tail_w + (static_cast<size_t>(issued - gmin) << 9)); \
      br_cp_async16_ef(ring_lane + ((static_cast<uint32_t>(issued) << 9) & (kRB - 1u)), src);                 \
    }                                                                                                         \
    if (issued < gmin) ilv_ptr += stride_g;                                                                   \
    asm volatile("cp.async.commit_group;" ::: "memory");                                                      \
    x_commits++;                                                                                              \
    issued++;                                                                                                 \
  }
    const int R_max = (NB + pb.tw - 1) / pb.tw;
    int cur_buf = 0;
    for (int r_i = 0; r_i < R_max; ++r_i) {
      const int nb = w + r_i * pb.tw;
      const bool do_nb = active && (nb < NB);
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      const int trig_s = (do_nb && (pb.pdl_mode & 2) && (nb + pb.tw >= NB)) ? ((S > 16) ? (S - 16) : 0) : -1;
      if constexpr (kXsmem) {
        if (!xsmem_async) {
          cur_buf = 0;
          load_x_chunk_sync(0, 0);
          __syncthreads();
        }
      }
      const int cs_step = kXsmem ? 16 : S;
      for (int cs = 0; cs < S; cs += cs_step) {
        const bool has_next = kXsmem && ((cs + 16 < S) || (xsmem_async && (r_i + 1 < R_max)));
        const int next_s = (cs + 16 < S) ? (cs + 16) : 0;
        int x_commits = 0;
        if constexpr (kXsmem) {
          if (has_next) {
            if (xsmem_async) load_x_chunk_async(cur_buf ^ 1, next_s);
            else load_x_chunk_sync(cur_buf ^ 1, next_s);
          }
        }
        if (do_nb) {
          const int s_end = kXsmem ? (cs + 16) : S;
          for (int s = cs; s < s_end; ++s) {
            if (s == trig_s) {
              asm volatile("griddepcontrol.launch_dependents;");
            }
            // Early-issue cp.async ring refill before LUT/ALU/xpose4 so DRAM/L2 latency overlaps compute.
            const int curg = static_cast<int>(pwb >> 9);
            const bool need_refill = (issued <= curg + static_cast<int>(kRG) - 1);
            if (need_refill) {
              __syncwarp();
              do {
                BR_ISSUE_FAST();
              } while (issued <= curg + static_cast<int>(kRG) - 1);
            }
            // Pre-issue 1 (or 2 for kM16) vectorized uint4 activation loads (SMEM or L1/L2) + 4 LUT loads.
            uint4 v_lo = make_uint4(0u, 0u, 0u, 0u);
            uint4 v_hi = make_uint4(0u, 0u, 0u, 0u);
            if constexpr (kXsmem) {
              const uint32_t xbase = (cur_buf * 16 + (s - cs)) * kBrXStrideU4;
              if (valid_m) v_lo = x_smem_u4[xbase + g * 4 + t];
              if constexpr (kM16) {
                if (valid_m_hi) v_hi = x_smem_u4[xbase + (g + 8) * 4 + t];
              }
            } else {
              if (valid_m) v_lo = __ldg(xrow_u4 + 4 * s);
              if constexpr (kM16) {
                if (valid_m_hi) v_hi = __ldg(xrow_u4_hi + 4 * s);
              }
            }

            const uint32_t le0 = sh_lut[st[0] & 4095u];
            const uint32_t le1 = sh_lut[st[1] & 4095u];
            const uint32_t le2 = sh_lut[st[2] & 4095u];
            const uint32_t le3 = sh_lut[st[3] & 4095u];

            // Execute all ring-independent rANS updates, ballots, renorm addresses, exponent decode,
            // and activation transpose before waiting on the weight ring group.
            const uint32_t pwb_n = pwb;
            const uint32_t na = pwb_n + 8u * static_cast<uint32_t>(lane);
            const uint32_t xh0 = st[0] >> 12;
            const uint32_t x0 = ((le0 >> 8) & 4095u) * xh0 + xh0 + (le0 >> 20);
            const uint32_t xh1 = st[1] >> 12;
            const uint32_t x1 = ((le1 >> 8) & 4095u) * xh1 + xh1 + (le1 >> 20);
            const uint32_t xh2 = st[2] >> 12;
            const uint32_t x2 = ((le2 >> 8) & 4095u) * xh2 + xh2 + (le2 >> 20);
            const uint32_t xh3 = st[3] >> 12;
            const uint32_t x3 = ((le3 >> 8) & 4095u) * xh3 + xh3 + (le3 >> 20);
            const bool n0 = x0 < kRansL;
            const bool n1 = x1 < kRansL;
            const bool n2 = x2 < kRansL;
            const bool n3 = x3 < kRansL;
            const uint32_t m0 = __ballot_sync(0xffffffffu, n0);
            const uint32_t m1 = __ballot_sync(0xffffffffu, n1);
            const uint32_t m2 = __ballot_sync(0xffffffffu, n2);
            const uint32_t m3 = __ballot_sync(0xffffffffu, n3);
            const uint32_t p0 = pwb_n + 256u;
            const uint32_t p1 = p0 + (__popc(m0) << 1);
            const uint32_t p2 = p1 + (__popc(m1) << 1);
            const uint32_t p3 = p2 + (__popc(m2) << 1);
            pwb = p3 + (__popc(m3) << 1);
            const uint32_t ro0 = ring_s + ((p0 + (__popc(m0 & lt) << 1)) & (kRB - 1u));
            const uint32_t ro1 = ring_s + ((p1 + (__popc(m1 & lt) << 1)) & (kRB - 1u));
            const uint32_t ro2 = ring_s + ((p2 + (__popc(m2 & lt) << 1)) & (kRB - 1u));
            const uint32_t ro3 = ring_s + ((p3 + (__popc(m3 & lt) << 1)) & (kRB - 1u));

            const uint32_t Pw01 = __byte_perm(le0, le1, 0x7740);
            const uint32_t Pw23 = __byte_perm(le2, le3, 0x7740);
            const uint32_t Pw = __byte_perm(Pw01, Pw23, 0x5410);

            const uint32_t hi4 = (Pw >> 4) & 0x0F0F0F0Fu;
            const uint32_t lo4 = Pw & 0x0F0F0F0Fu;
            const uint32_t esc_bits = ((hi4 + 0x01010101u) | (lo4 + 0x01010101u)) & 0x10101010u;

            uint32_t c0_ = hi4 & 15u, c1_ = lo4 & 15u;
            uint32_t c2_ = (hi4 >> 8) & 15u, c3_ = (lo4 >> 8) & 15u;
            uint32_t c4_ = (hi4 >> 16) & 15u, c5_ = (lo4 >> 16) & 15u;
            uint32_t c6_ = (hi4 >> 24) & 15u, c7_ = (lo4 >> 24) & 15u;

            uint32_t e0 = c0_ ? (base_m1 + c0_) : 0u;
            uint32_t e1 = c1_ ? (base_m1 + c1_) : 0u;
            uint32_t e2 = c2_ ? (base_m1 + c2_) : 0u;
            uint32_t e3 = c3_ ? (base_m1 + c3_) : 0u;
            uint32_t e4 = c4_ ? (base_m1 + c4_) : 0u;
            uint32_t e5 = c5_ ? (base_m1 + c5_) : 0u;
            uint32_t e6 = c6_ ? (base_m1 + c6_) : 0u;
            uint32_t e7 = c7_ ? (base_m1 + c7_) : 0u;

            uint32_t a0, a2, a4, a6;
            br_xpose4(v_lo, t, a0, a2, a4, a6);
            uint32_t a1 = 0u, a3 = 0u, a5 = 0u, a7 = 0u;
            if constexpr (kM16) {
              br_xpose4(v_hi, t, a1, a3, a5, a7);
            }

            // Wait for weight ring group right before issuing sm_w0/sm_w1 and r0..r3 LDS in parallel.
            if (need_refill) {
              asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 2) : "memory");
              __syncwarp();
            }

            uint32_t sm_w0, sm_w1;
            if ((pwb_n & 2u) == 0u) {
              sm_w0 = br_lds32(ring_s + (na & (kRB - 1u)));
              sm_w1 = br_lds32(ring_s + ((na + 4u) & (kRB - 1u)));
            } else {
              sm_w0 = br_lds16(ring_s + (na & (kRB - 1u))) | (br_lds16(ring_s + ((na + 2u) & (kRB - 1u))) << 16);
              sm_w1 = br_lds16(ring_s + ((na + 4u) & (kRB - 1u))) | (br_lds16(ring_s + ((na + 6u) & (kRB - 1u))) << 16);
            }

            // Issue all 4 conditional lds16 renorms in parallel.
            const uint32_t r0 = n0 ? br_lds16(ro0) : 0u;
            const uint32_t r1 = n1 ? br_lds16(ro1) : 0u;
            const uint32_t r2 = n2 ? br_lds16(ro2) : 0u;
            const uint32_t r3 = n3 ? br_lds16(ro3) : 0u;

            const bool had_esc = __any_sync(0xffffffffu, esc_bits != 0u);
            if (__builtin_expect(had_esc, 0)) {
              // Commit renorms before BR_ISSUE_FAST() can overwrite ring_s.
              st[0] = n0 ? ((x0 << 16) | r0) : x0;
              st[1] = n1 ? ((x1 << 16) | r1) : x1;
              st[2] = n2 ? ((x2 << 16) | r2) : x2;
              st[3] = n3 ? ((x3 << 16) | r3) : x3;
              const int cg2 = static_cast<int>(pwb >> 9);
              if (issued <= cg2 + static_cast<int>(kRG) - 1) {
                __syncwarp();
                do {
                  BR_ISSUE_FAST();
                } while (issued <= cg2 + static_cast<int>(kRG) - 1);
                asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 2) : "memory");
                __syncwarp();
              }
#define BR_ESC_P(ci, ei)                                                                             \
  {                                                                                                  \
    const bool is_e = ((ci) == 15u);                                                                 \
    const uint32_t em = __ballot_sync(0xffffffffu, is_e);                                            \
    if (em) {                                                                                        \
      if (is_e) (ei) = br_lds16(ring_s + ((pwb + (__popc(em & lt) << 1)) & (kRB - 1u)));             \
      pwb += __popc(em) << 1;                                                                        \
    }                                                                                                \
  }
              BR_ESC_P(c0_, e0) BR_ESC_P(c1_, e1) BR_ESC_P(c2_, e2) BR_ESC_P(c3_, e3)
              BR_ESC_P(c4_, e4) BR_ESC_P(c5_, e5) BR_ESC_P(c6_, e6) BR_ESC_P(c7_, e7)
#undef BR_ESC_P
            }

            const uint32_t sm01 = __byte_perm(sm_w0, 0u, 0x4140);
            const uint32_t sm23 = __byte_perm(sm_w0, 0u, 0x4342);
            const uint32_t b0 = ((sm01 & 0x00800080u) << 8) | (sm01 & 0x007F007Fu) | (e0 << 7) | (e1 << 23);
            const uint32_t b1 = ((sm23 & 0x00800080u) << 8) | (sm23 & 0x007F007Fu) | (e2 << 7) | (e3 << 23);
            if constexpr (kM16) {
              bf16_mma_m16n8k16_m16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            } else {
              bf16_mma_m16n8k16(c0, c1, c2, c3, a0, a2, b0, b1);
            }

            const uint32_t sm45 = __byte_perm(sm_w1, 0u, 0x4140);
            const uint32_t sm67 = __byte_perm(sm_w1, 0u, 0x4342);
            const uint32_t b2 = ((sm45 & 0x00800080u) << 8) | (sm45 & 0x007F007Fu) | (e4 << 7) | (e5 << 23);
            const uint32_t b3 = ((sm67 & 0x00800080u) << 8) | (sm67 & 0x007F007Fu) | (e6 << 7) | (e7 << 23);
            if constexpr (kM16) {
              bf16_mma_m16n8k16_m16(c0, c1, c2, c3, a4, a5, a6, a7, b2, b3);
            } else {
              bf16_mma_m16n8k16(c0, c1, c2, c3, a4, a6, b2, b3);
            }

            if (!had_esc) {
              st[0] = n0 ? ((x0 << 16) | r0) : x0;
              st[1] = n1 ? ((x1 << 16) | r1) : x1;
              st[2] = n2 ? ((x2 << 16) | r2) : x2;
              st[3] = n3 ? ((x3 << 16) | r3) : x3;
            }
          }
        }
        if constexpr (kXsmem) {
          if (has_next) {
            if (xsmem_async && (!do_nb || x_commits < static_cast<int>(kRG) - 1)) {
              if (x_commits == 0) asm volatile("cp.async.commit_group;" ::: "memory");
              asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            cur_buf ^= 1;
          }
        }
      }
      if (do_nb && valid_m) {
        const bool is_out2 = (pb.out2 != nullptr && nb >= pb.nb_split);
        __nv_bfloat16* const dst_out = is_out2 ? pb.out2 : pb.out;
        const int dst_N = (pb.out2 != nullptr) ? (is_out2 ? pb.N2 : pb.N1) : pb.N;
        const int dst_nb = is_out2 ? (nb - pb.nb_split) : nb;
        __nv_bfloat162 r;
        r.x = __float2bfloat16_rn(c0);
        r.y = __float2bfloat16_rn(c1);
        *reinterpret_cast<__nv_bfloat162*>(dst_out + static_cast<size_t>(g) * dst_N + dst_nb * 8 + 2 * t) = r;
        if constexpr (kM16) {
          if (valid_m_hi) {
            __nv_bfloat162 r2;
            r2.x = __float2bfloat16_rn(c2);
            r2.y = __float2bfloat16_rn(c3);
            *reinterpret_cast<__nv_bfloat162*>(dst_out + static_cast<size_t>(g + 8) * dst_N + dst_nb * 8 + 2 * t) = r2;
          }
        }
      }
      if constexpr (kXsmem) {
        if (!xsmem_async && r_i + 1 < R_max) __syncthreads();
      }
    }
    if (active) asm volatile("cp.async.wait_group 0;" ::: "memory");
#undef BR_ISSUE_FAST
    return;
  }

  for (int nb = w; nb < NB; nb += pb.tw) {
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    const int trig_s = ((pb.pdl_mode & 2) && (nb + pb.tw >= NB)) ? ((S > 16) ? (S - 16) : 0) : -1;
    for (int s = 0; s < S; ++s) {
      if (s == trig_s) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
      // Pre-issue 4 (or 8 for kM16) L1/L2 activation loads for the two K16 MMAs
      const uint32_t a0 = valid_m ? __ldg(xrow32 + 16 * s) : 0u;
      const uint32_t a2 = valid_m ? __ldg(xrow32 + 16 * s + 4) : 0u;
      const uint32_t a4 = valid_m ? __ldg(xrow32 + 16 * s + 8) : 0u;
      const uint32_t a6 = valid_m ? __ldg(xrow32 + 16 * s + 12) : 0u;
      const uint32_t a1 = (kM16 && valid_m_hi) ? __ldg(xrow32_hi + 16 * s) : 0u;
      const uint32_t a3 = (kM16 && valid_m_hi) ? __ldg(xrow32_hi + 16 * s + 4) : 0u;
      const uint32_t a5 = (kM16 && valid_m_hi) ? __ldg(xrow32_hi + 16 * s + 8) : 0u;
      const uint32_t a7 = (kM16 && valid_m_hi) ? __ldg(xrow32_hi + 16 * s + 12) : 0u;

      __syncwarp();
      const int curg = static_cast<int>(pwb >> 9);
      while (issued <= curg + static_cast<int>(kRG) - 1) BR_ISSUE();
      asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 2) : "memory");
      __syncwarp();

      const uint32_t na = pwb + 8u * static_cast<uint32_t>(lane);
      uint32_t sm_w0, sm_w1;
      if ((pwb & 2u) == 0u) {
        sm_w0 = br_lds32(ring_s + (na & (kRB - 1u)));
        sm_w1 = br_lds32(ring_s + ((na + 4u) & (kRB - 1u)));
      } else {
        sm_w0 = br_lds16(ring_s + (na & (kRB - 1u))) | (br_lds16(ring_s + ((na + 2u) & (kRB - 1u))) << 16);
        sm_w1 = br_lds16(ring_s + ((na + 4u) & (kRB - 1u))) | (br_lds16(ring_s + ((na + 6u) & (kRB - 1u))) << 16);
      }
      pwb += 256u;

      // 4 independent 12-bit rANS decodes -> 8 4-bit exponent codes in Pw
      uint32_t Pw = 0u;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t& x = st[q];
        const uint32_t e = sh_lut[x & 4095u];
        const uint32_t xh = x >> 12;
        x = ((e >> 8) & 4095u) * xh + xh + (e >> 20);
        Pw = __byte_perm(Pw, e, (q == 0) ? 0x3214 : (q == 1) ? 0x3240 : (q == 2) ? 0x3410 : 0x4210);
        const bool need = x < kRansL;
        const uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) x = (x << 16) | br_lds16(ring_s + ((pwb + (__popc(m & lt) << 1)) & (kRB - 1u)));
        pwb += __popc(m) << 1;
      }

      const uint32_t hi4 = (Pw >> 4) & 0x0F0F0F0Fu;
      const uint32_t lo4 = Pw & 0x0F0F0F0Fu;
      const uint32_t esc_bits = ((hi4 + 0x01010101u) | (lo4 + 0x01010101u)) & 0x10101010u;

      uint32_t c0_ = hi4 & 15u, c1_ = lo4 & 15u;
      uint32_t c2_ = (hi4 >> 8) & 15u, c3_ = (lo4 >> 8) & 15u;
      uint32_t c4_ = (hi4 >> 16) & 15u, c5_ = (lo4 >> 16) & 15u;
      uint32_t c6_ = (hi4 >> 24) & 15u, c7_ = (lo4 >> 24) & 15u;

      uint32_t e0 = c0_ ? (base_m1 + c0_) : 0u;
      uint32_t e1 = c1_ ? (base_m1 + c1_) : 0u;
      uint32_t e2 = c2_ ? (base_m1 + c2_) : 0u;
      uint32_t e3 = c3_ ? (base_m1 + c3_) : 0u;
      uint32_t e4 = c4_ ? (base_m1 + c4_) : 0u;
      uint32_t e5 = c5_ ? (base_m1 + c5_) : 0u;
      uint32_t e6 = c6_ ? (base_m1 + c6_) : 0u;
      uint32_t e7 = c7_ ? (base_m1 + c7_) : 0u;

      if (__any_sync(0xffffffffu, esc_bits != 0u)) {
        // Rare escape slow-path (<10% of warp-steps): ensure ring buffer has room and read raw u16 exponents
        const int cg2 = static_cast<int>(pwb >> 9);
        while (issued <= cg2 + static_cast<int>(kRG) - 1) BR_ISSUE();
        asm volatile("cp.async.wait_group %0;" ::"n"(kRG - 2) : "memory");
        __syncwarp();
#define BR_ESC(ci, ei)                                                                             \
  {                                                                                                \
    const bool is_e = ((ci) == 15u);                                                               \
    const uint32_t em = __ballot_sync(0xffffffffu, is_e);                                          \
    if (em) {                                                                                      \
      if (is_e) (ei) = br_lds16(ring_s + ((pwb + (__popc(em & lt) << 1)) & (kRB - 1u)));           \
      pwb += __popc(em) << 1;                                                                      \
    }                                                                                              \
  }
        BR_ESC(c0_, e0) BR_ESC(c1_, e1) BR_ESC(c2_, e2) BR_ESC(c3_, e3)
        BR_ESC(c4_, e4) BR_ESC(c5_, e5) BR_ESC(c6_, e6) BR_ESC(c7_, e7)
#undef BR_ESC
      }

      const uint32_t sm01 = __byte_perm(sm_w0, 0u, 0x4140);
      const uint32_t sm23 = __byte_perm(sm_w0, 0u, 0x4342);
      const uint32_t b0 = ((sm01 & 0x00800080u) << 8) | (sm01 & 0x007F007Fu) | (e0 << 7) | (e1 << 23);
      const uint32_t b1 = ((sm23 & 0x00800080u) << 8) | (sm23 & 0x007F007Fu) | (e2 << 7) | (e3 << 23);
      if constexpr (kM16) {
        bf16_mma_m16n8k16_m16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
      } else {
        bf16_mma_m16n8k16(c0, c1, c2, c3, a0, a2, b0, b1);
      }

      const uint32_t sm45 = __byte_perm(sm_w1, 0u, 0x4140);
      const uint32_t sm67 = __byte_perm(sm_w1, 0u, 0x4342);
      const uint32_t b2 = ((sm45 & 0x00800080u) << 8) | (sm45 & 0x007F007Fu) | (e4 << 7) | (e5 << 23);
      const uint32_t b3 = ((sm67 & 0x00800080u) << 8) | (sm67 & 0x007F007Fu) | (e6 << 7) | (e7 << 23);
      if constexpr (kM16) {
        bf16_mma_m16n8k16_m16(c0, c1, c2, c3, a4, a5, a6, a7, b2, b3);
      } else {
        bf16_mma_m16n8k16(c0, c1, c2, c3, a4, a6, b2, b3);
      }
    }
    if (valid_m) {
      const bool is_out2 = (pb.out2 != nullptr && nb >= pb.nb_split);
      __nv_bfloat16* const dst_out = is_out2 ? pb.out2 : pb.out;
      const int dst_N = (pb.out2 != nullptr) ? (is_out2 ? pb.N2 : pb.N1) : pb.N;
      const int dst_nb = is_out2 ? (nb - pb.nb_split) : nb;
      __nv_bfloat162 r;
      r.x = __float2bfloat16_rn(c0);
      r.y = __float2bfloat16_rn(c1);
      *reinterpret_cast<__nv_bfloat162*>(dst_out + static_cast<size_t>(g) * dst_N + dst_nb * 8 + 2 * t) = r;
      if constexpr (kM16) {
        if (valid_m_hi) {
          __nv_bfloat162 r2;
          r2.x = __float2bfloat16_rn(c2);
          r2.y = __float2bfloat16_rn(c3);
          *reinterpret_cast<__nv_bfloat162*>(dst_out + static_cast<size_t>(g + 8) * dst_N + dst_nb * 8 + 2 * t) = r2;
        }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#undef BR_ISSUE
}

// ---------------------------------------------------------------- host helpers
static inline size_t br_woff_off() { return kHdrBytes + kLutBytes; }
static inline size_t br_stream_off(int tw) {
  return (br_woff_off() + static_cast<size_t>(2 * tw + 1) * 4u + 511u) & ~static_cast<size_t>(511u);
}

struct BrCfg {
  int tw;
  int blocks;
  int wpb;
};

static BrCfg br_pick_cfg(int N, int K) {
  const int NB = N / 8;
  static const bool gate32 = []() {
    const char* env = std::getenv("MUSE_BF16_GATE_32_BLOCKS");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  if (NB == 4992) return BrCfg{624, 48, 13};   // gate_up (39936): exact R=8 chunks/warp
  if (NB == 768)  return BrCfg{768, 48, 16};   // qkv (6144): exact R=1
  if (NB == 1280) return BrCfg{640, 48, 14};   // fkv (10240): exact R=2
  if (NB == 1088) return BrCfg{544, 48, 12};   // stacked target qkv+output_gate (8704): exact R=2, 48 SMs
  if (NB == 832)  return BrCfg{832, 48, 18};   // o_proj / down / fc (6656)
  if (NB == 576)  return BrCfg{576, 48, 12};   // target qkv (4608)
  if (NB == 512) {
    // Target output_gate (4096x6656): 32 blocks x 16 warps leaves 16 SMs free for side-stream FlashAttention
    if (gate32 && K == 6656) return BrCfg{512, 32, 16};
    return BrCfg{512, 48, 11};
  }
  const int blocks = std::min(NB, 48);
  for (int wcap = 18; wcap >= 1; --wcap) {
    const int R = (NB + blocks * wcap - 1) / (blocks * wcap);
    const int tw = (NB + R - 1) / R;
    const int wpb = (tw + blocks - 1) / blocks;
    if (wpb <= 20) return BrCfg{tw, blocks, wpb};
  }
  return BrCfg{NB, blocks, (NB + blocks - 1) / blocks};
}

static void br_normalize_4096(const int64_t* counts, int32_t* f) {
  int64_t total = 0;
  int active = 0;
  for (int i = 0; i < 256; ++i) {
    total += counts[i];
    if (counts[i] > 0) ++active;
  }
  TORCH_CHECK(total > 0 && active <= 4096);
  int sum = 0;
  for (int i = 0; i < 256; ++i) {
    if (counts[i] > 0) {
      long v = static_cast<long>(std::floor(static_cast<double>(counts[i]) * 4096.0 / static_cast<double>(total)));
      f[i] = static_cast<int>(std::max(1L, v));
    } else {
      f[i] = 0;
    }
    sum += f[i];
  }
  while (sum < 4096) {
    int best = -1;
    double best_gain = -1.0;
    for (int i = 0; i < 256; ++i) {
      if (f[i] <= 0) continue;
      double gain = static_cast<double>(counts[i]) * std::log((static_cast<double>(f[i]) + 1.0) / static_cast<double>(f[i]));
      if (gain > best_gain) { best_gain = gain; best = i; }
    }
    TORCH_CHECK(best >= 0, "cannot normalize bf16 pair rANS frequencies (+)");
    f[best]++;
    sum++;
  }
  while (sum > 4096) {
    int best = -1;
    double best_loss = 1e300;
    for (int i = 0; i < 256; ++i) {
      if (f[i] <= 1) continue;
      double loss = static_cast<double>(counts[i]) * std::log(static_cast<double>(f[i]) / (static_cast<double>(f[i]) - 1.0));
      if (loss < best_loss) { best_loss = loss; best = i; }
    }
    TORCH_CHECK(best >= 0, "cannot normalize bf16 pair rANS frequencies (-)");
    f[best]--;
    sum--;
  }
}

static void br_set_attrs_once() {
  static bool done = false;
  if (done) return;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
  // All production shapes use wpb <= 18 -> dsmem <= 40 KiB (+ 16 KiB LUT <= 64 KiB) for non-xsmem,
  // and <= 72 KiB (+ 16 KiB LUT <= 88 KiB <= 99 KiB) for cooperative K-chunked xsmem.
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<true, false, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(kMaxSmem - kLutBytes));
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<true, true, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(kMaxSmem - kLutBytes));
  cudaFuncSetAttribute(br_pair_mma_gemm_kernel<false, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(kMaxSmem - kLutBytes));
  cudaThreadExchangeStreamCaptureMode(&mode);
  done = true;
}

template <bool kPar4, bool kM16 = false, bool kXsmem = false>
static inline void br_launch_k(int blocks, int threads, size_t dsmem, cudaStream_t st, BrProb pb) {
  if ((pb.pdl_mode & 1) != 0) {
    cudaLaunchConfig_t lcfg = {};
    lcfg.gridDim = dim3(blocks, 1, 1);
    lcfg.blockDim = dim3(threads, 1, 1);
    lcfg.dynamicSmemBytes = dsmem;
    lcfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(&lcfg, br_pair_mma_gemm_kernel<kPar4, kM16, kXsmem>, pb);
    TORCH_CHECK(err == cudaSuccess, "br_pair_mma_gemm_kernel PDL launch failed: ", cudaGetErrorString(err));
  } else {
    br_pair_mma_gemm_kernel<kPar4, kM16, kXsmem><<<blocks, threads, dsmem, st>>>(pb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

torch::Tensor bf16_rans_pack(torch::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "W must be contiguous CUDA bfloat16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N > 0 && K > 0 && N % 8 == 0 && K % 32 == 0, "need N%8==0 and K%32==0");
  const int NB = N / 8;
  const BrCfg cfg = br_pick_cfg(N, K);
  const int tw = cfg.tw;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto dev_u8 = W.options().dtype(torch::kUInt8);

  // 1. Find best 14-wide contiguous exponent window [base .. base+13] (+ exp=0)
  auto ecounts_d = torch::zeros({256}, W.options().dtype(torch::kInt64));
  br_exp_hist_kernel<<<1024, 256, 0, st>>>(
      reinterpret_cast<const uint16_t*>(W.data_ptr()), static_cast<size_t>(W.numel()),
      reinterpret_cast<unsigned long long*>(ecounts_d.data_ptr()));
  auto ecounts_h = ecounts_d.cpu();
  const int64_t* ec = reinterpret_cast<const int64_t*>(ecounts_h.data_ptr());
  uint32_t best_base = 110u;
  int64_t best_sum = -1;
  for (uint32_t b = 1; b <= 241; ++b) {
    int64_t s = 0;
    for (uint32_t d = 0; d < 14; ++d) s += ec[b + d];
    if (s > best_sum) {
      best_sum = s;
      best_base = b;
    }
  }

  // 2. Pair histogram over 256 (c0<<4 | c1) symbols
  auto pcounts_d = torch::zeros({256}, W.options().dtype(torch::kInt64));
  br_pair_hist_kernel<<<1024, 256, 0, st>>>(
      reinterpret_cast<const uint16_t*>(W.data_ptr()), static_cast<size_t>(W.numel()), best_base,
      reinterpret_cast<unsigned long long*>(pcounts_d.data_ptr()));
  auto pcounts_h = pcounts_d.cpu();
  int32_t f[256], cum[256];
  br_normalize_4096(reinterpret_cast<const int64_t*>(pcounts_h.data_ptr()), f);
  int c = 0;
  for (int i = 0; i < 256; ++i) {
    cum[i] = c;
    c += f[i];
  }
  std::vector<uint32_t> lut(4096);
  for (int s = 0; s < 256; ++s) {
    for (int k = 0; k < f[s]; ++k) {
      lut[cum[s] + k] = static_cast<uint32_t>(s) | (static_cast<uint32_t>(f[s] - 1) << 8) | (static_cast<uint32_t>(k) << 20);
    }
  }
  std::vector<BrPairEncSym> tab(256);
  for (int s = 0; s < 256; ++s) {
    if (f[s] == 0) {
      tab[s] = BrPairEncSym{0ull, 0xFFFFFFFFu, 0u};
      continue;
    }
    unsigned long long rcp = (f[s] == 1)
                                 ? 0ull
                                 : static_cast<unsigned long long>(
                                       ((static_cast<unsigned __int128>(1) << 64) + static_cast<unsigned>(f[s]) - 1) /
                                       static_cast<unsigned>(f[s]));
    unsigned long long xm = (static_cast<unsigned long long>(f[s]) << 20) - 1ull;
    tab[s] = BrPairEncSym{rcp, static_cast<uint32_t>(std::min<unsigned long long>(xm, 0xFFFFFFFFull)),
                          static_cast<uint32_t>(cum[s]) | (static_cast<uint32_t>(4096 - f[s]) << 16)};
  }
  auto tab_d = torch::empty({256 * static_cast<int64_t>(sizeof(BrPairEncSym))}, dev_u8);
  cudaMemcpyAsync(tab_d.data_ptr(), tab.data(), 256 * sizeof(BrPairEncSym), cudaMemcpyHostToDevice, st);

  auto wwords_d = torch::empty({tw}, W.options().dtype(torch::kInt32));
  const int enc_blocks = (tw + 3) / 4;
  br_pair_enc_pass1_kernel<<<enc_blocks, 128, 0, st>>>(
      reinterpret_cast<const uint16_t*>(W.data_ptr()), NB, K, tw, best_base,
      reinterpret_cast<const BrPairEncSym*>(tab_d.data_ptr()),
      reinterpret_cast<uint32_t*>(wwords_d.data_ptr()));
  auto wwords_h = wwords_d.cpu();
  const uint32_t* ww = reinterpret_cast<const uint32_t*>(wwords_h.data_ptr());

  std::vector<int32_t> woff16(tw + 1, 0);
  for (int w = 0; w < tw; ++w) {
    woff16[w + 1] = woff16[w] + static_cast<int32_t>((128 * kNS + 2 * static_cast<int64_t>(ww[w]) + 15) / 16);
  }
  std::vector<int64_t> G(tw);
  int64_t sum_g = 0;
  for (int w = 0; w < tw; ++w) {
    G[w] = ((static_cast<int64_t>(woff16[w + 1] - woff16[w]) * 16) + 511) >> 9;
    sum_g += G[w];
  }
  std::vector<int64_t> Gs = G;
  std::sort(Gs.begin(), Gs.end());
  int64_t gmin64 = Gs[0];
  static const int br_ilv_pct_env = []() -> int {
    const char* env = std::getenv("MUSE_BF16_RANS_ILV_PCT");
    if (!env || !*env) return -1;
    long v = std::strtol(env, nullptr, 10);
    return (v >= 0 && v <= 50) ? static_cast<int>(v) : -1;
  }();
  const int br_ilv_pct = (br_ilv_pct_env >= 0) ? br_ilv_pct_env : ((N == 4608 || N == 8704) ? 12 : 0);
  if (br_ilv_pct > 0 && tw >= 16) {
    int idx = static_cast<int>(static_cast<int64_t>(tw) * br_ilv_pct / 100);
    if (idx >= tw) idx = tw - 1;
    const int64_t max_pad = sum_g / 50;  // <= 2% virtual padding cap; short warps skip via boff < wbytes
    while (idx > 0) {
      const int64_t cand = Gs[idx];
      int64_t pad = 0;
      for (int j = 0; j < idx; ++j) pad += (cand - Gs[j]);
      if (pad <= max_pad) {
        gmin64 = cand;
        break;
      }
      --idx;
    }
  }
  const int32_t gmin = static_cast<int32_t>(gmin64);
  std::vector<int32_t> tab_i(static_cast<size_t>(2 * tw + 1), 0);
  for (int w = 0; w <= tw; ++w) tab_i[w] = woff16[w];
  int64_t tg = static_cast<int64_t>(gmin) * tw;
  for (int w = 0; w < tw; ++w) {
    tab_i[tw + 1 + w] = static_cast<int32_t>(tg);
    if (G[w] > gmin) tg += G[w] - gmin;
  }
  const size_t stream_bytes = static_cast<size_t>(tg) << 9;
  const size_t soff = br_stream_off(tw);
  const size_t total = soff + stream_bytes + 64;
  auto blob = torch::zeros({static_cast<int64_t>(total)}, dev_u8);
  uint8_t* base_ptr = reinterpret_cast<uint8_t*>(blob.data_ptr());

  uint32_t hdr[64] = {0};
  hdr[0] = kMagic;
  hdr[1] = kVersion;
  hdr[2] = static_cast<uint32_t>(N);
  hdr[3] = static_cast<uint32_t>(K);
  hdr[4] = static_cast<uint32_t>(tw);
  hdr[5] = static_cast<uint32_t>(kNS);
  hdr[6] = static_cast<uint32_t>(NB);
  hdr[7] = 9u;
  hdr[8] = kHdrBytes;
  hdr[9] = static_cast<uint32_t>(br_woff_off());
  hdr[10] = static_cast<uint32_t>(soff);
  hdr[11] = static_cast<uint32_t>(stream_bytes & 0xFFFFFFFFu);
  hdr[12] = static_cast<uint32_t>(total & 0xFFFFFFFFu);
  hdr[13] = static_cast<uint32_t>(total >> 32);
  hdr[14] = static_cast<uint32_t>(gmin);
  hdr[15] = static_cast<uint32_t>(stream_bytes >> 32);
  hdr[16] = best_base;

  cudaMemcpyAsync(base_ptr, hdr, sizeof(hdr), cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(base_ptr + kHdrBytes, lut.data(), kLutBytes, cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(base_ptr + br_woff_off(), tab_i.data(), tab_i.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st);

  const int* woff_dev = reinterpret_cast<const int*>(base_ptr + br_woff_off());
  br_pair_enc_pass2_kernel<<<enc_blocks, 128, 0, st>>>(
      reinterpret_cast<const uint16_t*>(W.data_ptr()), NB, K, tw, gmin, best_base,
      reinterpret_cast<const BrPairEncSym*>(tab_d.data_ptr()),
      reinterpret_cast<const uint32_t*>(wwords_d.data_ptr()), woff_dev, base_ptr + soff);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaStreamSynchronize(st);
  return blob;
}

torch::Tensor bf16_rans_unpack(torch::Tensor blob, int64_t N, int64_t K) {
  auto h = blob.narrow(0, 0, 256).cpu();
  const uint32_t* p = reinterpret_cast<const uint32_t*>(h.data_ptr());
  TORCH_CHECK(p[0] == kMagic && p[1] == kVersion && p[2] == N && p[3] == K, "bad bf16 pair rANS blob");
  const int tw = static_cast<int>(p[4]);
  const int NB = static_cast<int>(p[6]);
  const int gmin = static_cast<int>(p[14]);
  const size_t soff = static_cast<size_t>(p[10]);
  const uint32_t base = p[16];
  auto out = torch::empty({N, K}, blob.options().dtype(torch::kBFloat16));
  auto status = torch::zeros({1}, blob.options().dtype(torch::kInt32));
  const uint8_t* b = reinterpret_cast<const uint8_t*>(blob.data_ptr());
  auto st = c10::cuda::getCurrentCUDAStream();
  br_pair_unpack_kernel<<<(tw + 3) / 4, 128, 0, st>>>(
      b + soff, reinterpret_cast<const int*>(b + br_woff_off()),
      reinterpret_cast<const uint32_t*>(b + kHdrBytes),
      reinterpret_cast<uint16_t*>(out.data_ptr()), NB, static_cast<int>(K), tw, gmin, base,
      reinterpret_cast<int*>(status.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(status.item<int>() == 0, "bf16_rans_unpack: final rANS state mismatch");
  return out;
}

static void bf16_rans_gemm_launch(
    torch::Tensor x, torch::Tensor blob,
    __nv_bfloat16* out1_ptr, __nv_bfloat16* out2_ptr,
    int64_t N, int64_t N1, int64_t N2, int64_t K, int64_t base_exp) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
              "x must be contiguous CUDA bf16 [M, K]");
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && x.size(1) == K && N % 8 == 0 && K % 32 == 0, "bad shape");
  TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == torch::kUInt8 && blob.is_contiguous(), "bad blob");

  const BrCfg cfg = br_pick_cfg(static_cast<int>(N), static_cast<int>(K));
  const size_t soff = br_stream_off(cfg.tw);
  const size_t blob_bytes = static_cast<size_t>(blob.numel());
  const uint8_t* b = reinterpret_cast<const uint8_t*>(blob.data_ptr());

  BrProb pb;
  pb.hdr = b;
  pb.stream = b + soff;
  pb.woff16 = reinterpret_cast<const int*>(b + br_woff_off());
  pb.lut = reinterpret_cast<const uint32_t*>(b + kHdrBytes);
  pb.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  pb.out = out1_ptr;
  pb.out2 = out2_ptr;
  pb.M = M;
  pb.N = static_cast<int>(N);
  pb.K = static_cast<int>(K);
  pb.N1 = static_cast<int>(N1);
  pb.N2 = static_cast<int>(N2);
  pb.nb_split = static_cast<int>(N1 >> 3);
  pb.tw = cfg.tw;
  const int pdl_cfg = br_pdl_cfg();
  pb.pdl_mode = (pdl_cfg == 2) ? 3 : ((pdl_cfg == 1) ? 1 : 0);
  static const bool xsmem_async_en = []() {
    const char* env = std::getenv("MUSE_BF16_RANS_XSMEM_ASYNC");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  pb.xsmem_async = xsmem_async_en ? 1 : 0;
  pb.base_m1 = static_cast<uint32_t>(base_exp) - 1u;
  pb.stream_bytes = (blob_bytes > soff + 64u) ? (blob_bytes - soff - 64u) : 0u;
  pb.tpf_stride = br_tpf_stride();

  TORCH_CHECK(cfg.wpb <= 20, "bf16_rans_gemm_launch: wpb exceeds launch bounds (20)");
  size_t dsmem = kRB + static_cast<size_t>(cfg.wpb) * kRB;
  if (cfg.blocks == 48 && dsmem < 40960u) dsmem = 40960u;
  const size_t dsmem_xsmem = kBrXsmemBytes + kRB + static_cast<size_t>(cfg.wpb) * kRB;
  br_set_attrs_once();

  static const bool par4_en = []() {
    const char* env = std::getenv("MUSE_BF16_RANS_PAR4");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  static const bool xsmem_en = []() {
    const char* env = std::getenv("MUSE_BF16_RANS_XSMEM");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const bool x_aligned16 = ((reinterpret_cast<uintptr_t>(pb.x) & 15u) == 0u);
  if (par4_en && x_aligned16) {
    if (M > 8) {
      if (xsmem_en && ((K >> 5) % 16 == 0) && dsmem_xsmem <= static_cast<size_t>(kMaxSmem - kLutBytes)) {
        br_launch_k<true, true, true>(cfg.blocks, cfg.wpb * 32, dsmem_xsmem, st, pb);
      } else if (dsmem <= 48u * 1024u) {
        br_launch_k<true, true, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
      } else {
        br_launch_k<false, true, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
      }
    } else if (dsmem <= 48u * 1024u) {
      br_launch_k<true, false, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
    } else {
      br_launch_k<false, false, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
    }
  } else {
    if (M > 8) {
      br_launch_k<false, true, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
    } else {
      br_launch_k<false, false, false>(cfg.blocks, cfg.wpb * 32, dsmem, st, pb);
    }
  }
}

static void bf16_rans_gemm_into(torch::Tensor x, torch::Tensor blob, torch::Tensor out, int64_t N, int64_t K, int64_t base_exp) {
  const int64_t M = x.size(0);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.numel() == M * N,
              "bad out");
  bf16_rans_gemm_launch(x, blob, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr, N, N, 0, K, base_exp);
}

torch::Tensor bf16_rans_gemm(torch::Tensor x, torch::Tensor blob, int64_t N, int64_t K, int64_t base_exp) {
  auto out = torch::empty({x.size(0), N}, x.options());
  bf16_rans_gemm_into(x, blob, out, N, K, base_exp);
  return out;
}

void bf16_rans_gemm_out(torch::Tensor x, torch::Tensor blob, torch::Tensor out, int64_t N, int64_t K, int64_t base_exp) {
  bf16_rans_gemm_into(x, blob, out, N, K, base_exp);
}

void bf16_rans_gemm_dual_out(
    torch::Tensor x, torch::Tensor blob, torch::Tensor out1, torch::Tensor out2,
    int64_t N1, int64_t N2, int64_t K, int64_t base_exp) {
  const int64_t M = x.size(0);
  TORCH_CHECK(N1 > 0 && N2 > 0 && N1 % 8 == 0 && N2 % 8 == 0, "bad dual N split");
  TORCH_CHECK(out1.is_cuda() && out1.scalar_type() == torch::kBFloat16 && out1.is_contiguous() &&
              out1.numel() == M * N1, "bad out1");
  TORCH_CHECK(out2.is_cuda() && out2.scalar_type() == torch::kBFloat16 && out2.is_contiguous() &&
              out2.numel() == M * N2, "bad out2");
  bf16_rans_gemm_launch(
      x, blob,
      reinterpret_cast<__nv_bfloat16*>(out1.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out2.data_ptr()),
      N1 + N2, N1, N2, K, base_exp);
}

std::tuple<torch::Tensor, torch::Tensor> bf16_rans_gemm_dual(
    torch::Tensor x, torch::Tensor blob, int64_t N1, int64_t N2, int64_t K, int64_t base_exp) {
  auto out1 = torch::empty({x.size(0), N1}, x.options());
  auto out2 = torch::empty({x.size(0), N2}, x.options());
  bf16_rans_gemm_dual_out(x, blob, out1, out2, N1, N2, K, base_exp);
  return std::make_tuple(out1, out2);
}

int64_t bf16_rans_base(torch::Tensor blob) {
  auto h = blob.narrow(0, 0, 256).cpu();
  const uint32_t* p = reinterpret_cast<const uint32_t*>(h.data_ptr());
  TORCH_CHECK(p[0] == kMagic && p[1] == kVersion, "bad bf16 pair rANS blob");
  return static_cast<int64_t>(p[16]);
}
