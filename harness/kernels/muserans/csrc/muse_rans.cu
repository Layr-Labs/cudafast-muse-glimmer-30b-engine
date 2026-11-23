// =====================================================================================================================
// muse_rans.cu -- lossless-compressed FP8 (e4m3) weights + bit-exact M=1 GEMV for GB10 (sm_121a).
//
//   rans_pack(W)            uint8 cuda [N,K] (FP8 e4m3 bytes, contiguous, N%8==0, K%64==0)  ->  ONE self-describing uint8 cuda blob (~0.82-0.84 x N*K)
//   rans_gemv_auto(blob, x, sb, sa, out, N, K)   out[n] = bf16_rn( sa * ( sb[n] * acc_n ) ),  acc_n = fp32 MMA chain over k32 steps (k ascending)
//   rans_unpack(blob, N, K) -> the original bytes (bit-exact round trip; debug / verification)
//
// Output is BIT-IDENTICAL to vLLM cutlass_scaled_mm (SM120 FP8, M<=16 tile) and to the uncompressed kernel M1 (muse_fp8.cu fp8_gemv): the weights are
// decoded losslessly and fed to the same mma.sync.m16n8k32 e4m3 chain (one accumulator per 8-row chunk, k32 steps in order, activation in row 0).
//
// Format v7 (continuous per-warp exponent-pair rANS + raw nibble plane, 2-byte step alignment):
//   Each virtual warp w carries ONE continuous rANS stream across all its R chunks (nb = w, w + tw, ...):
//   [128*NS B initial rANS states ONCE per warp], then for every chunk nb and every k32 step s:
//   [32 x u32 nibble row (2-byte aligned, no 4-byte step pad)][rANS renorm u16 words of the 4 pair iterations,
//   in (iteration, lane) order with ballot+popc], with no inter-chunk state reset or 16-byte chunk padding.
// =====================================================================================================================
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef MUSE_MMA_KIND
#define MUSE_MMA_KIND 0
#endif

#define RANS_L 65536u
#define RANS_LUT_BYTES 16384
#define RANS_HDR_BYTES 256
#define RANS_MAX_SMEM (99 * 1024)
#ifndef RANS_PREISSUE
#define RANS_PREISSUE 1
#endif

__device__ __forceinline__ void rans_mma_e4m3(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
#if MUSE_MMA_KIND == 0
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
#else
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint64_t rans_policy_evict_first() { uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ void rans_cp_async16_s(uint32_t smem_dst, const void* gsrc, uint64_t pol) { asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(smem_dst), "l"(gsrc), "l"(pol)); }
__device__ __forceinline__ void rans_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void rans_cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ uint32_t rans_lds32(uint32_t addr) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }
__device__ __forceinline__ uint32_t rans_lds16(uint32_t addr) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr)); return (uint32_t)v; }

__device__ __forceinline__ size_t rans_phys(const int* __restrict__ woff16, int tw, int gmin, int ilv, int w, uint32_t lb) {
  if (ilv == 0) return (size_t)woff16[w] * 16 + lb;
  const uint32_t g = lb >> ilv;
  const size_t gi = (g < (uint32_t)gmin) ? ((size_t)g * (size_t)tw + (size_t)w) : ((size_t)woff16[tw + 1 + w] + (size_t)(g - (uint32_t)gmin));
  return (gi << ilv) | (size_t)(lb & ((1u << ilv) - 1u));
}

// ------------------------------------------------------------------------------------------------ pair histogram
__global__ void rans_hist_kernel(const uint8_t* __restrict__ W, size_t n, unsigned long long* counts) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const uint4* W4 = (const uint4*)W;
  size_t n16 = n / 16;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = W4[i];
    uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
      atomicAdd(&h[(((a[q] >> 3) & 15u) << 4) | ((a[q] >> 11) & 15u)], 1u);
      atomicAdd(&h[(((a[q] >> 19) & 15u) << 4) | ((a[q] >> 27) & 15u)], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&counts[i], (unsigned long long)h[i]);
}

// ------------------------------------------------------------------------------------------------ continuous-warp encoder
struct RansEncSym { unsigned long long rcp; uint32_t xmax_m1; uint32_t cum_nf; };

__device__ __forceinline__ uint32_t rans_enc_step(uint32_t x, const RansEncSym& e, bool emit) {
  if (emit) x >>= 16;
  uint32_t q = e.rcp ? (uint32_t)__umul64hi((unsigned long long)x, e.rcp) : x;
  return x + q * (e.cum_nf >> 16) + (e.cum_nf & 0xffffu);
}
__device__ __forceinline__ uint32_t rans_pair(uint32_t b0, uint32_t b1, int q) {
  const uint32_t w = (q < 2) ? b0 : b1;
  const int sh = 16 * (q & 1);
  return (((w >> (sh + 3)) & 15u) << 4) | ((w >> (sh + 11)) & 15u);
}

template <int NS>
__global__ void __launch_bounds__(128) rans_enc_pass1(const uint8_t* __restrict__ W, int NB, int K, int tw,
                                                      const RansEncSym* __restrict__ gtab, uint32_t* __restrict__ wwords) {
  __shared__ RansEncSym tab[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = gtab[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  int nblast = w;
  while (nblast + tw < NB) nblast += tw;
  uint32_t states[NS], total = 0;
#pragma unroll
  for (int stream = 0; stream < NS; ++stream) states[stream] = RANS_L;
  for (int nb = nblast; nb >= w; nb -= tw) {
    const uint8_t* row = W + (size_t)(nb * 8 + g) * K;
    for (int s = S - 1; s >= 0; --s) {
      uint32_t b0 = *(const uint32_t*)(row + 32 * s + 4 * t), b1 = *(const uint32_t*)(row + 32 * s + 16 + 4 * t);
      uint32_t ws = 0;
#pragma unroll
      for (int q = 3; q >= 0; --q) {
        uint32_t& x = states[q % NS];
        RansEncSym e = tab[rans_pair(b0, b1, q)];
        bool emit = x > e.xmax_m1;
        ws += __popc(__ballot_sync(0xffffffffu, emit));
        x = rans_enc_step(x, e, emit);
      }
      total += 64u + ws;
    }
  }
  if (lane == 0) wwords[w] = total;
}

template <int NS>
__global__ void __launch_bounds__(128) rans_enc_pass2(const uint8_t* __restrict__ W, int NB, int K, int tw, int gmin, int ilv,
                                                      const RansEncSym* __restrict__ gtab, const uint32_t* __restrict__ wwords,
                                                      const int* __restrict__ woff16, uint8_t* __restrict__ stream) {
  __shared__ RansEncSym tab[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = gtab[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  const uint32_t lt = (1u << lane) - 1u;
  int nblast = w;
  while (nblast + tw < NB) nblast += tw;
  const uint32_t lbase = 128u * NS;
  uint32_t posw = wwords[w];
  uint32_t states[NS];
#pragma unroll
  for (int st = 0; st < NS; ++st) states[st] = RANS_L;
  for (int nb = nblast; nb >= w; nb -= tw) {
    const uint8_t* row = W + (size_t)(nb * 8 + g) * K;
    for (int s = S - 1; s >= 0; --s) {
      uint32_t b0 = *(const uint32_t*)(row + 32 * s + 4 * t), b1 = *(const uint32_t*)(row + 32 * s + 16 + 4 * t);
#pragma unroll
      for (int q = 3; q >= 0; --q) {
        uint32_t& x = states[q % NS];
        RansEncSym e = tab[rans_pair(b0, b1, q)];
        bool emit = x > e.xmax_m1;
        uint32_t m = __ballot_sync(0xffffffffu, emit);
        posw -= __popc(m);
        if (emit) *(uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, lbase + 2u * (posw + __popc(m & lt)))) = (uint16_t)(x & 0xffffu);
        x = rans_enc_step(x, e, emit);
      }
      posw -= 64u;
      uint32_t nw = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t by = (((i < 4) ? b0 : b1) >> (8 * (i & 3))) & 255u;
        nw |= ((((by >> 7) << 3) | (by & 7u))) << (4 * i);
      }
      const uint32_t nboff = lbase + 2u * posw + 4u * lane;
      *(uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, nboff)) = (uint16_t)(nw & 0xffffu);
      *(uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, nboff + 2u)) = (uint16_t)(nw >> 16);
    }
  }
#pragma unroll
  for (int state = 0; state < NS; ++state) {
    *(uint32_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, 128u * state + 4u * lane)) = states[state];
  }
}

// debug unpack: scalar reassembly, direct global reads
template <int NS>
__global__ void __launch_bounds__(128) rans_unpack_kernel(const uint8_t* __restrict__ stream, const int* __restrict__ woff16, const uint32_t* __restrict__ lut, uint8_t* __restrict__ out, int NB, int K, int tw, int gmin, int ilv,
                                                          int* __restrict__ status) {
  __shared__ uint32_t lut_s[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) lut_s[i] = lut[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const int g = lane >> 2, t = lane & 3, S = K >> 5;
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t states[NS];
#pragma unroll
  for (int state = 0; state < NS; ++state) {
    states[state] = *(const uint32_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, 128u * state + 4u * lane));
  }
  uint32_t pos = 128u * NS;
  for (int nb = w; nb < NB; nb += tw) {
    uint8_t* orow = out + (size_t)(nb * 8 + g) * K;
    for (int s = 0; s < S; s++) {
      const uint32_t nboff = pos + 4u * lane;
      const uint32_t nw = (uint32_t)(*(const uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, nboff))) |
                          ((uint32_t)(*(const uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, nboff + 2u))) << 16);
      pos += 128u;
      uint32_t ex[8];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t& x = states[q % NS];
        uint32_t e = lut_s[x & 4095u];
        uint32_t xh = x >> 12;
        x = ((e >> 8) & 4095u) * xh + xh + (e >> 20);
        ex[2 * q] = (e >> 4) & 15u; ex[2 * q + 1] = e & 15u;
        bool need = x < RANS_L;
        uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) x = (x << 16) | *(const uint16_t*)(stream + rans_phys(woff16, tw, gmin, ilv, w, pos + 2u * __popc(m & lt)));
        pos += 2u * __popc(m);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t nib = (nw >> (4 * i)) & 15u;
        orow[32 * s + ((i < 4) ? (4 * t + i) : (12 + 4 * t + i))] = (uint8_t)(((nib >> 3) << 7) | (ex[i] << 3) | (nib & 7u));
      }
    }
  }
  bool bad_state = false;
#pragma unroll
  for (int state = 0; state < NS; ++state) bad_state |= states[state] != RANS_L;
  if (__any_sync(0xffffffffu, bad_state) && lane == 0) atomicAdd(status, 1);
}

// ------------------------------------------------------------------------------------------------ decode + MMA GEMV
struct RansProb {
  const uint8_t* hdr; const uint8_t* stream; const int* woff16; const uint32_t* lut; const uint8_t* x; const float* sb; int sb_scalar; const float* sa;
  __nv_bfloat16* out; int N; int K; int tw; int ilv; size_t stream_bytes; size_t tpf_stride;
};

static size_t rans_tpf_stride() {
  static size_t stride = []() -> size_t {
    const char* env = std::getenv("MUSERANS_TLB_PREFETCH");
    if (!env || !*env) return 2048u << 10;  // 2 MiB default
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return stride;
}

template <int NS, int GBS, int RG>
__global__ void __launch_bounds__(1024, 1) rans_gemv_kernel(RansProb pb) {
  constexpr uint32_t GB = 1u << GBS;
  constexpr uint32_t RB = GB * (uint32_t)RG;
  __shared__ __align__(16) uint32_t lut_s[4096];
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, wib = tid >> 5, wpb = blockDim.x >> 5;
  const int K = pb.K, S = K >> 5;
  uint2* xs = (uint2*)smem;
  const int xbytes = (K + 15) & ~15;
  const uint32_t sbase0 = (uint32_t)__cvta_generic_to_shared(smem) + (uint32_t)xbytes;
  const uint32_t ring_s = ((sbase0 + RB - 1u) & ~(RB - 1u)) + (uint32_t)wib * RB;
  const int w = blockIdx.x * wpb + wib;
  const bool active = w < pb.tw;
  const uint8_t* ws = pb.stream;
  uint32_t wbytes = 0;
  int gmin = 0, toffw = 0, issued = 0;
  const uint64_t pol = rans_policy_evict_first();
#define RANS_ISSUE()                                                                                          \
  {                                                                                                         \
    _Pragma("unroll") for (uint32_t pc = 0; pc < GB / 512u; pc++) {                                         \
      uint32_t boff = (uint32_t)issued * GB + pc * 512u + (uint32_t)lane * 16u;                             \
      if (boff < wbytes) rans_cp_async16_s(ring_s + (boff & (RB - 1u)), pb.ilv ? (pb.stream + ((((issued < gmin) ? ((size_t)issued * pb.tw + w) : (size_t)(toffw + issued - gmin)) << GBS) + pc * 512u + (uint32_t)lane * 16u)) : (ws + boff), pol); \
    }                                                                                                       \
    rans_cp_commit();                                                                                         \
    issued++;                                                                                               \
  }
#if RANS_PREISSUE
  const bool fast_preissue = (pb.ilv != 0 && pb.K >= 2048 && RG <= 8);
#endif
  if (active) {
#if RANS_PREISSUE
    if (fast_preissue) {
#pragma unroll
      for (int i = 0; i < RG - 1; i++) {
        _Pragma("unroll") for (uint32_t pc = 0; pc < GB / 512u; pc++) {
          uint32_t boff = (uint32_t)i * GB + pc * 512u + (uint32_t)lane * 16u;
          rans_cp_async16_s(
              ring_s + (boff & (RB - 1u)),
              pb.stream + ((((size_t)i * pb.tw + w) << GBS) + pc * 512u + (uint32_t)lane * 16u),
              pol);
        }
        rans_cp_commit();
      }
      issued = RG - 1;
    } else
#endif
    {
      ws = pb.stream + (size_t)pb.woff16[w] * 16;
      wbytes = (uint32_t)(pb.woff16[w + 1] - pb.woff16[w]) * 16u;
      gmin = (int)((const uint32_t*)pb.hdr)[14];
      toffw = pb.woff16[pb.tw + 1 + w];
#if RANS_PREISSUE
#pragma unroll
      for (int i = 0; i < RG - 1; i++) RANS_ISSUE();
#endif
    }
  }
  if (blockIdx.x == 0 && pb.tpf_stride > 0) {
    // Issue one L2 prefetch per 2 MiB of the weight stream to overlap cold GMMU address-translation walks
    // with the prologue's LUT/activation loads and initial cp.async fills (ArthurOstapenko 75967574 idea).
    for (size_t off = (size_t)tid * pb.tpf_stride; off < pb.stream_bytes; off += (size_t)blockDim.x * pb.tpf_stride) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(pb.stream + off));
    }
  }
  for (int i = tid; i < RANS_LUT_BYTES / 16; i += blockDim.x) ((uint4*)lut_s)[i] = ((const uint4*)pb.lut)[i];
  for (int i = tid; i < S * 4; i += blockDim.x) {
    int s = i >> 2, tt = i & 3;
    xs[i] = make_uint2(*(const uint32_t*)(pb.x + 32 * s + 4 * tt), *(const uint32_t*)(pb.x + 32 * s + 16 + 4 * tt));
  }
#if RANS_PREISSUE
  if (active && fast_preissue) {
    wbytes = (uint32_t)(pb.woff16[w + 1] - pb.woff16[w]) * 16u;
    gmin = (int)((const uint32_t*)pb.hdr)[14];
    toffw = pb.woff16[pb.tw + 1 + w];
  }
#endif
  __syncthreads();
  if (!active) return;
  const int g = lane >> 2, t = lane & 3;
  const uint32_t lt = (1u << lane) - 1u;
  const int NB = pb.N >> 3;

  // Read initial rANS states once for the continuous warp stream
  rans_cp_wait<RG - 2>();
  __syncwarp();
  uint32_t states[NS];
#pragma unroll
  for (int state = 0; state < NS; ++state) {
    states[state] = rans_lds32(ring_s | ((128u * state + 4u * lane) & (RB - 1u)));
  }
  uint32_t pwb = 128u * NS;

  for (int nb = w; nb < NB; nb += pb.tw) {
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int s = 0; s < S; s++) {
      __syncwarp();
      const int curg = (int)(pwb >> GBS);
      while (issued <= curg + (RG - 1)) RANS_ISSUE();
      rans_cp_wait<RG - 2>();
      __syncwarp();
      const uint32_t na = pwb + 4u * (uint32_t)lane;
      uint32_t nw;
      if ((pwb & 2u) == 0u) {
        nw = rans_lds32(ring_s | (na & (RB - 1u)));
      } else {
        nw = rans_lds16(ring_s | (na & (RB - 1u))) | (rans_lds16(ring_s | ((na + 2u) & (RB - 1u))) << 16);
      }
      pwb += 128u;
      uint32_t Pw = 0;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t& x = states[q % NS];
        uint32_t e = ((uint32_t*)lut_s)[x & 4095u];
        uint32_t xh = x >> 12;
        x = ((e >> 8) & 4095u) * xh + xh + (e >> 20);
        Pw = __byte_perm(Pw, e, (q == 0) ? 0x3214 : (q == 1) ? 0x3240 : (q == 2) ? 0x3410 : 0x4210);
        bool need = x < RANS_L;
        uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) x = (x << 16) | rans_lds16(ring_s | ((pwb + (__popc(m & lt) << 1)) & (RB - 1u)));
        pwb += __popc(m) << 1;
      }
      const uint32_t hiE = (Pw >> 4) & 0x0F0F0F0Fu, loE = Pw & 0x0F0F0F0Fu;
      const uint32_t E0 = __byte_perm(hiE, loE, 0x5140), E1 = __byte_perm(hiE, loE, 0x7362);
      const uint32_t u0 = __byte_perm(nw, 0u, 0x1100), u1 = __byte_perm(nw, 0u, 0x3322);
      const uint32_t bw0 = (u0 & 0x80078007u) | ((u0 >> 4) & 0x07800780u) | (E0 << 3);
      const uint32_t bw1 = (u1 & 0x80078007u) | ((u1 >> 4) & 0x07800780u) | (E1 << 3);
      uint2 a = xs[s * 4 + t];
      a.x = (g == 0) ? a.x : 0u; a.y = (g == 0) ? a.y : 0u;
      rans_mma_e4m3(c0, c1, c2, c3, a.x, a.y, bw0, bw1);
    }
    if (lane < 4) {
      const float sa_v = __ldg(pb.sa);
      int n = nb * 8 + 2 * t;
      float s0 = pb.sb_scalar ? pb.sb[0] : pb.sb[n], s1 = pb.sb_scalar ? pb.sb[0] : pb.sb[n + 1];
      __nv_bfloat162 r; r.x = __float2bfloat16_rn(sa_v * (s0 * c0)); r.y = __float2bfloat16_rn(sa_v * (s1 * c1));
      *(__nv_bfloat162*)(pb.out + n) = r;
    }
  }
  rans_cp_wait<0>();
#undef RANS_ISSUE
}

// ------------------------------------------------------------------------------------------------ host side
struct RansCfg { int tw; int wpb; int ns; int rg; int ilv; };
static inline size_t rans_round16(size_t v) { return (v + 15) & ~(size_t)15; }
static inline size_t rans_woff_off() { return RANS_HDR_BYTES + RANS_LUT_BYTES; }
static inline size_t rans_stream_off(int tw) { return (rans_woff_off() + (size_t)(2 * tw + 1) * 4 + 511) & ~(size_t)511; }
static inline size_t rans_dsmem(int K, int wpb, int gbs, int rg) { size_t rb = (size_t)rg << gbs; return rans_round16((size_t)K) + rb + (size_t)wpb * rb; }
static inline bool rans_fits(int K, int wpb, int gbs, int rg) { return rans_dsmem(K, wpb, gbs, rg) + RANS_LUT_BYTES + 1024 <= (size_t)RANS_MAX_SMEM; }

static RansCfg rans_pick_cfg(int N, int K) {
  const int NB = N / 8;
  // Attention output_gate (4096x6656, 28 layers) runs concurrently with split-KV flash attention on a side stream:
  // launching 32 blocks of 16 warps (512 warps total) leaves 16 free SMs for flash attention (idea: ArthurOstapenko a9002906).
  static const bool gate32 = []() {
    const char* env = std::getenv("MUSERANS_GATE_32_BLOCKS");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  if (gate32 && N == 4096 && K == 6656 && NB == 512 && rans_fits(K, 16, 9, 4)) {
    return RansCfg{512, 16, 2, 4, 9};
  }
  for (int wcap = 18; wcap >= 1; wcap--) {
    const int R = (NB + 48 * wcap - 1) / (48 * wcap);
    const int tw = (NB + R - 1) / R, wpb = (tw + 47) / 48;
    if (rans_fits(K, wpb, 9, 4)) {
      const int states = (N == 6656 && K == 19968) ? 4 : 2;
      return RansCfg{tw, wpb, states, 4, 9};
    }
  }
  return RansCfg{NB, 1, 2, 4, 9};
}

static void rans_normalize(const int64_t* counts, int32_t* f) {
  int64_t total = 0;
  for (int i = 0; i < 256; i++) total += counts[i];
  TORCH_CHECK(total > 0);
  int sum = 0;
  for (int i = 0; i < 256; i++) {
    if (counts[i] > 0) { long v = std::lround((double)counts[i] * 4096.0 / (double)total); f[i] = (int)std::max(1L, v); } else f[i] = 0;
    sum += f[i];
  }
  int diff = 4096 - sum;
  while (diff != 0) {
    int k = 0;
    for (int i = 1; i < 256; i++) if (f[i] > f[k]) k = i;
    int step = diff > 0 ? diff : std::max(diff, -(f[k] - 1));
    TORCH_CHECK(step != 0, "cannot normalize frequencies");
    f[k] += step; diff -= step;
  }
}

torch::Tensor rans_pack_cfg(torch::Tensor W, int64_t tw, int64_t ns, int64_t ilv) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 2 && W.is_contiguous(), "W must be a contiguous uint8 cuda [N,K] tensor");
  const int N = (int)W.size(0), K = (int)W.size(1);
  TORCH_CHECK(N % 8 == 0 && K % 64 == 0, "need N%8==0 and K%64==0");
  TORCH_CHECK(ns == 1 || ns == 2 || ns == 4);
  TORCH_CHECK(ilv == 0 || ilv == 9, "ilv must be 0 or 9");
  const int NB = N / 8;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto dev_u8 = W.options();
  auto counts_d = torch::zeros({256}, W.options().dtype(torch::kInt64));
  rans_hist_kernel<<<1024, 256, 0, st>>>((const uint8_t*)W.data_ptr(), (size_t)W.numel(), (unsigned long long*)counts_d.data_ptr());
  auto counts_h = counts_d.cpu();
  int32_t f[256], cum[256];
  rans_normalize((const int64_t*)counts_h.data_ptr(), f);
  int c = 0;
  for (int i = 0; i < 256; i++) { cum[i] = c; c += f[i]; }
  std::vector<uint32_t> lut(4096);
  for (int s = 0; s < 256; s++)
    for (int k = 0; k < f[s]; k++) lut[cum[s] + k] = (uint32_t)s | ((uint32_t)(f[s] - 1) << 8) | ((uint32_t)k << 20);
  std::vector<RansEncSym> tab(256);
  for (int s = 0; s < 256; s++) {
    if (f[s] == 0) { tab[s] = RansEncSym{0ull, 0xFFFFFFFFu, 0u}; continue; }
    unsigned long long rcp = (f[s] == 1) ? 0ull : (unsigned long long)((((unsigned __int128)1 << 64) + (unsigned)f[s] - 1) / (unsigned)f[s]);
    unsigned long long xm = ((unsigned long long)f[s] << 20) - 1ull;
    tab[s] = RansEncSym{rcp, (uint32_t)std::min(xm, 0xFFFFFFFFull), (uint32_t)cum[s] | ((uint32_t)(4096 - f[s]) << 16)};
  }
  auto tab_d = torch::empty({256 * (int64_t)sizeof(RansEncSym)}, dev_u8);
  cudaMemcpyAsync(tab_d.data_ptr(), tab.data(), 256 * sizeof(RansEncSym), cudaMemcpyHostToDevice, st);
  auto wwords_d = torch::empty({tw}, W.options().dtype(torch::kInt32));
  const int blocks = ((int)tw + 3) / 4;
  if (ns == 4) rans_enc_pass1<4><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (const RansEncSym*)tab_d.data_ptr(), (uint32_t*)wwords_d.data_ptr());
  else if (ns == 2) rans_enc_pass1<2><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (const RansEncSym*)tab_d.data_ptr(), (uint32_t*)wwords_d.data_ptr());
  else rans_enc_pass1<1><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (const RansEncSym*)tab_d.data_ptr(), (uint32_t*)wwords_d.data_ptr());
  auto wwords_h = wwords_d.cpu();
  const uint32_t* ww = (const uint32_t*)wwords_h.data_ptr();
  std::vector<int32_t> woff16(tw + 1, 0);
  for (int w = 0; w < tw; w++) woff16[w + 1] = woff16[w] + (int32_t)((128 * (int)ns + 2 * (int64_t)ww[w] + 15) / 16);
  size_t stream_bytes = (size_t)woff16[tw] * 16;
  int32_t gmin = 0;
  std::vector<int32_t> tab_i((size_t)(2 * tw + 1), 0);
  for (int w = 0; w <= tw; w++) tab_i[w] = woff16[w];
  if (ilv > 0) {
    std::vector<int64_t> G(tw);
    for (int w = 0; w < tw; w++) G[w] = ((int64_t)(woff16[w + 1] - woff16[w]) * 16 + ((int64_t)1 << ilv) - 1) >> ilv;
    gmin = (int32_t)(*std::min_element(G.begin(), G.end()));
    int64_t t = (int64_t)gmin * tw;
    for (int w = 0; w < tw; w++) { tab_i[tw + 1 + w] = (int32_t)t; t += G[w] - gmin; }
    stream_bytes = (size_t)t << ilv;
  }
  const size_t soff = rans_stream_off((int)tw);
  const size_t total = soff + stream_bytes + 64;
  auto blob = torch::zeros({(int64_t)total}, dev_u8);
  uint8_t* base = (uint8_t*)blob.data_ptr();
  uint32_t hdr[64] = {0};
  hdr[0] = 0x534E5241u; hdr[1] = 7; hdr[2] = N; hdr[3] = K; hdr[4] = (uint32_t)tw; hdr[5] = (uint32_t)ns; hdr[6] = NB; hdr[7] = (uint32_t)ilv;
  hdr[8] = (uint32_t)RANS_HDR_BYTES; hdr[9] = (uint32_t)rans_woff_off(); hdr[10] = (uint32_t)soff; hdr[11] = (uint32_t)stream_bytes; hdr[12] = (uint32_t)(total & 0xffffffffu); hdr[13] = (uint32_t)(total >> 32); hdr[14] = (uint32_t)gmin;
  cudaMemcpyAsync(base, hdr, sizeof(hdr), cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(base + RANS_HDR_BYTES, lut.data(), RANS_LUT_BYTES, cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(base + rans_woff_off(), tab_i.data(), tab_i.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st);
  const int* woff_dev = (const int*)(base + rans_woff_off());
  if (ns == 4) rans_enc_pass2<4><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (int)gmin, (int)ilv, (const RansEncSym*)tab_d.data_ptr(), (const uint32_t*)wwords_d.data_ptr(), woff_dev, base + soff);
  else if (ns == 2) rans_enc_pass2<2><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (int)gmin, (int)ilv, (const RansEncSym*)tab_d.data_ptr(), (const uint32_t*)wwords_d.data_ptr(), woff_dev, base + soff);
  else rans_enc_pass2<1><<<blocks, 128, 0, st>>>((const uint8_t*)W.data_ptr(), NB, K, (int)tw, (int)gmin, (int)ilv, (const RansEncSym*)tab_d.data_ptr(), (const uint32_t*)wwords_d.data_ptr(), woff_dev, base + soff);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaStreamSynchronize(st);
  return blob;
}

torch::Tensor rans_pack(torch::Tensor W) {
  TORCH_CHECK(W.dim() == 2);
  RansCfg cfg = rans_pick_cfg((int)W.size(0), (int)W.size(1));
  return rans_pack_cfg(W, cfg.tw, cfg.ns, cfg.ilv);
}

std::vector<int64_t> rans_blob_info(torch::Tensor blob) {
  auto h = blob.narrow(0, 0, 256).cpu();
  const uint32_t* p = (const uint32_t*)h.data_ptr();
  TORCH_CHECK(p[0] == 0x534E5241u, "bad blob magic");
  TORCH_CHECK(p[1] == 7 && (p[5] == 1 || p[5] == 2 || p[5] == 4), "bad rANS stream format");
  return {(int64_t)p[2], (int64_t)p[3], (int64_t)p[4], (int64_t)p[5], (int64_t)(((uint64_t)p[13] << 32) | p[12]), (int64_t)p[7], (int64_t)p[14]};
}

torch::Tensor rans_unpack(torch::Tensor blob, int64_t N, int64_t K) {
  auto info = rans_blob_info(blob);
  TORCH_CHECK(info[0] == N && info[1] == K);
  const int tw = (int)info[2], ns = (int)info[3], ilv = (int)info[5], gmin = (int)info[6];
  auto out = torch::empty({N, K}, blob.options());
  auto status = torch::zeros({1}, blob.options().dtype(torch::kInt32));
  const uint8_t* b = (const uint8_t*)blob.data_ptr();
  auto st = c10::cuda::getCurrentCUDAStream();
  const int NB = (int)(N / 8);
  if (ns == 4) rans_unpack_kernel<4><<<(tw + 3) / 4, 128, 0, st>>>(b + rans_stream_off(tw), (const int*)(b + rans_woff_off()), (const uint32_t*)(b + RANS_HDR_BYTES), (uint8_t*)out.data_ptr(), NB, (int)K, tw, gmin, ilv, (int*)status.data_ptr());
  else if (ns == 2) rans_unpack_kernel<2><<<(tw + 3) / 4, 128, 0, st>>>(b + rans_stream_off(tw), (const int*)(b + rans_woff_off()), (const uint32_t*)(b + RANS_HDR_BYTES), (uint8_t*)out.data_ptr(), NB, (int)K, tw, gmin, ilv, (int*)status.data_ptr());
  else rans_unpack_kernel<1><<<(tw + 3) / 4, 128, 0, st>>>(b + rans_stream_off(tw), (const int*)(b + rans_woff_off()), (const uint32_t*)(b + RANS_HDR_BYTES), (uint8_t*)out.data_ptr(), NB, (int)K, tw, gmin, ilv, (int*)status.data_ptr());
  TORCH_CHECK(status.item<int>() == 0, "rans unpack: final states mismatch (corrupt blob)");
  return out;
}

static void rans_set_attrs_once() {
  static bool done = false;
  if (done) return;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
#define RANS_PREP(NSV, RGV) cudaFuncSetAttribute(rans_gemv_kernel<NSV, 9, RGV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(RANS_MAX_SMEM - RANS_LUT_BYTES));
  RANS_PREP(1, 4) RANS_PREP(1, 8) RANS_PREP(2, 4) RANS_PREP(2, 8) RANS_PREP(4, 4) RANS_PREP(4, 8)
#undef RANS_PREP
  cudaThreadExchangeStreamCaptureMode(&mode);
  done = true;
}

static void rans_check_io(const torch::Tensor& blob, const torch::Tensor& x, const torch::Tensor& sb, const torch::Tensor& sa, const torch::Tensor& out, int64_t N, int64_t K) {
  TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == torch::kUInt8 && blob.is_contiguous(), "blob must be a contiguous uint8 cuda tensor");
  TORCH_CHECK(N > 0 && K > 0 && N % 8 == 0 && K % 64 == 0, "need N%8==0 and K%64==0");
  TORCH_CHECK(x.is_cuda() && x.element_size() == 1 && x.numel() == K && x.is_contiguous(), "x must be a contiguous 1-byte (FP8 e4m3) cuda tensor with K elements");
  TORCH_CHECK(sb.is_cuda() && sb.scalar_type() == torch::kFloat && sb.is_contiguous() && (sb.numel() == N || sb.numel() == 1), "sb must be a contiguous float32 cuda tensor with N (per-channel) or 1 (scalar) elements");
  TORCH_CHECK(sa.is_cuda() && sa.scalar_type() == torch::kFloat && sa.numel() == 1, "sa must be a float32 cuda tensor with 1 element");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.numel() == N, "out must be a contiguous bfloat16 cuda tensor with N elements");
}

static void rans_launch(const uint8_t* blob, size_t blob_bytes, int tw, int ns, int wpb, int rg, int ilv, const uint8_t* x, const float* sb, int sb_scalar, const float* sa, __nv_bfloat16* out, int N, int K, cudaStream_t st) {
  const size_t soff = rans_stream_off(tw);
  RansProb pb;
  pb.hdr = blob; pb.stream = blob + soff; pb.woff16 = (const int*)(blob + rans_woff_off()); pb.lut = (const uint32_t*)(blob + RANS_HDR_BYTES);
  pb.x = x; pb.sb = sb; pb.sb_scalar = sb_scalar; pb.sa = sa; pb.out = out; pb.N = N; pb.K = K; pb.tw = tw; pb.ilv = ilv;
  pb.stream_bytes = (blob_bytes > soff + 64u) ? (blob_bytes - soff - 64u) : 0u;
  pb.tpf_stride = rans_tpf_stride();
  TORCH_CHECK(tw >= 1 && tw <= N / 8 && wpb >= 1 && wpb <= 32 && (ns == 1 || ns == 2 || ns == 4) && (rg == 4 || rg == 8) && (ilv == 0 || ilv == 9), "bad launch config");
  const int blocks = (tw + wpb - 1) / wpb;
  const size_t dsmem = rans_dsmem(K, wpb, 9, rg);
  TORCH_CHECK(rans_fits(K, wpb, 9, rg), "rans smem too large (K too large for smem staging of x): ", dsmem);
  rans_set_attrs_once();
#define RANS_L2(NSV, RGV) \
  if (ns == NSV && rg == RGV) rans_gemv_kernel<NSV, 9, RGV><<<blocks, wpb * 32, dsmem, st>>>(pb);
  RANS_L2(1, 4) RANS_L2(1, 8) RANS_L2(2, 4) RANS_L2(2, 8) RANS_L2(4, 4) RANS_L2(4, 8)
#undef RANS_L2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rans_gemv_cfg(torch::Tensor blob, torch::Tensor x, torch::Tensor sb, torch::Tensor sa, torch::Tensor out, int64_t N, int64_t K, int64_t tw, int64_t ns, int64_t wpb, int64_t rg, int64_t ilv) {
  rans_check_io(blob, x, sb, sa, out, N, K);
  rans_launch((const uint8_t*)blob.data_ptr(), (size_t)blob.numel(), (int)tw, (int)ns, (int)wpb, (int)rg, (int)ilv, (const uint8_t*)x.data_ptr(), (const float*)sb.data_ptr(), sb.numel() == 1, (const float*)sa.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
              (int)N, (int)K, c10::cuda::getCurrentCUDAStream());
}

void rans_gemv_auto(torch::Tensor blob, torch::Tensor x, torch::Tensor sb, torch::Tensor sa, torch::Tensor out, int64_t N, int64_t K) {
  rans_check_io(blob, x, sb, sa, out, N, K);
  RansCfg c = rans_pick_cfg((int)N, (int)K);
  rans_launch((const uint8_t*)blob.data_ptr(), (size_t)blob.numel(), c.tw, c.ns, c.wpb, c.rg, c.ilv, (const uint8_t*)x.data_ptr(), (const float*)sb.data_ptr(), sb.numel() == 1, (const float*)sa.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
              (int)N, (int)K, c10::cuda::getCurrentCUDAStream());
}

int64_t rans_tpf_kb() { return static_cast<int64_t>(rans_tpf_stride() >> 10); }

std::vector<int64_t> rans_cfg_info(int64_t N, int64_t K) { RansCfg c = rans_pick_cfg((int)N, (int)K); return {c.tw, c.wpb, c.ns, c.rg, c.ilv}; }

std::vector<int64_t> rans_blob_cfg(torch::Tensor blob) {
  auto info = rans_blob_info(blob);
  RansCfg c = rans_pick_cfg((int)info[0], (int)info[1]);
  if (c.tw == (int)info[2] && c.ns == (int)info[3] && c.ilv == (int)info[5]) {
    return {c.tw, c.wpb, c.ns, c.rg, c.ilv};
  }
  const int tw = (int)info[2];
  return {tw, (tw + 47) / 48, info[3], 4, info[5]};
}

void rans_prepare() { rans_set_attrs_once(); }

#ifdef MUSE_RANS_STANDALONE_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rans_pack", &rans_pack);
  m.def("rans_pack_cfg", &rans_pack_cfg);
  m.def("rans_unpack", &rans_unpack);
  m.def("rans_blob_info", &rans_blob_info);
  m.def("rans_gemv_auto", &rans_gemv_auto);
  m.def("rans_gemv_cfg", &rans_gemv_cfg);
  m.def("rans_cfg_info", &rans_cfg_info);
  m.def("rans_blob_cfg", &rans_blob_cfg);
  m.def("rans_prepare", &rans_prepare);
  m.def("rans_tpf_kb", &rans_tpf_kb);
}
#endif
