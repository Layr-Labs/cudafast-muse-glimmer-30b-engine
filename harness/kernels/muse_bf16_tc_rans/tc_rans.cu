#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

namespace {

constexpr int kCTAs = 48;
constexpr int kWPC = 8;
constexpr int kTW = kCTAs * kWPC;  // 384 total workers
constexpr int kThreads = kWPC * 32;  // 256 threads per CTA
constexpr uint32_t kRansL = 65536u;
constexpr int kNS = 4;
constexpr int kRG = 4;
constexpr uint32_t kRB = kRG * 512u;  // 2048 B per warp ring buffer

__device__ __forceinline__ uint32_t lds32(const uint8_t* s) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(v)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))));
  return v;
}

__device__ __forceinline__ uint16_t lds16(const uint8_t* s) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];"
               : "=h"(v)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))));
  return v;
}

__device__ __forceinline__ uint64_t pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ void cp_async16_ef(uint8_t* dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
               :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                  "l"(src), "l"(pol));
}

template <class Fn>
__device__ __forceinline__ void for_each_worker_item(int w, int B, int k_tiles, Fn&& fn) {
  const int full_waves = B / kTW;
  for (int wave = 0; wave < full_waves; ++wave) {
    const int b = wave * kTW + w;
    fn(b, 0, k_tiles, false);
  }
  const int rem = B - full_waves * kTW;
  const int cta = w / kWPC;
  const int lw = w & (kWPC - 1);
  const int kt_s = (k_tiles * lw) / kWPC;
  const int kt_e = (k_tiles * (lw + 1)) / kWPC;
  const int r_waves = (rem + kCTAs - 1) / kCTAs;
  const int base_b = full_waves * kTW;
  for (int rw = 0; rw < r_waves; ++rw) {
    const int b = base_b + rw * kCTAs + cta;
    fn(b, kt_s, kt_e, true);  // caller checks (b < B)
  }
}

template <class Fn>
__device__ __forceinline__ void for_each_worker_item_rev(int w, int B, int k_tiles, Fn&& fn) {
  const int full_waves = B / kTW;
  const int rem = B - full_waves * kTW;
  const int cta = w / kWPC;
  const int lw = w & (kWPC - 1);
  const int kt_s = (k_tiles * lw) / kWPC;
  const int kt_e = (k_tiles * (lw + 1)) / kWPC;
  const int r_waves = (rem + kCTAs - 1) / kCTAs;
  const int base_b = full_waves * kTW;
  for (int rw = r_waves - 1; rw >= 0; --rw) {
    const int b = base_b + rw * kCTAs + cta;
    if (b < B) {
      fn(b, kt_s, kt_e);
    }
  }
  for (int wave = full_waves - 1; wave >= 0; --wave) {
    const int b = wave * kTW + w;
    fn(b, 0, k_tiles);
  }
}

__device__ __forceinline__ void load_frag_a_u16(
    const uint16_t* __restrict__ W, int K, int b, int kt, int g, int t, uint16_t (&u)[8]) {
  const int row0 = b * 16;
  const uint32_t* wptr0 = reinterpret_cast<const uint32_t*>(W + static_cast<size_t>(row0 + g) * K + kt * 16 + 2 * t);
  const uint32_t* wptr1 = reinterpret_cast<const uint32_t*>(W + static_cast<size_t>(row0 + g + 8) * K + kt * 16 + 2 * t);
  const uint32_t p0 = __ldg(wptr0);
  const uint32_t p1 = __ldg(wptr1);
  const uint32_t p2 = __ldg(wptr0 + 4);
  const uint32_t p3 = __ldg(wptr1 + 4);
  u[0] = static_cast<uint16_t>(p0);
  u[1] = static_cast<uint16_t>(p0 >> 16);
  u[2] = static_cast<uint16_t>(p1);
  u[3] = static_cast<uint16_t>(p1 >> 16);
  u[4] = static_cast<uint16_t>(p2);
  u[5] = static_cast<uint16_t>(p2 >> 16);
  u[6] = static_cast<uint16_t>(p3);
  u[7] = static_cast<uint16_t>(p3 >> 16);
}

__global__ void tc_hist_kernel(const uint16_t* __restrict__ W, size_t n, unsigned long long* counts) {
  __shared__ unsigned long long local_hist[256];
  const int tid = threadIdx.x;
  local_hist[tid] = 0ULL;
  __syncthreads();
  const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t n2 = n >> 1;
  const uint32_t* W2 = reinterpret_cast<const uint32_t*>(W);
  for (size_t i = gid; i < n2; i += stride) {
    const uint32_t v = W2[i];
    atomicAdd(&local_hist[(v >> 7) & 0xFFu], 1ULL);
    atomicAdd(&local_hist[(v >> 23) & 0xFFu], 1ULL);
  }
  if ((n & 1u) != 0u && gid == 0) {
    atomicAdd(&local_hist[(W[n - 1] >> 7) & 0xFFu], 1ULL);
  }
  __syncthreads();
  if (local_hist[tid] != 0ULL) {
    atomicAdd(&counts[tid], local_hist[tid]);
  }
}

__global__ void __launch_bounds__(128) tc_enc_pass1_kernel(
    const uint16_t* __restrict__ W, int N, int K,
    const uint32_t* __restrict__ f_in, const uint32_t* __restrict__ cum_in,
    uint32_t* __restrict__ wb_exact) {
  __shared__ uint32_t sh_f[256], sh_cum[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    sh_f[i] = f_in[i];
    sh_cum[i] = cum_in[i];
  }
  __syncthreads();

  const int w = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (w >= kTW) return;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int B = N / 16;
  const int k_tiles = K / 16;

  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  uint32_t tot_u16 = 0;
  uint32_t nw = 0;

  for_each_worker_item_rev(w, B, k_tiles, [&](int b, int kt_s, int kt_e) {
    for (int kt = kt_e - 1; kt >= kt_s; --kt) {
      ++nw;
      uint16_t u[8];
      load_frag_a_u16(W, K, b, kt, g, t, u);
#pragma unroll
      for (int step = 7; step >= 0; --step) {
        const uint32_t sym = (u[step] >> 7) & 0xFFu;
        const uint32_t fs = sh_f[sym];
        const uint32_t cs = sh_cum[sym];
        const int q = step & 3;
        const uint64_t limit = static_cast<uint64_t>(fs) << 22;
        const bool ren = static_cast<uint64_t>(st[q]) >= limit;
        const uint32_t msk = __ballot_sync(0xffffffffu, ren);
        tot_u16 += static_cast<uint32_t>(__popc(msk));
        if (ren) st[q] >>= 16;
        st[q] = (st[q] / fs) * 1024u + cs + (st[q] % fs);
      }
    }
  });

  if (lane == 0) {
    wb_exact[w] = 128u * kNS + nw * 256u + tot_u16 * 2u;
  }
}

__global__ void __launch_bounds__(128) tc_enc_pass2_kernel(
    const uint16_t* __restrict__ W, int N, int K, int gmin,
    const uint32_t* __restrict__ f_in, const uint32_t* __restrict__ cum_in,
    const int* __restrict__ tab, const uint32_t* __restrict__ wb_exact, uint8_t* __restrict__ stream) {
  __shared__ uint32_t sh_f[256], sh_cum[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    sh_f[i] = f_in[i];
    sh_cum[i] = cum_in[i];
  }
  __syncthreads();

  const int w = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (w >= kTW) return;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const uint32_t lt = (1u << lane) - 1u;
  const int B = N / 16;
  const int k_tiles = K / 16;
  const int toffw = tab[kTW + 1 + w];

  auto sw_ptr = [&](uint32_t bpos) -> uint8_t* {
    const int grp = static_cast<int>(bpos >> 9);
    const size_t gi = (grp < gmin)
        ? (static_cast<size_t>(grp) * kTW + w)
        : (static_cast<size_t>(toffw) + grp - gmin);
    return stream + (gi << 9) + (bpos & 511u);
  };

  uint32_t pos = wb_exact[w];
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;

  for_each_worker_item_rev(w, B, k_tiles, [&](int b, int kt_s, int kt_e) {
    for (int kt = kt_e - 1; kt >= kt_s; --kt) {
      uint16_t u[8];
      load_frag_a_u16(W, K, b, kt, g, t, u);
#pragma unroll
      for (int step = 7; step >= 0; --step) {
        const uint32_t sym = (u[step] >> 7) & 0xFFu;
        const uint32_t fs = sh_f[sym];
        const uint32_t cs = sh_cum[sym];
        const int q = step & 3;
        const uint64_t limit = static_cast<uint64_t>(fs) << 22;
        const bool ren = static_cast<uint64_t>(st[q]) >= limit;
        const uint32_t msk = __ballot_sync(0xffffffffu, ren);
        pos -= static_cast<uint32_t>(__popc(msk)) << 1;
        if (ren) {
          const uint16_t out16 = static_cast<uint16_t>(st[q] & 0xFFFFu);
          const uint32_t p = pos + (static_cast<uint32_t>(__popc(msk & lt)) << 1);
          *reinterpret_cast<uint16_t*>(sw_ptr(p)) = out16;
          st[q] >>= 16;
        }
        st[q] = (st[q] / fs) * 1024u + cs + (st[q] % fs);
      }
      pos -= 256u;
      uint32_t lo0 = 0u, lo1 = 0u;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t sm0 = (((u[s] >> 8) & 0x80u) | (u[s] & 0x7Fu));
        const uint32_t sm1 = (((u[s + 4] >> 8) & 0x80u) | (u[s + 4] & 0x7Fu));
        lo0 |= sm0 << (s * 8);
        lo1 |= sm1 << (s * 8);
      }
      const uint32_t sm_base = pos + static_cast<uint32_t>(lane) * 8u;
      *reinterpret_cast<uint16_t*>(sw_ptr(sm_base)) = static_cast<uint16_t>(lo0);
      *reinterpret_cast<uint16_t*>(sw_ptr(sm_base + 2u)) = static_cast<uint16_t>(lo0 >> 16);
      *reinterpret_cast<uint16_t*>(sw_ptr(sm_base + 4u)) = static_cast<uint16_t>(lo1);
      *reinterpret_cast<uint16_t*>(sw_ptr(sm_base + 6u)) = static_cast<uint16_t>(lo1 >> 16);
    }
  });

#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    *reinterpret_cast<uint32_t*>(sw_ptr(128u * s + 4u * lane)) = st[s];
  }
}

__global__ void __launch_bounds__(kThreads) tc_decode_kernel(
    const uint8_t* __restrict__ blob, uint16_t* __restrict__ W_out, int N, int K) {
  __shared__ alignas(16) uint32_t sh_lut[1024];
  __shared__ alignas(16) uint8_t sh_ring[kWPC][kRB];

  const int tid = threadIdx.x;
  const int lw = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int cta = blockIdx.x;
  const int w = cta * kWPC + lw;

  const int32_t* meta = reinterpret_cast<const int32_t*>(blob + 4096);
  const int32_t* tab = reinterpret_cast<const int32_t*>(blob + 4112);
  const int gmin = meta[2];
  const uint32_t soff = static_cast<uint32_t>(meta[3]);
  const uint8_t* stream = blob + soff;
  const uint32_t wbytes = static_cast<uint32_t>(tab[w + 1] - tab[w]) * 16u;
  const int toffw = tab[kTW + 1 + w];

  const uint32_t lut_sa = static_cast<uint32_t>(__cvta_generic_to_shared(sh_lut));
  for (int i = tid; i < 256; i += blockDim.x) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(lut_sa + 16u * static_cast<uint32_t>(i)),
                    "l"(reinterpret_cast<const uint4*>(blob) + i));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");

  uint8_t* ring_s = sh_ring[lw];
  const uint64_t pol = pol_ef();
  int issued = 0;

#define TC_DEC_ISSUE() { \
    const uint32_t boff = static_cast<uint32_t>(issued) * 512u + static_cast<uint32_t>(lane) * 16u; \
    if (boff < wbytes) { \
      const size_t gi = (issued < gmin) \
          ? (static_cast<size_t>(issued) * kTW + w) \
          : (static_cast<size_t>(toffw) + issued - gmin); \
      cp_async16_ef(ring_s + (boff & (kRB - 1u)), stream + (gi << 9) + static_cast<uint32_t>(lane) * 16u, pol); \
    } \
    asm volatile("cp.async.commit_group;" ::: "memory"); \
    ++issued; \
  }

#pragma unroll
  for (int i = 0; i < kRG - 1; ++i) TC_DEC_ISSUE();

  asm volatile("cp.async.wait_group %0;" :: "n"(kRG - 2) : "memory");
  __syncthreads();

  const uint32_t lt = (1u << lane) - 1u;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    st[s] = lds32(ring_s + ((128u * s + 4u * lane) & (kRB - 1u)));
  }
  uint32_t pwb = 128u * kNS;

  const int B = N / 16;
  const int k_tiles = K / 16;
  for_each_worker_item(w, B, k_tiles, [&](int b, int kt_s, int kt_e, bool /*is_ksplit*/) {
    if (b >= B) return;
    const int row0 = b * 16;
    for (int kt = kt_s; kt < kt_e; ++kt) {
      __syncwarp();
      const int curg = static_cast<int>(pwb >> 9);
      while (issued < curg + kRG) TC_DEC_ISSUE();
      asm volatile("cp.async.wait_group %0;" :: "n"(1) : "memory");
      __syncwarp();

      uint32_t lo0, lo1;
      const uint32_t a0_off = pwb + 8u * static_cast<uint32_t>(lane);
      if ((pwb & 2u) == 0u) {
        lo0 = lds32(ring_s + (a0_off & (kRB - 1u)));
        lo1 = lds32(ring_s + ((a0_off + 4u) & (kRB - 1u)));
      } else {
        lo0 = lds16(ring_s + (a0_off & (kRB - 1u))) | (lds16(ring_s + ((a0_off + 2u) & (kRB - 1u))) << 16);
        lo1 = lds16(ring_s + ((a0_off + 4u) & (kRB - 1u))) | (lds16(ring_s + ((a0_off + 6u) & (kRB - 1u))) << 16);
      }
      pwb += 256u;

      uint16_t out_u[8];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint32_t e0 = sh_lut[st[0] & 1023u];
        const uint32_t e1 = sh_lut[st[1] & 1023u];
        const uint32_t e2 = sh_lut[st[2] & 1023u];
        const uint32_t e3 = sh_lut[st[3] & 1023u];
        const uint32_t es[4] = {e0, e1, e2, e3};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int step = half * 4 + q;
          const uint32_t e = es[q];
          uint32_t& rx = st[q];
          rx = ((e >> 8) & 2047u) * (rx >> 10) + (e >> 19);
          const bool need = rx < kRansL;
          const uint32_t msk = __ballot_sync(0xffffffffu, need);
          if (need) rx = (rx << 16) | lds16(ring_s + ((pwb + (__popc(msk & lt) << 1)) & (kRB - 1u)));
          pwb += __popc(msk) << 1;
          const uint32_t sm_dup = __byte_perm(lo0, lo1, (step << 4) | step);
          out_u[step] = static_cast<uint16_t>((sm_dup & 0x807Fu) | ((e & 255u) << 7));
        }
      }
      *reinterpret_cast<uint32_t*>(W_out + static_cast<size_t>(row0 + g) * K + kt * 16 + 2 * t) =
          static_cast<uint32_t>(out_u[0]) | (static_cast<uint32_t>(out_u[1]) << 16);
      *reinterpret_cast<uint32_t*>(W_out + static_cast<size_t>(row0 + g + 8) * K + kt * 16 + 2 * t) =
          static_cast<uint32_t>(out_u[2]) | (static_cast<uint32_t>(out_u[3]) << 16);
      *reinterpret_cast<uint32_t*>(W_out + static_cast<size_t>(row0 + g) * K + kt * 16 + 2 * t + 8) =
          static_cast<uint32_t>(out_u[4]) | (static_cast<uint32_t>(out_u[5]) << 16);
      *reinterpret_cast<uint32_t*>(W_out + static_cast<size_t>(row0 + g + 8) * K + kt * 16 + 2 * t + 8) =
          static_cast<uint32_t>(out_u[6]) | (static_cast<uint32_t>(out_u[7]) << 16);
    }
  });
#undef TC_DEC_ISSUE
}

template <bool HasBias>
__global__ void __launch_bounds__(kThreads, 1) tc_rans_gemm_kernel(
    const uint16_t* __restrict__ X,
    const uint8_t* __restrict__ blob,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ out,
    int M, int N, int K, size_t blob_bytes, size_t tpf_stride, int pdl_mode) {
  __shared__ alignas(16) uint32_t sh_lut[1024];
  __shared__ alignas(16) uint8_t sh_ring[kWPC][kRB];
  __shared__ alignas(16) float4 sh_red[kWPC][32];

  const int tid = threadIdx.x;
  const int lw = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int cta = blockIdx.x;
  const int m0 = blockIdx.y * 8;
  const int w = cta * kWPC + lw;

  const int32_t* meta = reinterpret_cast<const int32_t*>(blob + 4096);
  const int32_t* tab = reinterpret_cast<const int32_t*>(blob + 4112);
  const int gmin = __ldg(&meta[2]);
  const uint32_t soff = static_cast<uint32_t>(__ldg(&meta[3]));
  const uint8_t* stream = blob + soff;
  const uint32_t wbytes = static_cast<uint32_t>(__ldg(&tab[w + 1]) - __ldg(&tab[w])) * 16u;
  const int toffw = __ldg(&tab[kTW + 1 + w]);

  const uint32_t lut_sa = static_cast<uint32_t>(__cvta_generic_to_shared(sh_lut));
  for (int i = tid; i < 256; i += blockDim.x) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(lut_sa + 16u * static_cast<uint32_t>(i)),
                    "l"(reinterpret_cast<const uint4*>(blob) + i));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");

  uint8_t* ring_s = sh_ring[lw];
  const uint64_t pol = pol_ef();
  int issued = 0;

#define TC_ISSUE() { \
    const uint32_t boff = static_cast<uint32_t>(issued) * 512u + static_cast<uint32_t>(lane) * 16u; \
    if (boff < wbytes) { \
      const size_t gi = (issued < gmin) \
          ? (static_cast<size_t>(issued) * kTW + w) \
          : (static_cast<size_t>(toffw) + issued - gmin); \
      cp_async16_ef(ring_s + (boff & (kRB - 1u)), stream + (gi << 9) + static_cast<uint32_t>(lane) * 16u, pol); \
    } \
    asm volatile("cp.async.commit_group;" ::: "memory"); \
    ++issued; \
  }

#pragma unroll
  for (int i = 0; i < kRG - 1; ++i) TC_ISSUE();

  if (cta == 0 && blockIdx.y == 0 && tpf_stride > 0 && blob_bytes > static_cast<size_t>(soff) + 64u) {
    const size_t sbytes = blob_bytes - static_cast<size_t>(soff) - 64u;
    for (size_t off = static_cast<size_t>(tid) * tpf_stride; off < sbytes;
         off += static_cast<size_t>(blockDim.x) * tpf_stride) {
      asm volatile("prefetch.global.L2 [%0];" :: "l"(stream + off));
    }
  }

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  asm volatile("cp.async.wait_group %0;" :: "n"(kRG - 2) : "memory");
  __syncthreads();

  const uint32_t lt = (1u << lane) - 1u;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    st[s] = lds32(ring_s + ((128u * s + 4u * lane) & (kRB - 1u)));
  }
  uint32_t pwb = 128u * kNS;

  const bool valid_x_row = (m0 + g) < M;
  const uint16_t* x_row = valid_x_row ? (X + static_cast<size_t>(m0 + g) * K) : nullptr;
  const bool valid_m0 = (m0 + 2 * t) < M;
  const bool valid_m1 = (m0 + 2 * t + 1) < M;

  const int B = N / 16;
  const int k_tiles = K / 16;
  const int out_stride = N;

  auto write_band = [&](int b, float d0, float d1, float d2, float d3) {
    const int n0 = b * 16 + g;
    const int n1 = n0 + 8;
    float bv0 = 0.0f, bv1 = 0.0f;
    if constexpr (HasBias) {
      bv0 = __bfloat162float(__ushort_as_bfloat16(__ldg(bias + n0)));
      bv1 = __bfloat162float(__ushort_as_bfloat16(__ldg(bias + n1)));
    }
    if (valid_m0) {
      const size_t base0 = static_cast<size_t>(m0 + 2 * t) * out_stride;
      out[base0 + n0] = __bfloat16_as_ushort(__float2bfloat16_rn(d0 + bv0));
      out[base0 + n1] = __bfloat16_as_ushort(__float2bfloat16_rn(d2 + bv1));
    }
    if (valid_m1) {
      const size_t base1 = static_cast<size_t>(m0 + 2 * t + 1) * out_stride;
      out[base1 + n0] = __bfloat16_as_ushort(__float2bfloat16_rn(d1 + bv0));
      out[base1 + n1] = __bfloat16_as_ushort(__float2bfloat16_rn(d3 + bv1));
    }
  };

  for_each_worker_item(w, B, k_tiles, [&](int b, int kt_s, int kt_e, bool is_ksplit) {
    if (is_ksplit && b >= B) return;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int kt = kt_s; kt < kt_e; ++kt) {
      __syncwarp();
      const int curg = static_cast<int>(pwb >> 9);
      while (issued < curg + kRG) TC_ISSUE();
      asm volatile("cp.async.wait_group %0;" :: "n"(1) : "memory");
      __syncwarp();

      uint32_t lo0, lo1;
      const uint32_t a0_off = pwb + 8u * static_cast<uint32_t>(lane);
      if ((pwb & 2u) == 0u) {
        lo0 = lds32(ring_s + (a0_off & (kRB - 1u)));
        lo1 = lds32(ring_s + ((a0_off + 4u) & (kRB - 1u)));
      } else {
        lo0 = lds16(ring_s + (a0_off & (kRB - 1u))) | (lds16(ring_s + ((a0_off + 2u) & (kRB - 1u))) << 16);
        lo1 = lds16(ring_s + ((a0_off + 4u) & (kRB - 1u))) | (lds16(ring_s + ((a0_off + 6u) & (kRB - 1u))) << 16);
      }
      pwb += 256u;

      const int k_base = kt * 16 + 2 * t;
      const uint32_t b0 = valid_x_row ? __ldg(reinterpret_cast<const uint32_t*>(x_row + k_base)) : 0u;
      const uint32_t b1 = valid_x_row ? __ldg(reinterpret_cast<const uint32_t*>(x_row + k_base + 8)) : 0u;

      uint32_t a[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint32_t e0 = sh_lut[st[0] & 1023u];
        const uint32_t e1 = sh_lut[st[1] & 1023u];
        const uint32_t e2 = sh_lut[st[2] & 1023u];
        const uint32_t e3 = sh_lut[st[3] & 1023u];
        const uint32_t es[4] = {e0, e1, e2, e3};
        uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int step = half * 4 + q;
          const uint32_t e = es[q];
          uint32_t& rx = st[q];
          rx = ((e >> 8) & 2047u) * (rx >> 10) + (e >> 19);
          const bool need = rx < kRansL;
          const uint32_t msk = __ballot_sync(0xffffffffu, need);
          if (need) rx = (rx << 16) | lds16(ring_s + ((pwb + (__popc(msk & lt) << 1)) & (kRB - 1u)));
          pwb += __popc(msk) << 1;
          const uint32_t sm_dup = __byte_perm(lo0, lo1, (step << 4) | step);
          const uint32_t w16 = (sm_dup & 0x807Fu) | ((e & 255u) << 7);
          if (q < 2) {
            w_lo |= w16 << (q * 16);
          } else {
            w_hi |= w16 << ((q - 2) * 16);
          }
        }
        if (half == 0) {
          a[0] = w_lo;
          a[1] = w_hi;
        } else {
          a[2] = w_lo;
          a[3] = w_hi;
        }
      }

      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }

    if (!is_ksplit) {
      write_band(b, d0, d1, d2, d3);
    } else {
      sh_red[lw][lane] = make_float4(d0, d1, d2, d3);
      __syncthreads();
      if (lw == 0) {
        float s0 = d0, s1 = d1, s2 = d2, s3 = d3;
#pragma unroll
        for (int wi = 1; wi < kWPC; ++wi) {
          const float4 f = sh_red[wi][lane];
          s0 += f.x;
          s1 += f.y;
          s2 += f.z;
          s3 += f.w;
        }
        write_band(b, s0, s1, s2, s3);
      }
      __syncthreads();
    }
  });

  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
#undef TC_ISSUE
}

static size_t tc_tpf_stride() {
  static const size_t val = []() -> size_t {
    const char* env = std::getenv("MUSE_TC_RANS_TLB_PREFETCH");
    if (!env || !*env) return static_cast<size_t>(2048) << 10;
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return val;
}

static int g_tc_pdl_override = -1;
static int tc_pdl_cfg() {
  if (g_tc_pdl_override >= 0) return g_tc_pdl_override;
  static const int cfg = []() -> int {
    const char* env = std::getenv("MUSE_TC_RANS_PDL");
    if (!env || !*env) return 0;
    long v = std::strtol(env, nullptr, 10);
    return (v >= 0 && v <= 3) ? static_cast<int>(v) : 0;
  }();
  return cfg;
}

}  // namespace

extern "C" int muse_bf16_tc_rans_set_pdl(int mode) {
  const int prev = tc_pdl_cfg();
  g_tc_pdl_override = (mode >= 0 && mode <= 3) ? mode : -1;
  return prev;
}

extern "C" int muse_bf16_tc_rans_hist(
    const uint16_t* W, size_t n, unsigned long long* counts, cudaStream_t stream) {
  tc_hist_kernel<<<256, 256, 0, stream>>>(W, n, counts);
  return cudaGetLastError();
}

extern "C" int muse_bf16_tc_rans_pass1(
    const uint16_t* W, int N, int K,
    const uint32_t* f, const uint32_t* cum, uint32_t* wb_exact, cudaStream_t stream) {
  if (N <= 0 || K <= 0 || (N % 16) != 0 || (K % 16) != 0) return cudaErrorInvalidValue;
  tc_enc_pass1_kernel<<<kTW / 4, 128, 0, stream>>>(W, N, K, f, cum, wb_exact);
  return cudaGetLastError();
}

extern "C" int muse_bf16_tc_rans_pass2(
    const uint16_t* W, int N, int K, int gmin,
    const uint32_t* f, const uint32_t* cum, const int* tab,
    const uint32_t* wb_exact, uint8_t* stream_out, cudaStream_t stream) {
  if (N <= 0 || K <= 0 || (N % 16) != 0 || (K % 16) != 0) return cudaErrorInvalidValue;
  tc_enc_pass2_kernel<<<kTW / 4, 128, 0, stream>>>(W, N, K, gmin, f, cum, tab, wb_exact, stream_out);
  return cudaGetLastError();
}

extern "C" int muse_bf16_tc_rans_decode(
    const void* blob, void* W_out, int N, int K, cudaStream_t stream) {
  if (N <= 0 || K <= 0 || (N % 16) != 0 || (K % 16) != 0) return cudaErrorInvalidValue;
  tc_decode_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(blob), static_cast<uint16_t*>(W_out), N, K);
  return cudaGetLastError();
}

extern "C" int muse_bf16_tc_rans_gemm(
    int M, int N, int K, size_t blob_bytes,
    const void* X, const void* blob, const void* bias, void* out, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || (N % 16) != 0 || (K % 16) != 0) return cudaErrorInvalidValue;
  if (((reinterpret_cast<uintptr_t>(X) | reinterpret_cast<uintptr_t>(blob) | reinterpret_cast<uintptr_t>(out)) & 3u) != 0u) {
    return cudaErrorInvalidValue;
  }
  const int grid_y = (M + 7) >> 3;
  const dim3 grid(kCTAs, grid_y, 1);
  const auto* x_arg = static_cast<const uint16_t*>(X);
  const auto* b_arg = static_cast<const uint8_t*>(blob);
  const auto* bias_arg = static_cast<const uint16_t*>(bias);
  auto* o_arg = static_cast<uint16_t*>(out);
  const size_t tpf_arg = tc_tpf_stride();
  const int pdl_mode = tc_pdl_cfg();
  const bool has_bias = (bias != nullptr);

  auto launch = [&](auto kernel) -> int {
    if ((pdl_mode & 1) != 0) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(kThreads, 1, 1);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      return static_cast<int>(cudaLaunchKernelEx(
          &cfg, kernel, x_arg, b_arg, bias_arg, o_arg, M, N, K, blob_bytes, tpf_arg, pdl_mode));
    }
    kernel<<<grid, kThreads, 0, stream>>>(
        x_arg, b_arg, bias_arg, o_arg, M, N, K, blob_bytes, tpf_arg, pdl_mode);
    return cudaGetLastError();
  };

  if (has_bias) {
    return launch(tc_rans_gemm_kernel<true>);
  }
  return launch(tc_rans_gemm_kernel<false>);
}
