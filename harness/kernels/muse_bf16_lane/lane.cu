// SPDX-License-Identifier: Apache-2.0
// Lossless one-row BF16 attention projections for Muse Glimmer:
//   - Primary path: persistent-warp NS=4 10-bit rANS on BF16 high byte (sign + 7-bit exp) + raw low byte,
//     512B-interleaved warp streams prefetched via cp.async with L2::evict_first (~10.8 bits/weight).
//   - Fallback path: original 12-bit per-tile exponent-base lane kernel.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace {
constexpr int kWidth = 6656;
constexpr int kLanes = 32;
constexpr int kSteps = 8;
constexpr int kValues = kSteps * kLanes;
constexpr int kTiles = kWidth / kValues;  // 26
constexpr int kDenseBytes = kValues * sizeof(uint16_t);
constexpr int kCompactBytes = kValues + kValues / 2;
constexpr uint32_t kRansL = 65536u;
constexpr int kNS = 4;
static_assert(kWidth % kValues == 0 && kTiles < 32);

struct RowIndex {
  uint32_t offset;
  uint32_t compact;
  uint8_t exponent_base[28];
};
static_assert(sizeof(RowIndex) == 36);

__device__ __forceinline__ float bf16_word_to_float(uint32_t bits) {
  return __uint_as_float((bits & 65535u) << 16);
}

template <bool CacheGlobal>
__device__ __forceinline__ uint4 load_dense(const void* ptr) {
  if constexpr (CacheGlobal) {
    uint4 result;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w) : "l"(ptr));
    return result;
  }
  return *static_cast<const uint4*>(ptr);
}

template <bool CacheGlobal>
__device__ __forceinline__ uint2 load_low(const void* ptr) {
  if constexpr (CacheGlobal) {
    uint2 result;
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                 : "=r"(result.x), "=r"(result.y) : "l"(ptr));
    return result;
  }
  return *static_cast<const uint2*>(ptr);
}

template <bool CacheGlobal>
__device__ __forceinline__ uint32_t load_codes(const void* ptr) {
  if constexpr (CacheGlobal) {
    uint32_t result;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(result) : "l"(ptr));
    return result;
  }
  return *static_cast<const uint32_t*>(ptr);
}

template <bool Compact, bool CacheGlobal>
__global__ void lane_gemv(const uint16_t* x, const uint8_t* weights,
                          const RowIndex* index, uint16_t* out) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const uint8_t* current;
  uint32_t compact = 0;
  if constexpr (Compact) {
    current = weights + __ldg(&index[row].offset);
    compact = __ldg(&index[row].compact);
  } else {
    current = weights + row * kTiles * kDenseBytes;
  }

  float accum = 0.0f;
  for (int tile = 0; tile < kTiles; ++tile) {
    const uint16_t* input = x + tile * kValues + lane;
    if constexpr (Compact) {
      if (compact & 1u) {
        const uint32_t base = __ldg(&index[row].exponent_base[tile]);
        const uint2 low = load_low<CacheGlobal>(current + 8 * lane);
        const uint32_t codes = load_codes<CacheGlobal>(current + kValues + 4 * lane);
        #pragma unroll
        for (int step = 0; step < kSteps; ++step) {
          const uint32_t lo = ((step < 4 ? low.x : low.y) >> (8 * (step & 3))) & 255u;
          const uint32_t code = codes >> (step * 4);
          const uint32_t hi = (base + (code & 7u)) | ((code & 8u) << 4);
          const float w = __uint_as_float((hi << 24) | (lo << 16));
          const float xx = bf16_word_to_float(__ldg(input + kLanes * step));
          accum = __fmaf_rn(xx, w, accum);
        }
        current += kCompactBytes;
        compact >>= 1;
        continue;
      }
      compact >>= 1;
    }

    const uint4 vector = load_dense<CacheGlobal>(current + 16 * lane);
    const uint32_t pairs[4] = {vector.x, vector.y, vector.z, vector.w};
    #pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      const float w = bf16_word_to_float(pairs[step >> 1] >> (16 * (step & 1)));
      const float xx = bf16_word_to_float(__ldg(input + kLanes * step));
      accum = __fmaf_rn(xx, w, accum);
    }
    current += kDenseBytes;
  }
  #pragma unroll
  for (int distance = kLanes / 2; distance > 0; distance >>= 1) {
    accum = __fadd_rn(accum, __shfl_down_sync(0xffffffffu, accum, distance));
  }
  if (lane == 0) {
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(out[row]) : "f"(accum));
  }
}

// ---------------------------------------------------------------- rANS high-byte BF16 GEMV
__device__ __forceinline__ uint64_t pol_ef() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16_ef(uint32_t sd, const void* gs, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "l"(pol));
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds16(uint32_t a) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return static_cast<uint32_t>(v);
}

__device__ __forceinline__ size_t phys_off(const int* __restrict__ woff16, int tw, int gmin, int w, uint32_t lb) {
  const uint32_t g = lb >> 9;
  const size_t gi = (g < static_cast<uint32_t>(gmin))
                        ? (static_cast<size_t>(g) * static_cast<size_t>(tw) + static_cast<size_t>(w))
                        : (static_cast<size_t>(woff16[tw + 1 + w]) + static_cast<size_t>(g - static_cast<uint32_t>(gmin)));
  return (gi << 9) | static_cast<size_t>(lb & 511u);
}

__global__ void bf16_hist_kernel(const uint16_t* __restrict__ W, size_t n, unsigned long long* counts) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    atomicAdd(&h[W[i] >> 8], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&counts[i], static_cast<unsigned long long>(h[i]));
}

__global__ void bf16_enc_pass1_kernel(const uint16_t* __restrict__ W, int rows, int tw,
                                      const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
                                      uint32_t* __restrict__ warp_bytes16) {
  __shared__ uint32_t sf[256], sc[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { sf[i] = enc_f[i]; sc[i] = enc_c[i]; }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  int rlast = w;
  while (rlast + tw < rows) rlast += tw;
  uint32_t tot_u16 = 0;
  for (int row = rlast; row >= w; row -= tw) {
    const uint16_t* src = W + static_cast<size_t>(row) * kWidth;
    for (int tile = kTiles - 1; tile >= 0; --tile) {
      const uint16_t* tsrc = src + tile * kValues + lane;
      uint32_t ws = 0;
#pragma unroll
      for (int step = kSteps - 1; step >= 0; --step) {
        uint32_t& x = st[step & 3];
        const uint32_t sy = tsrc[step * kLanes] >> 8;
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        ws += __popc(__ballot_sync(0xffffffffu, emit));
        if (emit) x >>= 16;
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
      tot_u16 += 128u + ws;
    }
  }
  if (lane == 0) warp_bytes16[w] = (128u * kNS + 2u * tot_u16 + 15u) / 16u;
}

__global__ void bf16_enc_pass2_kernel(const uint16_t* __restrict__ W, int rows, int tw, int gmin,
                                      const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
                                      const int* __restrict__ woff16, uint8_t* __restrict__ stream) {
  __shared__ uint32_t sf[256], sc[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { sf[i] = enc_f[i]; sc[i] = enc_c[i]; }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (w >= tw) return;
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  int rlast = w;
  while (rlast + tw < rows) rlast += tw;
  uint32_t tot_u16 = 0;
  for (int row = rlast; row >= w; row -= tw) {
    const uint16_t* src = W + static_cast<size_t>(row) * kWidth;
    for (int tile = kTiles - 1; tile >= 0; --tile) {
      const uint16_t* tsrc = src + tile * kValues + lane;
      uint32_t ws = 0;
#pragma unroll
      for (int step = kSteps - 1; step >= 0; --step) {
        uint32_t& x = st[step & 3];
        const uint32_t sy = tsrc[step * kLanes] >> 8;
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        ws += __popc(__ballot_sync(0xffffffffu, emit));
        if (emit) x >>= 16;
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
      tot_u16 += 128u + ws;
    }
  }
  const uint32_t lbase = 128u * kNS;
  uint32_t posw = tot_u16;
#pragma unroll
  for (int s = 0; s < kNS; ++s) st[s] = kRansL;
  for (int row = rlast; row >= w; row -= tw) {
    const uint16_t* src = W + static_cast<size_t>(row) * kWidth;
    for (int tile = kTiles - 1; tile >= 0; --tile) {
      const uint16_t* tsrc = src + tile * kValues + lane;
      uint16_t lo_u16[4] = {0, 0, 0, 0};
#pragma unroll
      for (int step = kSteps - 1; step >= 0; --step) {
        const uint16_t word = tsrc[step * kLanes];
        lo_u16[step >> 1] |= static_cast<uint16_t>((word & 255u) << (8 * (step & 1)));
        uint32_t& x = st[step & 3];
        const uint32_t sy = word >> 8;
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        const uint32_t msk = __ballot_sync(0xffffffffu, emit);
        posw -= __popc(msk);
        if (emit) {
          *reinterpret_cast<uint16_t*>(stream + phys_off(woff16, tw, gmin, w, lbase + 2u * (posw + __popc(msk & lt)))) =
              static_cast<uint16_t>(x & 0xFFFFu);
          x >>= 16;
        }
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
      posw -= 128u;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        *reinterpret_cast<uint16_t*>(stream + phys_off(woff16, tw, gmin, w, lbase + 2u * (posw + 4u * lane + q))) = lo_u16[q];
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    *reinterpret_cast<uint32_t*>(stream + phys_off(woff16, tw, gmin, w, 128u * s + 4u * lane)) = st[s];
  }
}

__global__ void __launch_bounds__(512, 1) bf16_rans_gemv_kernel(
    const uint16_t* __restrict__ x_in,
    const uint8_t* __restrict__ blob,
    uint16_t* __restrict__ out,
    int rows) {
  // Blob layout (self-describing, no host sync needed on launch):
  //   [0..4096):    u32 lut[1024] = hi_byte | (freq << 8) | ((slot - cum) << 19)
  //   [4096..4112): u32 meta[4]   = {rows, tw, gmin, soff}
  //   [4112..soff): int32 woff16[tw + 1], int32 toff[tw], padded to 512 B
  //   [soff..):     512 B interleaved warp streams
  constexpr uint32_t GB = 512u;
  constexpr uint32_t RG = 4u;
  constexpr uint32_t RB = GB * RG;
  __shared__ __align__(16) uint32_t sh_lut[1024];
  __shared__ __align__(16) uint16_t sh_x[kWidth];
  extern __shared__ __align__(16) unsigned char smem[];

  const int tid = threadIdx.x, lane = tid & 31, wib = tid >> 5, wpb = blockDim.x >> 5;
  const uint32_t* meta = reinterpret_cast<const uint32_t*>(blob + 4096);
  const int tw = static_cast<int>(__ldg(&meta[1]));
  const int gmin = static_cast<int>(__ldg(&meta[2]));
  const int soff = static_cast<int>(__ldg(&meta[3]));

  const uint32_t sbase0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t ring_s = ((sbase0 + RB - 1u) & ~(RB - 1u)) + static_cast<uint32_t>(wib) * RB;
  const int w = blockIdx.x * wpb + wib;
  const bool active = w < tw;

  const int* woff16 = reinterpret_cast<const int*>(blob + 4112);
  const uint8_t* stream = blob + soff;
  uint32_t wbytes = 0;
  int toffw = 0, issued = 0;
  const uint64_t pol = pol_ef();

#define BF_ISSUE() { \
    uint32_t boff = static_cast<uint32_t>(issued) * 512u + static_cast<uint32_t>(lane) * 16u; \
    if (boff < wbytes) { \
      size_t gi = (issued < gmin) ? (static_cast<size_t>(issued) * tw + w) : static_cast<size_t>(toffw + issued - gmin); \
      cp_async16_ef(ring_s + (boff & (RB - 1u)), stream + (gi << 9) + static_cast<uint32_t>(lane) * 16u, pol); \
    } \
    asm volatile("cp.async.commit_group;" ::: "memory"); \
    issued++; \
  }

  if (active) {
    wbytes = static_cast<uint32_t>(woff16[w + 1] - woff16[w]) * 16u;
    toffw = woff16[tw + 1 + w];
#pragma unroll
    for (int i = 0; i < static_cast<int>(RG) - 1; ++i) BF_ISSUE();
  }
  for (int i = tid; i < 256; i += blockDim.x) {
    reinterpret_cast<uint4*>(sh_lut)[i] = reinterpret_cast<const uint4*>(blob)[i];
  }
  for (int i = tid; i < kWidth / 8; i += blockDim.x) {
    reinterpret_cast<uint4*>(sh_x)[i] = reinterpret_cast<const uint4*>(x_in)[i];
  }
  __syncthreads();
  if (!active) return;

  const uint32_t lt = (1u << lane) - 1u;
  asm volatile("cp.async.wait_group %0;" ::"n"(RG - 2) : "memory");
  __syncwarp();
  uint32_t st[kNS];
#pragma unroll
  for (int s = 0; s < kNS; ++s) {
    st[s] = lds32(ring_s + ((128u * s + 4u * lane) & (RB - 1u)));
  }
  uint32_t pwb = 128u * kNS;

  for (int row = w; row < rows; row += tw) {
    float accum = 0.0f;
    for (int tile = 0; tile < kTiles; ++tile) {
      __syncwarp();
      const int curg = static_cast<int>(pwb >> 9);
      while (issued <= curg + static_cast<int>(RG) - 1) BF_ISSUE();
      asm volatile("cp.async.wait_group %0;" ::"n"(RG - 2) : "memory");
      __syncwarp();
      uint32_t lo0, lo1;
      const uint32_t a0 = pwb + 8u * static_cast<uint32_t>(lane);
      if ((pwb & 2u) == 0u) {
        lo0 = lds32(ring_s + (a0 & (RB - 1u)));
        lo1 = lds32(ring_s + ((a0 + 4u) & (RB - 1u)));
      } else {
        lo0 = lds16(ring_s + (a0 & (RB - 1u))) | (lds16(ring_s + ((a0 + 2u) & (RB - 1u))) << 16);
        lo1 = lds16(ring_s + ((a0 + 4u) & (RB - 1u))) | (lds16(ring_s + ((a0 + 6u) & (RB - 1u))) << 16);
      }
      pwb += 256u;
      const uint16_t* xtile = sh_x + tile * kValues + lane;
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        uint32_t& rx = st[step & 3];
        const uint32_t e = sh_lut[rx & 1023u];
        rx = ((e >> 8) & 2047u) * (rx >> 10) + (e >> 19);
        const uint32_t hi = e & 255u;
        const uint32_t lo = ((step < 4 ? lo0 : lo1) >> (8 * (step & 3))) & 255u;
        const bool need = rx < kRansL;
        const uint32_t msk = __ballot_sync(0xffffffffu, need);
        if (need) rx = (rx << 16) | lds16(ring_s + ((pwb + (__popc(msk & lt) << 1)) & (RB - 1u)));
        pwb += __popc(msk) << 1;
        const float wv = __uint_as_float((hi << 24) | (lo << 16));
        const float xv = __uint_as_float(static_cast<uint32_t>(xtile[kLanes * step]) << 16);
        accum = __fmaf_rn(xv, wv, accum);
      }
    }
#pragma unroll
    for (int dist = 16; dist > 0; dist >>= 1) {
      accum = __fadd_rn(accum, __shfl_down_sync(0xffffffffu, accum, dist));
    }
    if (lane == 0) {
      asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(out[row]) : "f"(accum));
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#undef BF_ISSUE
}

bool valid_rows(int rows) {
  return rows > 0 && rows <= 4608 && rows % 4 == 0;
}
}  // namespace

extern "C" size_t muse_bf16_lane_pack(const uint16_t* source, int rows, uint8_t* encoded,
                                      size_t capacity, void* output_indices, int compress) {
  if (!valid_rows(rows) || capacity < size_t(rows) * kWidth * 2) return 0;
  auto* index = static_cast<RowIndex*>(output_indices);
  size_t bytes = 0;
  for (int row = 0; row < rows; ++row) {
    RowIndex entry{};
    entry.offset = static_cast<uint32_t>(bytes);
    for (int tile = 0; tile < kTiles; ++tile) {
      const uint16_t* original = source + row * kWidth + tile * kValues;
      uint32_t low = 127, high = 0;
      if (compress) {
        for (int i = 0; i < kValues; ++i) {
          const uint32_t exponent = (original[i] >> 8) & 127u;
          if (exponent < low) low = exponent;
          if (exponent > high) high = exponent;
        }
      }
      if (compress && high - low < 8) {
        entry.exponent_base[tile] = low;
        entry.compact |= 1u << tile;
        for (int lane = 0; lane < kLanes; ++lane) {
          uint32_t codes = 0;
          for (int step = 0; step < kSteps; ++step) {
            const uint16_t word = original[step * kLanes + lane];
            const uint32_t code = (((word >> 8) & 127u) - low) | ((word >> 12) & 8u);
            encoded[bytes + lane * kSteps + step] = static_cast<uint8_t>(word);
            codes |= code << (step * 4);
          }
          std::memcpy(encoded + bytes + kValues + 4 * lane, &codes, sizeof(codes));
        }
        bytes += kCompactBytes;
      } else {
        for (int lane = 0; lane < kLanes; ++lane) {
          for (int step = 0; step < kSteps; ++step) {
            const size_t destination = bytes + (lane * kSteps + step) * 2;
            std::memcpy(encoded + destination, original + step * kLanes + lane, sizeof(uint16_t));
          }
        }
        bytes += kDenseBytes;
      }
    }
    index[row] = entry;
  }
  return bytes;
}

extern "C" int muse_bf16_lane_verify(const uint16_t* source, int rows, const uint8_t* encoded,
                                     const void* source_indices, size_t bytes) {
  if (!valid_rows(rows)) return 0;
  const auto* index = static_cast<const RowIndex*>(source_indices);
  size_t offset = 0;
  for (int row = 0; row < rows; ++row) {
    const RowIndex entry = index[row];
    if (entry.offset != offset || entry.compact >> kTiles) return 0;
    for (int tile = 0; tile < kTiles; ++tile) {
      const bool compact = entry.compact & (1u << tile);
      const uint16_t* original = source + row * kWidth + tile * kValues;
      const size_t length = compact ? kCompactBytes : kDenseBytes;
      if (offset + length > bytes) return 0;
      for (int lane = 0; lane < kLanes; ++lane) {
        uint32_t codes;
        if (compact) std::memcpy(&codes, encoded + offset + kValues + 4 * lane, sizeof(codes));
        for (int step = 0; step < kSteps; ++step) {
          const size_t position = lane * kSteps + step;
          uint16_t reconstructed;
          if (compact) {
            const uint32_t code = (codes >> (step * 4)) & 15u;
            const uint32_t high = (entry.exponent_base[tile] + (code & 7u)) | ((code & 8u) << 4);
            reconstructed = static_cast<uint16_t>((high << 8) | encoded[offset + position]);
          } else {
            std::memcpy(&reconstructed, encoded + offset + 2 * position, sizeof(reconstructed));
          }
          if (reconstructed != original[step * kLanes + lane]) return 0;
        }
      }
      offset += length;
    }
  }
  return offset == bytes;
}

extern "C" int muse_bf16_lane_gemv(int variant, int rows, const void* input,
                                   const void* weights, const void* index, void* output,
                                   cudaStream_t stream) {
  if (!valid_rows(rows)) return cudaErrorInvalidValue;
  const dim3 threads(kLanes, 4);
  const auto* x = static_cast<const uint16_t*>(input);
  const auto* packed = static_cast<const uint8_t*>(weights);
  const auto* lookup = static_cast<const RowIndex*>(index);
  auto* out = static_cast<uint16_t*>(output);
  switch (variant) {
    case 0: lane_gemv<false, false><<<rows/4, threads, 0, stream>>>(x, packed, lookup, out); break;
    case 1: lane_gemv<true, false><<<rows/4, threads, 0, stream>>>(x, packed, lookup, out); break;
    case 2: lane_gemv<true, true><<<rows/4, threads, 0, stream>>>(x, packed, lookup, out); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" int muse_bf16_rans_hist(const uint16_t* W, size_t n, unsigned long long* counts, cudaStream_t stream) {
  bf16_hist_kernel<<<256, 256, 0, stream>>>(W, n, counts);
  return cudaGetLastError();
}

extern "C" int muse_bf16_rans_pass1(const uint16_t* W, int rows, int tw,
                                    const uint32_t* f, const uint32_t* c, uint32_t* wb, cudaStream_t stream) {
  bf16_enc_pass1_kernel<<<(tw + 3) / 4, 128, 0, stream>>>(W, rows, tw, f, c, wb);
  return cudaGetLastError();
}

extern "C" int muse_bf16_rans_pass2(const uint16_t* W, int rows, int tw, int gmin,
                                    const uint32_t* f, const uint32_t* c, const int* woff, uint8_t* s,
                                    cudaStream_t stream) {
  bf16_enc_pass2_kernel<<<(tw + 3) / 4, 128, 0, stream>>>(W, rows, tw, gmin, f, c, woff, s);
  return cudaGetLastError();
}

extern "C" int muse_bf16_rans_gemv(int rows, const void* input, const void* blob, void* output, cudaStream_t stream) {
  if (!valid_rows(rows)) return cudaErrorInvalidValue;
  const int tw = (rows == 4608) ? 576 : ((rows == 4096) ? 512 : (rows / 8));
  if (tw <= 0) return cudaErrorInvalidValue;
  const int wpb = (tw + 47) / 48;
  const int blocks = (tw + wpb - 1) / wpb;
  const size_t rb = 4u * 512u;
  const size_t dsmem = rb + static_cast<size_t>(wpb) * rb;
  static bool init = false;
  if (!init) {
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    cudaFuncSetAttribute(bf16_rans_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 75 * 1024);
    cudaThreadExchangeStreamCaptureMode(&mode);
    init = true;
  }
  bf16_rans_gemv_kernel<<<blocks, wpb * 32, dsmem, stream>>>(
      static_cast<const uint16_t*>(input), static_cast<const uint8_t*>(blob),
      static_cast<uint16_t*>(output), rows);
  return cudaGetLastError();
}
