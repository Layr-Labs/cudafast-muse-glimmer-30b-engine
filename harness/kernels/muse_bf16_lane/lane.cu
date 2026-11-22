// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace {
constexpr int kWidth = 6656;
constexpr int kLanes = 32;
constexpr int kSteps = 8;
constexpr int kValues = kSteps * kLanes;
constexpr int kTiles = kWidth / kValues;
constexpr int kDenseBytes = kValues * sizeof(uint16_t);
constexpr int kCompactBytes = kValues + kValues / 2;
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

  // Each lane must retain columns lane, lane+32, ... in that order. This
  // reproduces the exact single FP32 accumulator and descending warp fold
  // verified against both stock cuBLAS GEMVs.
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
