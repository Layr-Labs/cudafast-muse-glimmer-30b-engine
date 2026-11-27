#pragma once

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <vector_types.h>

namespace muse_nvfp4 {

__device__ __forceinline__ float2 widen_pair_x2(unsigned byte) {
  __nv_fp4x2_e2m1 packed;
  packed.__x = static_cast<__nv_fp4x2_storage_t>(byte);
  const __half2 w = static_cast<__half2>(packed);
  return __half22float2(__hadd2(w, w));
}

__device__ __forceinline__ float group_dot(
    unsigned wlo, unsigned whi, const uint4& a0, const uint4& a1) {
  const auto* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const auto* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float gsum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap0[i]);
    const float2 wf = widen_pair_x2(wlo);
    gsum = __fmaf_rn(wf.x, af.x, gsum);
    gsum = __fmaf_rn(wf.y, af.y, gsum);
    wlo >>= 8;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap1[i]);
    const float2 wf = widen_pair_x2(whi);
    gsum = __fmaf_rn(wf.x, af.x, gsum);
    gsum = __fmaf_rn(wf.y, af.y, gsum);
    whi >>= 8;
  }
  return gsum;
}

}
