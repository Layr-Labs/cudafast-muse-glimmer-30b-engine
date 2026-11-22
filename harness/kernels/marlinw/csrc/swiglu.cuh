// The separate baseline first rounds each projection to BF16 and then computes
// gate / (1 + exp(-gate)) * up in FP32. These operations match the pinned
// CUDA 13 libdevice lowering and the Inductor div.full.f32 rounding.
#pragma once

__device__ __forceinline__ float swiglu_silu(float gate) {
  float result;
  asm("{\n\t"
      ".reg .f32 x, a, b, c, d, e, f, g, h;\n\t"
      ".reg .b32 exp2_scale;\n\t"
      "sub.f32 x, 0f00000000, %1;\n\t"
      "fma.rn.ftz.f32 a, x, 0f3BBB989D, 0f3F000000;\n\t"
      "cvt.ftz.sat.f32.f32 b, a;\n\t"
      "fma.rm.ftz.f32 c, b, 0f437C0000, 0f4B400001;\n\t"
      "add.f32 d, c, 0fCB40007F;\n\t"
      "neg.f32 d, d;\n\t"
      "fma.rn.ftz.f32 e, x, 0f3FB8AA3B, d;\n\t"
      "fma.rn.ftz.f32 f, x, 0f32A57060, e;\n\t"
      "shl.b32 exp2_scale, c, 23;\n\t"
      "ex2.approx.ftz.f32 g, f;\n\t"
      "fma.rn.f32 h, g, exp2_scale, 0f3F800000;\n\t"
      "div.full.f32 %0, %1, h;\n\t"
      "}" : "=f"(result) : "f"(gate));
  return result;
}

__global__ void swiglu_initialize() {}

__device__ __forceinline__ __nv_bfloat16 swiglu_product(__nv_bfloat16 gate,
                                                       __nv_bfloat16 up) {
  float silu = swiglu_silu(__bfloat162float(gate));
  return __float2bfloat16_rn(__fmul_rn(silu, __bfloat162float(up)));
}

#ifdef MARLINW_SWIGLU_FP8
__device__ __forceinline__ uint16_t swiglu_fp8_pair(uint32_t gates, uint32_t ups, float inverse) {
      float scaled[2];
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        const auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        const float silu = swiglu_silu(__bfloat162float(gate));
        const float product = __fmul_rn(silu, __bfloat162float(up));
        const float q = __fmul_rn(product, inverse);
        scaled[e] = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
      }
      uint16_t converted;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(converted) : "f"(scaled[1]), "f"(scaled[0]));
      return converted;
}
#endif
