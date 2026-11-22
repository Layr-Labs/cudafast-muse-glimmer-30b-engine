// silu(g) * u for a pair of bf16 elements, with the exact instruction chain of
// the TorchInductor (Triton) kernel that the vLLM model's compiled graph runs
// for `SiluAndMul` on bf16 inputs: g, u widened to fp32; silu as
// g / (exp(-g) + 1) with libdevice's ftz expf (Cody-Waite reduction, ex2.approx)
// whose final scaling is contracted with the + 1 into one fma; div.full.f32;
// fp32 multiply by u; one round to nearest-even bf16 at the end (no rounding of
// silu(g) to bf16 in between). Inputs and output: two bf16 packed in a 32-bit
// word, element 0 in the low half.
#pragma once

#include <cstdint>

namespace marlinw {

__device__ __forceinline__ uint32_t silu_mul_one(uint32_t g_bits16,
                                                 uint32_t u_bits16) {
  uint32_t r;
  asm("{\n\t"
      ".reg .b16 gh, uh;\n\t"
      ".reg .b32 g, u, n, t, j, jr, f, si, e, d, q;\n\t"
      "cvt.u16.u32 gh, %1;\n\t"
      "cvt.u16.u32 uh, %2;\n\t"
      "cvt.f32.bf16 g, gh;\n\t"
      "cvt.f32.bf16 u, uh;\n\t"
      "sub.f32 n, 0f00000000, g;\n\t"
      "fma.rn.ftz.f32 t, n, 0f3BBB989D, 0f3F000000;\n\t"
      "cvt.ftz.sat.f32.f32 t, t;\n\t"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n\t"
      "add.f32 jr, j, 0fCB40007F;\n\t"
      "neg.f32 jr, jr;\n\t"
      "fma.rn.ftz.f32 f, n, 0f3FB8AA3B, jr;\n\t"
      "fma.rn.ftz.f32 f, n, 0f32A57060, f;\n\t"
      "shl.b32 si, j, 23;\n\t"
      "ex2.approx.ftz.f32 e, f;\n\t"
      "fma.rn.f32 d, e, si, 0f3F800000;\n\t"
      "div.full.f32 q, g, d;\n\t"
      "mul.rn.f32 %0, q, u;\n\t"
      "}"
      : "=r"(r)
      : "r"(g_bits16), "r"(u_bits16));
  return r;  // fp32 bits
}

__device__ __forceinline__ int silu_mul_bf16x2(int g2, int u2) {
  uint32_t g = static_cast<uint32_t>(g2), u = static_cast<uint32_t>(u2);
  uint32_t lo = silu_mul_one(g & 0xffffu, u & 0xffffu);
  uint32_t hi = silu_mul_one(g >> 16, u >> 16);
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "r"(hi), "r"(lo));
  return static_cast<int>(r);
}

}  // namespace marlinw
