/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "dequant.h"
#include "marlin_mma.h"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId b_type_id,  // weight MarlinScalarType id
          const vllm::ScalarTypeId s_type_id,  // weight scale ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,  // whether act_order is enabled
          const int group_blocks,    // number of consecutive 16x16 blocks
                                     // with a separate quantization scale
          const bool is_zp_float     // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {}

}  // namespace marlin

#else

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <vllm::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s,
                             int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_and_sub(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t s,
    typename MarlinScalarType<type_id>::scalar_t zp) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s2 = MarlinScalarType<type_id>::num2num2(s);
  scalar_t2 zp2 = MarlinScalarType<type_id>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <vllm::ScalarTypeId type_id>
__device__ inline void sub_zp(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 zp = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Same as above, but for act_order (each K is multiplied individually)
template <vllm::ScalarTypeId type_id>
__device__ inline void scale4(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragS& frag_s_1,
    typename MarlinScalarType<type_id>::FragS& frag_s_2,
    typename MarlinScalarType<type_id>::FragS& frag_s_3,
    typename MarlinScalarType<type_id>::FragS& frag_s_4, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;

  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Given 2 floats multiply by 2 scales (halves)
template <vllm::ScalarTypeId type_id>
__device__ inline void scale_float(
    float* c, typename MarlinScalarType<type_id>::FragS& s) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], MarlinScalarType<type_id>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], MarlinScalarType<type_id>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

__device__ __forceinline__ float marlin_swiglu_silu(float gate) {
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

__device__ __forceinline__ float marlin_full_rcp(float s) {
  float r;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(s));
  return r;
}

__device__ __forceinline__ uint16_t marlin_pack_fp8x2(float v0, float v1) {
  auto cl = [](float v) {
    return (v != v) ? 448.0f : (v < -448.0f ? -448.0f : (v > 448.0f ? 448.0f : v));
  };
  uint16_t lo;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(cl(v1)), "f"(cl(v0)));
  return lo;
}

__device__ __forceinline__ float marlin_triton_sigmoid(float g) {
  float sig;
  asm("{\n\t"
      ".reg .f32 ng, t, e, d;\n\t"
      "sub.f32 ng, 0f00000000, %1;\n\t"
      "mul.f32 t, ng, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 e, t;\n\t"
      "add.f32 d, e, 0f3F800000;\n\t"
      "div.full.f32 %0, 0f3F800000, d;\n\t"
      "}" : "=f"(sig) : "f"(g));
  return sig;
}

template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float,  // is zero point of float16 type?
          // muse scale6: the group-16 FP8 block scales arrive as 6-bit codes
          // into a 64-entry per-layer palette (3/4 of the bytes); only the
          // M = 1 NVFP4 decode configuration supports it.
          const bool scale6 = false,
          const bool gate_up_grid = false,
          const bool down_ilv = false,
          const bool gate_v48 = false,
          const bool gate_up_swiglu = false,
          const bool fuse_lmpost = false,
          const bool fuse_mul_sigmoid = false,
          const bool split2 = false,
          const int fixed_shape = 0,
          const bool gate_up_swiglu_fp8 = false,
          const bool tma_ovlp = false,
          const bool half_split = false,
          const bool fast_pal16 = false,
          const bool multi_m = false,
          const bool kloop_ovlp = false>
__global__ void Marlin(
    const int4* __restrict__ A0,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,   // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C0,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,     // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    // float scales of input matrix, only used when is_a_8bit == true.
    // shape (m,)
    const float* __restrict__ a_scales_ptr,
    // fp16 quantization scales. shape (k/groupsize, n)
    const int4* __restrict__ scales_ptr,
    // float global scale (for nvfp4// only)
    const float* __restrict__ global_scale_ptr,
    // 4bit packed zero-points of shape
    // (k/groupsize, n/pack_factor)
    const int4* __restrict__ zp_ptr,
    // int32 group indices of shape k
    const int* __restrict__ g_idx,
    int num_groups,  // number of scale groups per output channel
    int prob_m,      // batch dimension m
    int prob_n,      // output dimension n
    int prob_k,      // reduction dimension k
    int lda,         // A.stride(0), equal to prob_k is A is contiguous
    int* locks,      // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,   // whether to use atomic add to reduce
    bool use_fp32_reduce,  // whether to use fp32 global reduce
    int max_shared_mem,
    // scale6 only: 64 palette bytes (16-byte aligned) of this layer's codes
    const int4* __restrict__ scale6_palette,
    size_t tpf_stride = 0,
    int* pair_locks = nullptr,
    const uint16_t* __restrict__ lmpost_lut = nullptr,
    int64_t* __restrict__ lmpost_out = nullptr,
    int org_vocab_size = 0,
    const int4* __restrict__ A_aux = nullptr
#ifdef MUSE_SCALE_TMA4
    ,
    int tma_pdl_mode = 0
#endif
) {
  if constexpr (scale6) {
    if constexpr (!multi_m) prob_m = 1;
    lda = prob_k;
    num_groups = prob_k >> 4;
    has_bias = false;
    use_atomic_add = false;
    use_fp32_reduce = true;
  }
  if constexpr (gate_up_grid) {
    static_assert(scale6 && threads == 256 && thread_n_blocks == 8 && thread_k_blocks == 8);
    prob_n = 39936;
    prob_k = lda = 6656;
    num_groups = 416;
  }
  if constexpr (gate_up_swiglu) {
    static_assert(scale6 && threads == 256 && thread_n_blocks == 8 && thread_k_blocks == 8 &&
                  (gate_up_grid || down_ilv) && !gate_v48 && !half_split && !fuse_lmpost && !fuse_mul_sigmoid);
    prob_n = 39936;
    prob_k = lda = 6656;
    num_groups = 416;
  }
  if constexpr (fuse_lmpost) {
    static_assert(scale6 && m_block_size_8 && !gate_up_grid && !gate_v48 && !half_split &&
                  ((thread_n_blocks == 4 && threads == 128) ||
                   (thread_n_blocks == 8 && threads == 256)) &&
                  c_type_id == vllm::kBFloat16.id());
  }
  if constexpr (fuse_mul_sigmoid) {
    static_assert(scale6 &&
                  ((threads == 256 && thread_n_blocks == 8) ||
                   (threads == 128 && thread_n_blocks == 4 && half_split)) &&
                  thread_k_blocks == 8 &&
                  !gate_up_grid && !gate_v48 && !gate_up_swiglu && !fuse_lmpost);
    prob_n = 6656;
    prob_k = lda = 4096;
    num_groups = 256;
  }
  if constexpr (half_split) {
    static_assert(scale6 && threads == 128 && thread_n_blocks == 4 && thread_k_blocks == 8 &&
                  down_ilv && !gate_up_grid && !gate_v48 && !split2 && !gate_up_swiglu && !fuse_lmpost);
  }
  if constexpr (fixed_shape == 1) {
    prob_n = 6656;
    prob_k = lda = 19968;
    num_groups = 1248;
  } else if constexpr (fixed_shape == 2) {
    prob_n = 202048;
    prob_k = lda = 6656;
    num_groups = 416;
  } else if constexpr (fixed_shape == 3) {
    static_assert(scale6 && m_block_size_8 && down_ilv && !gate_up_grid && !gate_v48 &&
                  !split2 && !half_split && !gate_up_swiglu && !fuse_mul_sigmoid &&
                  thread_n_blocks == 8 && threads == 256 && thread_k_blocks == 8);
    prob_n = 202112;
    prob_k = lda = 6656;
    num_groups = 416;
  }
  [[maybe_unused]] float inv_fp8_scale = 1.0f;
  [[maybe_unused]] uint8_t* const fp8_out_ptr = reinterpret_cast<uint8_t*>(lmpost_out);
  if constexpr (gate_up_swiglu_fp8) {
    static_assert(gate_up_swiglu && !fuse_lmpost);
    inv_fp8_scale = marlin_full_rcp(__ldg(a_scales_ptr));
  }
  [[maybe_unused]] float lm_bv = -__int_as_float(0x7f800000);
  [[maybe_unused]] int lm_bi = 0x7fffffff;
  [[maybe_unused]] float lm_bv1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int lm_bi1 = 0x7fffffff;
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  // FP8 computation is only supported for Ada Lovelace or newer architectures.
  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  // Turing TensorCore only supports fp16 and int8
  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id())
    return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  constexpr auto num_bits = vllm::ScalarType::from_id(b_type_id).size_bits();
  // Disable use_fp16_accum for NVFP4 and cases when group_size == -1 &&
  // num_bits == 4
  constexpr bool use_fp16_accum =
      a_type_id == vllm::kFloat16.id() &&
      (!(b_type_id == vllm::kFE2M1f.id() && s_type_id == vllm::kFE4M3fn.id()) &&
       !(group_blocks == -1 && num_bits == 4));
  #else
  constexpr bool use_fp16_accum = false;
  #endif
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  const int4* A = A0;
  int4* C = C0;

  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;

  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;

  using FragA = typename MarlinScalarType<a_type_id>::FragA;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using FragC = typename MarlinScalarType<a_type_id>::FragC;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;

  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (s_type == vllm::kFE8M0fnu) {
    // MXFP8: FP8 weights with e8m0 microscaling block scales
    static_assert(b_type == vllm::kFE4M3fn && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  constexpr bool is_a_8bit = a_type.size_bits() == 8;
  constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  if constexpr (!is_a_8bit) {
    static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  }
  constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                               b_type == vllm::kS4 || b_type == vllm::kS8 ||
                               b_type == vllm::kU4B8 || b_type == vllm::kU8B128;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop =
      is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);

  float global_scale_f32 = 1.0f;

  if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
    global_scale_f32 = global_scale_ptr[0];
  }

  constexpr bool has_act_order = group_blocks == 0;
  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);

  extern __shared__ int4 sh[];
  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_new = sh + (is_a_8bit ? (4 * thread_m_blocks) : 0);
  constexpr int pack_factor = 32 / b_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  const int sched_n_tiles = half_split ? (n_tiles >> 1) : n_tiles;

  int global_mn_tiles = parallel * sched_n_tiles;
  const int vgrid = (gate_v48 || split2 || half_split) ? 48 : gridDim.x;
  int half_id = (half_split && static_cast<int>(gridDim.x) == 96)
                    ? (static_cast<int>(blockIdx.x) & 1)
                    : 0;
  int vblk = gate_v48
                 ? (47 - static_cast<int>(blockIdx.x))
                 : (split2 ? (static_cast<int>(blockIdx.x) >= 48
                                  ? static_cast<int>(blockIdx.x) - 48
                                  : static_cast<int>(blockIdx.x))
                           : ((half_split && static_cast<int>(gridDim.x) == 96)
                                  ? (static_cast<int>(blockIdx.x) >> 1)
                                  : static_cast<int>(blockIdx.x)));
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;

  if (global_mn_tiles > vgrid) {
    part2_mn_tiles = global_mn_tiles % vgrid;
    if (part2_mn_tiles * 3 <= vgrid) part2_mn_tiles += vgrid;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / vgrid;
  }
  const int orig_p1_iters = part1_mn_iters;
  [[maybe_unused]] const int split2_lo_p1 = (orig_p1_iters + 1) / 2;
  [[maybe_unused]] const int split2_hi_p1 = orig_p1_iters - split2_lo_p1;
  if constexpr (split2) {
    part1_mn_iters = (static_cast<int>(blockIdx.x) < 48) ? split2_lo_p1 : split2_hi_p1;
  }

  int iters = div_ceil(k_tiles * part2_mn_tiles, vgrid);

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = 0;
  int slice_col_par = (split2 && static_cast<int>(blockIdx.x) >= 48)
                          ? (vblk + split2_lo_p1 * vgrid)
                          : vblk;
  int slice_col;
  int slice_iters =
      k_tiles;  // number of threadblock tiles in the current slice
  // total number of active threadblocks in the current slice
  int slice_count = 1;
  // index of threadblock in current slice; numbered bottom to top
  int slice_idx = 0;

  int par_id = 0;
  int locks_off = 0;

  if (part2_mn_tiles >= vgrid) {
    // when part2_mn_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = vblk;
  } else {
    locks_off = (iters * vblk) / k_tiles - 1;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters =
        iters * (vblk + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * vblk - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= vgrid) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(thread_m_blocks * 16, threads / threads_per_m);
      if (m_block_size_8) m_per_thread = div_ceil(8, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < prob_m) {
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * lda / (is_a_8bit ? 16 : 8);
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    if (is_a_8bit && (first_init || slice_col == 0)) {
      __syncthreads();
      int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
      cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                        threadIdx.x < prob_m);
    }
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / sched_n_tiles;
      const int base_sc = slice_col_par % sched_n_tiles;
      slice_col = half_split ? (2 * base_sc + half_id) : base_sc;
      slice_iters = k_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
      if (is_a_8bit) {
        __syncthreads();
        int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
        cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                          threadIdx.x < prob_m);
      }
    }
  };

  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      if constexpr (split2) {
        if (static_cast<int>(blockIdx.x) < 48) {
          slice_iters = 0;
          return;
        }
      }
      in_part2 = true;
      slice_col_par = (iters * vblk) / k_tiles;
      slice_row = (iters * vblk) % k_tiles;
      const int base_sc = (slice_col_par + global_mn_tiles - part2_mn_tiles) % sched_n_tiles;
      slice_col = half_split ? (2 * base_sc + half_id) : base_sc;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / sched_n_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };

  if constexpr (gate_up_grid) {
    // At this shape the 48-CTA schedule has 288 full columns and 24 split
    // columns. Issue one independent CTA per parent slice. For every split,
    // the high-K CTA produces the original FP32 partial and the low-K CTA
    // consumes it through the same lock and final reduction.
    const int task = blockIdx.x;
    const int tail = task - 288;
    const bool split = task >= 288;
    slice_col = split ? 288 + tail % 24 : task;
    slice_row = split && tail < 24 ? 26 : 0;
    slice_iters = split ? 26 : 52;
    slice_count = split ? 2 : 1;
    slice_idx = split && tail >= 24 ? 1 : 0;
    locks_off = split ? tail % 24 : 0;
  } else if constexpr (fixed_shape == 3) {
    slice_col = static_cast<int>(blockIdx.x);
    slice_row = 0;
    slice_iters = (slice_col < 1579) ? 52 : 0;
    slice_count = 1;
    slice_idx = 0;
    locks_off = 0;
  } else {
    init_slice();
  }

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = lda / (is_a_8bit ? 16 : 8);
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  constexpr int a_sh_stage_step =
      (!multi_m && scale6 && m_block_size_8 && (tma_ovlp || stages != 4 || half_split || fast_pal16 || fixed_shape == 3))
          ? a_sh_stride
          : a_sh_stage;
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // At M=1, each output tile stores its full K range contiguously.
  // The Marlin threadblock geometry and reduction order are unchanged.
  const int tile_column_stride = prob_k / 16;
  // B sizes/strides
  constexpr int b_gl_stride = 16 * (16 * thread_n_blocks) / (pack_factor * (is_a_8bit ? 2 : 4));
  constexpr int b_sh_stride =
      ((thread_n_blocks * 16) * 16 / pack_factor) / (is_a_8bit ? 2 : 4);
  constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides without act_order
  constexpr int s_gl_stride = (16 * thread_n_blocks) / (is_8bit_scale ? 16 : 8);
#ifdef MUSE_SCALE_TMA4
  constexpr int s_sh_stride = 16 * thread_n_blocks / ((is_8bit_scale && !fast_pal16) ? 16 : 8);
#else
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
#endif
  constexpr int s_tb_groups =
      !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;
  constexpr int act_s_max_num_groups = 32;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;

  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  const int a_gl_rd_base = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                           (threadIdx.x % a_gl_rd_delta_o);
  int a_gl_rd = a_gl_rd_base + a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;

  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  [[maybe_unused]] int ilv_a_kt = slice_row;
  [[maybe_unused]] int ilv_f = 0;
  [[maybe_unused]] int ilv_jump_f = 0;
  [[maybe_unused]] int ilv_jump_kt = 0;
  [[maybe_unused]] int ilv_step_p1end = 0;
  [[maybe_unused]] int ilv_step_switch = 0;
  [[maybe_unused]] int ilv_step_p2end = 0;
  [[maybe_unused]] int ilv_delta_p1 = 0;
  [[maybe_unused]] int ilv_delta_hi = 0;
  [[maybe_unused]] int ilv_delta_lo = 0;
  [[maybe_unused]] int ilv_delta_tail = 0;
#ifdef MUSE_GATE_UP_ILV
  [[maybe_unused]] int ilv_base = 0;
  if constexpr (gate_up_grid) {
    ilv_base = blockIdx.x < 288 ? (blockIdx.x / 48) * 2496 + blockIdx.x % 48
                                : 14976 + (blockIdx.x - 288);
    b_gl_rd += b_gl_rd_delta_o * ilv_base;
    b_gl_rd_delta_o *= 48;
  } else {
    b_gl_rd += b_sh_stride * tile_column_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
  }
#else
  if constexpr (down_ilv) {
    static_assert(scale6 && ((threads == 256 && thread_n_blocks == 8) ||
                             (threads == 128 && thread_n_blocks == 4)) && !gate_up_grid,
                  "down_ilv requires M=1 256T/128-col or 128T/64-col decode config");
    b_gl_rd += b_gl_rd_delta_o * blockIdx.x;
    const int total_p2 = k_tiles * part2_mn_tiles;
    if constexpr (gate_v48) {
      ilv_jump_f = min(iters, max(0, total_p2 - vblk * iters));
      ilv_jump_kt = (vblk >= static_cast<int>(gridDim.x))
                        ? ((iters * (vblk - static_cast<int>(gridDim.x))) % k_tiles)
                        : 0;
      ilv_step_switch = 53;
      ilv_delta_hi = 24 * b_gl_rd_delta_o;
      ilv_delta_lo = 23 * b_gl_rd_delta_o;
    } else if constexpr (split2) {
      const int hi_base = split2_hi_p1 * k_tiles;
      const int lo_active = (split2_lo_p1 > split2_hi_p1) ? 48 : 0;
      const int ilv_full_b = total_p2 / iters;
      const int ilv_rem_s = total_p2 - ilv_full_b * iters;
      ilv_jump_f = (static_cast<int>(blockIdx.x) < 48) ? (split2_lo_p1 * k_tiles) : hi_base;
      ilv_jump_kt = (static_cast<int>(blockIdx.x) < 48) ? 0 : ((iters * vblk) % k_tiles);
      ilv_step_p1end = lo_active ? hi_base : (hi_base - 1);
      ilv_step_switch = lo_active ? (hi_base + ilv_rem_s) : (hi_base + (ilv_rem_s > 0 ? ilv_rem_s : -1));
      ilv_step_p2end = hi_base + iters;
      ilv_delta_p1 = 96 * b_gl_rd_delta_o;
      ilv_delta_hi = (lo_active + ilv_full_b + 1) * b_gl_rd_delta_o;
      ilv_delta_lo = (lo_active + ilv_full_b) * b_gl_rd_delta_o;
      ilv_delta_tail = 48 * b_gl_rd_delta_o;
    } else if constexpr (half_split) {
      const int ilv_full_b48 = total_p2 / iters;
      const int ilv_rem_s = total_p2 - ilv_full_b48 * iters;
      if (gridDim.x == 96) {
        ilv_jump_f = orig_p1_iters * k_tiles;
        ilv_jump_kt = (iters * vblk) % k_tiles;
        ilv_step_p1end = ilv_jump_f;
        ilv_step_switch = ilv_jump_f + ilv_rem_s;
        ilv_delta_p1 = 96 * b_gl_rd_delta_o;
        ilv_delta_hi = 2 * (ilv_full_b48 + 1) * b_gl_rd_delta_o;
        ilv_delta_lo = 2 * ilv_full_b48 * b_gl_rd_delta_o;
      } else {
        const int p2_len = min(iters, max(0, total_p2 - vblk * iters));
        ilv_jump_f = p2_len;
        ilv_jump_kt = (iters * vblk) % k_tiles;
        ilv_step_p1end = 0;
        ilv_step_switch = 2 * ilv_rem_s;
        ilv_delta_p1 = 48 * b_gl_rd_delta_o;
        ilv_delta_hi = (ilv_full_b48 + 1) * b_gl_rd_delta_o;
        ilv_delta_lo = ilv_full_b48 * b_gl_rd_delta_o;
      }
    } else if constexpr (fixed_shape == 3) {
      const int full_p = 1579 / static_cast<int>(gridDim.x);
      const int rem_c = 1579 - full_p * static_cast<int>(gridDim.x);
      ilv_jump_f = -1;
      ilv_jump_kt = 0;
      ilv_step_p1end = full_p * 52;
      ilv_step_switch = (full_p + 1) * 52;
      ilv_delta_p1 = static_cast<int>(gridDim.x) * b_gl_rd_delta_o;
      ilv_delta_hi = rem_c * b_gl_rd_delta_o;
      ilv_delta_lo = rem_c * b_gl_rd_delta_o;
    } else {
      ilv_jump_f = orig_p1_iters * k_tiles;
      ilv_jump_kt = (iters * vblk) % k_tiles;
      const int ilv_full_b = total_p2 / iters;
      const int ilv_rem_s = total_p2 - ilv_full_b * iters;
      ilv_step_p1end = ilv_jump_f;
      ilv_step_switch = ilv_jump_f + ilv_rem_s;
      ilv_delta_p1 = 48 * b_gl_rd_delta_o;
      ilv_delta_hi = (ilv_full_b + 1) * b_gl_rd_delta_o;
      ilv_delta_lo = ilv_full_b * b_gl_rd_delta_o;
    }
  } else {
    b_gl_rd += b_sh_stride * tile_column_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
  }
#endif
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));

  // For act_order
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

  // No act_order
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * tile_column_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * tile_column_stride * slice_col + threadIdx.x;
    } else {
      if constexpr (scale6) {
        // codes: each column tile's K range is 3/4 as long; a stage (k-tile)
        // is 3/4 of s_sh_stage int4, stages in K order
        s_gl_rd = (3 * s_sh_stride * tile_column_stride / 4) * slice_col +
                  (3 * s_sh_stage / 4) * slice_row + threadIdx.x;
  #ifdef MUSE_GATE_UP_ILV
        if constexpr (gate_up_grid) {
          s_gl_rd = (3 * s_sh_stage / 4) * ilv_base + threadIdx.x;
          s_gl_rd_delta *= 48;
        }
  #endif
      } else {
        s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / s_sh_stride) +
                  s_sh_stride * tile_column_stride * slice_col + threadIdx.x % s_sh_stride;
      }
    }
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred =
      threadIdx.x < (scale6 ? (s_sh_stage * 3) / 4 : s_sh_stage);

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (is_a_8bit) {
    s_sh_rd = 4 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 4);
  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = (is_a_8bit ? 4 : 8) * ((threadIdx.x / 32) % tb_n_warps) +
                 (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd =
            8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
      }
    } else if (is_a_8bit) {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps / 2) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_g_idx = sh_new + sh_b_red_bias_size;
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
#ifdef MUSE_SCALE_TMA4
  constexpr int sh_s_stages = 4;
#else
  constexpr int sh_s_stages = stages;
#endif
  constexpr int sh_s_size = has_act_order ? (act_s_max_num_groups * s_sh_stride)
                                          : (sh_s_stages * s_sh_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;
  // scale6: the 64-byte palette (or 128-byte BF16 palette when fast_pal16) lives after the A stages.
  [[maybe_unused]] uint8_t* sh_pal =
      reinterpret_cast<uint8_t*>(sh_a + (stages - 1) * a_sh_stage_step + a_sh_stage);
  [[maybe_unused]] uint16_t* sh_pal16 = reinterpret_cast<uint16_t*>(sh_pal);
  if constexpr (scale6) {
    static_assert(group_blocks == 1 && is_8bit_scale && !has_act_order &&
                      !has_zp && !is_a_8bit && thread_k_blocks == 8,
                  "scale6 is only built for the M = 1 NVFP4 decode config");
    if constexpr (fast_pal16) {
      if (threadIdx.x < 64) {
        const uint8_t b = reinterpret_cast<const uint8_t*>(scale6_palette)[threadIdx.x];
        sh_pal16[threadIdx.x] = static_cast<uint16_t>(((b & 0x80u) << 7) | ((b & 0x7Fu) << 4));
      }
    } else {
      // visible to all threads at the __syncthreads of the first wait_for_stage
      if (threadIdx.x < 4)
        reinterpret_cast<int4*>(sh_pal)[threadIdx.x] = scale6_palette[threadIdx.x];
    }
  }
  constexpr bool fast_m1_red =
      !multi_m && scale6 && m_block_size_8 && (tma_ovlp || fast_pal16 || stages == 3 || half_split || fixed_shape == 3);
#ifdef MUSE_SCALE_TMA4
  static_assert(scale6 && (threads == 256 || threads == 128),
                "TMA4 rANS scales require M = 1 decode config (128 or 256 threads)");
  constexpr int kTmaWarps = threads / 32;
  constexpr int RANS_CAP = (fixed_shape == 3) ? 4096 : ((threads == 256) ? 3200 : 2048);
  uint32_t* sh_states = reinterpret_cast<uint32_t*>(sh_pal + (fast_pal16 ? 128 : 64));
  unsigned char* sh_rslot = reinterpret_cast<unsigned char*>(sh_states + threads);
  uint32_t* sh_rlut = reinterpret_cast<uint32_t*>(sh_rslot + 2 * RANS_CAP);
  uint64_t* sh_rbar = reinterpret_cast<uint64_t*>(sh_rlut + 1024);
  sh_red = reinterpret_cast<int4*>(sh_rbar + 4);
  const uint32_t* tma_u32 = reinterpret_cast<const uint32_t*>(g_idx);
  const uint32_t tma_tot_sbs = tma_u32[1024];
  const uint32_t tma_max_sbs = tma_u32[1026];
  const uint32_t* tma_blk_it = tma_u32 + 1028;
  const uint32_t* tma_sb_row = tma_blk_it + gridDim.x;
  const size_t tma_meta_u32 = ((static_cast<size_t>(4 + gridDim.x + tma_max_sbs) + 3) / 4) * 4;
  const uint32_t* tma_off16 = tma_u32 + 1024 + tma_meta_u32;
  const int tma_iters = static_cast<int>(tma_blk_it[blockIdx.x]);
  const int tma_num_sbs = (tma_iters + 3) >> 2;
  const unsigned char* tma_var_g = reinterpret_cast<const unsigned char*>(zp_ptr);
  int tma_slice_base = 0, tma_done = 0, tma_q = 0;
  bool tma_started = false;
  uint32_t tma_x = 0;
  const uint16_t* tma_wp = nullptr;
  auto tma_bar = [&](int i) { return static_cast<uint32_t>(__cvta_generic_to_shared(sh_rbar + i)); };
  auto tma_bulk = [&](void* dst, const void* src, int bytes, int bar) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(tma_bar(bar)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "l"(src), "r"(bytes), "r"(tma_bar(bar))
                 : "memory");
  };
  auto tma_wait = [&](int bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\tTMA4_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@!P1 bra TMA4_WAIT_%=;\n\t}" ::"r"(tma_bar(bar)), "r"(parity) : "memory");
  };
  auto tma_issue = [&](int sb) {
    uint32_t gsb;
    if constexpr (gate_up_grid) {
      gsb = static_cast<uint32_t>(((blockIdx.x / 48) * 13 + sb) * 48 + (blockIdx.x % 48));
    } else {
      gsb = __ldg(tma_sb_row + sb) + blockIdx.x;
    }
    const uint32_t o0 = __ldg(tma_off16 + gsb);
    const uint32_t o1 = __ldg(tma_off16 + gsb + 1);
    const int bytes = static_cast<int>(o1 - o0) * 16;
    const int slot = sb & 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    tma_bulk(sh_rslot + slot * RANS_CAP, tma_var_g + static_cast<size_t>(o0) * 16, bytes, 1 + slot);
  };
  if (threadIdx.x == 0) {
  #pragma unroll
    for (int i = 0; i < 3; i++)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tma_bar(i)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(tma_bar(0)),
                 "r"(4096 + threads * 4) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(sh_rlut))),
                 "l"(g_idx), "r"(4096), "r"(tma_bar(0)) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(sh_states))),
                 "l"(reinterpret_cast<const unsigned char*>(scales_ptr) +
                     static_cast<size_t>(blockIdx.x) * threads * 4),
                 "r"(threads * 4), "r"(tma_bar(0)) : "memory");
    if (tma_num_sbs > 0) tma_issue(0);
    if (tma_num_sbs > 1) tma_issue(1);
  }
  auto tma_decode = [&](int X) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if ((X & 3) == 0) {
      const int sb = X >> 2;
      const int slot = sb & 1;
      if (sb > 0 && threadIdx.x == 0 && sb + 1 < tma_num_sbs) tma_issue(sb + 1);
      tma_wait(1 + slot, static_cast<uint32_t>((sb >> 1) & 1));
      const unsigned char* blk = sh_rslot + slot * RANS_CAP;
      const uint16_t* cw = reinterpret_cast<const uint16_t*>(blk);
      int start = kTmaWarps;
  #pragma unroll
      for (int i = 0; i < kTmaWarps - 1; i++) start += i < w ? cw[i] : 0;
      tma_wp = reinterpret_cast<const uint16_t*>(blk) + start;
    }
    const uint32_t lt = (1u << lane) - 1u;
    if constexpr (fast_pal16) {
      const int pl = (lane & ~3) | ((lane & 1) << 1) | ((lane >> 1) & 1);
      uint16_t* dst16 = reinterpret_cast<uint16_t*>(sh_s + s_sh_stage * (X & 3)) + 128 * w + pl;
  #pragma unroll
      for (int m = 0; m < 4; m++) {
        const uint32_t e = sh_rlut[tma_x & 1023u];
        tma_x = ((e >> 6) & 2047u) * (tma_x >> 10) + (e >> 17);
        dst16[32 * m] = sh_pal16[e & 63u];
        const bool need = tma_x < 65536u;
        const uint32_t msk = __ballot_sync(0xffffffffu, need);
        if (need) tma_x = (tma_x << 16) | tma_wp[__popc(msk & lt)];
        tma_wp += __popc(msk);
      }
    } else {
      uint8_t* dst = reinterpret_cast<uint8_t*>(sh_s + s_sh_stage * (X & 3)) + 128 * w + lane;
  #pragma unroll
      for (int m = 0; m < 4; m++) {
        const uint32_t e = sh_rlut[tma_x & 1023u];
        tma_x = ((e >> 6) & 2047u) * (tma_x >> 10) + (e >> 17);
        dst[32 * m] = sh_pal[e & 63u];
        const bool need = tma_x < 65536u;
        const uint32_t msk = __ballot_sync(0xffffffffu, need);
        if (need) tma_x = (tma_x << 16) | tma_wp[__popc(msk & lt)];
        tma_wp += __popc(msk);
      }
    }
  };
#endif
#ifdef MUSE_SCALE_RANS
  // rANS-coded block scales (fixed gate/up grid only; muse_marlin6_gemv.encode_gate_up_scales, gemv6_srans.cu).
  // Global operands, carried in pointer arguments this configuration does not otherwise use:
  //   scales_ptr: per-CTA headers, RANS_HDR bytes each: u32 states[256] (warp-major), u32 off16[64]
  //               (block X >= 2 starts at var + 16 off16[X]; off16[iters] ends the last), block 0, block 1
  //   zp_ptr:     the variable region (blocks X >= 2 of every CTA)
  //   g_idx:      the layer's decode table, u32[1024]: symbol | freq << 6 | (slot - cum) << 17
  // A block (one k-stage of the CTA's slice): u8 count[8] (16-bit words per warp), then the words of warp 0..7.
  // Warp w decodes bytes 128 w + lane + 32 m (m = 0..3) of the stage's 1 KB raw scale block into the 8-bit
  // scale slot of that stage; consumers then read raw scale bytes exactly as the uncompressed kernel does.
#ifdef MUSE_SCALE_RANS_SK
  // stream-K variant (one-row down projection, 48 CTAs): a CTA's stages run across up to two column slices; its
  // rANS lanes, block stream and scale-slot ring continue across the slice boundary (global stage index X), the
  // header holds 192 block offsets, and the consumers address the slot of global stage X.
  static_assert(!gate_up_grid && scale6 && threads == 256 && thread_n_blocks == 8,
                "stream-K rANS scales are built for the one-row 128-column decode configuration");
  constexpr int RANS_HDR = 3840, RANS_CAP = 1024, RANS_BLK0 = 1792;
#else
  static_assert(gate_up_grid && scale6, "rANS scales are built for the fixed gate/up grid only");
  constexpr int RANS_HDR = 3328, RANS_CAP = 1024;
#endif
  unsigned char* sh_rhdr = sh_pal + 64;
  unsigned char* sh_rslot = sh_rhdr + RANS_HDR;
  uint32_t* sh_rlut = reinterpret_cast<uint32_t*>(sh_rslot + stages * RANS_CAP);
  uint64_t* sh_rbar = reinterpret_cast<uint64_t*>(sh_rlut + 1024);
  const unsigned char* rans_var_g = reinterpret_cast<const unsigned char*>(zp_ptr);
#ifdef MUSE_SCALE_RANS_SK
  const int rans_iters = min(iters, k_tiles * part2_mn_tiles - static_cast<int>(blockIdx.x) * iters);
  int rans_slice_base = 0, rans_done = 0;
  bool rans_started = false;
#else
  const int rans_iters = slice_iters;
#endif
  uint32_t rans_x = 0;
  int rans_q = 0;
  auto rans_bar = [&](int i) { return static_cast<uint32_t>(__cvta_generic_to_shared(sh_rbar + i)); };
  auto rans_bulk = [&](void* dst, const void* src, int bytes, int bar) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(rans_bar(bar)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "l"(src), "r"(bytes), "r"(rans_bar(bar))
                 : "memory");
  };
  auto rans_wait = [&](int bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\tRANS_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@!P1 bra RANS_WAIT_%=;\n\t}" ::"r"(rans_bar(bar)), "r"(parity) : "memory");
  };
  // block X >= 2 into slot X % stages; the slot's previous block was decoded two barriers earlier
  auto rans_issue = [&](int X) {
    const uint32_t* off = reinterpret_cast<const uint32_t*>(sh_rhdr + 1024);
    const int bytes = static_cast<int>(off[X + 1] - off[X]) * 16;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    rans_bulk(sh_rslot + (X % stages) * RANS_CAP, rans_var_g + static_cast<size_t>(off[X]) * 16, bytes,
              1 + X % stages);
  };
  if (threadIdx.x == 0) {
  #pragma unroll
    for (int i = 0; i <= stages; i++)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(rans_bar(i)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(rans_bar(0)), "r"(RANS_HDR + 4096)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(sh_rhdr))),
                 "l"(reinterpret_cast<const unsigned char*>(scales_ptr) + static_cast<size_t>(blockIdx.x) * RANS_HDR),
                 "r"(RANS_HDR), "r"(rans_bar(0))
                 : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(sh_rlut))),
                 "l"(g_idx), "r"(4096), "r"(rans_bar(0))
                 : "memory");
  }
  auto rans_decode = [&](int X) {
#ifdef MUSE_SCALE_RANS_SK
    const unsigned char* blk = X < 2 ? sh_rhdr + RANS_BLK0 + X * RANS_CAP : sh_rslot + (X % stages) * RANS_CAP;
#else
    const unsigned char* blk = X < 2 ? sh_rhdr + 1280 + X * RANS_CAP : sh_rslot + (X % stages) * RANS_CAP;
#endif
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    int start = 4;
  #pragma unroll
    for (int i = 0; i < 7; i++) start += i < w ? blk[i] : 0;
    const uint16_t* words = reinterpret_cast<const uint16_t*>(blk) + start;
    uint8_t* dst = reinterpret_cast<uint8_t*>(sh_s + s_sh_stage * (X % stages)) + 128 * w + lane;
    const uint32_t lt = (1u << lane) - 1u;
    int pos = 0;
  #pragma unroll
    for (int m = 0; m < 4; m++) {
      const uint32_t e = sh_rlut[rans_x & 1023u];
      rans_x = ((e >> 6) & 2047u) * (rans_x >> 10) + (e >> 17);
      dst[32 * m] = sh_pal[e & 63u];
      const bool need = rans_x < 65536u;
      const uint32_t msk = __ballot_sync(0xffffffffu, need);
      if (need) rans_x = (rans_x << 16) | words[pos + __popc(msk & lt)];
      pos += __popc(msk);
    }
  };
#endif

  [[maybe_unused]] int4* sh_a_gated = nullptr;
  if constexpr (fuse_mul_sigmoid && !multi_m) {
#ifdef MUSE_SCALE_TMA4
    constexpr int sh_red_tma_max = fast_m1_red ? ((threads == 256) ? 128 : 64)
                                               : ((threads == 256) ? 512 : 256);
    sh_a_gated = sh_red + sh_red_tma_max;
#else
    sh_a_gated = reinterpret_cast<int4*>((reinterpret_cast<uintptr_t>(sh_pal + 64) + 15u) & ~uintptr_t(15u)) + 1024;
#endif
    const uint4* g_ptr = reinterpret_cast<const uint4*>(A0);
    const uint4* a_ptr = reinterpret_cast<const uint4*>(A_aux);
    uint4* dst_ptr = reinterpret_cast<uint4*>(sh_a_gated);
  #pragma unroll
    for (int idx = threadIdx.x; idx < 512; idx += threads) {
      const uint4 g = g_ptr[idx];
      const uint4 av = a_ptr[idx];
      const uint32_t gw[4] = {g.x, g.y, g.z, g.w};
      const uint32_t aw[4] = {av.x, av.y, av.z, av.w};
      uint32_t ow[4];
  #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float gl = __uint_as_float(gw[j] << 16);
        const float gh = __uint_as_float(gw[j] & 0xffff0000u);
        const float al = __uint_as_float(aw[j] << 16);
        const float ah = __uint_as_float(aw[j] & 0xffff0000u);
        __nv_bfloat162 p = __floats2bfloat162_rn(
            __fmul_rn(marlin_triton_sigmoid(gl), al),
            __fmul_rn(marlin_triton_sigmoid(gh), ah));
        ow[j] = *reinterpret_cast<uint32_t*>(&p);
      }
      dst_ptr[idx] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    __syncthreads();
  }

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragS frag_s[2][4];  // No act-order
  FragS frag_bias[2][4];
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

  if constexpr (is_a_8bit) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][0][g] = 0.0f;
        }

  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][1][g] = 0.0f;
        }
      }
    }
  }

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;

  auto fetch_act_order_scales_to_shared = [&](bool is_async, int first_group_id,
                                              int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups > act_s_max_num_groups) {
      sh_num_groups = act_s_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(&sh_s[(i * s_sh_stride) + threadIdx.x],
                         &scales_ptr[row_offset + (i * s_gl_stride) +
                                     slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset +
                         threadIdx.x];
        }
      }
    }
  };
#ifdef MUSE_B_EVICT_FIRST
  // The weight and scale-code streams are read once: insert them in L2 as evict-first with 256B sector fetch
  // so they do not displace L2-resident activations or l2pf prefetched weight heads. Policy is materialized
  // locally per copy to avoid ptxas odd-UR descriptor allocation on sm_121a.
  auto cp_async4_stream = [&](void* smem_ptr, const void* glob_ptr) {
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if constexpr (fuse_lmpost && down_ilv && (fixed_shape != 3 || multi_m)) {
      asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(smem), "l"(glob_ptr));
    } else {
      uint64_t stream_policy;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(stream_policy));
      asm volatile("cp.async.cg.shared.global.L2::256B.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(glob_ptr),
                   "l"(stream_policy));
    }
  };
#else
  auto cp_async4_stream = [&](void* smem_ptr, const void* glob_ptr) { cp_async4(smem_ptr, glob_ptr); };
#endif
  auto load_mul_sigmoid_int4 = [&](int idx) -> int4 {
    const uint4 g = reinterpret_cast<const uint4*>(A)[idx];
    const uint4 av = reinterpret_cast<const uint4*>(A_aux)[idx];
    const uint32_t gw[4] = {g.x, g.y, g.z, g.w};
    const uint32_t aw[4] = {av.x, av.y, av.z, av.w};
    uint32_t ow[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float gl = __uint_as_float(gw[j] << 16);
      const float gh = __uint_as_float(gw[j] & 0xffff0000u);
      const float al = __uint_as_float(aw[j] << 16);
      const float ah = __uint_as_float(aw[j] & 0xffff0000u);
      __nv_bfloat162 p = __floats2bfloat162_rn(
          __fmul_rn(marlin_triton_sigmoid(gl), al),
          __fmul_rn(marlin_triton_sigmoid(gh), ah));
      ow[j] = *reinterpret_cast<uint32_t*>(&p);
    }
    return make_int4(static_cast<int>(ow[0]), static_cast<int>(ow[1]),
                     static_cast<int>(ow[2]), static_cast<int>(ow[3]));
  };
  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage_step * pipe;
      const int cur_a_off = down_ilv ? (a_gl_rd_base + a_gl_rd_delta_o * ilv_a_kt)
                                     : (a_gl_rd + a_gl_rd_delta_o * a_off);
      if constexpr (fuse_mul_sigmoid) {
        if constexpr (multi_m) {
  #pragma unroll
          for (int i = 0; i < a_sh_wr_iters; i++) {
            if (a_sh_wr_pred[i]) {
              sh_a_stage[a_sh_wr_trans[i]] =
                  load_mul_sigmoid_int4(a_gl_rd_delta_i * i + cur_a_off);
            }
          }
        } else {
  #pragma unroll
          for (int i = 0; i < a_sh_wr_iters; i++) {
            if (a_sh_wr_pred[i]) {
              sh_a_stage[a_sh_wr_trans[i]] = sh_a_gated[(a_gl_rd_delta_i * i + cur_a_off) & 511];
            }
          }
        }
      } else {
  #pragma unroll
        for (int i = 0; i < a_sh_wr_iters; i++) {
          cp_async4_pred(
              &sh_a_stage[a_sh_wr_trans[i]],
              &A[a_gl_rd_delta_i * i + cur_a_off],
              a_sh_wr_pred[i]);
        }
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

        cp_async4_stream(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }

      if constexpr (down_ilv) {
        if constexpr (gate_v48) {
          b_gl_rd += (ilv_f < ilv_step_switch) ? ilv_delta_hi : ilv_delta_lo;
        } else if constexpr (split2) {
          b_gl_rd += (ilv_f < ilv_step_p1end)
                         ? ilv_delta_p1
                         : ((ilv_f < ilv_step_switch)
                                ? ilv_delta_hi
                                : ((ilv_f < ilv_step_p2end) ? ilv_delta_lo : ilv_delta_tail));
        } else {
          b_gl_rd += (ilv_f < ilv_step_p1end)
                         ? ilv_delta_p1
                         : ((ilv_f < ilv_step_switch) ? ilv_delta_hi : ilv_delta_lo);
        }
        ilv_f++;
        if (ilv_f == ilv_jump_f) {
          ilv_a_kt = ilv_jump_kt;
        } else if (++ilv_a_kt == k_tiles) {
          ilv_a_kt = 0;
        }
      } else {
        b_gl_rd += b_gl_rd_delta_o;
      }

      if constexpr (has_act_order) {
        // Fetch g_idx thread-block portion
        int full_pipe = a_off;
        int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
        if (cur_k < prob_k && cur_k < slice_k_finish) {
          int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

          int4 const* cur_g_idx_stage_ptr =
              reinterpret_cast<int4 const*>(&g_idx[cur_k]);

          if (threadIdx.x < g_idx_stage) {
            cp_async4_pred(&sh_g_idx_stage[threadIdx.x],
                           &cur_g_idx_stage_ptr[threadIdx.x]);
          }
        }
      } else {
        if constexpr (group_blocks != -1) {
          // Only fetch scales if this tile starts a new group
          if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
#if !defined(MUSE_SCALE_RANS) && !defined(MUSE_SCALE_TMA4)
            int4* sh_s_stage = sh_s + s_sh_stage * pipe;
            if (s_sh_wr_pred) {
              cp_async4_stream(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
            }
#endif
            s_gl_rd += scale6 ? (s_gl_rd_delta * s_tb_groups * 3) / 4
                              : s_gl_rd_delta * s_tb_groups;
          }
        }

        if constexpr (has_zp && group_blocks != -1) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          // Only fetch zero points if this tile starts a new group
          if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
            if (zp_sh_wr_pred) {
              cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
            }
            zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };
#ifdef MUSE_SCALE_TMA4
  // SM121 PDL consumer pre-wait fetch: issue B weight cp.async and advance ILV pointers without touching A.
  auto fetch_b_only_to_shared = [&](int pipe, bool pred = true) {
    if (pred) {
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);
        cp_async4_stream(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }
      if constexpr (down_ilv) {
        if constexpr (gate_v48) {
          b_gl_rd += (ilv_f < ilv_step_switch) ? ilv_delta_hi : ilv_delta_lo;
        } else if constexpr (split2) {
          b_gl_rd += (ilv_f < ilv_step_p1end)
                         ? ilv_delta_p1
                         : ((ilv_f < ilv_step_switch)
                                ? ilv_delta_hi
                                : ((ilv_f < ilv_step_p2end) ? ilv_delta_lo : ilv_delta_tail));
        } else {
          b_gl_rd += (ilv_f < ilv_step_p1end)
                         ? ilv_delta_p1
                         : ((ilv_f < ilv_step_switch) ? ilv_delta_hi : ilv_delta_lo);
        }
        ilv_f++;
        if (ilv_f == ilv_jump_f) {
          ilv_a_kt = ilv_jump_kt;
        } else if (++ilv_a_kt == k_tiles) {
          ilv_a_kt = 0;
        }
      } else {
        b_gl_rd += b_gl_rd_delta_o;
      }
    }
    cp_async_fence();
  };
#endif

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage_step * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    using IT1 = typename std::conditional_t<is_a_8bit, int2, int4>;
    using IT0 = typename std::conditional_t<is_a_8bit, int, int2>;
    constexpr int group_blocks2 = div_ceil(group_blocks, is_a_8bit ? 2 : 1);

    if constexpr (!has_act_order) {
      // No act-order case
      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      } else if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0) {
            if (k % b_sh_wr_iters == 0) {
              int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
              reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
            } else {
              reinterpret_cast<int4*>(&frag_s[1])[0] =
                  reinterpret_cast<int4*>(&frag_s[0])[0];
            }
          }
        } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;
          int warp_row = warp_id / tb_n_warps;

          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks2;

#ifdef MUSE_SCALE_RANS_SK
          int4* sh_s_stage = sh_s + s_sh_stage * ((rans_slice_base + pipe) & (stages - 1));
#elif defined(MUSE_SCALE_TMA4)
          int4* sh_s_stage = sh_s + s_sh_stage * (tma_q & 3);
#else
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
#endif

          if constexpr (!is_8bit_scale) {
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
                sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
          } else if constexpr (scale6) {
#ifdef MUSE_SCALE_TMA4
            if constexpr (fast_pal16) {
              reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
                  sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
            } else {
              reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(
                  sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
            }
#elif defined(MUSE_SCALE_RANS)
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(
                sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
#else
            // raw byte b of the stage has its code at bits [6b, 6b + 6) of the
            // compressed stage (LSB first); these 8 bytes start at b = 8m, so
            // their 48 code bits sit in 32-bit words floor(1.5m), floor(1.5m)+1
            const int m = s_sh_rd + cur_group_id * (2 * s_sh_stride);
            const uint32_t* cw = reinterpret_cast<const uint32_t*>(sh_s_stage);
            const int wi = (3 * m) >> 1;
            uint64_t v = static_cast<uint64_t>(cw[wi]) |
                         (static_cast<uint64_t>(cw[wi + 1]) << 32);
            v >>= (m & 1) * 16;
            uint32_t lo = 0, hi = 0;
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              lo |= static_cast<uint32_t>(sh_pal[(v >> (6 * j)) & 63]) << (8 * j);
              hi |= static_cast<uint32_t>(sh_pal[(v >> (6 * (j + 4))) & 63])
                    << (8 * j);
            }
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                make_int2(static_cast<int>(lo), static_cast<int>(hi));
#endif
          } else {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(
                    sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
          }
        } else if (group_blocks >= b_sh_wr_iters) {
          if constexpr (!is_8bit_scale) {
            reinterpret_cast<int4*>(&frag_s[1])[0] =
                reinterpret_cast<int4*>(&frag_s[0])[0];
          } else {
            reinterpret_cast<int2*>(&frag_s[1])[0] =
                reinterpret_cast<int2*>(&frag_s[0])[0];
          }
        }
      }

      return;
    }

    // Act-order case

    // Determine K of the "current" thread-block
    int cur_k = slice_k_start + tb_k * full_pipe;
    if (cur_k >= prob_k || cur_k >= slice_k_finish) {
      return;
    }

    // Reset (to current thread-block) since we read g_idx portion from the
    // shared memory
    cur_k = 0;

    // Progress to current iteration
    cur_k += k % b_sh_wr_iters;

    // Determine "position" inside the thread-block (based on warp and
    // thread-id)
    auto warp_id = threadIdx.x / 32;
    int warp_row = warp_id / tb_n_warps;
    int warp_col = warp_id % tb_n_warps;

    cur_k += warp_row * 16 * b_sh_wr_iters;

    auto th_id = threadIdx.x % 32;
    cur_k += (th_id % 4) * 2;  // Due to tensor-core layout for fp16 B matrix

    int s_col_shift =
        /*slice_n_offset +*/ (act_s_col_warp_stride * warp_col) +
        (th_id / 4) * act_s_col_stride;

    if (is_same_group[pipe]) {
      if (k % 2 == 0) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride +
                 s_col_shift];
      } else {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
      }

      for (int i = 1; i < 4; i++) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
      }
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    constexpr int k_frag_offsets[4] = {0, 1, 8,
                                       9};  // Tensor core offsets per thread

  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int actual_k = cur_k + k_frag_offsets[i];

      int group_id = sh_g_idx_int_ptr[actual_k];
      int rel_group_id = group_id - sh_first_group_id;

      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
          sh_s[rel_group_id * s_sh_stride + s_col_shift];
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    // This code does not handle group_blocks == 0,
    // which signifies act_order.
    // has_zp implies AWQ, which doesn't have act_order,
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0 || is_a_8bit) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }
      } else if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0 || is_a_8bit) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / div_ceil(group_blocks, is_a_8bit ? 2 : 1);

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;

          int warp_row = warp_id / tb_n_warps;
          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits()) {
      if constexpr (is_a_8bit && has_zp) {
        sub_zp_and_dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(
            q, frag_b_ptr, zp);
      } else {
        dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    if (is_a_8bit) return;
    int k2 = k % 2;
    constexpr int g =
        group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
    const bool is_new_zp =
        (group_blocks == 0) ||
        ((group_blocks > 0) && (group_blocks < b_sh_wr_iters || k == 0)) &&
            (pipe % g == 0) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (b_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(b_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
        dequant_data(zp_quant_1,
                     reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (!fast_pal16 && (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu)) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (b_type_id == vllm::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (b_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(b_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float && !is_a_8bit) {
        sub_zp<a_type_id>(frag_b0, frag_zp[j], 0);
        sub_zp<a_type_id>(frag_b1, frag_zp[j], 1);
      }

      // Apply scale to frag_b0
      if constexpr (has_act_order && !is_a_8bit) {
        static_assert(group_blocks != -1);
        scale4<a_type_id>(frag_b0, act_frag_s[k2][0][j], act_frag_s[k2][1][j],
                          act_frag_s[k2][2][j], act_frag_s[k2][3][j], 0);
        scale4<a_type_id>(frag_b1, act_frag_s[k2][0][j], act_frag_s[k2][1][j],
                          act_frag_s[k2][2][j], act_frag_s[k2][3][j], 1);
      } else if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                           group_blocks == -1 && !is_a_8bit) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Adtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<a_type_id>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1 &&
                           !is_a_8bit) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<a_type_id>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1 && !is_a_8bit) {
        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0, frag_b1,
                                               frag_c[i][j][0]);
        } else {
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0,
                                         frag_c[i][j][0]);
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b1,
                                         frag_c[i][j][1]);
        }
      }
    }
  };

  auto matmul_a8 = [&](int k) {
    int k2 = k % 2;
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      FragB frag_b[2];

      if (is_a_8bit && b_type.size_bits() == 4 && !has_zp) {
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b));
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2);
      } else if (is_a_8bit && b_type.size_bits() == 4 && has_zp) {
        int off = (threadIdx.x / 32) % 2 * 2 + j;
        int zp = (frag_qzp[k2][0] >> (off * 8)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
        zp = (frag_qzp[k2][0] >> (off * 8 + 4)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2, zp);
      } else {
        reinterpret_cast<int2*>(&frag_b)[0] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[0];
        reinterpret_cast<int2*>(&frag_b)[1] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[1];
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[0],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][0]);
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[1],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][1]);
      }

      if constexpr (group_blocks != -1) {
        if (group_blocks == 2 || k == 1) {
          if constexpr (a_type == vllm::kS8) {
            int2 s_vals[2];
            s_vals[0] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[1]};
            s_vals[1] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[1]};

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[0])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) *
                    scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[1])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) *
                    scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          } else {
            float2 s_vals[2];
            if constexpr (s_type_id != vllm::kFE8M0fnu.id()) {
              static_assert(a_type.size_bits() == 16 ||
                            s_type.size_bits() == 16);
              s_vals[0] = Cdtype::num22float2(frag_s[k2][j * 2][0]);
              s_vals[1] = Cdtype::num22float2(frag_s[k2][j * 2 + 1][0]);
            } else {
              int32_t* s_vals_int = reinterpret_cast<int32_t*>(&s_vals[0]);
              int32_t s_vals_e8m0 =
                  *reinterpret_cast<int32_t*>(&frag_s[k2][j][0]);

              s_vals_int[0] = (s_vals_e8m0 & 0xFF) << 23;
              s_vals_int[1] = (s_vals_e8m0 & 0xFF00) << 15;
              s_vals_int[2] = (s_vals_e8m0 & 0xFF0000) << 7;
              s_vals_int[3] = (s_vals_e8m0 & 0xFF000000) >> 1;
            }

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[0])[g % 2];
                frag_c[i][j][0][g] += frag_c_tmp[i][j][0][g] * scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[1])[g % 2];
                frag_c[i][j][1][g] += frag_c_tmp[i][j][1][g] * scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          }
        }
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    if constexpr (fast_m1_red) {
      constexpr int red_off = threads / b_sh_stride_threads / 2;
      if constexpr (red_off >= 1) {
        const int red_idx = threadIdx.x / b_sh_stride_threads;
        constexpr int lane_grp = b_sh_stride_threads >> 2;
        const int tid_grp = (threadIdx.x % b_sh_stride_threads) >> 2;
        const bool is_row0 = (threadIdx.x & 3) == 0;
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (is_row0 && i <= red_idx && red_idx < 2 * i) {
            const int wr_slot = (red_idx - i) * 2 * lane_grp + tid_grp;
            if (i < red_off) {
              const int rd_slot = red_idx * 2 * lane_grp + tid_grp;
              const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[rd_slot];
              const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[rd_slot + lane_grp];
              const float4 wr0 = reinterpret_cast<const float4*>(sh_red)[wr_slot];
              const float4 wr1 = reinterpret_cast<const float4*>(sh_red)[wr_slot + lane_grp];
              frag_c[0][0][0][0] += rd0.x + wr0.x;
              frag_c[0][0][0][2] += rd0.y + wr0.y;
              frag_c[0][1][0][0] += rd0.z + wr0.z;
              frag_c[0][1][0][2] += rd0.w + wr0.w;
              frag_c[0][2][0][0] += rd1.x + wr1.x;
              frag_c[0][2][0][2] += rd1.y + wr1.y;
              frag_c[0][3][0][0] += rd1.z + wr1.z;
              frag_c[0][3][0][2] += rd1.w + wr1.w;
            }
            reinterpret_cast<float4*>(sh_red)[wr_slot] = make_float4(
                frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
            reinterpret_cast<float4*>(sh_red)[wr_slot + lane_grp] = make_float4(
                frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
          }
          __syncthreads();
        }
        if (is_row0 && red_idx == 0) {
          const int rd_slot = tid_grp;
          const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[rd_slot];
          const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[rd_slot + lane_grp];
          frag_c[0][0][0][0] += rd0.x;
          frag_c[0][0][0][2] += rd0.y;
          frag_c[0][1][0][0] += rd0.z;
          frag_c[0][1][0][2] += rd0.w;
          frag_c[0][2][0][0] += rd1.x;
          frag_c[0][2][0][2] += rd1.y;
          frag_c[0][3][0][0] += rd1.z;
          frag_c[0][3][0][2] += rd1.w;
        }
        __syncthreads();
      }
      return;
    }
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_j_shift = m_block_size_8 ? 1 : 0;
      constexpr int red_sh_stride =
          b_sh_stride_threads * ((is_a_8bit ? 2 : 4) * 2 >> red_j_shift);
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < (is_a_8bit ? 2 : 4) * 2;
                 j += (m_block_size_8 ? 2 : 1)) {
              const int jj = j >> red_j_shift;
              int red_sh_wr =
                  red_sh_delta * jj + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(
                    &sh_red[red_sh_delta * jj + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(
                      frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < (is_a_8bit ? 2 : 4) * 2;
               i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd =
                reinterpret_cast<float*>(&sh_red[red_sh_delta * (i >> red_j_shift) + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(
                  frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * tb_n_warps;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride * (is_a_8bit ? 2 : 1);
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr;
      if constexpr (m_block_size_8) {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) +
                  4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      } else {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) * (is_a_8bit ? 2 : 1) +
                  4 * (threadIdx.x / 32) + threadIdx.x % 4;
        c_gl_wr += (2 * thread_n_blocks) * slice_col * (is_a_8bit ? 2 : 1);
      }
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
  // Interestingly, doing direct global accesses here really seems to mess up
  // the compiler and lead to slowdowns, hence we also use async-copies even
  // though these fetches are not actually asynchronous.
  #pragma unroll
        for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
          if constexpr (m_block_size_8) {
            cp_async4_pred(&sh_red[c_sh_wr + c_sh_wr_delta * i],
                           &C[c_gl_wr + i * c_gl_stride +
                              (threadIdx.x % 8) / 4 * c_gl_wr_delta_i],
                           (threadIdx.x % 4) * 2 + i < prob_m);
          } else if constexpr (is_a_8bit) {
            int2* sh_red_int2 = reinterpret_cast<int2*>(sh_red);
            int2* c_int2 = reinterpret_cast<int2*>(C);
            cp_async2_ca_pred(
                &sh_red_int2[c_sh_wr + c_sh_wr_delta * i],
                &c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                        c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          } else {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                   c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          }
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        bool mask = (!m_block_size_8) && (i < (thread_m_blocks - 1) * 4 ||
                                          8 * (i / 2) + row < prob_m) ||
                    (m_block_size_8) && ((threadIdx.x % 4) * 2 + i < prob_m);
        if (mask) {
          if (!first) {
            c_scalar_t* c_red_f16;
            if constexpr (is_a_8bit) {
              int2 tmp =
                  reinterpret_cast<int2*>(sh_red)[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            } else {
              int4 tmp = sh_red[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            }
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta] += Cdtype::num2float(c_red_f16[j]);
            }
          }
          if (!last) {
            c_scalar_t c_f16[is_a_8bit ? 4 : 8];
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              c_f16[j] = Cdtype::float2num(reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta]);
            }
            if constexpr (m_block_size_8) {
              C[c_gl_wr + i * c_gl_stride +
                (threadIdx.x % 8) / 4 * c_gl_wr_delta_i] =
                  *reinterpret_cast<int4*>(c_f16);
            } else if constexpr (is_a_8bit) {
              int2* c_int2 = reinterpret_cast<int2*>(C);
              c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                     c_gl_wr_delta_i * (i % 2)] =
                  *reinterpret_cast<int2*>(c_f16);
            } else {
              C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                c_gl_wr_delta_i * (i % 2)] = *reinterpret_cast<int4*>(c_f16);
            }
          }
        }
      }
    }
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    const int eff_lock = half_split ? (2 * locks_off + half_id) : locks_off;
    if constexpr (fast_m1_red) {
      constexpr int active_threads = 32 * tb_n_warps;
      constexpr int red_m1_c_size = (active_threads >> 2) * 2;
      if (threadIdx.x < active_threads && (threadIdx.x & 3) == 0) {
        const int red_tid = threadIdx.x >> 2;
        const int c_cur_offset = eff_lock * red_m1_c_size;
        if (!first) {
          const float4 rd0 = reinterpret_cast<const float4*>(C_tmp)[c_cur_offset + red_tid * 2 + 0];
          const float4 rd1 = reinterpret_cast<const float4*>(C_tmp)[c_cur_offset + red_tid * 2 + 1];
          frag_c[0][0][0][0] += rd0.x;
          frag_c[0][0][0][2] += rd0.y;
          frag_c[0][1][0][0] += rd0.z;
          frag_c[0][1][0][2] += rd0.w;
          frag_c[0][2][0][0] += rd1.x;
          frag_c[0][2][0][2] += rd1.y;
          frag_c[0][3][0][0] += rd1.z;
          frag_c[0][3][0][2] += rd1.w;
        }
        if (!last) {
          reinterpret_cast<float4*>(C_tmp)[c_cur_offset + red_tid * 2 + 0] = make_float4(
              frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
          reinterpret_cast<float4*>(C_tmp)[c_cur_offset + red_tid * 2 + 1] = make_float4(
              frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
        }
      }
      return;
    }
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * tb_n_warps;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = eff_lock * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&](bool last) {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = ((fixed_shape == 3 && !multi_m) ? (202048 / 8) : c_gl_stride) * prob_m;
    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        c0 *= global_scale_f32;
        c1 *= global_scale_f32;
      }
      c_scalar_t2 res =
          Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (!has_act_order && group_blocks == -1 && !is_a_8bit &&
                    b_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        c_scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }
      if (has_bias && last) {
        c_scalar_t2 tmp_bias = b_bias[0];
        if constexpr (m_block_size_8) {
          tmp_bias = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hadd2(res, tmp_bias);
      }

      if constexpr (m_block_size_8) {
        ((c_scalar_t*)sh_red)[idx] = res.x;
        ((c_scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((c_scalar_t2*)sh_red)[idx] = res;
      }
    };

    if constexpr (fast_m1_red) {
      if (threadIdx.x < 32 * tb_n_warps && (threadIdx.x & 3) == 0) {
        const int wr_base = (threadIdx.x % 32) / 4 + 64 * (threadIdx.x / 32);
        c_scalar_t* sh_red_s = reinterpret_cast<c_scalar_t*>(sh_red);
  #pragma unroll
        for (int j = 0; j < 4; j++) {
          sh_red_s[wr_base + 16 * j + 0] = Cdtype::float2num(frag_c[0][j][0][0] * global_scale_f32);
          sh_red_s[wr_base + 16 * j + 8] = Cdtype::float2num(frag_c[0][j][0][2] * global_scale_f32);
        }
      }
    } else if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1],
                  frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3],
                  frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                  frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                  frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                  frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                  frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    if constexpr (gate_up_swiglu) {
      if constexpr (multi_m) {
        const uint32_t* const sh_u32 = reinterpret_cast<const uint32_t*>(sh_red);
        if (slice_col < 264) {
          if (threadIdx.x < prob_m * 32) {
            const int r = threadIdx.x >> 5;
            const int c = threadIdx.x & 31;
            const uint32_t gw = sh_u32[r * (4 * c_sh_stride) + c];
            const uint32_t uw = sh_u32[r * (4 * c_sh_stride) + c + 32];
            const float g0 = __uint_as_float(gw << 16);
            const float g1 = __uint_as_float(gw & 0xffff0000u);
            const float u0 = __uint_as_float(uw << 16);
            const float u1 = __uint_as_float(uw & 0xffff0000u);
            __nv_bfloat162 res = __floats2bfloat162_rn(
                __fmul_rn(marlin_swiglu_silu(g0), u0),
                __fmul_rn(marlin_swiglu_silu(g1), u1));
            reinterpret_cast<uint32_t*>(C0 + r * 2496 + slice_col * 8)[c] =
                *reinterpret_cast<uint32_t*>(&res);
          }
        } else {
          const bool is_up = (slice_col >= 288);
          const int m_pair = is_up ? (slice_col - 288) : (slice_col - 264);
          int4* const my_slot = C0 + prob_m * 2496 + m_pair * (prob_m * 32) + (is_up ? prob_m * 16 : 0);
          const int4* const other_slot = C0 + prob_m * 2496 + m_pair * (prob_m * 32) + (is_up ? 0 : prob_m * 16);
          int* const plock = (pair_locks != nullptr ? pair_locks : (locks + 24)) + m_pair;
          if (threadIdx.x < prob_m * 16) {
            const int r_s = threadIdx.x >> 4;
            const int c_s = threadIdx.x & 15;
            const int4 v = sh_red[r_s * c_sh_stride + c_s];
            asm volatile("st.global.wb.v4.b32 [%0], {%1, %2, %3, %4};\n"
                         :
                         : "l"(my_slot + threadIdx.x), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                         : "memory");
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            int old = 0;
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n"
                         : "=r"(old)
                         : "l"(plock)
                         : "memory");
            if (old == 1) {
              asm volatile("st.global.wb.b32 [%0], 0;\n" ::"l"(plock) : "memory");
            }
            sh_red[prob_m * c_sh_stride].x = old;
          }
          __syncthreads();
          if (sh_red[prob_m * c_sh_stride].x == 1) {
            const uint32_t* const other_u32 = reinterpret_cast<const uint32_t*>(other_slot);
            for (int idx = threadIdx.x; idx < prob_m * 64; idx += threads) {
              const int r = idx >> 6;
              const int c = idx & 63;
              const uint32_t own_w = sh_u32[r * (4 * c_sh_stride) + c];
              uint32_t oth_w;
              asm volatile("ld.global.ca.b32 %0, [%1];\n"
                           : "=r"(oth_w)
                           : "l"(other_u32 + r * 64 + c));
              const uint32_t gw = is_up ? oth_w : own_w;
              const uint32_t uw = is_up ? own_w : oth_w;
              const float g0 = __uint_as_float(gw << 16);
              const float g1 = __uint_as_float(gw & 0xffff0000u);
              const float u0 = __uint_as_float(uw << 16);
              const float u1 = __uint_as_float(uw & 0xffff0000u);
              __nv_bfloat162 res = __floats2bfloat162_rn(
                  __fmul_rn(marlin_swiglu_silu(g0), u0),
                  __fmul_rn(marlin_swiglu_silu(g1), u1));
              reinterpret_cast<uint32_t*>(C0 + r * 2496 + 2112 + m_pair * 16)[c] =
                  *reinterpret_cast<uint32_t*>(&res);
            }
          }
        }
      } else if (slice_col < 264) {
        if (threadIdx.x < 32) {
          const uint32_t gw = reinterpret_cast<const uint32_t*>(sh_red)[threadIdx.x];
          const uint32_t uw = reinterpret_cast<const uint32_t*>(sh_red)[threadIdx.x + 32];
          const float g0 = __uint_as_float(gw << 16);
          const float g1 = __uint_as_float(gw & 0xffff0000u);
          const float u0 = __uint_as_float(uw << 16);
          const float u1 = __uint_as_float(uw & 0xffff0000u);
          if constexpr (gate_up_swiglu_fp8) {
            reinterpret_cast<uint16_t*>(fp8_out_ptr + slice_col * 64)[threadIdx.x] =
                marlin_pack_fp8x2(
                    __fmul_rn(__fmul_rn(marlin_swiglu_silu(g0), u0), inv_fp8_scale),
                    __fmul_rn(__fmul_rn(marlin_swiglu_silu(g1), u1), inv_fp8_scale));
          } else {
            __nv_bfloat162 r = __floats2bfloat162_rn(
                __fmul_rn(marlin_swiglu_silu(g0), u0),
                __fmul_rn(marlin_swiglu_silu(g1), u1));
            reinterpret_cast<uint32_t*>(C0 + slice_col * 8)[threadIdx.x] =
                *reinterpret_cast<uint32_t*>(&r);
          }
        }
      } else {
        const bool is_up = (slice_col >= 288);
        const int m = is_up ? (slice_col - 288) : (slice_col - 264);
        int4* const my_slot = C0 + 2496 + m * 32 + (is_up ? 16 : 0);
        const int4* const other_slot = C0 + 2496 + m * 32 + (is_up ? 0 : 16);
        int* const plock = (pair_locks != nullptr ? pair_locks : (locks + 24)) + m;
        if (threadIdx.x < 16) {
          const int4 v = sh_red[threadIdx.x];
          asm volatile("st.global.wb.v4.b32 [%0], {%1, %2, %3, %4};\n"
                       :
                       : "l"(my_slot + threadIdx.x), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                       : "memory");
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          int old = 0;
          asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n"
                       : "=r"(old)
                       : "l"(plock)
                       : "memory");
          if (old == 1) {
            asm volatile("st.global.wb.b32 [%0], 0;\n" ::"l"(plock) : "memory");
          }
          sh_red[16].x = old;
        }
        __syncthreads();
        if (sh_red[16].x == 1 && threadIdx.x < 64) {
          const uint32_t own_w = reinterpret_cast<const uint32_t*>(sh_red)[threadIdx.x];
          uint32_t oth_w;
          asm volatile("ld.global.ca.b32 %0, [%1];\n"
                       : "=r"(oth_w)
                       : "l"(reinterpret_cast<const uint32_t*>(other_slot) + threadIdx.x));
          const uint32_t gw = is_up ? oth_w : own_w;
          const uint32_t uw = is_up ? own_w : oth_w;
          const float g0 = __uint_as_float(gw << 16);
          const float g1 = __uint_as_float(gw & 0xffff0000u);
          const float u0 = __uint_as_float(uw << 16);
          const float u1 = __uint_as_float(uw & 0xffff0000u);
          if constexpr (gate_up_swiglu_fp8) {
            reinterpret_cast<uint16_t*>(fp8_out_ptr + 16896 + m * 128)[threadIdx.x] =
                marlin_pack_fp8x2(
                    __fmul_rn(__fmul_rn(marlin_swiglu_silu(g0), u0), inv_fp8_scale),
                    __fmul_rn(__fmul_rn(marlin_swiglu_silu(g1), u1), inv_fp8_scale));
          } else {
            __nv_bfloat162 r = __floats2bfloat162_rn(
                __fmul_rn(marlin_swiglu_silu(g0), u0),
                __fmul_rn(marlin_swiglu_silu(g1), u1));
            reinterpret_cast<uint32_t*>(C0 + 2112 + m * 16)[threadIdx.x] =
                *reinterpret_cast<uint32_t*>(&r);
          }
        }
      }
    } else {
  #pragma unroll
      for (int i = 0;
           i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
           i++) {
        if (c_gl_wr < c_gl_wr_end) {
          if (use_atomic_add && slice_count > 1) {
            c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_gl_wr]);
            c_scalar_t2* sh_red_half2 =
                reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
            for (int a = 0; a < 4; a++) {
              atomicAdd(&C_half2[a], sh_red_half2[a]);
            }
          } else {
            C[c_gl_wr] = sh_red[c_sh_rd];
          }
          c_gl_wr += c_gl_wr_delta;
          c_sh_rd += c_sh_rd_delta;
        }
      }
    }
    __syncthreads();
  };

#ifdef MUSE_SCALE_TMA4
  constexpr bool cont_pipe = down_ilv;
  [[maybe_unused]] int total_rem = cont_pipe ? tma_iters : 0;
#else
  constexpr bool cont_pipe = false;
  [[maybe_unused]] int total_rem = 0;
#endif
  [[maybe_unused]] int pipe_start = 0;

  // Start global fetch and register load pipelines.
  [[maybe_unused]] bool tlb_done = false;
  auto start_pipes = [&]() {
#ifdef MUSE_SCALE_RANS_SK
    rans_slice_base = rans_done;
    rans_done += slice_iters;
#endif
#ifdef MUSE_SCALE_TMA4
    tma_slice_base = tma_done;
    tma_done += slice_iters;
    const bool pdl_cons = ((tma_pdl_mode & 1) != 0) && !tma_started && cont_pipe &&
                          (!fuse_mul_sigmoid || multi_m) && !has_act_order && !has_zp && !is_a_8bit;
    int pre_a_off[stages - 1];
    bool pre_a_pred[stages - 1];
    if (pdl_cons) {
  #pragma unroll
      for (int i = 0; i < stages - 1; i++) {
        const bool pred = (i < total_rem);
        pre_a_pred[i] = pred;
        pre_a_off[i] = a_gl_rd_base + a_gl_rd_delta_o * ilv_a_kt;
        fetch_b_only_to_shared(i, pred);
      }
    } else
#endif
    {
  #pragma unroll
      for (int i = 0; i < stages - 1; i++) {
        if (has_act_order && i == 0) {
          int last_g_idx = slice_k_start + stages * tb_k * 2;
          if (last_g_idx >= prob_k) {
            last_g_idx = prob_k - 1;
          }
          fetch_act_order_scales_to_shared(true, g_idx[slice_k_start],
                                           g_idx[last_g_idx]);
        }

        if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
          if (i == 0) {
            fetch_col_zp_to_shared();
            if constexpr (!dequant_skip_flop) {
              fetch_col_scale_to_shared();
            }
          }
        }
        fetch_to_shared(i, i, cont_pipe ? (i < total_rem) : (i < slice_iters));
      }
    }

    if constexpr (scale6) {
      if (blockIdx.x == 0 && !tlb_done && tpf_stride > 0) {
        tlb_done = true;
        const uint8_t* b_base = reinterpret_cast<const uint8_t*>(B);
        const size_t b_bytes = (static_cast<size_t>(prob_n) * static_cast<size_t>(prob_k)) >> 1;
        for (size_t off = static_cast<size_t>(threadIdx.x) * tpf_stride; off < b_bytes;
             off += static_cast<size_t>(threads) * tpf_stride) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(b_base + off));
        }
#ifdef MUSE_SCALE_TMA4
        const size_t s_bytes = static_cast<size_t>(tma_off16[tma_tot_sbs]) * 16u;
        for (size_t off = static_cast<size_t>(threadIdx.x) * tpf_stride; off < s_bytes;
             off += static_cast<size_t>(threads) * tpf_stride) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(tma_var_g + off));
        }
#elif !defined(MUSE_SCALE_RANS)
        const uint8_t* s_base = reinterpret_cast<const uint8_t*>(scales_ptr);
        const size_t s_bytes = ((static_cast<size_t>(prob_n) * static_cast<size_t>(prob_k)) >> 4) * 3u / 4u;
        for (size_t off = static_cast<size_t>(threadIdx.x) * tpf_stride; off < s_bytes;
             off += static_cast<size_t>(threads) * tpf_stride) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(s_base + off));
        }
#endif
      }
    }

    zero_accums();
#ifdef MUSE_SCALE_TMA4
    if constexpr (tma_ovlp) {
      if (!tma_started) {
        tma_started = true;
        __syncthreads();
        tma_wait(0, 0);
        tma_x = sh_states[threadIdx.x];
        if (tma_iters > 0) tma_decode(0);
        if (tma_iters > 1) tma_decode(1);
      }
    }
    if (pdl_cons) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
  #pragma unroll
      for (int i = 0; i < stages - 1; i++) {
        if (pre_a_pred[i]) {
          int4* sh_a_stage = sh_a + a_sh_stage_step * i;
  #pragma unroll
          for (int j = 0; j < a_sh_wr_iters; j++) {
            if (a_sh_wr_pred[j]) {
              if constexpr (fuse_mul_sigmoid && multi_m) {
                sh_a_stage[a_sh_wr_trans[j]] = load_mul_sigmoid_int4(a_gl_rd_delta_i * j + pre_a_off[i]);
              } else {
                sh_a_stage[a_sh_wr_trans[j]] = A[a_gl_rd_delta_i * j + pre_a_off[i]];
              }
            }
          }
        }
      }
    }
#endif
    wait_for_stage();
#ifdef MUSE_SCALE_TMA4
    if constexpr (!tma_ovlp) {
      if (!tma_started) {
        tma_started = true;
        tma_wait(0, 0);
        tma_x = sh_states[threadIdx.x];
        if (tma_iters > 0) tma_decode(0);
        if (tma_iters > 1) tma_decode(1);
        __syncthreads();
      }
    }
#endif
#ifdef MUSE_SCALE_RANS
  #ifdef MUSE_SCALE_RANS_SK
    if (rans_started) {
      // a later slice: entering its first stage is one more stage boundary of the CTA's stream
      rans_q++;
      if (rans_q + 1 < rans_iters) {
        const int X = rans_q + 1;
        if (X >= 2) rans_wait(1 + X % stages, static_cast<uint32_t>(((X - 2) / 4) & 1));
        rans_decode(X);
      }
      if (threadIdx.x == 0 && rans_q + 3 < rans_iters) rans_issue(rans_q + 3);
      __syncthreads();
    } else {
      rans_started = true;
  #endif
    rans_wait(0, 0);
    rans_x = reinterpret_cast<const uint32_t*>(sh_rhdr)[threadIdx.x];
    rans_decode(0);
    if (rans_iters > 1) rans_decode(1);
    if (threadIdx.x == 0) {
      if (rans_iters > 2) rans_issue(2);
      if (rans_iters > 3) rans_issue(3);
    }
    __syncthreads();
  #ifdef MUSE_SCALE_RANS_SK
    }
  #endif
#endif
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    if constexpr (has_act_order) {
      slice_k_start_shared_fetch += tb_k * (stages - 1);
    }
  };
  if (slice_iters) {
    start_pipes();
#ifdef MUSE_SCALE_TMA4
    if ((tma_pdl_mode & 2) && (!cont_pipe || total_rem <= 8)) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
#endif
  }
#ifdef MUSE_SCALE_TMA4
  else {
    if (tma_pdl_mode & 1) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tma_pdl_mode & 2) asm volatile("griddepcontrol.launch_dependents;");
  }
#endif

  // Main loop.
  while (slice_iters) {
    if constexpr (cont_pipe) {
  #pragma unroll
      for (int p = 0; p < stages; p++) {
        if (p < pipe_start) continue;
        int pipe = p;
        if constexpr (kloop_ovlp && b_sh_wr_iters == 2 && !has_act_order && !is_a_8bit && !has_zp) {
          // Overlap matmul(0) (pure registers) and next-stage TMA scale decode ahead of
          // cp_async_wait + __syncthreads(), hiding DRAM fetch and LDS latency on Stages=3/4.
          fetch_to_registers(1, pipe % stages);
          fetch_scales_to_registers(1, pipe);
          fetch_to_shared((pipe + stages - 1) % stages, pipe, total_rem >= stages);
          matmul(0, pipe);
#ifdef MUSE_SCALE_TMA4
          tma_q++;
          if (tma_q + 1 < tma_iters) tma_decode(tma_q + 1);
#endif
          pipe++;
          wait_for_stage();
          fetch_to_registers(2, pipe % stages);
          fetch_scales_to_registers(2, pipe);
          matmul(1, pipe - 1);
        } else {
  #pragma unroll
          for (int k = 0; k < b_sh_wr_iters; k++) {
            fetch_to_registers(k + 1, pipe % stages);
            fetch_scales_to_registers(k + 1, pipe);
            fetch_zp_to_registers(k + 1, pipe);
            if (k == b_sh_wr_iters - 2) {
              fetch_to_shared((pipe + stages - 1) % stages, pipe,
                              total_rem >= stages);
              pipe++;
              wait_for_stage();
#ifdef MUSE_SCALE_TMA4
              tma_q++;
              if (tma_q + 1 < tma_iters) tma_decode(tma_q + 1);
#endif
              init_same_group(pipe % stages);
            }
            matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
          }
        }
        total_rem--;
#ifdef MUSE_SCALE_TMA4
        if ((tma_pdl_mode & 2) && total_rem == 8) {
          asm volatile("griddepcontrol.launch_dependents;");
        }
#endif
        slice_iters--;
        if constexpr (fixed_shape == 3 && !multi_m) {
          if (__builtin_expect(slice_col >= 1560 && slice_iters > 0, 0)) {
            const int kt_done = 52 - slice_iters;
            const int c_L = (slice_col - 1560) * 2;
            const int rem_L = (52 * c_L) % 41;
            const int c1_L = (rem_L == 0) ? 41 : (41 - rem_L);
            const int c2_L = (c1_L <= 10) ? (c1_L + 41) : 52;
            const int c_R = c_L + 1;
            int c1_R = 52, c2_R = 52;
            if (c_R < 37) {
              const int rem_R = (52 * c_R) % 41;
              c1_R = (rem_R == 0) ? 41 : (41 - rem_R);
              c2_R = (c1_R <= 10) ? (c1_R + 41) : 52;
            }
            if (kt_done == c1_L || kt_done == c2_L || kt_done == c1_R || kt_done == c2_R) {
              const int n_warp = (threadIdx.x >> 5) & 1;
              const int my_c1 = (n_warp == 0) ? c1_L : c1_R;
              const int my_c2 = (n_warp == 0) ? c2_L : c2_R;
              const bool do_my_half = (kt_done == my_c1 || kt_done == my_c2);
              const int save_slot = (kt_done >= 42) ? 1 : 0;
              constexpr int lane_grp = b_sh_stride_threads >> 2;
              const int red_idx = threadIdx.x / b_sh_stride_threads;
              const int tid_grp = (threadIdx.x % b_sh_stride_threads) >> 2;
              const bool is_row0 = (threadIdx.x & 3) == 0;
              if (do_my_half && is_row0 && red_idx >= 2) {
                const int wr_slot = (red_idx - 2) * 2 * lane_grp + tid_grp;
                reinterpret_cast<float4*>(sh_red)[wr_slot] = make_float4(
                    frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
                reinterpret_cast<float4*>(sh_red)[wr_slot + lane_grp] = make_float4(
                    frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
              }
              __syncthreads();
              if (do_my_half && is_row0 && red_idx == 1) {
                const int wr_slot = tid_grp;
                const int rd_slot = 2 * lane_grp + tid_grp;
                const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[rd_slot];
                const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[rd_slot + lane_grp];
                const float4 wr0 = reinterpret_cast<const float4*>(sh_red)[wr_slot];
                const float4 wr1 = reinterpret_cast<const float4*>(sh_red)[wr_slot + lane_grp];
                frag_c[0][0][0][0] += rd0.x + wr0.x;
                frag_c[0][0][0][2] += rd0.y + wr0.y;
                frag_c[0][1][0][0] += rd0.z + wr0.z;
                frag_c[0][1][0][2] += rd0.w + wr0.w;
                frag_c[0][2][0][0] += rd1.x + wr1.x;
                frag_c[0][2][0][2] += rd1.y + wr1.y;
                frag_c[0][3][0][0] += rd1.z + wr1.z;
                frag_c[0][3][0][2] += rd1.w + wr1.w;
                reinterpret_cast<float4*>(sh_red)[wr_slot] = make_float4(
                    frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
                reinterpret_cast<float4*>(sh_red)[wr_slot + lane_grp] = make_float4(
                    frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
              }
              __syncthreads();
              if (do_my_half && is_row0 && red_idx == 0) {
                const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[tid_grp];
                const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[tid_grp + lane_grp];
                float4* dst = reinterpret_cast<float4*>(C_tmp) +
                              static_cast<size_t>(blockIdx.x) * 64 + save_slot * 32 + tid_grp * 2;
                dst[0] = make_float4(
                    frag_c[0][0][0][0] + rd0.x, frag_c[0][0][0][2] + rd0.y,
                    frag_c[0][1][0][0] + rd0.z, frag_c[0][1][0][2] + rd0.w);
                dst[1] = make_float4(
                    frag_c[0][2][0][0] + rd1.x, frag_c[0][2][0][2] + rd1.y,
                    frag_c[0][3][0][0] + rd1.z, frag_c[0][3][0][2] + rd1.w);
              }
              if (do_my_half) zero_accums();
              __syncthreads();
            }
          }
        }
        if (slice_iters == 0) {
          pipe_start = (p + 1 == stages) ? 0 : (p + 1);
          break;
        }
      }
      if (slice_iters != 0) {
        pipe_start = 0;
      }
    } else {
  #pragma unroll
      for (int pipe = 0; pipe < stages;) {
  #pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          fetch_to_registers(k + 1, pipe % stages);
          fetch_scales_to_registers(k + 1, pipe);
          fetch_zp_to_registers(k + 1, pipe);
          if (k == b_sh_wr_iters - 2) {
            fetch_to_shared((pipe + stages - 1) % stages, pipe,
                            slice_iters >= stages);
            pipe++;
            wait_for_stage();
#ifdef MUSE_SCALE_TMA4
            tma_q++;
            if (tma_q + 1 < tma_iters) tma_decode(tma_q + 1);
#endif
#ifdef MUSE_SCALE_RANS
            rans_q++;
            if (rans_q + 1 < rans_iters) {
              const int X = rans_q + 1;
              if (X >= 2) rans_wait(1 + X % stages, static_cast<uint32_t>(((X - 2) / 4) & 1));
              rans_decode(X);
            }
            if (threadIdx.x == 0 && rans_q + 3 < rans_iters) rans_issue(rans_q + 3);
#endif
            init_same_group(pipe % stages);
          }

          if constexpr (!is_a_8bit) {
            matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
          } else {
            static_assert(group_blocks != 0 && group_blocks != 1);
            matmul_a8(k);
          }
        }
        slice_iters--;
        if constexpr (fixed_shape == 3 && !multi_m) {
          if (__builtin_expect(slice_col >= 1560 && slice_iters > 0, 0)) {
            const int kt_done = 52 - slice_iters;
            const int c_L = (slice_col - 1560) * 2;
            const int rem_L = (52 * c_L) % 41;
            const int c1_L = (rem_L == 0) ? 41 : (41 - rem_L);
            const int c2_L = (c1_L <= 10) ? (c1_L + 41) : 52;
            const int c_R = c_L + 1;
            int c1_R = 52, c2_R = 52;
            if (c_R < 37) {
              const int rem_R = (52 * c_R) % 41;
              c1_R = (rem_R == 0) ? 41 : (41 - rem_R);
              c2_R = (c1_R <= 10) ? (c1_R + 41) : 52;
            }
            if (kt_done == c1_L || kt_done == c2_L || kt_done == c1_R || kt_done == c2_R) {
              const int n_warp = (threadIdx.x >> 5) & 1;
              const int my_c1 = (n_warp == 0) ? c1_L : c1_R;
              const int my_c2 = (n_warp == 0) ? c2_L : c2_R;
              const bool do_my_half = (kt_done == my_c1 || kt_done == my_c2);
              const int save_slot = (kt_done >= 42) ? 1 : 0;
              constexpr int lane_grp = b_sh_stride_threads >> 2;
              const int red_idx = threadIdx.x / b_sh_stride_threads;
              const int tid_grp = (threadIdx.x % b_sh_stride_threads) >> 2;
              const bool is_row0 = (threadIdx.x & 3) == 0;
              if (do_my_half && is_row0 && red_idx >= 2) {
                const int wr_slot = (red_idx - 2) * 2 * lane_grp + tid_grp;
                reinterpret_cast<float4*>(sh_red)[wr_slot] = make_float4(
                    frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
                reinterpret_cast<float4*>(sh_red)[wr_slot + lane_grp] = make_float4(
                    frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
              }
              __syncthreads();
              if (do_my_half && is_row0 && red_idx == 1) {
                const int wr_slot = tid_grp;
                const int rd_slot = 2 * lane_grp + tid_grp;
                const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[rd_slot];
                const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[rd_slot + lane_grp];
                const float4 wr0 = reinterpret_cast<const float4*>(sh_red)[wr_slot];
                const float4 wr1 = reinterpret_cast<const float4*>(sh_red)[wr_slot + lane_grp];
                frag_c[0][0][0][0] += rd0.x + wr0.x;
                frag_c[0][0][0][2] += rd0.y + wr0.y;
                frag_c[0][1][0][0] += rd0.z + wr0.z;
                frag_c[0][1][0][2] += rd0.w + wr0.w;
                frag_c[0][2][0][0] += rd1.x + wr1.x;
                frag_c[0][2][0][2] += rd1.y + wr1.y;
                frag_c[0][3][0][0] += rd1.z + wr1.z;
                frag_c[0][3][0][2] += rd1.w + wr1.w;
                reinterpret_cast<float4*>(sh_red)[wr_slot] = make_float4(
                    frag_c[0][0][0][0], frag_c[0][0][0][2], frag_c[0][1][0][0], frag_c[0][1][0][2]);
                reinterpret_cast<float4*>(sh_red)[wr_slot + lane_grp] = make_float4(
                    frag_c[0][2][0][0], frag_c[0][2][0][2], frag_c[0][3][0][0], frag_c[0][3][0][2]);
              }
              __syncthreads();
              if (do_my_half && is_row0 && red_idx == 0) {
                const float4 rd0 = reinterpret_cast<const float4*>(sh_red)[tid_grp];
                const float4 rd1 = reinterpret_cast<const float4*>(sh_red)[tid_grp + lane_grp];
                float4* dst = reinterpret_cast<float4*>(C_tmp) +
                              static_cast<size_t>(blockIdx.x) * 64 + save_slot * 32 + tid_grp * 2;
                dst[0] = make_float4(
                    frag_c[0][0][0][0] + rd0.x, frag_c[0][0][0][2] + rd0.y,
                    frag_c[0][1][0][0] + rd0.z, frag_c[0][1][0][2] + rd0.w);
                dst[1] = make_float4(
                    frag_c[0][2][0][0] + rd1.x, frag_c[0][2][0][2] + rd1.y,
                    frag_c[0][3][0][0] + rd1.z, frag_c[0][3][0][2] + rd1.w);
              }
              if (do_my_half) zero_accums();
              __syncthreads();
            }
          }
        }
        if (slice_iters == 0) {
          break;
        }
      }

      a_gl_rd += a_gl_rd_delta_o * stages;
    }

    if constexpr (has_act_order) {
      slice_k_start += tb_k * stages;

      if (slice_k_start < prob_k) {
        slice_k_start_shared_fetch += tb_k * stages;
        int first_group_id = g_idx[slice_k_start];
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        int last_group_id = g_idx[last_g_idx];
        if (last_group_id >= sh_first_group_id + sh_num_groups) {
          fetch_act_order_scales_to_shared(false, first_group_id,
                                           last_group_id);
          __syncthreads();
        }
      }
    }

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      // convert fp16 accum to fp32 for reduction
      if constexpr (use_fp16_accum) {
  #pragma unroll
        for (int i = 0; i < (thread_m_blocks * (is_a_8bit ? 2 : 4) * 2); i++) {
          float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
          scalar_t* frag_c_part_half =
              reinterpret_cast<scalar_t*>(frag_c_part_float);

  #pragma unroll
          for (int i = 3; i >= 0; i--) {
            frag_c_part_float[i] = Cdtype::num2float(frag_c_part_half[i]);
          }
        }
      }

      if constexpr (is_a_8bit) {
        float frag_a_s[2 * thread_m_blocks];

        for (int i = 0; i < 2 * thread_m_blocks; i++)
          frag_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];

  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][0][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][0][g] = c_val * s_val;
            }
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][1][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][1][g] = c_val * s_val;
            }
          }
        }
      }

      if constexpr (!cont_pipe) {
        cp_async_wait<0>();
      } else if (total_rem == 0) {
        cp_async_wait<0>();
      }
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add) || is_a_8bit) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if constexpr (fixed_shape == 3 && !multi_m) {
        if (__builtin_expect(slice_col >= 1560, 0) && threadIdx.x < 64 && (threadIdx.x & 3) == 0) {
          const int c_p2 = (slice_col - 1560) * 2 + ((threadIdx.x >> 5) & 1);
          if (c_p2 < 37) {
            const int rem = (52 * c_p2) % 41;
            const int c1 = (rem == 0) ? 41 : (41 - rem);
            const int tid_grp = threadIdx.x >> 2;
            const float4* src = reinterpret_cast<const float4*>(C_tmp) +
                                static_cast<size_t>(blockIdx.x) * 64 + tid_grp * 2;
            if (c1 <= 10) {
              const float4 m0 = src[32];
              const float4 m1 = src[33];
              frag_c[0][0][0][0] = __fadd_rn(m0.x, frag_c[0][0][0][0]);
              frag_c[0][0][0][2] = __fadd_rn(m0.y, frag_c[0][0][0][2]);
              frag_c[0][1][0][0] = __fadd_rn(m0.z, frag_c[0][1][0][0]);
              frag_c[0][1][0][2] = __fadd_rn(m0.w, frag_c[0][1][0][2]);
              frag_c[0][2][0][0] = __fadd_rn(m1.x, frag_c[0][2][0][0]);
              frag_c[0][2][0][2] = __fadd_rn(m1.y, frag_c[0][2][0][2]);
              frag_c[0][3][0][0] = __fadd_rn(m1.z, frag_c[0][3][0][0]);
              frag_c[0][3][0][2] = __fadd_rn(m1.w, frag_c[0][3][0][2]);
            }
            const float4 l0 = src[0];
            const float4 l1 = src[1];
            frag_c[0][0][0][0] = __fadd_rn(l0.x, frag_c[0][0][0][0]);
            frag_c[0][0][0][2] = __fadd_rn(l0.y, frag_c[0][0][0][2]);
            frag_c[0][1][0][0] = __fadd_rn(l0.z, frag_c[0][1][0][0]);
            frag_c[0][1][0][2] = __fadd_rn(l0.w, frag_c[0][1][0][2]);
            frag_c[0][2][0][0] = __fadd_rn(l1.x, frag_c[0][2][0][0]);
            frag_c[0][2][0][2] = __fadd_rn(l1.y, frag_c[0][2][0][2]);
            frag_c[0][3][0][0] = __fadd_rn(l1.z, frag_c[0][3][0][0]);
            frag_c[0][3][0][2] = __fadd_rn(l1.w, frag_c[0][3][0][2]);
          }
        }
      }

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp || is_a_8bit)) {
        if constexpr (is_a_8bit) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          }
        } else if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              c_scalar_t2* frag_s_half2 =
                  reinterpret_cast<c_scalar_t2*>(frag_s);
  #pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Cdtype::num2num2(
                    reinterpret_cast<c_scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (!has_act_order && group_blocks == -1 && is_a_8bit) {
  #pragma unroll
        for (int j = 0; j < 2; j++) {
          float2 aa[2];
          aa[0] = Cdtype::num22float2(frag_s[0][j * 2][0]);
          aa[1] = Cdtype::num22float2(frag_s[0][j * 2 + 1][0]);

  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[0])[g % 2];
              frag_c[i][j][0][g] *= scale;
            }

  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[1])[g % 2];
              frag_c[i][j][1][g] *= scale;
            }
          }
        }
      } else if (!has_act_order && group_blocks == -1 &&
                 b_type.size_bits() == 8 &&
                 (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      const int eff_lock = half_split ? (2 * locks_off + half_id) : locks_off;
      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[eff_lock], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[eff_lock], last);
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        if constexpr (!is_a_8bit)
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[eff_lock]);
      if constexpr (fuse_lmpost) {
        constexpr int active_threads = 32 * tb_n_warps;
        if constexpr (multi_m) {
          if (last && threadIdx.x < active_threads) {
            const int lane = threadIdx.x & 31;
            const int row0 = (lane & 3) << 1;
            const bool has_r0 = row0 < prob_m;
            const bool has_r1 = (row0 + 1) < prob_m;
            const int base_col = slice_col * (thread_n_blocks * 16) + 64 * (threadIdx.x >> 5);
            const int g = lane >> 2;
  #pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int col0 = base_col + 16 * j + g;
              const int col2 = col0 + 8;
              if (col0 < org_vocab_size) {
                if (has_r0) {
                  const __nv_bfloat16 b0 = Cdtype::float2num(frag_c[0][j][0][0] * global_scale_f32);
                  const uint16_t m0 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b0));
                  const float v0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m0));
                  if ((v0 > lm_bv) || (v0 == lm_bv && col0 < lm_bi)) {
                    lm_bv = v0;
                    lm_bi = col0;
                  }
                }
                if (has_r1) {
                  const __nv_bfloat16 b1 = Cdtype::float2num(frag_c[0][j][0][1] * global_scale_f32);
                  const uint16_t m1 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b1));
                  const float v1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m1));
                  if ((v1 > lm_bv1) || (v1 == lm_bv1 && col0 < lm_bi1)) {
                    lm_bv1 = v1;
                    lm_bi1 = col0;
                  }
                }
              }
              if (col2 < org_vocab_size) {
                if (has_r0) {
                  const __nv_bfloat16 b2 = Cdtype::float2num(frag_c[0][j][0][2] * global_scale_f32);
                  const uint16_t m2 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b2));
                  const float v2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m2));
                  if ((v2 > lm_bv) || (v2 == lm_bv && col2 < lm_bi)) {
                    lm_bv = v2;
                    lm_bi = col2;
                  }
                }
                if (has_r1) {
                  const __nv_bfloat16 b3 = Cdtype::float2num(frag_c[0][j][0][3] * global_scale_f32);
                  const uint16_t m3 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b3));
                  const float v3 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m3));
                  if ((v3 > lm_bv1) || (v3 == lm_bv1 && col2 < lm_bi1)) {
                    lm_bv1 = v3;
                    lm_bi1 = col2;
                  }
                }
              }
            }
          }
        } else if (last && threadIdx.x < active_threads && (threadIdx.x & 3) == 0) {
          const int base_col = slice_col * (thread_n_blocks * 16) + 64 * (threadIdx.x >> 5);
          const int g = (threadIdx.x & 31) >> 2;
  #pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int col0 = base_col + 16 * j + g;
            const int col2 = col0 + 8;
            if (col0 < org_vocab_size) {
              const __nv_bfloat16 b0 = Cdtype::float2num(frag_c[0][j][0][0] * global_scale_f32);
              const uint16_t m0 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b0));
              const float v0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m0));
              if ((v0 > lm_bv) || (v0 == lm_bv && col0 < lm_bi)) {
                lm_bv = v0;
                lm_bi = col0;
              }
            }
            if (col2 < org_vocab_size) {
              const __nv_bfloat16 b2 = Cdtype::float2num(frag_c[0][j][0][2] * global_scale_f32);
              const uint16_t m2 = __ldg(lmpost_lut + *reinterpret_cast<const uint16_t*>(&b2));
              const float v2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&m2));
              if ((v2 > lm_bv) || (v2 == lm_bv && col2 < lm_bi)) {
                lm_bv = v2;
                lm_bi = col2;
              }
            }
          }
        }
      } else {
        if (last || use_atomic_add)
          // only the last block in a slice actually writes the result
          write_result(last);
      }
      if constexpr (gate_up_grid) return;
      if constexpr (fixed_shape == 3) {
        slice_row = 0;
        slice_col += static_cast<int>(gridDim.x);
        slice_iters = (slice_col < 1579) ? 52 : 0;
        is_first_matmul_in_slice = true;
      } else {
        slice_row = 0;
        if (!in_part2) {
          slice_col_par += ((split2 || half_split) ? vgrid : gridDim.x);
        } else {
          slice_col_par++;
          slice_col += (half_split ? 2 : 1);
        }
        is_first_matmul_in_slice = true;
        init_slice();
      }
      if constexpr (gate_v48) {
        if (slice_iters == 0 && vblk >= static_cast<int>(gridDim.x)) {
          __syncthreads();
          vblk -= static_cast<int>(gridDim.x);
          in_part2 = false;
          first_init = true;
          slice_row = 0;
          slice_col_par = vblk;
          locks_off = (iters * vblk) / k_tiles - 1;
          init_slice();
        }
      }
      if constexpr (half_split) {
        if (slice_iters == 0 && static_cast<int>(gridDim.x) == 48 && half_id == 0) {
          __syncthreads();
          half_id = 1;
          in_part2 = false;
          first_init = true;
          slice_row = 0;
          slice_col_par = vblk;
          locks_off = (part2_mn_tiles >= vgrid) ? vblk : ((iters * vblk) / k_tiles - 1);
          init_slice();
        }
      }

      if (slice_iters) {
        if constexpr (cont_pipe) {
          zero_accums();
        } else {
          a_gl_rd = a_gl_rd_base + a_gl_rd_delta_o * slice_row;
          if constexpr (!down_ilv) {
            b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                      (threadIdx.x % b_sh_stride);
            b_gl_rd += b_sh_stride * tile_column_stride * slice_col + b_gl_rd_delta_o * slice_row;
          }

          bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
          // Update slice k/n for scales loading
          if constexpr (has_act_order) {
            slice_k_start = tb_k * slice_row;
            slice_k_finish = slice_k_start + tb_k * slice_iters;
            slice_k_start_shared_fetch = slice_k_start;
            slice_n_offset = act_s_col_tb_stride * slice_col;
          } else {
            if constexpr (group_blocks == -1) {
              s_gl_rd = s_sh_stride * tile_column_stride * slice_col + threadIdx.x;
              zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
            } else if constexpr (group_blocks >= thread_k_blocks) {
              s_gl_rd =
                  s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                  s_sh_stride * tile_column_stride * slice_col + threadIdx.x;
              zp_gl_rd =
                  zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                  zp_sh_stride * slice_col + threadIdx.x;
            } else {
              if constexpr (scale6) {
                s_gl_rd = (3 * s_sh_stride * tile_column_stride / 4) * slice_col +
                          (3 * s_sh_stage / 4) * slice_row + threadIdx.x;
              } else {
                s_gl_rd =
                    s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                   threadIdx.x / s_sh_stride) +
                    s_sh_stride * tile_column_stride * slice_col + threadIdx.x % s_sh_stride;
              }
              zp_gl_rd =
                  zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                  threadIdx.x / zp_sh_stride) +
                  zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
            }
          }
          start_pipes();
        }
      }
    }
  }
  if constexpr (fuse_lmpost) {
    if constexpr (multi_m) {
      __syncthreads();
      float* const sh_bv = reinterpret_cast<float*>(sh);
      int* const sh_bi = reinterpret_cast<int*>(sh_bv + 64);
      int* const sh_last = sh_bi + 64;

      constexpr int active_threads = 32 * tb_n_warps;
      if (threadIdx.x < active_threads) {
        const int lane = threadIdx.x & 31;
        const int w = threadIdx.x >> 5;
#pragma unroll
        for (int off = 16; off >= 4; off >>= 1) {
          const float ov0 = __shfl_xor_sync(0xffffffffu, lm_bv, off);
          const int oi0 = __shfl_xor_sync(0xffffffffu, lm_bi, off);
          if ((ov0 > lm_bv) || (ov0 == lm_bv && oi0 < lm_bi)) {
            lm_bv = ov0;
            lm_bi = oi0;
          }
          const float ov1 = __shfl_xor_sync(0xffffffffu, lm_bv1, off);
          const int oi1 = __shfl_xor_sync(0xffffffffu, lm_bi1, off);
          if ((ov1 > lm_bv1) || (ov1 == lm_bv1 && oi1 < lm_bi1)) {
            lm_bv1 = ov1;
            lm_bi1 = oi1;
          }
        }
        if (lane < 4) {
          sh_bv[w * 8 + (lane << 1)] = lm_bv;
          sh_bi[w * 8 + (lane << 1)] = lm_bi;
          sh_bv[w * 8 + (lane << 1) + 1] = lm_bv1;
          sh_bi[w * 8 + (lane << 1) + 1] = lm_bi1;
        }
      }
      __syncthreads();

      float* const g_bv = reinterpret_cast<float*>(C0);
      int* const g_bi = reinterpret_cast<int*>(g_bv + 1024);
      int* const g_done = g_bi + 1024;
      if (static_cast<int>(threadIdx.x) < prob_m) {
        const int r = static_cast<int>(threadIdx.x);
        float bv = sh_bv[r];
        int bi = sh_bi[r];
        for (int w = 1; w < tb_n_warps; ++w) {
          const float ov = sh_bv[w * 8 + r];
          const int oi = sh_bi[w * 8 + r];
          if ((ov > bv) || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
          }
        }
        g_bv[r * 128 + blockIdx.x] = bv;
        g_bi[r * 128 + blockIdx.x] = bi;
        __threadfence();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        *sh_last = (atomicAdd(g_done, 1) == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
      }
      __syncthreads();

      if (*sh_last) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int nwarps = threads >> 5;
        for (int r = warp; r < prob_m; r += nwarps) {
          float fv = -__int_as_float(0x7f800000);
          int fi = 0x7fffffff;
          for (int b = lane; b < static_cast<int>(gridDim.x); b += 32) {
            const float ov = const_cast<const volatile float*>(g_bv)[r * 128 + b];
            const int oi = const_cast<const volatile int*>(g_bi)[r * 128 + b];
            if ((ov > fv) || (ov == fv && oi < fi)) {
              fv = ov;
              fi = oi;
            }
          }
#pragma unroll
          for (int off = 16; off >= 1; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, fv, off);
            const int oi = __shfl_down_sync(0xffffffffu, fi, off);
            if ((ov > fv) || (ov == fv && oi < fi)) {
              fv = ov;
              fi = oi;
            }
          }
          if (lane == 0) {
            lmpost_out[r] = static_cast<int64_t>(fi);
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          *g_done = 0;
        }
      }
    } else {
    __syncthreads();
    float* const sh_bv = reinterpret_cast<float*>(sh);
    int* const sh_bi = reinterpret_cast<int*>(sh_bv + 128);
    int* const sh_last = sh_bi + 128;

    constexpr int active_threads = 32 * tb_n_warps;
    if (threadIdx.x < active_threads) {
      if ((threadIdx.x & 3) != 0) {
        lm_bv = -__int_as_float(0x7f800000);
        lm_bi = 0x7fffffff;
      }
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, lm_bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, lm_bi, off);
        if ((ov > lm_bv) || (ov == lm_bv && oi < lm_bi)) {
          lm_bv = ov;
          lm_bi = oi;
        }
      }
      if constexpr (tb_n_warps > 1) {
        if ((threadIdx.x & 31) == 0) {
          sh_bv[threadIdx.x >> 5] = lm_bv;
          sh_bi[threadIdx.x >> 5] = lm_bi;
        }
      }
    }
    if constexpr (tb_n_warps > 1) {
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int w = 1; w < tb_n_warps; ++w) {
          const float ov = sh_bv[w];
          const int oi = sh_bi[w];
          if ((ov > lm_bv) || (ov == lm_bv && oi < lm_bi)) {
            lm_bv = ov;
            lm_bi = oi;
          }
        }
      }
    }

    float* const g_bv = reinterpret_cast<float*>(C0);
    int* const g_bi = reinterpret_cast<int*>(g_bv + 128);
    int* const g_done = g_bi + 128;

    if (threadIdx.x == 0) {
      g_bv[blockIdx.x] = lm_bv;
      g_bi[blockIdx.x] = lm_bi;
      __threadfence();
      *sh_last = (atomicAdd(g_done, 1) == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
    }
    __syncthreads();

    if (*sh_last) {
      float fv = (static_cast<int>(threadIdx.x) < static_cast<int>(gridDim.x))
                     ? const_cast<const volatile float*>(g_bv)[threadIdx.x]
                     : -__int_as_float(0x7f800000);
      int fi = (static_cast<int>(threadIdx.x) < static_cast<int>(gridDim.x))
                   ? const_cast<const volatile int*>(g_bi)[threadIdx.x]
                   : 0x7fffffff;
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, fv, off);
        const int oi = __shfl_down_sync(0xffffffffu, fi, off);
        if ((ov > fv) || (ov == fv && oi < fi)) {
          fv = ov;
          fi = oi;
        }
      }
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      if (lane == 0) {
        sh_bv[warp] = fv;
        sh_bi[warp] = fi;
      }
      __syncthreads();
      if (warp == 0) {
        fv = (lane < (threads >> 5)) ? sh_bv[lane] : -__int_as_float(0x7f800000);
        fi = (lane < (threads >> 5)) ? sh_bi[lane] : 0x7fffffff;
#pragma unroll
        for (int off = (threads >> 5) >> 1; off >= 1; off >>= 1) {
          const float ov = __shfl_down_sync(0xffffffffu, fv, off);
          const int oi = __shfl_down_sync(0xffffffffu, fi, off);
          if ((ov > fv) || (ov == fv && oi < fi)) {
            fv = ov;
            fi = oi;
          }
        }
        if (lane == 0) {
          *g_done = 0;
          *lmpost_out = static_cast<int64_t>(fi);
        }
      }
    }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME

#endif
