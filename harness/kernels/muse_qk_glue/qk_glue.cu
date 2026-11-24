// SPDX-License-Identifier: Apache-2.0
// One-pass fused Q/K RMSNorm + query pre-scale + NEOX RoPE (+ optional KV cache scatter)
// for MuseGlimmerAttention (head_dim = 128, 32 Q heads + 2 K heads + 2 V heads,
// BF16 I/O, FP32 internal arithmetic).
//
// Evaluates the exact same FP32 sum-of-squares reduction order (4 elements per lane,
// xor-shuffle 16/8/4/2/1), div.full.f32(/128), rsqrt.approx.ftz.f32, query pre-scale,
// and NEOX RoPE FMA chain as the compiled Inductor kernels in a single global pass
// for both prompt prefill (M > 1) and single-token decode (M = 1).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace muse_qk_glue {

template <bool CacheFP32>
__device__ __forceinline__ void load_cos_sin(
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    int t,
    int lane,
    int cache_rows,
    float& c0, float& c1, float& c2, float& c3,
    float& s0, float& s1, float& s2, float& s3) {
  int64_t pos = __ldg(positions + t);
  if (pos < 0) pos += cache_rows;
  const int d_lo = (lane & 15) * 4;
  if constexpr (CacheFP32) {
    const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
    const float4 cv = __ldg(reinterpret_cast<const float4*>(row + d_lo));
    const float4 sv = __ldg(reinterpret_cast<const float4*>(row + 64 + d_lo));
    c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
    s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
  } else {
    const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
    const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
    const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
    const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
    const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
    c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
    c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
    s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
    s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
  }
}

template <bool UseRope>
__device__ __forceinline__ uint2 norm_rope_pack(
    uint2 raw,
    int h,
    int lane,
    float eps,
    float qscale,
    float c0, float c1, float c2, float c3,
    float s0, float s1, float s2, float s3) {
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw);
  const float x0 = __bfloat162float(vb[0]);
  const float x1 = __bfloat162float(vb[1]);
  const float x2 = __bfloat162float(vb[2]);
  const float x3 = __bfloat162float(vb[3]);

  // Exact Triton reduction order: ((x0^2 + x1^2) + x2^2) + x3^2, then xor-shuffle 16,8,4,2,1.
  float ss = __fmaf_rn(x0, x0, 0.0f);
  ss = __fmaf_rn(x1, x1, ss);
  ss = __fmaf_rn(x2, x2, ss);
  ss = __fmaf_rn(x3, x3, ss);
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 16));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 8));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));

  const float mean_v = __fmul_rn(ss, 0.0078125f);
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean_v + eps));

  float y0 = __fmul_rn(x0, r);
  float y1 = __fmul_rn(x1, r);
  float y2 = __fmul_rn(x2, r);
  float y3 = __fmul_rn(x3, r);
  if (h < 32) {
    y0 = __fmul_rn(y0, qscale);
    y1 = __fmul_rn(y1, qscale);
    y2 = __fmul_rn(y2, qscale);
    y3 = __fmul_rn(y3, qscale);
  }

  float o0 = y0, o1 = y1, o2 = y2, o3 = y3;
  if constexpr (UseRope) {
    // Lane l holds elements 4*l..4*l+3, so lane (l ^ 16) holds the exact +/-64 NEOX partner.
    const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
    const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
    const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
    const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
    if (lane < 16) {
      o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
    } else {
      o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
    }
  }

  uint2 out_v;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
  ob[0] = __float2bfloat16_rn(o0);
  ob[1] = __float2bfloat16_rn(o1);
  ob[2] = __float2bfloat16_rn(o2);
  ob[3] = __float2bfloat16_rn(o3);
  return out_v;
}

template <bool Block16>
__device__ __forceinline__ int64_t compute_kv_slot_offset(
    const int64_t* __restrict__ slot_mapping,
    int t,
    int num_slot_tokens,
    int block_size,
    int64_t stride_blk,
    int64_t stride_slot) {
  if (t >= num_slot_tokens) return -1;
  const int64_t slot = __ldg(slot_mapping + t);
  if (slot < 0) return -1;
  const int64_t blk = Block16 ? (slot >> 4) : (slot / block_size);
  const int64_t off = Block16 ? (slot & 15) : (slot - blk * block_size);
  return blk * stride_blk + off * stride_slot;
}

// ---------------------------------------------------------------------------------------------
// Paged-KV L2 prefetch (read-only, no numerics).
//
// FA2 decode runs this model's GQA-16 (32 Q heads, 2 KV heads) as 2 CTAs with num_splits=1, so
// the whole layer's KV window (640 slots x 2 KV heads x 2 (K+V) x 128 x 2 B = 655 KB at the
// benchmark's seq_k) is pulled by two CTAs: ~38 GB/s standalone, and 72.8 us in-engine once it
// queues behind output_gate_proj's 18 GB/token weight stream. Warming those same 655 KB into L2
// from 80 CTAs costs ~2.5 us and turns FA2's reads into L2 hits, which is bit-exact because
// nothing but the cache lines changes.
//
// One CTA per (sequence, K|V, block index); the physical block id and the sequence length are
// read on the device, so the launch geometry is a pure function of the captured shapes and the
// kernel is CUDA-graph safe.
template <int kThreads>
__global__ void kv_window_prefetch_kernel(
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    int bt_stride,
    const int32_t* __restrict__ seqused_k,
    int num_seqs,
    int block_size,
    int max_blocks,
    int num_cache_blocks,
    int window,
    int parts,
    int64_t stride_blk,
    int64_t span_elems,
    int elem_bytes) {
  const int per_seq = parts * max_blocks;
  const int seq = blockIdx.x / per_seq;
  if (seq >= num_seqs) return;
  const int rem = blockIdx.x - seq * per_seq;
  const int part = (parts == 2) ? (rem / max_blocks) : 0;
  const int bidx = rem - part * max_blocks;
  const int seqlen = __ldg(seqused_k + seq);
  if (bidx * block_size >= seqlen) return;
  // Sliding-window layers only read the last `window` slots; skip blocks entirely below it.
  if (window > 0 && (bidx + 1) * block_size <= seqlen - window) return;
  const int32_t blk = __ldg(block_table + seq * bt_stride + bidx);
  if (blk < 0 || blk >= num_cache_blocks) return;
  const char* base = static_cast<const char*>(part ? value_cache : key_cache) +
                     static_cast<int64_t>(blk) * stride_blk * elem_bytes;
  const int64_t nbytes = span_elems * elem_bytes;
  int64_t chunk = (nbytes + kThreads - 1) / kThreads;
  chunk = (chunk + 15) & ~static_cast<int64_t>(15);
  if (chunk <= 0) return;
  const int64_t step = static_cast<int64_t>(kThreads) * chunk;
  for (int64_t off = static_cast<int64_t>(threadIdx.x) * chunk; off < nbytes; off += step) {
    const int64_t sz = (nbytes - off < chunk) ? (nbytes - off) : chunk;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Fire-and-forget bulk prefetch: no destination registers, so the loop never serialises.
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 :
                 : "l"(base + off), "r"(static_cast<uint32_t>(sz))
                 : "memory");
#else
    for (int64_t o = 0; o + 16 <= sz; o += 16) {
      uint32_t a, b, c, d;
      asm volatile("ld.global.L2::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                   : "l"(base + off + o)
                   : "memory");
    }
#endif
  }
}

template <bool UseRope, bool CacheFP32>
__global__ void qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total_warps = num_tokens * 34;
  if (warp_id >= total_warps) return;
  const int lane = threadIdx.x & 31;
  const int t = warp_id / 34;
  const int h = warp_id - t * 34;

  const uint2 raw = *reinterpret_cast<const uint2*>(
      qkv + static_cast<size_t>(t) * 4608 + h * 128 + lane * 4);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    load_cos_sin<CacheFP32>(positions, cos_sin_cache, t, lane, cache_rows,
                            c0, c1, c2, c3, s0, s1, s2, s3);
  }
  const uint2 out_v = norm_rope_pack<UseRope>(raw, h, lane, eps, qscale,
                                              c0, c1, c2, c3, s0, s1, s2, s3);
  if (h < 32) {
    *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
  } else {
    *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(t) * 256 + (h - 32) * 128 + lane * 4) = out_v;
  }
}

template <bool UseRope, bool CacheFP32, int HPW, bool OneTokenCTA = false>
__global__ void qk_norm_rope_multi_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const int lane = threadIdx.x & 31;
  int t, h0;
  if constexpr (OneTokenCTA) {
    const int w = threadIdx.x >> 5;
    if (w >= kGroups) return;
    t = blockIdx.x;
    h0 = w * HPW;
  } else {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_tokens * kGroups) return;
    t = (num_tokens == 1) ? 0 : (warp_id / kGroups);
    h0 = (warp_id - t * kGroups) * HPW;
  }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    load_cos_sin<CacheFP32>(positions, cos_sin_cache, t, lane, cache_rows,
                            c0, c1, c2, c3, s0, s1, s2, s3);
  }
  uint2 raw[HPW];
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    if (h0 + j < 34)
      raw[j] = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(t) * 4608 + (h0 + j) * 128 + lane * 4);
  }
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const int h = h0 + j;
    if (h >= 34) break;
    const uint2 out_v = norm_rope_pack<UseRope>(raw[j], h, lane, eps, qscale,
                                                c0, c1, c2, c3, s0, s1, s2, s3);
    if (h < 32) {
      *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
    } else {
      *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(t) * 256 + (h - 32) * 128 + lane * 4) = out_v;
    }
  }
}

template <bool UseRope, bool CacheFP32, bool Block16>
__global__ void qk_norm_rope_cache_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_slot_tokens,
    int cache_rows,
    int block_size,
    int64_t stride_blk,
    int64_t stride_slot,
    int64_t stride_head,
    float eps,
    float qscale) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total_warps = num_tokens * 36;
  if (warp_id >= total_warps) return;
  const int lane = threadIdx.x & 31;
  const int t = warp_id / 36;
  const int h = warp_id - t * 36;

  const uint2 raw = *reinterpret_cast<const uint2*>(
      qkv + static_cast<size_t>(t) * 4608 + h * 128 + lane * 4);
  if (h >= 34) {
    const int64_t kv_off = compute_kv_slot_offset<Block16>(
        slot_mapping, t, num_slot_tokens, block_size, stride_blk, stride_slot);
    if (kv_off >= 0) {
      *reinterpret_cast<uint2*>(value_cache + kv_off + (h - 34) * stride_head + lane * 4) = raw;
    }
    return;
  }

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    load_cos_sin<CacheFP32>(positions, cos_sin_cache, t, lane, cache_rows,
                            c0, c1, c2, c3, s0, s1, s2, s3);
  }
  const uint2 out_v = norm_rope_pack<UseRope>(raw, h, lane, eps, qscale,
                                              c0, c1, c2, c3, s0, s1, s2, s3);
  if (h < 32) {
    *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
  } else {
    const int64_t kv_off = compute_kv_slot_offset<Block16>(
        slot_mapping, t, num_slot_tokens, block_size, stride_blk, stride_slot);
    if (kv_off >= 0) {
      *reinterpret_cast<uint2*>(key_cache + kv_off + (h - 32) * stride_head + lane * 4) = out_v;
    }
  }
}

template <bool UseRope, bool CacheFP32, bool Block16, int HPW, bool OneTokenCTA = false>
__global__ void qk_norm_rope_cache_multi_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_slot_tokens,
    int cache_rows,
    int block_size,
    int64_t stride_blk,
    int64_t stride_slot,
    int64_t stride_head,
    float eps,
    float qscale) {
  constexpr int kGroups = (36 + HPW - 1) / HPW;
  const int lane = threadIdx.x & 31;
  int t, h0;
  if constexpr (OneTokenCTA) {
    const int w = threadIdx.x >> 5;
    if (w >= kGroups) return;
    t = blockIdx.x;
    h0 = w * HPW;
  } else {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_tokens * kGroups) return;
    t = (num_tokens == 1) ? 0 : (warp_id / kGroups);
    h0 = (warp_id - t * kGroups) * HPW;
  }

  int64_t kv_off = -1;
  if (h0 + HPW > 32) {
    kv_off = compute_kv_slot_offset<Block16>(
        slot_mapping, t, num_slot_tokens, block_size, stride_blk, stride_slot);
  }

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    if (h0 < 34) {
      load_cos_sin<CacheFP32>(positions, cos_sin_cache, t, lane, cache_rows,
                              c0, c1, c2, c3, s0, s1, s2, s3);
    }
  }

  uint2 raw[HPW];
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    if (h0 + j < 36)
      raw[j] = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(t) * 4608 + (h0 + j) * 128 + lane * 4);
  }

#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const int h = h0 + j;
    if (h >= 36) break;
    if (h >= 34) {
      if (kv_off >= 0) {
        *reinterpret_cast<uint2*>(value_cache + kv_off + (h - 34) * stride_head + lane * 4) = raw[j];
      }
    } else {
      const uint2 out_v = norm_rope_pack<UseRope>(raw[j], h, lane, eps, qscale,
                                                  c0, c1, c2, c3, s0, s1, s2, s3);
      if (h < 32) {
        *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
      } else if (kv_off >= 0) {
        *reinterpret_cast<uint2*>(key_cache + kv_off + (h - 32) * stride_head + lane * 4) = out_v;
      }
    }
  }
}

template <int HPW>
int launch_multi(const void* qkv, const void* positions, const void* cos_sin_cache, void* q_out, void* k_out,
                 int num_tokens, int cache_rows, float eps, float qscale, int use_rope, int cache_fp32,
                 cudaStream_t stream) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* ko = static_cast<__nv_bfloat16*>(k_out);
  if (num_tokens > 1 && kGroups >= 2 && kGroups * 32 <= 576) {
    const int threads = kGroups * 32;
    const int blocks = num_tokens;
    if (use_rope) {
      if (cache_fp32)
        qk_norm_rope_multi_kernel<true, true, HPW, true><<<blocks, threads, 0, stream>>>(
            q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
      else
        qk_norm_rope_multi_kernel<true, false, HPW, true><<<blocks, threads, 0, stream>>>(
            q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
    } else {
      qk_norm_rope_multi_kernel<false, true, HPW, true><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
    }
    return static_cast<int>(cudaGetLastError());
  }
  const int threads = (HPW == 2 && num_tokens == 1) ? 288 : 128;
  const int wpb = threads >> 5;
  const int blocks = (num_tokens * kGroups + wpb - 1) / wpb;
  if (use_rope) {
    if (cache_fp32)
      qk_norm_rope_multi_kernel<true, true, HPW, false><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
    else
      qk_norm_rope_multi_kernel<true, false, HPW, false><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
  } else {
    qk_norm_rope_multi_kernel<false, true, HPW, false><<<blocks, threads, 0, stream>>>(
        q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

template <bool UseRope, bool CacheFP32, bool Block16>
void launch_cache_1(
    const __nv_bfloat16* q, const int64_t* pos, const void* cache,
    __nv_bfloat16* qo, __nv_bfloat16* kc, __nv_bfloat16* vc, const int64_t* slots,
    int num_tokens, int num_slot_tokens, int cache_rows, int block_size,
    int64_t s_blk, int64_t s_slot, int64_t s_head,
    float eps, float qscale, int blocks, int threads, cudaStream_t stream) {
  qk_norm_rope_cache_kernel<UseRope, CacheFP32, Block16><<<blocks, threads, 0, stream>>>(
      q, pos, cache, qo, kc, vc, slots,
      num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
}

template <int HPW, bool Block16>
int launch_cache_multi(
    const void* qkv, const void* positions, const void* cos_sin_cache,
    void* q_out, void* key_cache, void* value_cache, const void* slot_mapping,
    int num_tokens, int num_slot_tokens, int cache_rows, int block_size,
    int64_t s_blk, int64_t s_slot, int64_t s_head,
    float eps, float qscale, int use_rope, int cache_fp32, cudaStream_t stream) {
  constexpr int kGroups = (36 + HPW - 1) / HPW;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* kc = static_cast<__nv_bfloat16*>(key_cache);
  auto* vc = static_cast<__nv_bfloat16*>(value_cache);
  const auto* slots = static_cast<const int64_t*>(slot_mapping);
  if (num_tokens > 1 && kGroups >= 2 && kGroups * 32 <= 576) {
    const int threads = kGroups * 32;
    const int blocks = num_tokens;
    if (use_rope) {
      if (cache_fp32)
        qk_norm_rope_cache_multi_kernel<true, true, Block16, HPW, true><<<blocks, threads, 0, stream>>>(
            q, pos, cos_sin_cache, qo, kc, vc, slots,
            num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
      else
        qk_norm_rope_cache_multi_kernel<true, false, Block16, HPW, true><<<blocks, threads, 0, stream>>>(
            q, pos, cos_sin_cache, qo, kc, vc, slots,
            num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
    } else {
      qk_norm_rope_cache_multi_kernel<false, true, Block16, HPW, true><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, kc, vc, slots,
          num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
    }
    return static_cast<int>(cudaGetLastError());
  }
  const int threads = (HPW == 2 && num_tokens == 1) ? 288 : 128;
  const int wpb = threads >> 5;
  const int blocks = (num_tokens * kGroups + wpb - 1) / wpb;
  if (use_rope) {
    if (cache_fp32)
      qk_norm_rope_cache_multi_kernel<true, true, Block16, HPW, false><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, kc, vc, slots,
          num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
    else
      qk_norm_rope_cache_multi_kernel<true, false, Block16, HPW, false><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, kc, vc, slots,
          num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
  } else {
    qk_norm_rope_cache_multi_kernel<false, true, Block16, HPW, false><<<blocks, threads, 0, stream>>>(
        q, pos, cos_sin_cache, qo, kc, vc, slots,
        num_tokens, num_slot_tokens, cache_rows, block_size, s_blk, s_slot, s_head, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace muse_qk_glue

extern "C" int muse_kv_prefetch(
    const void* key_cache,
    const void* value_cache,
    const void* block_table,
    int bt_stride,
    const void* seqused_k,
    int num_seqs,
    int block_size,
    int max_blocks,
    int num_cache_blocks,
    int window,
    int parts,
    int64_t stride_blk,
    int64_t span_elems,
    int elem_bytes,
    cudaStream_t stream) {
  // Read-only: any layout we do not recognise is a silent no-op, never an error.
  if (num_seqs <= 0 || max_blocks <= 0 || block_size <= 0 || num_cache_blocks <= 0) return 0;
  if (parts != 1 && parts != 2) return 0;
  if (elem_bytes != 1 && elem_bytes != 2) return 0;
  if ((span_elems * elem_bytes) % 16 != 0) return 0;
  if (stride_blk * elem_bytes % 16 != 0) return 0;
  const int blocks = num_seqs * parts * max_blocks;
  constexpr int kThreads = 128;
  muse_qk_glue::kv_window_prefetch_kernel<kThreads><<<blocks, kThreads, 0, stream>>>(
      key_cache, value_cache, static_cast<const int32_t*>(block_table), bt_stride,
      static_cast<const int32_t*>(seqused_k), num_seqs, block_size, max_blocks, num_cache_blocks,
      window, parts, stride_blk, span_elems, elem_bytes);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_qk_norm_rope_multi(
    const void* qkv, const void* positions, const void* cos_sin_cache, void* q_out, void* k_out,
    int num_tokens, int cache_rows, float eps, float qscale, int use_rope, int cache_fp32, int hpw,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  switch (hpw) {
    case 2: return muse_qk_glue::launch_multi<2>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 3: return muse_qk_glue::launch_multi<3>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 4: return muse_qk_glue::launch_multi<4>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 6: return muse_qk_glue::launch_multi<6>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 9: return muse_qk_glue::launch_multi<9>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 17: return muse_qk_glue::launch_multi<17>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 34: return muse_qk_glue::launch_multi<34>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    default: return 1;
  }
}

extern "C" int muse_qk_norm_rope(
    const void* qkv,
    const void* positions,
    const void* cos_sin_cache,
    void* q_out,
    void* k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale,
    int use_rope,
    int cache_fp32,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  const int total_warps = num_tokens * 34;
  const int threads = (num_tokens == 1) ? 32 : 256;
  const int wpb = threads >> 5;
  const int blocks = (total_warps + wpb - 1) / wpb;
  if (use_rope) {
    if (cache_fp32) {
      muse_qk_glue::qk_norm_rope_kernel<true, true><<<blocks, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(qkv),
          static_cast<const int64_t*>(positions),
          cos_sin_cache,
          static_cast<__nv_bfloat16*>(q_out),
          static_cast<__nv_bfloat16*>(k_out),
          num_tokens, cache_rows, eps, qscale);
    } else {
      muse_qk_glue::qk_norm_rope_kernel<true, false><<<blocks, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(qkv),
          static_cast<const int64_t*>(positions),
          cos_sin_cache,
          static_cast<__nv_bfloat16*>(q_out),
          static_cast<__nv_bfloat16*>(k_out),
          num_tokens, cache_rows, eps, qscale);
    }
  } else {
    muse_qk_glue::qk_norm_rope_kernel<false, true><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const int64_t*>(positions),
        cos_sin_cache,
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_out),
        num_tokens, cache_rows, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_qk_norm_rope_cache(
    const void* qkv,
    const void* positions,
    const void* cos_sin_cache,
    void* q_out,
    void* key_cache,
    void* value_cache,
    const void* slot_mapping,
    int num_tokens,
    int num_slot_tokens,
    int cache_rows,
    int block_size,
    int64_t stride_blk,
    int64_t stride_slot,
    int64_t stride_head,
    float eps,
    float qscale,
    int use_rope,
    int cache_fp32,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  const int total_warps = num_tokens * 36;
  const int threads = (num_tokens == 1) ? 32 : 256;
  const int wpb = threads >> 5;
  const int blocks = (total_warps + wpb - 1) / wpb;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* kc = static_cast<__nv_bfloat16*>(key_cache);
  auto* vc = static_cast<__nv_bfloat16*>(value_cache);
  const auto* slots = static_cast<const int64_t*>(slot_mapping);
  const bool b16 = (block_size == 16);
  if (use_rope) {
    if (cache_fp32) {
      if (b16)
        muse_qk_glue::launch_cache_1<true, true, true>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
      else
        muse_qk_glue::launch_cache_1<true, true, false>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
    } else {
      if (b16)
        muse_qk_glue::launch_cache_1<true, false, true>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
      else
        muse_qk_glue::launch_cache_1<true, false, false>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
    }
  } else {
    if (b16)
      muse_qk_glue::launch_cache_1<false, true, true>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
    else
      muse_qk_glue::launch_cache_1<false, true, false>(q, pos, cos_sin_cache, qo, kc, vc, slots, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, blocks, threads, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_qk_norm_rope_cache_multi(
    const void* qkv,
    const void* positions,
    const void* cos_sin_cache,
    void* q_out,
    void* key_cache,
    void* value_cache,
    const void* slot_mapping,
    int num_tokens,
    int num_slot_tokens,
    int cache_rows,
    int block_size,
    int64_t stride_blk,
    int64_t stride_slot,
    int64_t stride_head,
    float eps,
    float qscale,
    int use_rope,
    int cache_fp32,
    int hpw,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  const bool b16 = (block_size == 16);
  switch (hpw) {
    case 2:
      return b16
          ? muse_qk_glue::launch_cache_multi<2, true>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream)
          : muse_qk_glue::launch_cache_multi<2, false>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream);
    case 3:
      return b16
          ? muse_qk_glue::launch_cache_multi<3, true>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream)
          : muse_qk_glue::launch_cache_multi<3, false>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream);
    case 4:
      return b16
          ? muse_qk_glue::launch_cache_multi<4, true>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream)
          : muse_qk_glue::launch_cache_multi<4, false>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream);
    case 6:
      return b16
          ? muse_qk_glue::launch_cache_multi<6, true>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream)
          : muse_qk_glue::launch_cache_multi<6, false>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream);
    case 9:
      return b16
          ? muse_qk_glue::launch_cache_multi<9, true>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream)
          : muse_qk_glue::launch_cache_multi<9, false>(qkv, positions, cos_sin_cache, q_out, key_cache, value_cache, slot_mapping, num_tokens, num_slot_tokens, cache_rows, block_size, stride_blk, stride_slot, stride_head, eps, qscale, use_rope, cache_fp32, stream);
    default:
      return 1;
  }
}
