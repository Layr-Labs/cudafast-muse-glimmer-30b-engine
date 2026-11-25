// SPDX-License-Identifier: Apache-2.0
// prefill3: the token embedding + weightless embedding RMSNorm of MuseGlimmerModel.embed_input_ids in two passes
// instead of eight eager kernels (index copy, gather, to-fp32 copy, pow, mean, add, rsqrt, mul, to-bf16 copy, and
// the copy into inputs_embeds). The reduction stays ATen's own mean kernel on the same fp32 tensor:
//   square: x2[r, c] = x * x with x = float(table[ids[r], c])        (ATen pow(x, 2) on float is base * base)
//   [ATen]  inv = rsqrt(mean(x2, -1, keepdim=True) + eps)              (unchanged ops)
//   scale : out[r, c] = bf16_rn(x * inv[r])                            (ATen mul on float, then type_as bf16)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace {
constexpr int kThreads = 128;

static bool decode_tail_pdl_enabled() {
  static const bool on = []() {
    const char* env = std::getenv("MUSE_DECODE_TAIL_PDL");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return on;
}

template <typename... KArgs, typename... Args>
cudaError_t launch_pdl(void (*kernel)(KArgs...), int grid, int block, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(block, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

template <typename Id>
__global__ void __launch_bounds__(kThreads) embed_square(const Id* __restrict__ ids, const uint4* __restrict__ table,
                                                         float4* __restrict__ x2, int vec_per_row) {
  const int row = blockIdx.y;
  const int v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= vec_per_row) return;
  const long long id = static_cast<long long>(ids[row]);
  const uint4 raw = table[id * vec_per_row + v];
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  float f[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
  float4* dst = x2 + (static_cast<size_t>(row) * vec_per_row + v) * 2;
  dst[0] = make_float4(__fmul_rn(f[0], f[0]), __fmul_rn(f[1], f[1]), __fmul_rn(f[2], f[2]), __fmul_rn(f[3], f[3]));
  dst[1] = make_float4(__fmul_rn(f[4], f[4]), __fmul_rn(f[5], f[5]), __fmul_rn(f[6], f[6]), __fmul_rn(f[7], f[7]));
}

__device__ __forceinline__ unsigned bf16_rne(float a) {
  // c10::BFloat16 round_to_nearest_even (finite inputs; NaN -> 0x7fc0 as c10)
  const unsigned u = __float_as_uint(a);
  if (a != a) return 0x7fc0u;
  return (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

template <typename Id>
__global__ void __launch_bounds__(kThreads) embed_scale(const Id* __restrict__ ids, const uint4* __restrict__ table,
                                                        const float* __restrict__ inv, uint4* __restrict__ out,
                                                        int vec_per_row) {
  const int row = blockIdx.y;
  const int v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= vec_per_row) return;
  const long long id = static_cast<long long>(ids[row]);
  const uint4 raw = table[id * vec_per_row + v];
  const float r = inv[row];
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  unsigned o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned lo = bf16_rne(__fmul_rn(__uint_as_float(w[i] << 16), r));
    const unsigned hi = bf16_rne(__fmul_rn(__uint_as_float(w[i] & 0xffff0000u), r));
    o[i] = lo | (hi << 16);
  }
  out[static_cast<size_t>(row) * vec_per_row + v] = make_uint4(o[0], o[1], o[2], o[3]);
}

// Single-CTA exact M=1 decode embedding + weightless RMSNorm (width <= 8192, width % 8 == 0).
// Matches ATen's gpu_reduce_kernel<float, MeanOps, vt0=4, input_vec_size=4> for rows == 1:
// block_width = 512, step_input = 512, 4 vectorized accumulators per thread combined in order
// (((a0 + a1) + a2) + a3), shared tree reduction offset 256 -> 32, warp shfl_down 16 -> 1,
// mean = sum * float(1.0 / width), inv = rsqrtf(mean + eps), then bf16_rne(x * inv).
template <typename Id, bool kPdl>
__global__ void __launch_bounds__(512) embed_norm_m1(
    const Id* __restrict__ ids, const uint2* __restrict__ table, float eps,
    uint2* __restrict__ out, int width) {
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  } else {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  __shared__ float smem[512];
  const int t = threadIdx.x;
  const int vec4 = width >> 2;
  const long long id = static_cast<long long>(ids[0]);
  const uint2* row_ptr = table + id * static_cast<size_t>(vec4);

  float f[4][4];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const uint2 raw = row_ptr[idx];
      const float x0 = __uint_as_float(raw.x << 16);
      const float x1 = __uint_as_float(raw.x & 0xffff0000u);
      const float x2 = __uint_as_float(raw.y << 16);
      const float x3 = __uint_as_float(raw.y & 0xffff0000u);
      f[it][0] = x0;
      f[it][1] = x1;
      f[it][2] = x2;
      f[it][3] = x3;
      acc0 = __fadd_rn(acc0, __fmul_rn(x0, x0));
      acc1 = __fadd_rn(acc1, __fmul_rn(x1, x1));
      acc2 = __fadd_rn(acc2, __fmul_rn(x2, x2));
      acc3 = __fadd_rn(acc3, __fmul_rn(x3, x3));
    }
  }
  float v = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
  smem[t] = v;
#pragma unroll
  for (int offset = 256; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (t < offset) {
      v = __fadd_rn(v, smem[t + offset]);
      smem[t] = v;
    }
  }
  __syncthreads();
  if (t < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (t == 0) {
      const float factor = static_cast<float>(1.0 / static_cast<double>(width));
      const float mean = __fmul_rn(v, factor);
      smem[0] = rsqrtf(__fadd_rn(mean, eps));
    }
  }
  __syncthreads();
  const float r = smem[0];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const unsigned lo0 = bf16_rne(__fmul_rn(f[it][0], r));
      const unsigned hi0 = bf16_rne(__fmul_rn(f[it][1], r));
      const unsigned lo1 = bf16_rne(__fmul_rn(f[it][2], r));
      const unsigned hi1 = bf16_rne(__fmul_rn(f[it][3], r));
      out[idx] = make_uint2(lo0 | (hi0 << 16), lo1 | (hi1 << 16));
    }
  }
}
// Exact 1-request C++ CUDA replacement for Triton _post_update_kernel (input_batch.py).
// Overlaps metadata preloads behind lm_head+lmpost via SM121 PDL, waits before reading
// sampled_tokens, and immediately triggers single_decode_prep via launch_dependents.
template <bool kPdl>
__global__ void __launch_bounds__(32) post_update_m1_kernel(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ num_computed_tokens,
    int64_t* __restrict__ last_sampled_tokens,
    int32_t* __restrict__ output_bin_counts,
    int64_t output_bin_counts_stride,
    const int64_t* __restrict__ sampled_tokens,
    const int32_t* __restrict__ num_sampled_ptr,
    const int32_t* __restrict__ num_rejected_ptr,
    const int32_t* __restrict__ query_start_loc,
    int32_t* __restrict__ all_token_ids,
    int64_t all_token_ids_stride,
    int32_t* __restrict__ total_len_ptr) {
  int32_t req_state_idx = -1;
  int32_t total_len = 0;
  int32_t num_sampled = 0;
  int32_t num_rejected = 0;
  int32_t query_len = 0;
  int32_t num_computed = 0;
  if constexpr (kPdl) {
    if (threadIdx.x == 0) {
      req_state_idx = idx_mapping[0];
      if (req_state_idx >= 0) {
        total_len = total_len_ptr[req_state_idx];
        num_sampled = num_sampled_ptr[0];
        num_rejected = num_rejected_ptr[0];
        if (query_start_loc != nullptr) {
          query_len = query_start_loc[1] - query_start_loc[0];
        }
        num_computed = num_computed_tokens[req_state_idx];
      }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if (threadIdx.x != 0) return;
  if constexpr (!kPdl) {
    req_state_idx = idx_mapping[0];
    if (req_state_idx < 0) return;
    total_len = total_len_ptr[req_state_idx];
    num_sampled = num_sampled_ptr[0];
    num_rejected = num_rejected_ptr[0];
    if (query_start_loc != nullptr) {
      query_len = query_start_loc[1] - query_start_loc[0];
    }
    num_computed = num_computed_tokens[req_state_idx];
  } else {
    if (req_state_idx < 0) return;
  }
  const int32_t computed_delta = query_len - num_rejected;
  if (num_sampled == 1) {
    const int64_t tok = sampled_tokens[0];
    int32_t* const bin_ptr =
        output_bin_counts ? (output_bin_counts + static_cast<int64_t>(req_state_idx) * output_bin_counts_stride + tok)
                          : nullptr;
    int32_t bin_val = 0;
    if (bin_ptr != nullptr) {
      bin_val = *bin_ptr;
    }
    last_sampled_tokens[req_state_idx] = tok;
    total_len_ptr[req_state_idx] = total_len + 1;
    if (computed_delta != 0) {
      num_computed_tokens[req_state_idx] = num_computed + computed_delta;
    }
    all_token_ids[static_cast<int64_t>(req_state_idx) * all_token_ids_stride + total_len] =
        static_cast<int32_t>(tok);
    if (bin_ptr != nullptr) {
      *bin_ptr = bin_val + 1;
    }
    return;
  }
  if (num_sampled > 0) {
    const int64_t last_tok = sampled_tokens[num_sampled - 1];
    last_sampled_tokens[req_state_idx] = last_tok;
    total_len_ptr[req_state_idx] = total_len + num_sampled;
    int32_t* const all_row = all_token_ids + static_cast<int64_t>(req_state_idx) * all_token_ids_stride + total_len;
    int32_t* const bin_row =
        output_bin_counts ? (output_bin_counts + static_cast<int64_t>(req_state_idx) * output_bin_counts_stride) : nullptr;
    for (int32_t i = 0; i < num_sampled; ++i) {
      const int64_t tok = sampled_tokens[i];
      all_row[i] = static_cast<int32_t>(tok);
      if (bin_row != nullptr) {
        bin_row[tok] = bin_row[tok] + 1;
      }
    }
  }
  if (computed_delta != 0) {
    num_computed_tokens[req_state_idx] = num_computed + computed_delta;
  }
}

// Exact 1-request C++ CUDA replacement for Triton _fused_single_decode_prep_kernel (block_table.py).
// Preloads static group pointers/strides/block_sizes/num_blocks before PDL wait, then copies
// block tables and computes input_ids/positions/seq_lens/slot_mappings and triggers embed_norm_m1.
template <bool kPdl>
__global__ void __launch_bounds__(128) single_decode_prep_kernel(
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ num_computed_tokens,
    const int64_t* __restrict__ last_sampled_tokens,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ pos,
    int32_t* __restrict__ seq_lens,
    const uint64_t* __restrict__ src_bt_ptrs,
    const uint64_t* __restrict__ dst_bt_ptrs,
    const int64_t* __restrict__ bt_strides,
    const int32_t* __restrict__ num_blocks,
    int64_t num_blocks_stride,
    const int32_t* __restrict__ block_sizes,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_mappings_stride) {
  const int gid = static_cast<int>(blockIdx.x);
  const int32_t req_idx = idx_mapping[0];
  const int64_t stride = bt_strides[gid];
  const int32_t nblk = num_blocks[static_cast<int64_t>(gid) * num_blocks_stride + req_idx];
  const int32_t* const src_ptr =
      reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(src_bt_ptrs[gid])) +
      static_cast<int64_t>(req_idx) * stride;
  int32_t* const dst_ptr = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(dst_bt_ptrs[gid]));
  const int32_t bs = block_sizes[gid];
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  const int32_t num_comp = num_computed_tokens[req_idx];
  if (gid == 0 && threadIdx.x == 0) {
    input_ids[0] = static_cast<int32_t>(last_sampled_tokens[req_idx]);
    pos[0] = static_cast<int64_t>(num_comp);
    seq_lens[0] = num_comp + 1;
  }
  for (int i = static_cast<int>(threadIdx.x); i < nblk; i += 128) {
    dst_ptr[i] = src_ptr[i];
  }
  if (threadIdx.x == 0) {
    const int32_t b_idx = num_comp / bs;
    const int32_t b_off = num_comp % bs;
    const int64_t b_num = static_cast<int64_t>(src_ptr[b_idx]);
    slot_mappings[static_cast<int64_t>(gid) * slot_mappings_stride] =
        b_num * static_cast<int64_t>(bs) + static_cast<int64_t>(b_off);
  }
}

// Multi-row (1..8) exact embedding + weightless RMSNorm (1 CTA per row, 512 threads, same reduction order as m1).
template <typename Id, bool kPdl>
__global__ void __launch_bounds__(512) embed_norm_m(
    const Id* __restrict__ ids, const uint2* __restrict__ table, float eps,
    uint2* __restrict__ out, int width) {
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  } else {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  __shared__ float smem[512];
  const int row = static_cast<int>(blockIdx.x);
  const int t = threadIdx.x;
  const int vec4 = width >> 2;
  const long long id = static_cast<long long>(ids[row]);
  const uint2* row_ptr = table + id * static_cast<size_t>(vec4);
  uint2* out_row = out + static_cast<size_t>(row) * static_cast<size_t>(vec4);

  float f[4][4];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const uint2 raw = row_ptr[idx];
      const float x0 = __uint_as_float(raw.x << 16);
      const float x1 = __uint_as_float(raw.x & 0xffff0000u);
      const float x2 = __uint_as_float(raw.y << 16);
      const float x3 = __uint_as_float(raw.y & 0xffff0000u);
      f[it][0] = x0;
      f[it][1] = x1;
      f[it][2] = x2;
      f[it][3] = x3;
      acc0 = __fadd_rn(acc0, __fmul_rn(x0, x0));
      acc1 = __fadd_rn(acc1, __fmul_rn(x1, x1));
      acc2 = __fadd_rn(acc2, __fmul_rn(x2, x2));
      acc3 = __fadd_rn(acc3, __fmul_rn(x3, x3));
    }
  }
  float v = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
  smem[t] = v;
#pragma unroll
  for (int offset = 256; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (t < offset) {
      v = __fadd_rn(v, smem[t + offset]);
      smem[t] = v;
    }
  }
  __syncthreads();
  if (t < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (t == 0) {
      const float factor = static_cast<float>(1.0 / static_cast<double>(width));
      const float mean = __fmul_rn(v, factor);
      smem[0] = rsqrtf(__fadd_rn(mean, eps));
    }
  }
  __syncthreads();
  const float r = smem[0];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = t + it * 512;
    if (idx < vec4) {
      const unsigned lo0 = bf16_rne(__fmul_rn(f[it][0], r));
      const unsigned hi0 = bf16_rne(__fmul_rn(f[it][1], r));
      const unsigned lo1 = bf16_rne(__fmul_rn(f[it][2], r));
      const unsigned hi1 = bf16_rne(__fmul_rn(f[it][3], r));
      out_row[idx] = make_uint2(lo0 | (hi0 << 16), lo1 | (hi1 << 16));
    }
  }
}

// Single-CTA exact 1-request greedy rejection + get_num_sampled_and_rejected.
template <bool kPdl>
__global__ void __launch_bounds__(32) greedy_reject_m_kernel(
    const int64_t* __restrict__ target_tokens,
    const int32_t* __restrict__ input_ids,
    const int64_t* __restrict__ logits_indices,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ prefill_len,
    int32_t* __restrict__ num_sampled_out,
    int32_t* __restrict__ num_rejected_out,
    int num_logits) {
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if (threadIdx.x != 0) return;
  const int32_t req_state_idx = idx_mapping[0];
  const int32_t seq_len = seq_lens[0];
  const int32_t plen = prefill_len[req_state_idx];
  if (seq_len < plen) {
    num_sampled_out[0] = 0;
    num_rejected_out[0] = 0;
    return;
  }
  int32_t accepted = 0;
  for (int i = 0; i < num_logits - 1; ++i) {
    const int64_t t = target_tokens[i];
    const int64_t idx = logits_indices ? logits_indices[i + 1] : static_cast<int64_t>(i + 1);
    const int64_t d = static_cast<int64_t>(input_ids[idx]);
    if (t == d) {
      accepted += 1;
    } else {
      break;
    }
  }
  const int32_t ns = accepted + 1;
  num_sampled_out[0] = ns;
  num_rejected_out[0] = num_logits - ns;
}
}  // namespace

extern "C" int muse_embed_norm_m1(const void* ids, int ids_64, const void* table, float eps,
                                  void* out, int width, cudaStream_t stream) {
  if (width <= 0 || (width & 7) != 0 || width > 8192) return cudaErrorInvalidValue;
  const bool use_pdl = decode_tail_pdl_enabled();
  if (use_pdl) {
    if (ids_64)
      return static_cast<int>(launch_pdl(
          embed_norm_m1<long long, true>, 1, 512, stream,
          static_cast<const long long*>(ids), static_cast<const uint2*>(table), eps,
          static_cast<uint2*>(out), width));
    return static_cast<int>(launch_pdl(
        embed_norm_m1<int, true>, 1, 512, stream,
        static_cast<const int*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width));
  }
  if (ids_64)
    embed_norm_m1<long long, false><<<1, 512, 0, stream>>>(
        static_cast<const long long*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  else
    embed_norm_m1<int, false><<<1, 512, 0, stream>>>(
        static_cast<const int*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_post_update_m1(
    const void* idx_mapping, void* num_computed_tokens, void* last_sampled_tokens,
    void* output_bin_counts, int64_t output_bin_counts_stride,
    const void* sampled_tokens, const void* num_sampled, const void* num_rejected,
    const void* query_start_loc, void* all_token_ids, int64_t all_token_ids_stride,
    void* total_len, cudaStream_t stream) {
  if (decode_tail_pdl_enabled()) {
    return static_cast<int>(launch_pdl(
        post_update_m1_kernel<true>, 1, 32, stream,
        static_cast<const int32_t*>(idx_mapping), static_cast<int32_t*>(num_computed_tokens),
        static_cast<int64_t*>(last_sampled_tokens), static_cast<int32_t*>(output_bin_counts),
        output_bin_counts_stride, static_cast<const int64_t*>(sampled_tokens),
        static_cast<const int32_t*>(num_sampled), static_cast<const int32_t*>(num_rejected),
        static_cast<const int32_t*>(query_start_loc), static_cast<int32_t*>(all_token_ids),
        all_token_ids_stride, static_cast<int32_t*>(total_len)));
  }
  post_update_m1_kernel<false><<<1, 32, 0, stream>>>(
      static_cast<const int32_t*>(idx_mapping), static_cast<int32_t*>(num_computed_tokens),
      static_cast<int64_t*>(last_sampled_tokens), static_cast<int32_t*>(output_bin_counts),
      output_bin_counts_stride, static_cast<const int64_t*>(sampled_tokens),
      static_cast<const int32_t*>(num_sampled), static_cast<const int32_t*>(num_rejected),
      static_cast<const int32_t*>(query_start_loc), static_cast<int32_t*>(all_token_ids),
      all_token_ids_stride, static_cast<int32_t*>(total_len));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_single_decode_prep(
    int num_groups, const void* idx_mapping, const void* num_computed_tokens,
    const void* last_sampled_tokens, void* input_ids, void* pos, void* seq_lens,
    const void* src_bt_ptrs, const void* dst_bt_ptrs, const void* bt_strides,
    const void* num_blocks, int64_t num_blocks_stride, const void* block_sizes,
    void* slot_mappings, int64_t slot_mappings_stride, cudaStream_t stream) {
  if (num_groups <= 0) return cudaErrorInvalidValue;
  if (decode_tail_pdl_enabled()) {
    return static_cast<int>(launch_pdl(
        single_decode_prep_kernel<true>, num_groups, 128, stream,
        static_cast<const int32_t*>(idx_mapping), static_cast<const int32_t*>(num_computed_tokens),
        static_cast<const int64_t*>(last_sampled_tokens), static_cast<int32_t*>(input_ids),
        static_cast<int64_t*>(pos), static_cast<int32_t*>(seq_lens),
        static_cast<const uint64_t*>(src_bt_ptrs), static_cast<const uint64_t*>(dst_bt_ptrs),
        static_cast<const int64_t*>(bt_strides), static_cast<const int32_t*>(num_blocks),
        num_blocks_stride, static_cast<const int32_t*>(block_sizes),
        static_cast<int64_t*>(slot_mappings), slot_mappings_stride));
  }
  single_decode_prep_kernel<false><<<num_groups, 128, 0, stream>>>(
      static_cast<const int32_t*>(idx_mapping), static_cast<const int32_t*>(num_computed_tokens),
      static_cast<const int64_t*>(last_sampled_tokens), static_cast<int32_t*>(input_ids),
      static_cast<int64_t*>(pos), static_cast<int32_t*>(seq_lens),
      static_cast<const uint64_t*>(src_bt_ptrs), static_cast<const uint64_t*>(dst_bt_ptrs),
      static_cast<const int64_t*>(bt_strides), static_cast<const int32_t*>(num_blocks),
      num_blocks_stride, static_cast<const int32_t*>(block_sizes),
      static_cast<int64_t*>(slot_mappings), slot_mappings_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_embed_norm_m(const void* ids, int ids_64, const void* table, float eps,
                                 void* out, int rows, int width, cudaStream_t stream) {
  if (rows <= 0 || rows > 16 || width <= 0 || (width & 7) != 0 || width > 8192) {
    return cudaErrorInvalidValue;
  }
  const bool use_pdl = decode_tail_pdl_enabled();
  if (use_pdl) {
    if (ids_64)
      return static_cast<int>(launch_pdl(
          embed_norm_m<long long, true>, rows, 512, stream,
          static_cast<const long long*>(ids), static_cast<const uint2*>(table), eps,
          static_cast<uint2*>(out), width));
    return static_cast<int>(launch_pdl(
        embed_norm_m<int, true>, rows, 512, stream,
        static_cast<const int*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width));
  }
  if (ids_64)
    embed_norm_m<long long, false><<<rows, 512, 0, stream>>>(
        static_cast<const long long*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  else
    embed_norm_m<int, false><<<rows, 512, 0, stream>>>(
        static_cast<const int*>(ids), static_cast<const uint2*>(table), eps,
        static_cast<uint2*>(out), width);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_greedy_reject_m(
    const void* target_tokens, const void* input_ids, const void* logits_indices,
    const void* seq_lens, const void* idx_mapping, const void* prefill_len,
    void* num_sampled_out, void* num_rejected_out, int num_logits, cudaStream_t stream) {
  if (!target_tokens || !input_ids || !seq_lens || !idx_mapping || !prefill_len ||
      !num_sampled_out || !num_rejected_out || num_logits < 2 || num_logits > 16) {
    return cudaErrorInvalidValue;
  }
  if (decode_tail_pdl_enabled()) {
    return static_cast<int>(launch_pdl(
        greedy_reject_m_kernel<true>, 1, 32, stream,
        static_cast<const int64_t*>(target_tokens), static_cast<const int32_t*>(input_ids),
        static_cast<const int64_t*>(logits_indices), static_cast<const int32_t*>(seq_lens),
        static_cast<const int32_t*>(idx_mapping), static_cast<const int32_t*>(prefill_len),
        static_cast<int32_t*>(num_sampled_out), static_cast<int32_t*>(num_rejected_out),
        num_logits));
  }
  greedy_reject_m_kernel<false><<<1, 32, 0, stream>>>(
      static_cast<const int64_t*>(target_tokens), static_cast<const int32_t*>(input_ids),
      static_cast<const int64_t*>(logits_indices), static_cast<const int32_t*>(seq_lens),
      static_cast<const int32_t*>(idx_mapping), static_cast<const int32_t*>(prefill_len),
      static_cast<int32_t*>(num_sampled_out), static_cast<int32_t*>(num_rejected_out),
      num_logits);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_embed_square(const void* ids, int ids_64, const void* table, void* x2, int rows, int width,
                                 cudaStream_t stream) {
  if (rows <= 0 || width % 8) return cudaErrorInvalidValue;
  const int vec = width / 8;
  dim3 grid((vec + kThreads - 1) / kThreads, rows);
  if (ids_64)
    embed_square<long long><<<grid, kThreads, 0, stream>>>(static_cast<const long long*>(ids),
                                                            static_cast<const uint4*>(table),
                                                            static_cast<float4*>(x2), vec);
  else
    embed_square<int><<<grid, kThreads, 0, stream>>>(static_cast<const int*>(ids), static_cast<const uint4*>(table),
                                                      static_cast<float4*>(x2), vec);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_embed_scale(const void* ids, int ids_64, const void* table, const void* inv, void* out, int rows,
                                int width, cudaStream_t stream) {
  if (rows <= 0 || width % 8) return cudaErrorInvalidValue;
  const int vec = width / 8;
  dim3 grid((vec + kThreads - 1) / kThreads, rows);
  if (ids_64)
    embed_scale<long long><<<grid, kThreads, 0, stream>>>(static_cast<const long long*>(ids),
                                                           static_cast<const uint4*>(table),
                                                           static_cast<const float*>(inv), static_cast<uint4*>(out),
                                                           vec);
  else
    embed_scale<int><<<grid, kThreads, 0, stream>>>(static_cast<const int*>(ids), static_cast<const uint4*>(table),
                                                     static_cast<const float*>(inv), static_cast<uint4*>(out), vec);
  return static_cast<int>(cudaGetLastError());
}
