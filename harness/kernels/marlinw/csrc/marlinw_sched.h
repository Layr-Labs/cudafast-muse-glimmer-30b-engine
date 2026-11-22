// MarlinS: the Marlin W4A16 NVFP4 main loop (bf16 activations, fe2m1f codes,
// fe4m3fn group-16 scales, fp32 global scale, fp32 global reduce) of
// marlinw_template.h, restricted to that one type combination, with a
// configurable CTA shape and pipeline:
//
//   threads, thread_n_blocks  CTA tile of 64 rows x 16 * thread_n_blocks
//                             columns, run by 2 warp rows x tb_n_warps warps;
//   warp_j                    16-column groups per warp: 4 (a warp computes
//                             64 x 64 outputs, as Marlin) or 2 (64 x 32);
//   stages                    depth of the cp.async global->shared pipeline;
//   min_blocks                CTAs per SM the register budget is set for;
//   dq_ahead                  dequantize and scale the codes of the next k16
//                             step before the mma of the current one.
//
// What every output element computes does not depend on any of these. As in
// Marlin, a k-tile is 64 deep; in every k-tile warp row 0 issues
// mma.sync.m16n8k16.f32.bf16 for k16 blocks 0 and 1 and warp row 1 for blocks
// 2 and 3, each into its own fp32 accumulator, k-tiles in ascending order from
// +0. The B operand of every mma is the same dequant() of the same code word
// times the same dequant_fp8_scales() scale (__hmul2). After the K range of a
// slice the two warp rows' accumulators are added once in fp32
// (thread_block_reduce), slices of a k-split tile are folded in fp32 in
// slice_idx order through C_tmp (global_reduce_fp32), and the epilogue stores
// bf16_rn(acc * global_scale). The narrow warp tile only selects half of a
// thread's Marlin code word and scale word (the 32 columns that the two
// 16-column groups of the warp cover); the code and scale layouts in global and
// shared memory are Marlin's.
//
// Requires marlinw_template.h (ldsm, scale, barrier_acquire/release) and
// silu_mul.cuh to be included first, in namespace marlinw.

#pragma once

namespace marlinw {

template <const int threads, const int thread_n_blocks, const int warp_j,
          const int stages, const int min_blocks, const bool dq_ahead,
          const bool silu_mul>
__global__ void __launch_bounds__(threads, min_blocks)
    MarlinS(const int4* __restrict__ A0, const int4* __restrict__ B,
            int4* __restrict__ C0, int4* __restrict__ C_tmp,
            const int4* __restrict__ scales_ptr,
            const float* __restrict__ global_scale_ptr, int prob_m, int prob_n,
            int prob_k, int lda, int* locks, const int4* __restrict__ work,
            const int* __restrict__ work_ptr) {
  using Dt = MarlinScalarType<vllm::kBFloat16.id()>;
  using scalar_t2 = typename Dt::scalar_t2;
  using FragA = typename Dt::FragA;
  using FragB = typename Dt::FragB;
  using FragC = typename Dt::FragC;
  using FragS = typename Dt::FragS;
  constexpr auto a_id = vllm::kBFloat16.id();

  constexpr int thread_m_blocks = 4;  // 64-row chunks
  constexpr int thread_k_blocks = 4;  // 64-deep k-tiles
  constexpr int k_steps = 2;          // k16 steps of a warp per k-tile
  constexpr int tb_n_warps = thread_n_blocks / warp_j;
  static_assert(warp_j == 2 || warp_j == 4, "warp_j");
  static_assert(thread_n_blocks % warp_j == 0, "thread_n_blocks");
  static_assert(threads == 64 * tb_n_warps, "2 warp rows x tb_n_warps warps");
  static_assert(stages >= 3, "stages");

  const float global_scale_f32 = global_scale_ptr[0];
  extern __shared__ int4 sh[];

  if (prob_m > 16 * thread_m_blocks) prob_m = 16 * thread_m_blocks;
  const int c_row_n = silu_mul ? prob_n / 2 : prob_n;

  int slice_row = 0, slice_col = 0, slice_iters = 0, slice_count = 1;
  int slice_idx = 0, par_id = 0, locks_off = 0;
  int work_cur = work_ptr[blockIdx.x];
  const int work_end = work_ptr[blockIdx.x + 1];
  const int4* A = A0;
  int4* C = C0;

  auto init_slice = [&]() {
    if (work_cur >= work_end) {
      slice_iters = 0;
      return;
    }
    int4 w0 = work[2 * work_cur];
    int4 w1 = work[2 * work_cur + 1];
    work_cur++;
    par_id = w0.x;
    slice_col = w0.y;
    slice_row = w0.z;
    slice_iters = w0.w;
    slice_idx = w1.x;
    slice_count = w1.y;
    locks_off = w1.z;
    A = A0 + 2 * thread_m_blocks * par_id * lda;
    C = C0 + 2 * thread_m_blocks * par_id * c_row_n;
  };
  init_slice();

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int warp_row = warp_id / tb_n_warps;
  const int wn = warp_id % tb_n_warps;

  // A: 64 rows x 64 bf16 per stage (8 int4 per row), XOR-swizzled as Marlin.
  const int a_gl_stride = lda / 8;
  constexpr int a_sh_stride = 2 * thread_k_blocks;
  constexpr int a_gl_rd_delta_o = a_sh_stride;
  constexpr int a_sh_stage = a_sh_stride * 16 * thread_m_blocks;
  constexpr int a_sh_wr_iters = a_sh_stage / threads;
  static_assert(a_sh_stage % threads == 0, "A stage");
  const int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o) + a_gl_rd_delta_o * slice_row;
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return i ^ (row % 8);
  };
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(threads * i + threadIdx.x);
  // ldmatrix addresses: m-block j of k16 step k is a_sh_rd_trans[k] + 128 j
  // (128 = 16 rows; the swizzle depends on the row modulo 8 only).
  const int a_sh_rd = a_sh_stride * (lane % 16) + lane / 16 + 4 * warp_row;
  int a_sh_rd_trans[k_steps];
#pragma unroll
  for (int k = 0; k < k_steps; k++) a_sh_rd_trans[k] = transform_a(2 * k + a_sh_rd);

  // B: Marlin-packed codes, 16 * 16 * thread_n_blocks / 8 bytes per k16 row of
  // the tile; a 64-column group is 32 int4, one per lane (4 words, one per
  // 16-column group).
  const int b_gl_stride = prob_n / 2;
  constexpr int b_sh_stride = 8 * thread_n_blocks;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / threads;
  static_assert(b_sh_stage % threads == 0 && threads % b_sh_stride == 0, "B stage");
  const int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  const int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  const int b_gl_thr =
      b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  int b_gl_rd = b_gl_thr + b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;

  // Scales: fe4m3fn, one row of 16 * thread_n_blocks per k16 group.
  const int s_gl_stride = prob_n / 16;
  constexpr int s_sh_stride = thread_n_blocks;
  constexpr int s_sh_stage = thread_k_blocks * s_sh_stride;
  static_assert(s_sh_stage <= threads, "S stage");
  const bool s_sh_wr_pred = threadIdx.x < s_sh_stage;
  const int s_gl_thr = s_gl_stride * (threadIdx.x / s_sh_stride) +
                       (threadIdx.x % s_sh_stride);
  int s_gl_rd = s_gl_thr + s_sh_stride * slice_col +
                s_gl_stride * thread_k_blocks * slice_row;

  // Shared memory: [B stages | reduce / output staging] [scales] [A stages].
  constexpr int sh_red_size = 16 * thread_m_blocks * (2 * thread_n_blocks + 1);
  constexpr int sh_tbr_size = warp_j * threads;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int sh_r0 = sh_b_size > sh_red_size
                            ? (sh_b_size > sh_tbr_size ? sh_b_size : sh_tbr_size)
                            : (sh_red_size > sh_tbr_size ? sh_red_size : sh_tbr_size);
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + sh_r0;
  int4* sh_a = sh_s + stages * s_sh_stage;

  FragA frag_a[2][thread_m_blocks];
  int frag_bq[2][warp_j];
  int frag_sq[2][warp_j / 2];
  FragC frag_c[thread_m_blocks][warp_j][2];
  FragB frag_bd[dq_ahead ? 2 : 1][warp_j][2];

  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * warp_j * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++)
        cp_async4(&sh_a_stage[a_sh_wr_trans[i]],
                  &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off]);
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++)
        cp_async4(&sh_b_stage[threads * i + threadIdx.x],
                  &B[b_gl_rd + b_gl_rd_delta_i * i]);
      b_gl_rd += b_gl_rd_delta_o;
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      if (s_sh_wr_pred) cp_async4(&sh_s_stage[threadIdx.x], &scales_ptr[s_gl_rd]);
      s_gl_rd += s_gl_stride * thread_k_blocks;
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Operands of k16 step k (warp row r: k16 block 2 r + k % 2 of the k-tile).
  auto fetch_to_registers = [&](int k, int pipe) {
    const int k2 = k % 2;
    const int row = 2 * warp_row + k % k_steps;
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<4, a_id>(frag_a[k2][i],
                    &sh_a_stage[a_sh_rd_trans[k % k_steps] + 128 * i]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    int4* sh_s_stage = sh_s + s_sh_stage * pipe;
    if constexpr (warp_j == 4) {
      int4 q = sh_b_stage[b_sh_stride * row + 32 * wn + lane];
      frag_bq[k2][0] = q.x;
      frag_bq[k2][1] = q.y;
      frag_bq[k2][2] = q.z;
      frag_bq[k2][3] = q.w;
      int2 s = reinterpret_cast<int2*>(
          sh_s_stage)[2 * s_sh_stride * row + 8 * wn + lane / 4];
      frag_sq[k2][0] = s.x;
      frag_sq[k2][1] = s.y;
    } else {
      int2 q = reinterpret_cast<int2*>(
          sh_b_stage)[2 * (b_sh_stride * row + 32 * (wn / 2) + lane) + wn % 2];
      frag_bq[k2][0] = q.x;
      frag_bq[k2][1] = q.y;
      frag_sq[k2][0] = reinterpret_cast<int*>(
          sh_s_stage)[2 * (2 * s_sh_stride * row + 8 * (wn / 2) + lane / 4) +
                      wn % 2];
    }
  };

  // B operand of 16-column group j from code word frag_bq[k2][j] and the
  // scales fs: dequant<bf16, fe2m1f> of the word shifted by 8 (first n8 half)
  // and of the word itself, each times its scale (Marlin's matmul()).
  auto dequant_one = [&](int k2, int j, FragS* fs, FragB& b0, FragB& b1) {
    int b_quant_1 = frag_bq[k2][j];
    int b_quant_0 = b_quant_1 << 8;
    dequant<scalar_t2, vllm::kFE2M1f.id(), true>(
        b_quant_0, reinterpret_cast<scalar_t2*>(&b0));
    dequant<scalar_t2, vllm::kFE2M1f.id(), true>(
        b_quant_1, reinterpret_cast<scalar_t2*>(&b1));
    scale<a_id>(b0, fs[j], 0);
    scale<a_id>(b1, fs[j], 1);
  };
  auto dequant_scales = [&](int k2, FragS* fs) {
#pragma unroll
    for (int h = 0; h < warp_j / 2; h++)
      dequant_fp8_scales<scalar_t2, vllm::kFE4M3fn.id()>(
          frag_sq[k2][h], reinterpret_cast<scalar_t2*>(&fs[2 * h]));
  };
  auto mma_one = [&](int k2, int j, FragB& b0, FragB& b1) {
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      mma<a_id, false>(frag_a[k2][i], b0, frag_c[i][j][0]);
      mma<a_id, false>(frag_a[k2][i], b1, frag_c[i][j][1]);
    }
  };
  // Dequantized operands of step k into frag_bd[k % 2] (dq_ahead only).
  auto dequant_step = [&](int k) {
    const int k2 = k % 2;
    FragS fs[warp_j];
    dequant_scales(k2, fs);
#pragma unroll
    for (int j = 0; j < warp_j; j++)
      dequant_one(k2, j, fs, frag_bd[dq_ahead ? k2 : 0][j][0],
                  frag_bd[dq_ahead ? k2 : 0][j][1]);
  };

  auto matmul = [&](int k) {
    const int k2 = k % 2;
    if constexpr (dq_ahead) {
#pragma unroll
      for (int j = 0; j < warp_j; j++)
        mma_one(k2, j, frag_bd[k2][j][0], frag_bd[k2][j][1]);
      dequant_step(k + 1);
    } else {
      FragS fs[warp_j];
      dequant_scales(k2, fs);
#pragma unroll
      for (int j = 0; j < warp_j; j++) {
        FragB b0, b1;
        dequant_one(k2, j, fs, b0, b1);
        mma_one(k2, j, b0, b1);
      }
    }
  };

  // Warp row 1 hands its accumulators to warp row 0 through shared memory, one
  // m-block at a time; row 0 adds them (P = A0 + A1, one fp32 add).
  auto thread_block_reduce = [&]() {
    constexpr int half = threads / 2;
    const int t = threadIdx.x % half;
#pragma unroll
    for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
      if (warp_row == 1) {
#pragma unroll
        for (int j = 0; j < warp_j; j++)
#pragma unroll
          for (int h = 0; h < 2; h++)
            sh_red[(2 * j + h) * half + t] =
                *reinterpret_cast<int4*>(&frag_c[m_block][j][h]);
      }
      __syncthreads();
      if (warp_row == 0) {
#pragma unroll
        for (int j = 0; j < warp_j; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            float* c_rd = reinterpret_cast<float*>(&sh_red[(2 * j + h) * half + t]);
#pragma unroll
            for (int f = 0; f < 4; f++) frag_c[m_block][j][h][f] += c_rd[f];
          }
      }
      __syncthreads();
    }
  };

  // Slices of a k-split tile meet in C_tmp in slice_idx order (fp32).
  auto global_reduce_fp32 = [&](bool first, bool last) {
    constexpr int active = threads / 2;
    constexpr int c_size = 16 * thread_m_blocks * 16 * thread_n_blocks / 4;
    constexpr int th_size = thread_m_blocks * warp_j * 2;
    if (warp_row != 0) return;
    const int off = locks_off * c_size;
    float* fc = reinterpret_cast<float*>(&frag_c);
    if (!first) {
#pragma unroll
      for (int k = 0; k < th_size; k++) {
        int4 v = C_tmp[off + active * k + threadIdx.x];
        float* f = reinterpret_cast<float*>(&v);
#pragma unroll
        for (int e = 0; e < 4; e++) fc[k * 4 + e] += f[e];
      }
    }
    if (!last) {
      int4* fc4 = reinterpret_cast<int4*>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k++) C_tmp[off + active * k + threadIdx.x] = fc4[k];
    }
  };

  // Marlin's write_result: bf16_rn(acc * global_scale), staged in shared
  // memory, stored row-contiguous. silu_mul: the up tile replaces the gate
  // values that the same thread stored earlier with silu(gate) * up.
  auto write_result = [&]() {
    const int c_gl_stride = c_row_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    constexpr int rows_per_pass = threads / (2 * thread_n_blocks);
    const int c_gl_wr_delta = c_gl_stride * rows_per_pass;
    constexpr int c_sh_rd_delta = c_sh_stride * rows_per_pass;
    const int n_half = prob_n / (32 * thread_n_blocks);
    const bool up_tile = silu_mul && slice_col >= n_half;
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * (up_tile ? slice_col - n_half : slice_col);
    int c_sh_wr = (4 * c_sh_stride) * (lane / 4) + lane % 4 + 8 * warp_j * wn;
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    const int c_gl_wr_end = c_gl_stride * prob_m;
    auto write = [&](int idx, float c0, float c1) {
      c0 *= global_scale_f32;
      c1 *= global_scale_f32;
      scalar_t2 res = Dt::nums2num2(Dt::float2num(c0), Dt::float2num(c1));
      reinterpret_cast<scalar_t2*>(sh_red)[idx] = res;
    };
    if (warp_row == 0) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < warp_j; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < div_ceil(16 * thread_m_blocks, rows_per_pass); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        if (up_tile) {
          int4 g = C[c_gl_wr];
          int4 u = sh_red[c_sh_rd];
          int4 r;
          r.x = silu_mul_bf16x2(g.x, u.x);
          r.y = silu_mul_bf16x2(g.y, u.y);
          r.z = silu_mul_bf16x2(g.z, u.z);
          r.w = silu_mul_bf16x2(g.w, u.w);
          C[c_gl_wr] = r;
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  auto start_pipes = [&]() {
#pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    if constexpr (dq_ahead) dequant_step(0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters) start_pipes();

  while (slice_iters) {
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < k_steps; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == k_steps - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      const bool last = slice_idx == slice_count - 1;
      thread_block_reduce();
      if (slice_count > 1) {
        barrier_acquire(&locks[locks_off], slice_idx);
        global_reduce_fp32(slice_idx == 0, last);
        barrier_release(&locks[locks_off], last);
      }
      if (last) write_result();
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o) + a_gl_rd_delta_o * slice_row;
        b_gl_rd = b_gl_thr + b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
        s_gl_rd = s_gl_thr + s_sh_stride * slice_col +
                  s_gl_stride * thread_k_blocks * slice_row;
        start_pipes();
      }
    }
  }
}

// Dynamic shared memory of MarlinS<...> in bytes (the layout above).
template <const int threads, const int thread_n_blocks, const int warp_j,
          const int stages>
constexpr int marlin_s_smem() {
  constexpr int b_sh_stage = 8 * thread_n_blocks * 4;
  constexpr int sh_red_size = 64 * (2 * thread_n_blocks + 1);
  constexpr int sh_tbr_size = warp_j * threads;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int r0 = sh_b_size > sh_red_size
                         ? (sh_b_size > sh_tbr_size ? sh_b_size : sh_tbr_size)
                         : (sh_red_size > sh_tbr_size ? sh_red_size : sh_tbr_size);
  return 16 * (r0 + stages * 4 * thread_n_blocks + stages * 512);
}

}  // namespace marlinw
