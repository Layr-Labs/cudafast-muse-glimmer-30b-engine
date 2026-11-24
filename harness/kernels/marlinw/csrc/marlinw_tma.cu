// SPDX-License-Identifier: Apache-2.0
// 128-row NVFP4 W4A16 prompt GEMM tiles (M = 512) whose operands are brought into shared memory by TMA.
//
// Derived from e-man07's marlinw_down128 and marlinw_swiglu2 (128-row x 128-column tiles, each weight fragment
// dequantized once per 128 rows). Tiles, operand layouts in registers, per-element arithmetic and both epilogues
// are theirs: fe2m1 code -> bf16, times the bf16 group scale (__hmul2), chain A0 over k16 blocks {4t, 4t+1}
// (warps kw = 0) and A1 over {4t+2, 4t+3} (warps kw = 1) of every 64-deep k-tile in ascending t, each from +0 with
// mma.sync.m16n8k16 (bf16 x bf16 -> fp32), P = A0 + A1, stock K cuts folded through the fp32 scratch (down),
// bf16_rn(P * global_scale), and for gate/up the compiled SwiGLU product.
//
// What differs:
//  - How the operands reach shared memory. In the 128-row kernels every MMA warp issues six cp.async per k-tile
//    with their address arithmetic. Here one thread starts three (down) or five (gate/up) TMA bulk tensor copies
//    per k-tile (activations: a 128 B x rows box with SWIZZLE_128B, which is Marlin's xor layout; codes and scales:
//    plain boxes), stages are handed over with mbarriers (complete_tx), and the MMA warps keep ldmatrix, dequant
//    and HMMA. TMA mechanics follow SP5D's fp8ws.
//  - The fe2m1 -> bf16 placement of each 8 codes is written as mask, multiply, mask: the same four registers bit
//    for bit, three fewer integer instructions.
//  - Down projection, the 16 tiles left over after four full waves (208 tiles on 48 SMs): e-man07 splits them by
//    rows into 48 CTAs of 3/3/2 m-blocks, which run far below the tensor-core rate. Here they are split in K into
//    three full 128x128 pieces per tile (work list: marlinw_down128.tma_work_lists); a later piece continues from
//    the accumulator registers its predecessor stored, so every chain is the whole tile's chain, HMMA for HMMA.
//  - Down projection, K cuts: the hook that saves and clears the accumulators at a stock cut runs between loop
//    segments instead of inside the k-tile loop (where ptxas turned it into ~200 predicated instructions that
//    every k-tile issued).
//  - Gate/up epilogue: SiLU is evaluated with the instruction sequence that fills e-man07's 64K-entry table, on
//    the same BF16-rounded gate, instead of being read from the table; swiglu_prepare compares both for every BF16
//    pattern (FP32 bit for bit) and refuses the library on any difference.
// Measured on GB10 against e-man07's 128-row kernels (bit-exact on real layers): gate/up -2.2% in clock cycles per
// tile, down -4% per call.
//
// MARLINW_TMA_GATEUP selects the fused gate/up + SwiGLU kernel (N = 2 x 19968, K = 6656); default is the down
// projection (N = 6656, K = 19968, work list of marlinw_down128.tma_work_lists).
#ifdef MARLINW_TMA_GATEUP
#define MARLINW_SWIGLU
#ifdef MARLINW_TMA_FP8
#define MARLINW_SWIGLU_FP8
#define MARLIN_NAMESPACE_NAME marlinw_tma_gateup_fp8
#else
#define MARLIN_NAMESPACE_NAME marlinw_tma_gateup
#endif
#else
#define MARLIN_NAMESPACE_NAME marlinw_tma_down
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include "marlinw_template.h"


namespace MARLIN_NAMESPACE_NAME {

constexpr int kThreads = 256;
constexpr int kStages = 4;  // power of two
constexpr int kDist = 3;    // k-tiles copied ahead of the one being multiplied (<= kStages - 1)
static_assert((kStages & (kStages - 1)) == 0 && kDist >= 1 && kDist < kStages, "stages");
constexpr int kABytes = 128 * 128;  // 128 rows x 64 k (bf16), 128 B per row, swizzled
constexpr int kBBytes = 4096;       // 4 k16 rows x 128 columns of fe2m1 codes
constexpr int kSBytes = 512;        // 4 k16 rows x 128 fe4m3 scales
constexpr int kStageBytes = 21 * 1024;
// The epilogues stage their reduction in the same window: 64 KiB (down) / 34 KiB (gate/up) from its start.
constexpr int kTilesPerCta = 1;  // column tiles per CTA (gate/up)
// With several tiles per CTA the epilogue must not touch the stages that already hold the next tile: it stages in
// the last stage (free at a tile boundary) plus kEpiExtra bytes after it.
constexpr int kEpiBytes = 128 * 68 * 4;
constexpr int kEpiExtra = (kTilesPerCta > 1 && kEpiBytes > kStageBytes) ? kEpiBytes - kStageBytes : 0;
constexpr int kStagesBytes = kStages * kStageBytes + kEpiExtra + 1024 + 128;
constexpr int kShared = kStagesBytes > 66 * 1024 ? kStagesBytes : 66 * 1024;
static_assert(kShared <= 101376, "shared window");
constexpr auto kBF = vllm::kBFloat16.id();

using T = MarlinScalarType<kBF>;
using FragA = T::FragA;
using FragB = T::FragB;

__device__ __forceinline__ void mma16(const FragA& a, const FragB& b, float* c) {
  const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(&b);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ap[0]), "r"(ap[1]), "r"(ap[2]), "r"(ap[3]), "r"(bp[0]), "r"(bp[1]));
}
__device__ __forceinline__ void ldsm4(FragA& a, uint32_t smem) {
  uint32_t* r = reinterpret_cast<uint32_t*>(&a);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem));
}
__device__ __forceinline__ void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t tma_pol_el() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t tma_pol_ef() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma2d_hint(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar,
                                           uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}

// The k loop of one CTA: rows [16 MB] x 128 columns x `iters` 64-deep k-tiles, two accumulator sets per element
// (one per kw warp row). produce(g, dst, bar) (thread 0 only) starts the TMA copies of k-tile g into the stage at
// shared address dst, completing on mbarrier bar; on_tile(t) runs before k-tile t. b_rd / s_rd: this thread's
// int4 / int offsets of its codes and scales inside a stage, to which 4 b_row / s_row per k16 row are added.
// A CTA may run several tiles back to back (kTilesPerCta): q0 is the position of this tile's first k-tile in the CTA's
// k-tile stream of length `total`, and the copies keep running across the tile boundary.
template <int MB, class Produce, class OnTile>
__device__ __forceinline__ void k_loop(
    float (&acc)[MB][2][2][4], const int iters, const int b_rd, const int b_row, const int s_rd, const int s_row,
    Produce&& produce, OnTile&& on_tile, const int q0 = 0, const int total_or_0 = 0,
    const float4* const resume = nullptr, const int t0 = 0) {
  const int total = total_or_0 ? total_or_0 : iters;
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kw = warp >> 2;  // k16 pair of each k-tile: chain A0 / A1
  // Stages start on 1 KiB boundaries of the shared window (TMA swizzles by shared address).
  const uint32_t sbase = (static_cast<uint32_t>(__cvta_generic_to_shared(sh)) + 1023u) & ~1023u;
  const int4* shb = reinterpret_cast<const int4*>(
      reinterpret_cast<const char*>(sh) + (sbase - static_cast<uint32_t>(__cvta_generic_to_shared(sh))));
  // ldmatrix: m-block j of k-half kk sits at 16 (a_rd[kk] + 128 j) (the swizzle term does not depend on j)
  const uint32_t a_rd0 = 16 * (8 * (lane & 15) + (((lane >> 4) + 4 * kw) ^ (lane & 7)));
  const uint32_t a_rd1 = 16 * (8 * (lane & 15) + (((lane >> 4) + 4 * kw + 2) ^ (lane & 7)));
  const uint32_t full_bar = sbase + kStages * kStageBytes + kEpiExtra;
  const uint32_t empty_bar = full_bar + 8 * kStages;
  if (q0 == 0) {
    if (tid == 0) {
#pragma unroll
      for (int i = 0; i < kStages; i++) {
        mbar_init(full_bar + 8 * i, 1);
        mbar_init(empty_bar + 8 * i, kThreads / 32);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
  }
  // Thread 0: k-tile g goes to stage g mod kStages once every thread has finished with that stage's previous
  // occupant (k-tile g - kStages).
  auto start_copy = [&](const int g) {
    const int sg = g & (kStages - 1);
    if (g >= t0 + kStages) mbar_wait(empty_bar + 8 * sg, ((g - kStages) / kStages) & 1);
    produce(g, sbase + sg * kStageBytes, full_bar + 8 * sg);
  };
  if (tid == 0 && q0 == 0) {
#pragma unroll
    for (int g = 0; g < kDist; g++)
      if (t0 + g < total) start_copy(t0 + g);
  }

  FragA fa[2][MB];
  int2 fbq[2];
  int fsq[2];
  FragB bb[2][2][2];  // dequantized B operands [k-half][32-column quarter][n8 block]

  auto load_bs = [&](const int o, const int stage) {
    const int4* sa = shb + (kStageBytes / 16) * stage;
    const int g = 2 * kw + o;
    fbq[o] = *reinterpret_cast<const int2*>(reinterpret_cast<const int*>(sa) + b_rd + b_row * g);
    fsq[o] = reinterpret_cast<const int*>(sa)[s_rd + s_row * g];
  };
  auto dequant_to = [&](const int o) {
    nv_bfloat162 sc[2];
    dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(fsq[o], sc);
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      FragB& b0 = bb[o][jj][0];
      FragB& b1 = bb[o][jj][1];
      const int bq1 = jj == 0 ? fbq[o].x : fbq[o].y;
      // The four registers of dequant<.., kFE2M1f, true>(bq1 << 8) and (bq1), bit for bit: per register the two
      // nibbles are isolated, copied to the sign position and to the exponent/mantissa position by one multiply
      // (the copies cannot overlap) and masked: bit 15 <- sign, bits 8:6 <- e1 e0 m, the same in the upper half.
      {
        const unsigned u = static_cast<unsigned>(bq1);
        reinterpret_cast<unsigned*>(&b1)[1] = (u & 0x80008000u) | ((u >> 6) & 0x01c001c0u);
        reinterpret_cast<unsigned*>(&b1)[0] = (((u & 0x0f000f00u) >> 2) * 65u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[1] = ((u & 0x00f000f0u) * 260u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[0] = ((u & 0x000f000fu) * 4160u) & 0x81c081c0u;
      }
      const nv_bfloat162 s = sc[jj];
      const nv_bfloat162 s0 = __bfloat162bfloat162(reinterpret_cast<const nv_bfloat16*>(&s)[0]);
      const nv_bfloat162 s1 = __bfloat162bfloat162(reinterpret_cast<const nv_bfloat16*>(&s)[1]);
      b0[0] = __hmul2(b0[0], s0);
      b0[1] = __hmul2(b0[1], s0);
      b1[0] = __hmul2(b1[0], s1);
      b1[1] = __hmul2(b1[1], s1);
    }
  };

  if (resume) {  // a later K piece of a tile: both chains continue from the state the previous piece left
#pragma unroll
    for (int e = 0; e < MB * 4; e++) reinterpret_cast<float4*>(acc)[e] = resume[e * kThreads + tid];
  } else {
#pragma unroll
    for (int i = 0; i < MB * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.0f;
  }
  mbar_wait(full_bar + 8 * (q0 & (kStages - 1)), (q0 / kStages) & 1);
#pragma unroll
  for (int j = 0; j < MB; j++) ldsm4(fa[0][j], sbase + kStageBytes * (q0 & (kStages - 1)) + a_rd0 + 2048 * j);
  load_bs(0, q0 & (kStages - 1));
  dequant_to(0);

  // on_tile(t) returns the k-tile before which it wants to run next: the loop over the k-tiles in between carries
  // none of its code (compiled into the loop it becomes predicated instructions that every k-tile pays for).
#pragma unroll 1
  for (int t = t0; t < iters;) {
  const int stop = on_tile(t);
#pragma unroll 1
  for (; t < stop; t++) {
    const int q = q0 + t;
    const int s = q & (kStages - 1);
    const int ns = (q + 1) & (kStages - 1);
    if (tid == 0 && q + kDist < total) start_copy(q + kDist);
    // k-half 0 of k-tile t (operands already in registers); meanwhile k-half 1 of the same stage is loaded.
    load_bs(1, s);
    dequant_to(1);
    {
      const uint32_t sa = sbase + kStageBytes * s + a_rd1;
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[0][i], bb[0][0][0], acc[i][0][0]);
        mma16(fa[0][i], bb[0][0][1], acc[i][0][1]);
        ldsm4(fa[1][i], sa + 2048 * i);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_bar + 8 * s);  // warp-aggregated empty signal
#pragma unroll
    for (int i = 0; i < MB; i++) {
      mma16(fa[0][i], bb[0][1][0], acc[i][1][0]);
      mma16(fa[0][i], bb[0][1][1], acc[i][1][1]);
    }
    // k-half 1 of k-tile t; meanwhile k-half 0 of k-tile t + 1 is loaded.
    if (t + 1 < iters) mbar_wait(full_bar + 8 * ns, ((q + 1) / kStages) & 1);
    load_bs(0, ns);
    dequant_to(0);
    {
      const uint32_t sa = sbase + kStageBytes * ns + a_rd0;
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[1][i], bb[1][0][0], acc[i][0][0]);
        mma16(fa[1][i], bb[1][0][1], acc[i][0][1]);
        ldsm4(fa[0][i], sa + 2048 * i);
      }
    }
#pragma unroll
    for (int i = 0; i < MB; i++) {
      mma16(fa[1][i], bb[1][1][0], acc[i][1][0]);
      mma16(fa[1][i], bb[1][1][1], acc[i][1][1]);
    }
  }
  }
}

#ifndef MARLINW_TMA_GATEUP
// ---------------------------------------------------------------------------------------------------------------
// Down projection. One CTA = one work item of marlinw_down128: w[2b] = {row0, col128, mb, slot},
// w[2b+1] = {cut_a, cut_b, k_tiles, 0}; cut_a / cut_b are the k-tile at which the stock partition restarts the
// chains of the 64-row chunk containing row0 / the next chunk (-1: none).
constexpr int kColumns = 6656;
constexpr int kInner = 19968;

__device__ __forceinline__ uint32_t selp_u32(uint32_t a, uint32_t b, bool c) {
  uint32_t r;
  asm("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n selp.b32 %0, %1, %2, p;\n}"
      : "=r"(r) : "r"(a), "r"(b), "r"((uint32_t)c));
  return r;
}
__device__ __forceinline__ uint32_t sel4_u32(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, int i) {
  const uint32_t lo = selp_u32(v1, v0, i & 1), hi = selp_u32(v3, v2, i & 1);
  return selp_u32(hi, lo, i & 2);
}

template <int MB, bool kLater, int kCols, bool kShflEpi = true>  // kLater: the copy for the pieces that take over a state
__device__ __forceinline__ void down_tile(
    const CUtensorMap* tmA, const CUtensorMap* tmB, const CUtensorMap* tmS, int4* __restrict__ C0,
    float4* __restrict__ scratch, const float gs, const int row0, const int col, const int slot, const int cut_a,
    const int cut_b, const int k_tiles
    , const int piece, volatile int* const flags
    ) {
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kw = warp >> 2;
  const int nb = (warp >> 1) & 1;  // 64-column half of the tile
  const int sh2 = warp & 1;        // 32-column half of that

  float acc[MB][2][2][4];
  const int n_first = 4 - ((row0 >> 4) & 3);  // m-blocks in the chunk containing row0
  float4* my_scratch = scratch + ((size_t)slot * 256 + tid) * 32;
  // K pieces. piece = k0 | role << 10 | index << 12: this CTA runs k-tiles [k0, k0 + k_tiles) of its tile (k0 a
  // multiple of 8). role 0: the whole tile. 1: first piece (chains from +0, state stored at the end). 2: middle piece
  // (state loaded, state stored). 3: last piece (state loaded, epilogue). The state is every thread's accumulator
  // registers (scratch slot `index`), so the chains of a split tile are the chains of the whole tile, HMMA for HMMA.
  // A later piece waits until its predecessor has published the state: flags[index] goes 0 -> 1 (first piece
  // done) -> 2 (middle piece done) -> 0 (taken by the last piece).
  const int k0 = piece & 1023;
  const int k1 = k0 + k_tiles;
  if (tid == 0 && (piece & 2048)) {
    volatile int* const flag = flags + (piece >> 12);
    const int before = ((piece >> 10) & 3) - 1;
    while (*flag != before) {
    }
    __threadfence();  // acquire: the predecessor's state (stored before its release fence + flag) is visible
    if (before == 2) *flag = 0;
  }
  // Runs before k-tile t; returns the k-tile before which it has to run again (the next cut, or the end).
  auto cuts = [&](const int t) {
    if (t == cut_a || t == cut_b) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < MB; i++) {
        const int ci = i < n_first ? cut_a : cut_b;
        if (ci == t) {
#pragma unroll
          for (int e = 0; e < 4; e++) my_scratch[i * 4 + e] = reinterpret_cast<float4*>(acc[i])[e];
#pragma unroll
          for (int e = 0; e < 16; e++) reinterpret_cast<float*>(acc[i])[e] = 0.0f;
        }
      }
      __syncthreads();
    }
    int stop = k1;
    if (cut_a > t && cut_a < stop) stop = cut_a;
    if (cut_b > t && cut_b < stop) stop = cut_b;
    return stop;
  };
  auto produce = [&](const int g, const uint32_t dst, const uint32_t bar) {
    mbar_expect_tx(bar, MB * 16 * 128 + kBBytes + kSBytes);
    tma2d_hint(dst, tmA, 128 * g, row0, bar, tma_pol_el());                         // bytes, rows
    tma2d_hint(dst + kABytes, tmB, 256 * col, 4 * g, bar, tma_pol_ef());            // uint32, k16 rows
    tma2d_hint(dst + kABytes + kBBytes, tmS, 128 * col, 4 * g, bar, tma_pol_ef());  // bytes, k16 rows
  };
  // codes: int4 (64 g + 32 nb + lane), int2 sh2 of it; scales: int 32 g + 2 (8 nb + lane / 4) + sh2
  k_loop<MB>(acc, k1, kABytes / 4 + 4 * (32 * nb + lane) + 2 * sh2, 256,
             (kABytes + kBBytes) / 4 + 2 * (8 * nb + (lane >> 2)) + sh2, 32, produce, cuts, 0, 0,
             kLater ? scratch + (size_t)(piece >> 12) * (kThreads * 32) : nullptr, k0);
  __syncthreads();
  {
    const int role = (piece >> 10) & 3;
    if (role == 1 || role == 2) {
      float4* const hand = scratch + (size_t)(piece >> 12) * (kThreads * 32);
#pragma unroll
      for (int e = 0; e < MB * 4; e++) hand[e * kThreads + tid] = reinterpret_cast<float4*>(acc)[e];
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        flags[piece >> 12] = role;
        __threadfence();
      }
      return;
    }
  }

  // P = A0 + A1, fold of the K cuts, bf16_rn(P * global_scale): as marlinw_down128.
  float4* red = reinterpret_cast<float4*>(sh);
  if (kw == 1) {
#pragma unroll
    for (int k = 0; k < MB * 4; k++) red[k * 128 + (tid - 128)] = reinterpret_cast<float4*>(acc)[k];
  }
  __syncthreads();
  if (kw == 0) {
#pragma unroll
    for (int k = 0; k < MB * 4; k++) {
      const float4 r = red[k * 128 + tid];
      float* c = reinterpret_cast<float*>(acc) + 4 * k;
      c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
    }
    if (cut_a >= 0 || cut_b >= 0) {
      const float4* s0 = scratch + ((size_t)slot * 256 + tid) * 32;
      const float4* s1 = scratch + ((size_t)slot * 256 + tid + 128) * 32;
#pragma unroll
      for (int i = 0; i < MB; i++) {
        const int ci = i < n_first ? cut_a : cut_b;
        if (ci >= 0) {
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
            float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
            c[0] = (x.x + y.x) + c[0];
            c[1] = (x.y + y.y) + c[1];
            c[2] = (x.z + y.z) + c[2];
            c[3] = (x.w + y.w) + c[3];
          }
        }
      }
    }
    __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C0);
    if constexpr (kShflEpi) {
      const int q = lane & 3;
      const int r0 = row0 + (lane >> 2);
      const int base_col0 = 128 * col + 64 * nb + 32 * sh2;
#pragma unroll
      for (int i = 0; i < MB; i++) {
#pragma unroll
        for (int jj = 0; jj < 2; jj++) {
          const float* c = &acc[i][jj][0][0];
          const nv_bfloat162 lo0 = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
          const nv_bfloat162 hi0 = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
          const nv_bfloat162 lo1 = __halves2bfloat162(__float2bfloat16(c[4] * gs), __float2bfloat16(c[5] * gs));
          const nv_bfloat162 hi1 = __halves2bfloat162(__float2bfloat16(c[6] * gs), __float2bfloat16(c[7] * gs));
          const uint32_t v0 = *reinterpret_cast<const uint32_t*>(&lo0);
          const uint32_t v1 = *reinterpret_cast<const uint32_t*>(&hi0);
          const uint32_t v2 = *reinterpret_cast<const uint32_t*>(&lo1);
          const uint32_t v3 = *reinterpret_cast<const uint32_t*>(&hi1);
          const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 1), 1);
          const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 2), 2);
          const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 3), 3);
          const uint32_t own = sel4_u32(v0, v1, v2, v3, q);
          uint4 w;
          w.x = sel4_u32(own, r1, r2, r3, 0 ^ q);
          w.y = sel4_u32(own, r1, r2, r3, 1 ^ q);
          w.z = sel4_u32(own, r1, r2, r3, 2 ^ q);
          w.w = sel4_u32(own, r1, r2, r3, 3 ^ q);
          const int r_out = r0 + 16 * i + (q & 1) * 8;
          const int c_out = base_col0 + 16 * jj + (q >> 1) * 8;
          *reinterpret_cast<uint4*>(&C[(size_t)r_out * kCols + c_out]) = w;
        }
      }
    } else {
      const int r0 = row0 + (lane >> 2);
      const int col0 = 128 * col + 64 * nb + 32 * sh2 + 2 * (lane & 3);
#pragma unroll
      for (int i = 0; i < MB; i++)
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const float* c = acc[i][jj][h];
            const nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            const nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            const int cc = col0 + 16 * jj + 8 * h;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
          }
    }
  }
}


template <int kCols, bool kShflEpi = true>
__global__ void __launch_bounds__(kThreads, 1) down(
    const __grid_constant__ CUtensorMap tmA8, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmS, int4* __restrict__ C0, float4* __restrict__ scratch,
    const float* __restrict__ gs_ptr, int4* __restrict__ work
    ) {
  if (threadIdx.x == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA8) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS) : "memory");
  }
  const int4 w0 = work[2 * blockIdx.x];
  const int4 w1 = work[2 * blockIdx.x + 1];
  const float gs = gs_ptr[0];
  if (w1.w & 2048)
    down_tile<8, true, kCols, kShflEpi>(&tmA8, &tmB, &tmS, C0, scratch, gs, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, w1.w,
                       reinterpret_cast<volatile int*>(work + 2 * gridDim.x));
  else
    down_tile<8, false, kCols, kShflEpi>(&tmA8, &tmB, &tmS, C0, scratch, gs, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, w1.w,
                        reinterpret_cast<volatile int*>(work + 2 * gridDim.x));
}
#else
// ---------------------------------------------------------------------------------------------------------------
// Fused gate/up + SwiGLU. blockIdx = 4 * column tile + 128-row chunk; a column tile is 64 gate and the matching 64
// up columns. Epilogue as marlinw_swiglu2.
constexpr int kTiles = 6656 / 64;
constexpr int kFused = 39936;

// The epilogue evaluates SiLU with the instruction sequence that fills the table of the 128-row kernels
// (swiglu_silu, on the gate already rounded to BF16) instead of reading the table. silu_check runs both for every
// BF16 pattern and counts the patterns whose FP32 results differ in any bit; swiglu_prepare refuses the library
// unless that count is 0.
__device__ int silu_differences;
__global__ void silu_check() {
  const unsigned word = 256u * blockIdx.x + threadIdx.x;
  const float direct = swiglu_silu(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(word))));
  if (__float_as_uint(direct) != __float_as_uint(bf16_silu[word])) atomicAdd(&silu_differences, 1);
}

__global__ void __launch_bounds__(kThreads, 1) gateup(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmS, char* __restrict__ out,
    const float* __restrict__ global_scale_ptr
#ifdef MARLINW_TMA_FP8
    , const float* __restrict__ down_input_scale
#endif
    ) {
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kw = warp >> 2, cw = (warp >> 1) & 1, jh = warp & 1;
  const int chunk = blockIdx.x & 3, col0 = (blockIdx.x >> 2) * kTilesPerCta;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS) : "memory");
  }
  const float gs = global_scale_ptr[0];
  float c[8][2][2][4];
  // Stage: [activations 16 KiB][gate codes 4 x 512 B][up codes 4 x 512 B][gate scales 4 x 64 B][up scales 4 x 64 B]
  // q: position in this CTA's k-tile stream = kTiles * (tile within the CTA) + k-tile.
  auto produce = [&](const int q, const uint32_t dst, const uint32_t bar) {
    const int ti = q / kTiles;
    const int g = q - ti * kTiles;
    const int slice_col = col0 + ti;
    mbar_expect_tx(bar, kABytes + kBBytes + kSBytes);
    tma2d_hint(dst, &tmA, 128 * g, 128 * chunk, bar, tma_pol_el());                                               // bytes, rows
    tma2d_hint(dst + kABytes, &tmB, 128 * slice_col, 4 * g, bar, tma_pol_ef());                                   // uint32, k16 rows
    tma2d_hint(dst + kABytes + kBBytes / 2, &tmB, 128 * slice_col + kFused, 4 * g, bar, tma_pol_ef());
    tma2d_hint(dst + kABytes + kBBytes, &tmS, 64 * slice_col, 4 * g, bar, tma_pol_ef());                          // bytes, k16 rows
    tma2d_hint(dst + kABytes + kBBytes + kSBytes / 2, &tmS, 64 * slice_col + kFused / 2, 4 * g, bar, tma_pol_ef());
  };
  // codes: int4 (128 cw + 32 g + lane), int2 jh of it; scales: int 64 cw + 16 g + 2 (lane / 4) + jh
  // Epilogue staging: the start of the window, or (several tiles per CTA) the last stage plus the bytes after it.
  int4* const epi = kTilesPerCta == 1 ? sh : reinterpret_cast<int4*>(
      reinterpret_cast<char*>(sh) + (((static_cast<uint32_t>(__cvta_generic_to_shared(sh)) + 1023u) & ~1023u) -
                                     static_cast<uint32_t>(__cvta_generic_to_shared(sh))) + (kStages - 1) * kStageBytes);
#pragma unroll 1
  for (int ti = 0; ti < kTilesPerCta; ti++) {
  const int slice_col = col0 + ti;
  k_loop<8>(c, kTiles, kABytes / 4 + 4 * (128 * cw + lane) + 2 * jh, 128,
            (kABytes + kBBytes) / 4 + 64 * cw + 2 * (lane >> 2) + jh, 16, produce, [](int) { return kTiles; }, ti * kTiles,
            kTilesPerCta * kTiles);
  __syncthreads();

  // Reduce the two k-warps: P = A0 (kw 0) + A1 (kw 1).
  int4* red = epi;
  const int t128 = tid & 127;
  if constexpr (kTilesPerCta == 1) {
    if (kw == 1) {
#pragma unroll
      for (int q = 0; q < 32; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(c)[q];
    }
    __syncthreads();
    if (kw == 0) {
#pragma unroll
      for (int q = 0; q < 32; q++) {
        const int4 v = red[q * 128 + t128];
        float* cf = reinterpret_cast<float*>(c) + 4 * q;
        cf[0] += __int_as_float(v.x);
        cf[1] += __int_as_float(v.y);
        cf[2] += __int_as_float(v.z);
        cf[3] += __int_as_float(v.w);
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 16; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(&c[4 * half])[q];
      }
      __syncthreads();
      if (kw == 0) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[q * 128 + t128];
          float* cf = reinterpret_cast<float*>(&c[4 * half]) + 4 * q;
          cf[0] += __int_as_float(v.x);
          cf[1] += __int_as_float(v.y);
          cf[2] += __int_as_float(v.z);
          cf[3] += __int_as_float(v.w);
        }
      }
      __syncthreads();
    }
  }

  if (kw == 0) {
    // words (bf16x2) in rows of 68 words
    uint32_t* shw = reinterpret_cast<uint32_t*>(epi);
    int wr = 68 * (lane >> 2) + (lane & 3) + 32 * cw;
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int jl = 0; jl < 2; jl++) {
        const int j = 2 * jh + jl;
        auto put = [&](int idx, float c0, float c1) {
          __nv_bfloat162 r = __halves2bfloat162(__float2bfloat16_rn(c0 * gs), __float2bfloat16_rn(c1 * gs));
          shw[idx] = *reinterpret_cast<uint32_t*>(&r);
        };
        const int w = wr + 8 * j;
        put(w, c[i][jl][0][0], c[i][jl][0][1]);
        put(w + 68 * 8, c[i][jl][0][2], c[i][jl][0][3]);
        put(w + 4, c[i][jl][1][0], c[i][jl][1][1]);
        put(w + 68 * 8 + 4, c[i][jl][1][2], c[i][jl][1][3]);
      }
      wr += 16 * 68;
    }
  }
  __syncthreads();
  {
    const uint32_t* shw = reinterpret_cast<const uint32_t*>(epi);
#ifdef MARLINW_TMA_FP8
    float inverse;
    asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inverse) : "f"(*down_input_scale));
    uint16_t* o8 = reinterpret_cast<uint16_t*>(out);
#pragma unroll 4
    for (int row = warp; row < 128; row += 8) {
      uint32_t gates = shw[row * 68 + lane], ups = shw[row * 68 + lane + 32];
      float scaled[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        const float silu = swiglu_silu(__bfloat162float(gate));
        const float product = __fmul_rn(silu, __bfloat162float(up));
        const float q = __fmul_rn(product, inverse);
        scaled[e] = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
      }
      uint16_t converted;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(converted) : "f"(scaled[1]), "f"(scaled[0]));
      o8[(size_t)(chunk * 128 + row) * (kFused / 4) + slice_col * 32 + lane] = converted;
    }
#else
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll 4
    for (int row = warp; row < 128; row += 8) {
      uint32_t gates = shw[row * 68 + lane], ups = shw[row * 68 + lane + 32];
      uint32_t prod;
#pragma unroll
      for (int e = 0; e < 2; e++) {
        auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        // swiglu_product with the table entry replaced by its definition (silu_check: equal for all patterns)
        reinterpret_cast<__nv_bfloat16*>(&prod)[e] =
            __float2bfloat16_rn(__fmul_rn(swiglu_silu(__bfloat162float(gate)), __bfloat162float(up)));
      }
      o[(size_t)(chunk * 128 + row) * (kFused / 4) + slice_col * 32 + lane] = prod;
    }
#endif
  }
  }  // tiles of this CTA
}
#endif

// ---- host side: tensor maps --------------------------------------------------------------------------------------
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;

static cudaError_t find_encode() {
  if (encode) return cudaSuccess;
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q;
  cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
  if (e != cudaSuccess) return e;
  if (!fn || q != cudaDriverEntryPointSuccess) return cudaErrorNotSupported;
  encode = (EncodeFn)fn;
  return cudaSuccess;
}
// A 2-D map over `rows` rows of `width` elements of `esize` bytes, `stride` bytes apart; box = bw x bh elements.
static bool map2d(CUtensorMap* tm, const void* ptr, int esize, uint64_t width, uint64_t rows, uint64_t stride,
                  uint32_t bw, uint32_t bh, bool swizzle) {
  const cuuint64_t dims[2] = {width, rows};
  const cuuint64_t strides[1] = {stride};
  const cuuint32_t box[2] = {bw, bh};
  const cuuint32_t es[2] = {1, 1};
  const CUtensorMapDataType dt = esize == 4 ? CU_TENSOR_MAP_DATA_TYPE_UINT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  return encode(tm, dt, 2, const_cast<void*>(ptr), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

}  // namespace MARLIN_NAMESPACE_NAME


#ifndef MARLINW_TMA_GATEUP
// Same C ABI as libmarlinw_down128.so, plus multi-shape TMA entrypoints (qkv 4608x6656, o_proj 6656x4096, gate 4096x6656).
static bool g_shfl_epi = true;

extern "C" void marlinw_down128_set_shfl_epi(int enable) {
  g_shfl_epi = (enable != 0);
}

template <int kCols>
static int init_down_shape(int* active_blocks) {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  int occ_shfl = 0, occ_stock = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_shfl, down<kCols, true>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_stock, down<kCols, false>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  *active_blocks = occ_shfl < occ_stock ? occ_shfl : occ_stock;
  return cudaSuccess;
}

template <int kCols>
static int launch_down_shape(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int k, int row_stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  if (!encode || row_stride < k || row_stride % 8 || k % 64 != 0) return cudaErrorInvalidValue;
  if (((uintptr_t)input | (uintptr_t)weight | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA8, tmB, tmS;
  if (!map2d(&tmA8, input, 1, 2 * (uint64_t)k, 512, 2 * (uint64_t)row_stride, 128, 128, true) ||
      !map2d(&tmB, weight, 4, 2 * (uint64_t)kCols, (uint64_t)k / 16, 8 * (uint64_t)kCols, 256, 4, false) ||
      !map2d(&tmS, scales, 1, (uint64_t)kCols, (uint64_t)k / 16, (uint64_t)kCols, 128, 4, false))
    return cudaErrorInvalidValue;
  auto kernel = g_shfl_epi ? down<kCols, true> : down<kCols, false>;
  kernel<<<grid, kThreads, kShared, stream>>>(tmA8, tmB, tmS, output, scratch, global_scale,
                                              const_cast<int4*>(work));
  return cudaGetLastError();
}

extern "C" int marlinw_down128_init(int* active_blocks) {
  return init_down_shape<6656>(active_blocks);
}
extern "C" int marlinw_down128_main(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int row_stride, cudaStream_t stream) {
  return launch_down_shape<6656>(input, weight, output, scratch, scales, global_scale, work, grid, 19968,
                                 row_stride, stream);
}
extern "C" int marlinw_down128_shape_tma_init(int n, int k, int* active_blocks) {
  if (k % 64 != 0) return cudaErrorInvalidValue;
  if (n == 6656) return init_down_shape<6656>(active_blocks);
  if (n == 4608) return init_down_shape<4608>(active_blocks);
  if (n == 4096) return init_down_shape<4096>(active_blocks);
  return cudaErrorInvalidValue;
}
extern "C" int marlinw_down128_shape_tma_main(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int n, int k, int row_stride, cudaStream_t stream) {
  if (n == 6656)
    return launch_down_shape<6656>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  if (n == 4608)
    return launch_down_shape<4608>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  if (n == 4096)
    return launch_down_shape<4096>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  return cudaErrorInvalidValue;
}
#else
#ifdef MARLINW_TMA_FP8
extern "C" int swiglu_fp8_prepare() {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  int differences = 0;
  status = cudaMemcpyToSymbol(silu_differences, &differences, sizeof(int));
  if (status != cudaSuccess) return status;
  silu_check<<<256, 256>>>();
  status = cudaMemcpyFromSymbol(&differences, silu_differences, sizeof(int));
  if (status != cudaSuccess) return status;
  if (differences) return cudaErrorUnknown;
  return cudaDeviceSynchronize();
}
extern "C" int swiglu_fp8_gemm(const void* x, const void* w, const void* scales, const void* global,
                               const float* down_input_scale, void* out, int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w, 4, 2 * n, k / 16, 8 * n, 128, 4, false) ||
      !map2d(&tmS, scales, 1, n, k / 16, n, 64, 4, false))
    return cudaErrorInvalidValue;
  static_assert((n / 128) % kTilesPerCta == 0, "tiles per CTA");
  gateup<<<(m / 128) * (n / 128) / kTilesPerCta, kThreads, kShared, stream>>>(
      tmA, tmB, tmS, static_cast<char*>(out), static_cast<const float*>(global), down_input_scale);
  return cudaGetLastError();
}
#else
// Same C ABI as libmarlinw_swiglu2.so.
extern "C" int swiglu_prepare() {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  int differences = 0;
  status = cudaMemcpyToSymbol(silu_differences, &differences, sizeof(int));
  if (status != cudaSuccess) return status;
  silu_check<<<256, 256>>>();
  status = cudaMemcpyFromSymbol(&differences, silu_differences, sizeof(int));
  if (status != cudaSuccess) return status;
  if (differences) return cudaErrorUnknown;
  return cudaDeviceSynchronize();
}
extern "C" int swiglu_gemm(const void* x, const void* w, const void* scales, const void* global, void* out,
                           int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w, 4, 2 * n, k / 16, 8 * n, 128, 4, false) ||
      !map2d(&tmS, scales, 1, n, k / 16, n, 64, 4, false))
    return cudaErrorInvalidValue;
  static_assert((n / 128) % kTilesPerCta == 0, "tiles per CTA");
  gateup<<<(m / 128) * (n / 128) / kTilesPerCta, kThreads, kShared, stream>>>(
      tmA, tmB, tmS, static_cast<char*>(out), static_cast<const float*>(global)
      );
  return cudaGetLastError();
}
#endif
#endif
