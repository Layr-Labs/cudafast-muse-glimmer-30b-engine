// SPDX-License-Identifier: Apache-2.0
// 4-stage TMA super-block rANS block-scale M = 1 NVFP4 decode GEMV.
// Uses hardware TMA cp.async.bulk with 2 alternating super-block slots (4 k-stages per super-block),
// wave/stage-interleaved var super-blocks, and continuous cross-slice cp.async pipelines for
// gate_up (336 CTAs x 256T), down/qkv/o_proj (48 CTAs x 256T), GateV48 (24 CTAs x 256T), and lm_head (48 CTAs x 128T).
#define MARLIN_NAMESPACE_NAME muse_marlin6_tma4
#define MUSE_SCALE_TMA4
#define MUSE_B_EVICT_FIRST
#include <cstdlib>
#include "marlin/marlin_template.h"

extern "C" int muse_marlin6_get_gate_blocks();

namespace {

static size_t decode6_tpf_stride() {
  static size_t stride = []() -> size_t {
    const char* env = std::getenv("MUSE_DECODE6_TLB_PREFETCH");
    if (!env || !*env) return 0u;
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return stride;
}

// A stage holds tile_n * 8 scale bytes (8 group-16 codes per column of a 128-k tile), which is independent of the
// thread count: StageBytes is passed explicitly so a 64-column tile can run 8 warps (64 B each) instead of 4 (128 B
// each). The warps still cover the stage's bytes in ascending order, so the concatenated symbol stream -- and hence
// every decoded scale byte -- is unchanged.
template <int Threads, int StageBytes>
__global__ void tma4_pass1_kernel(
    const uint8_t* __restrict__ sym_tiled, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ blk_sb_off, const uint32_t* __restrict__ sb_gsb_map,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    uint32_t* __restrict__ sb_wc) {
  constexpr int kWarps = Threads / 32;
  constexpr int kPerWarp = StageBytes / kWarps;
  constexpr int kM = kPerWarp / 32;
  __shared__ uint32_t sf[64], sc[64];
  if (threadIdx.x < 64) {
    sf[threadIdx.x] = enc_f[threadIdx.x];
    sc[threadIdx.x] = enc_c[threadIdx.x];
  }
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int num_sbs = static_cast<int>(blk_sb_off[b + 1] - blk_sb_off[b]);
  const uint32_t sb_base = blk_sb_off[b];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t x = 65536u;

  for (int sb = num_sbs - 1; sb >= 0; --sb) {
    const int st_lo = s0 + sb * 4;
    const int st_hi = (st_lo + 4 < s1) ? (st_lo + 4) : s1;
    uint32_t wtot = 0;
    for (int s = st_hi - 1; s >= st_lo; --s) {
      const int col = stage_col_k[2 * s];
      const int kt = stage_col_k[2 * s + 1];
      const uint8_t* raw = sym_tiled + (static_cast<size_t>(col) * k_tiles + kt) * StageBytes + kPerWarp * w + lane;
#pragma unroll
      for (int m = kM - 1; m >= 0; --m) {
        const uint32_t sy = raw[32 * m];
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        wtot += __popc(__ballot_sync(0xffffffffu, emit));
        if (emit) x >>= 16;
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
    }
    if (lane == 0) sb_wc[sb_gsb_map[sb_base + sb] * kWarps + w] = wtot;
  }
}

template <int Threads, int StageBytes>
__global__ void tma4_pass2_kernel(
    const uint8_t* __restrict__ sym_tiled, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ blk_sb_off, const uint32_t* __restrict__ sb_gsb_map,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    const uint32_t* __restrict__ sb_wc, const uint32_t* __restrict__ off16,
    uint32_t* __restrict__ states_out, unsigned char* __restrict__ var_out) {
  constexpr int kWarps = Threads / 32;
  constexpr int kPerWarp = StageBytes / kWarps;
  constexpr int kM = kPerWarp / 32;
  __shared__ uint32_t sf[64], sc[64];
  if (threadIdx.x < 64) {
    sf[threadIdx.x] = enc_f[threadIdx.x];
    sc[threadIdx.x] = enc_c[threadIdx.x];
  }
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int num_sbs = static_cast<int>(blk_sb_off[b + 1] - blk_sb_off[b]);
  const uint32_t sb_base = blk_sb_off[b];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t x = 65536u;

  for (int sb = num_sbs - 1; sb >= 0; --sb) {
    const uint32_t gsb = sb_gsb_map[sb_base + sb];
    uint16_t* sb_ptr = reinterpret_cast<uint16_t*>(var_out + static_cast<size_t>(off16[gsb]) * 16);
    const uint32_t my_wc = sb_wc[gsb * kWarps + w];
    if (lane == 0) sb_ptr[w] = static_cast<uint16_t>(my_wc);
    uint32_t wpos = kWarps;
#pragma unroll
    for (int i = 0; i < kWarps - 1; ++i) {
      if (i < w) wpos += sb_wc[gsb * kWarps + i];
    }
    wpos += my_wc;

    const int st_lo = s0 + sb * 4;
    const int st_hi = (st_lo + 4 < s1) ? (st_lo + 4) : s1;
    for (int s = st_hi - 1; s >= st_lo; --s) {
      const int col = stage_col_k[2 * s];
      const int kt = stage_col_k[2 * s + 1];
      const uint8_t* raw = sym_tiled + (static_cast<size_t>(col) * k_tiles + kt) * StageBytes + kPerWarp * w + lane;
#pragma unroll
      for (int m = kM - 1; m >= 0; --m) {
        const uint32_t sy = raw[32 * m];
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        const uint32_t msk = __ballot_sync(0xffffffffu, emit);
        wpos -= __popc(msk);
        if (emit) {
          sb_ptr[wpos + __popc(msk & lt)] = static_cast<uint16_t>(x & 0xFFFFu);
          x >>= 16;
        }
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
    }
  }
  states_out[static_cast<size_t>(b) * Threads + threadIdx.x] = x;
}

template <int TileN, int Threads, bool GateUpGrid, bool DownIlv = false, bool GateV48 = false>
int launch_tma4(const int4* input, const int4* weight, int4* output, int4* partials,
                const int4* coded, const int4* palette, const float* global_scale,
                int* locks, int n, int k, int grid, int device, cudaStream_t stream,
                const int64_t* wptrs = nullptr, const int* wtiles = nullptr, int wchunks = 0) {
  auto kernel = muse_marlin6_tma4::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false, true, GateUpGrid, DownIlv, GateV48>;
  static int cached_shared[16] = {0};
  int shared = (device >= 0 && device < 16) ? cached_shared[device] : 0;
  if (shared == 0) {
    if constexpr (TileN == 64 && Threads == 128) {
      // Actual usage is 45,904 B (16 KiB B + 2 KiB S + 16 KiB A + 11.1 KiB TMA4/red); 48 KiB fits 2 CTAs/SM.
      // NOTE: the 256-thread instantiation must NOT take this branch. The cross-warp reduction region is
      // red_sh_stride * (Threads/32) = 256 * 8 = 2048 int4 = 32 KiB at 8 warps; at 4 warps it is 1024 int4 = 16 KiB,
      // which happens to fit inside the B-stage overlay that this 48 KiB figure was measured against. At 256 threads
      // the real footprint is 57,472 B, so requesting 48 KiB is an out-of-bounds dynamic shared memory request and
      // faults with an illegal memory access. 256 threads therefore takes the opt-in-maximum branch below; lm_head
      // runs 48 CTAs on 48 SMs, so 1 CTA/SM is what it gets either way and the larger request costs nothing.
      shared = 48 * 1024;
    } else {
      auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (status != cudaSuccess) return status;
    }
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    if (device >= 0 && device < 16) cached_shared[device] = shared;
  }
  size_t total_sbs = 0;
  size_t max_sbs = 0;
  if constexpr (GateUpGrid) {
    total_sbs = 288 * 13 + 48 * 7;  // 4080
    max_sbs = 0;
  } else {
    struct Entry { int n, k, grid; size_t total_sbs, max_sbs; };
    static Entry cache[8] = {
        {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0},
        {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0},
    };
    static int next_slot = 0;
    bool hit = false;
    for (int i = 0; i < 8; ++i) {
      if (cache[i].n == n && cache[i].k == k && cache[i].grid == grid) {
        total_sbs = cache[i].total_sbs;
        max_sbs = cache[i].max_sbs;
        hit = true;
        break;
      }
    }
    if (!hit) {
      const int vgrid = GateV48 ? 48 : grid;
      const int n_tiles = n / TileN;
      const int k_tiles = k / 128;
      int part2_mn_tiles = n_tiles;
      int part1_mn_iters = 0;
      if (n_tiles > vgrid) {
        part2_mn_tiles = n_tiles % vgrid;
        if (part2_mn_tiles * 3 <= vgrid) part2_mn_tiles += vgrid;
        part1_mn_iters = (n_tiles - part2_mn_tiles) / vgrid;
      }
      const int p2_iters = (k_tiles * part2_mn_tiles + vgrid - 1) / vgrid;
      const int total_p2 = k_tiles * part2_mn_tiles;
      for (int b = 0; b < grid; ++b) {
        int it = 0;
        if constexpr (GateV48) {
          const int v1 = 47 - b;
          const int rem1 = total_p2 - p2_iters * v1;
          const int p2_1 = rem1 > 0 ? (rem1 < p2_iters ? rem1 : p2_iters) : 0;
          int p2_2 = 0;
          if (v1 >= grid) {
            const int v2 = v1 - grid;
            const int rem2 = total_p2 - p2_iters * v2;
            p2_2 = rem2 > 0 ? (rem2 < p2_iters ? rem2 : p2_iters) : 0;
          }
          it = p2_1 + p2_2;
        } else {
          const int rem = total_p2 - p2_iters * b;
          const int p2 = rem > 0 ? (rem < p2_iters ? rem : p2_iters) : 0;
          it = part1_mn_iters * k_tiles + p2;
        }
        const size_t sbs = static_cast<size_t>((it + 3) / 4);
        total_sbs += sbs;
        if (sbs > max_sbs) max_sbs = sbs;
      }
      cache[next_slot] = {n, k, grid, total_sbs, max_sbs};
      next_slot = (next_slot + 1) & 7;
    }
  }
  const size_t meta_bytes = ((static_cast<size_t>(4 + grid) + max_sbs + 3) / 4) * 16;
  const size_t off16_bytes = ((total_sbs + 1) * 4 + 15) / 16 * 16;
  const size_t states_off = 4096 + meta_bytes + off16_bytes;
  const size_t var_off = states_off + static_cast<size_t>(grid) * Threads * 4;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut_and_meta = reinterpret_cast<const int*>(base);
  const auto* states = reinterpret_cast<const int4*>(base + states_off);
  const auto* var = reinterpret_cast<const int4*>(base + var_off);
  kernel<<<grid, Threads, shared, stream>>>(
      input, weight, output, partials, nullptr, nullptr, states, global_scale,
      var, lut_and_meta, k / 16, 1, n, k, k, locks, false, false, true, shared, palette,
      decode6_tpf_stride(), wptrs, wtiles, wchunks);
  return cudaGetLastError();
}

}  // namespace

extern "C" int muse_marlin6_gate_up_tma4(const int4* input, const int4* weight, int4* output,
                                         int4* partials, const int4* coded, const int4* palette,
                                         const float* global_scale, int* locks, int n, int k, int blocks,
                                         int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
  return launch_tma4<128, 256, true>(input, weight, output, partials, coded, palette,
                                     global_scale, locks, n, k, 336, device, stream);
}

extern "C" int muse_marlin6_sk_tma4(const int4* input, const int4* weight, int4* output,
                                    int4* partials, const int4* coded, const int4* palette,
                                    const float* global_scale, int* locks, int n, int k, int blocks,
                                    int device, cudaStream_t stream) {
  if (k % 128 != 0 || blocks < 48 || blocks > 1024) return cudaErrorInvalidValue;
  if (n == 4096 && k == 6656) {
    const int gb = muse_marlin6_get_gate_blocks();
    if (gb == 24) {
      return launch_tma4<128, 256, false, false, true>(input, weight, output, partials, coded, palette,
                                                       global_scale, locks, n, k, gb, device, stream);
    }
    return launch_tma4<128, 256, false, false, false>(input, weight, output, partials, coded, palette,
                                                      global_scale, locks, n, k, gb, device, stream);
  }
  if (n % 128 == 0)
    return launch_tma4<128, 256, false, false, false>(input, weight, output, partials, coded, palette,
                                                      global_scale, locks, n, k, 48, device, stream);
  if (n % 64 == 0)
    // 64-column tiles run 128 threads (4 warps), so lm_head takes the caller's wider grid: 336 CTAs give 2 CTAs
    // (8 warps) per SM and drop the 48-CTA wave tail, with each output column still reduced in ascending k order
    // inside a single CTA.
    return launch_tma4<64, 128, false, false, false>(input, weight, output, partials, coded, palette,
                                                     global_scale, locks, n, k, blocks, device, stream);
  return cudaErrorInvalidValue;
}

// 64-column tiles at 256 threads: identical grid, identical schedule, identical reduction tree, 8 warps/SM instead of
// 4. The rANS warp mapping is generalized (64 B per warp instead of 128 B) on both the encode and decode sides, so the
// decoded scale bytes are unchanged and the GEMV is bit-exact against the 128-thread instantiation by construction.
extern "C" int muse_marlin6_sk_tma4_t256(const int4* input, const int4* weight, int4* output,
                                         int4* partials, const int4* coded, const int4* palette,
                                         const float* global_scale, int* locks, int n, int k, int blocks,
                                         int device, cudaStream_t stream) {
  if (k % 128 != 0 || n % 64 != 0 || n % 128 == 0 || blocks < 48 || blocks > 1024)
    return cudaErrorInvalidValue;
  return launch_tma4<64, 256, false, false, false>(input, weight, output, partials, coded, palette,
                                                   global_scale, locks, n, k, blocks, device, stream);
}

// Same kernels with a row-chunked weight placement: the caller may split the 3157 output-column tiles across several
// separately allocated buffers (each small enough to land in the fast band) and pass the chunk bases plus the
// cumulative tile offsets. Only the weight base moves -- the tile index, the k-chain, the grid and the schedule are
// unchanged -- so the result is bit-exact. wptrs == nullptr is the contiguous path we ship today.
extern "C" int muse_marlin6_sk_tma4_chunked(const int4* input, const int4* weight, int4* output,
                                            int4* partials, const int4* coded, const int4* palette,
                                            const float* global_scale, int* locks, int n, int k, int blocks,
                                            int device, cudaStream_t stream,
                                            const int64_t* wptrs, const int* wtiles, int wchunks) {
  if (k % 128 != 0 || n % 64 != 0 || blocks < 48 || blocks > 1024 || wchunks < 1)
    return cudaErrorInvalidValue;
  if (n % 128 == 0)
    return launch_tma4<128, 256, false, false, false>(input, weight, output, partials, coded, palette,
                                                      global_scale, locks, n, k, 48, device, stream,
                                                      wptrs, wtiles, wchunks);
  return launch_tma4<64, 128, false, false, false>(input, weight, output, partials, coded, palette,
                                                   global_scale, locks, n, k, blocks, device, stream,
                                                   wptrs, wtiles, wchunks);
}

extern "C" int muse_marlin6_sk_tma4_ilv(const int4* input, const int4* weight, int4* output,
                                        int4* partials, const int4* coded, const int4* palette,
                                        const float* global_scale, int* locks, int n, int k, int blocks,
                                        int device, cudaStream_t stream) {
  if (k % 128 != 0 || n % 128 != 0 || blocks < 48 || blocks > 1024) return cudaErrorInvalidValue;
  if (n == 4096 && k == 6656) {
    const int gb = muse_marlin6_get_gate_blocks();
    if (gb == 24) {
      return launch_tma4<128, 256, false, true, true>(input, weight, output, partials, coded, palette,
                                                      global_scale, locks, n, k, 24, device, stream);
    }
    if (gb != 48) return cudaErrorInvalidValue;
    return launch_tma4<128, 256, false, true, false>(input, weight, output, partials, coded, palette,
                                                     global_scale, locks, n, k, 48, device, stream);
  } else if (n == 6656 && k == 19968) {
    // The down projection is the only 48-CTA stream-K ILV shape that is re-launched on the caller's wider grid;
    // its weight permutation is rebuilt from the same schedule, so the admitted bytes are unchanged.
    return launch_tma4<128, 256, false, true, false>(input, weight, output, partials, coded, palette,
                                                     global_scale, locks, n, k, blocks, device, stream);
  } else if (!((n == 39936 && k == 6656) || (n == 4608 && k == 6656) || (n == 6656 && k == 4096))) {
    return cudaErrorInvalidValue;
  }
  return launch_tma4<128, 256, false, true, false>(input, weight, output, partials, coded, palette,
                                                   global_scale, locks, n, k, 48, device, stream);
}

extern "C" int muse_tma4_pass1(const uint8_t* sym_tiled, int k_tiles,
                               const int* stage_col_k, const int* blk_stage_off,
                               const uint32_t* blk_sb_off, const uint32_t* sb_gsb_map,
                               const uint32_t* enc_f, const uint32_t* enc_c,
                               uint32_t* sb_wc, int num_blks, int threads, int stage_bytes,
                               cudaStream_t stream) {
  if (threads == 256 && stage_bytes == 1024)
    tma4_pass1_kernel<256, 1024><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c, sb_wc);
  else if (threads == 256 && stage_bytes == 512)
    tma4_pass1_kernel<256, 512><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c, sb_wc);
  else if (threads == 128 && stage_bytes == 512)
    tma4_pass1_kernel<128, 512><<<num_blks, 128, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c, sb_wc);
  else
    return cudaErrorInvalidValue;
  return cudaError_t(cudaGetLastError());
}

extern "C" int muse_tma4_pass2(const uint8_t* sym_tiled, int k_tiles,
                               const int* stage_col_k, const int* blk_stage_off,
                               const uint32_t* blk_sb_off, const uint32_t* sb_gsb_map,
                               const uint32_t* enc_f, const uint32_t* enc_c,
                               const uint32_t* sb_wc, const uint32_t* off16,
                               uint32_t* states_out, unsigned char* var_out,
                               int num_blks, int threads, int stage_bytes,
                               cudaStream_t stream) {
  if (threads == 256 && stage_bytes == 1024)
    tma4_pass2_kernel<256, 1024><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c,
        sb_wc, off16, states_out, var_out);
  else if (threads == 256 && stage_bytes == 512)
    tma4_pass2_kernel<256, 512><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c,
        sb_wc, off16, states_out, var_out);
  else if (threads == 128 && stage_bytes == 512)
    tma4_pass2_kernel<128, 512><<<num_blks, 128, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c,
        sb_wc, off16, states_out, var_out);
  else
    return cudaErrorInvalidValue;
  return cudaError_t(cudaGetLastError());
}
