// SPDX-License-Identifier: Apache-2.0
// draft_head_fast: Hadamard-64 Symmetric NF3 (3.125 bpw), 2D Pair-VQ2.5 (2.625 bpw),
// 4D Antipodal-VQ2.0 (2.125 bpw, R^4/{+-1}), and 4D Antipodal-VQ2.25 (2.375 bpw, R^4/{+-1})
// persistent wave-interleaved Hardware-TMA (`cp.async.bulk` + `L2::evict_first`) Tensor-Core
// `draft_head` with golden-ratio coprime row permutation (zero bucket collisions) and fused
// per-CTA Top-K candidate extraction (zero `static_logits` DRAM write/read), paired with
// exact-NVFP4 top-K rescore in `draft_vocab_tail_argmax`.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdint.h>
#include <tuple>

void vq4d_clear_all_prerot();
void vq4d_reset_norm_rot_table();

namespace {

constexpr int kDhWarps = 4;
constexpr int kDhThreads = kDhWarps * 32;            // 128
constexpr int kDhSubTilesPerCTA = kDhWarps * 2;      // 8 sub-tiles = 64 N-rows/CTA
constexpr int kDhRowsPerCTA = 64;
constexpr int kDhBK = 128;                           // 8 x m16n8k16 steps
constexpr int kDhStages = 4;
constexpr int kDhMaxM = 8;
constexpr int kDhMaxCandPool = 512;
constexpr int kDhSentinelCols = 32;

// Sub-tile byte sizes (all 16B aligned; 8 sub-tiles/CTA -> exact 128B TMA multiples):
//   Mode 0 (NF3, 3.125 bpw):          400 B/sub-tile -> 3200 B/step = 25 * 128 B
//   Mode 1 (2D VQ2.5, 2.625 bpw):     336 B/sub-tile -> 2688 B/step = 21 * 128 B
//   Mode 2 (4D Anti-VQ2.0, 2.125 bpw):272 B/sub-tile -> 2176 B/step = 17 * 128 B (0 funnelshift, 1x LDS.64)
//   Mode 3 (4D Anti-VQ2.25, 2.375 bpw):304 B/sub-tile -> 2432 B/step = 19 * 128 B (0 funnelshift, 1x LDS.64)
constexpr int kNf3SubTileBytes = 400;                // 384 B (3x u32 x 32) + 16 B (8x bf16 scale)
constexpr int kVq25SubTileBytes = 336;               // 320 B ((2x u32 + u16) x 32) + 16 B (8x bf16 scale)
constexpr int kVq20SubTileBytes = 272;               // 256 B (2x u32 x 32) + 16 B (8x bf16 scale)
constexpr int kVq225SubTileBytes = 304;              // 288 B ((2x u32 + u8) x 32) + 16 B (8x bf16 scale)

template <int kMode>
__host__ __device__ constexpr int dh_sub_bytes() {
  if constexpr (kMode == 0) return kNf3SubTileBytes;
  else if constexpr (kMode == 1) return kVq25SubTileBytes;
  else if constexpr (kMode == 2) return kVq20SubTileBytes;
  else return kVq225SubTileBytes;
}

template <int kMode>
__host__ __device__ constexpr int dh_step_bytes() {
  return kDhSubTilesPerCTA * dh_sub_bytes<kMode>();
}

static int64_t dh_sub_bytes_rt(int mode) {
  if (mode == 0) return kNf3SubTileBytes;
  if (mode == 1) return kVq25SubTileBytes;
  if (mode == 2) return kVq20SubTileBytes;
  return kVq225SubTileBytes;
}

static bool dh_dyn_env() {
  static const bool dyn = []() {
    const char* s = std::getenv("MUSE_DHEAD_DYN_ENV");
    return s && *s && std::atoi(s) != 0;
  }();
  return dyn;
}

static int dh_pdl_mode() {
  auto parse = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  };
  if (dh_dyn_env()) return parse();
  static const int cached = parse();
  return cached;
}

static int dh_rescore_k_cfg() {
  auto parse = []() -> int {
    const char* s = std::getenv("MUSE_DHEAD_RESCORE_K");
    if (!s || !*s) return 64;
    const int v = std::atoi(s);
    return std::max(16, std::min(kDhMaxCandPool, v));
  };
  if (dh_dyn_env()) return parse();
  static const int cached = parse();
  return cached;
}

// Golden-ratio coprime stride in Z_N: scatters consecutive sorted BPE/ASCII token IDs
// uniformly across all tiles, CTAs, warps, and t-lanes (< 1e-7 bucket collision rate!).
static uint32_t dh_perm_stride(int N) {
  const uint32_t un = static_cast<uint32_t>(N);
  uint32_t p = static_cast<uint32_t>(((static_cast<uint64_t>(un) * 2654435761ull) >> 32) | 1ull);
  if (p < 3u) p = 3u;
  while (std::gcd(p, un) != 1u) {
    p += 2u;
  }
  return p;
}

template <typename... KArgs, typename... Args>
static inline void dh_launch_pdl(
    void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = grid;
  lcfg.blockDim = dim3(static_cast<unsigned>(block), 1u, 1u);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = st;
  cudaLaunchAttribute attr[1];
  if ((pdl_mode & 1) != 0) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
  }
  cudaError_t err = cudaLaunchKernelEx(&lcfg, kernel, static_cast<KArgs>(args)...);
  TORCH_CHECK(err == cudaSuccess, "draft_head_fast launch failed: ", cudaGetErrorString(err));
}

// Deterministic Rademacher sign (+1.0f or -1.0f) from column index k.
__device__ __forceinline__ float dh_rad_sign(int k) {
  uint32_t x = static_cast<uint32_t>(k) * 0x9E3779B9u + 0x7F4A7C15u;
  x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
  x = (x ^ (x >> 13)) * 0xC2B2AE35u;
  x ^= (x >> 16);
  return (x & 1u) ? -1.0f : 1.0f;
}

// Exact normalized 64-point Walsh-Hadamard transform (H_64 / 8) across one warp (32 lanes x 2 floats).
__device__ __forceinline__ void dh_warp_fwht64(float& v0, float& v1, int lane) {
  {
    const float s = v0 + v1;
    const float d = v0 - v1;
    v0 = s;
    v1 = d;
  }
#pragma unroll
  for (int mask = 1; mask <= 16; mask <<= 1) {
    const float o0 = __shfl_xor_sync(0xffffffffu, v0, mask);
    const float o1 = __shfl_xor_sync(0xffffffffu, v1, mask);
    const bool hi = (lane & mask) != 0;
    v0 = hi ? (o0 - v0) : (v0 + o0);
    v1 = hi ? (o1 - v1) : (v1 + o1);
  }
  v0 *= 0.125f;
  v1 *= 0.125f;
}

// Exact NVFP4 E2M1x2 & FP8 E4M3 helpers (matching draft_vocab_tail.cu bitwise).
__device__ __forceinline__ float dh_e2m1_x2_to_f32(uint32_t nib) {
  const uint32_t mag_u = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
  const float mag = __uint_as_float(0x41800000u | (mag_u << 19)) - 16.0f;
  return __uint_as_float(__float_as_uint(mag) ^ ((nib & 8u) << 28));
}

__device__ __forceinline__ float dh_fp8_e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

// Symmetric 8-level Lloyd-Max Normal(0, 1) NF3 codebook.
__constant__ float kDhNf3Cb[8] = {
    -2.1519927f, -1.3439093f, -0.7560053f, -0.2450942f,
    +0.2450942f, +0.7560053f, +1.3439093f, +2.1519927f,
};

// Optimal 32-centroid 2D Gaussian Lloyd-Max VQ codebook (2.5 bits/weight).
__constant__ float2 kDhVq25Cb[32] = {
    {+0.336695f, +0.235905f}, {-0.221841f, +0.311697f}, {-0.410038f, -0.229322f}, {+0.157453f, -0.307426f},
    {+0.937875f, +0.342424f}, {+0.388188f, +0.849478f}, {-0.343749f, +0.863302f}, {-0.875162f, +0.305411f},
    {-1.020643f, -0.387199f}, {-0.401048f, -0.826984f}, {+0.334300f, -0.888789f}, {+0.824240f, -0.310415f},
    {+1.575516f, +0.022458f}, {+1.214165f, +0.978337f}, {+0.421903f, +1.533850f}, {-0.424192f, +1.515543f},
    {-1.186799f, +0.947133f}, {-1.643692f, +0.012782f}, {-1.203122f, -1.066539f}, {-0.311558f, -1.495596f},
    {+0.576900f, -1.490208f}, {+1.226518f, -0.875182f}, {+2.221101f, +0.807127f}, {+1.339640f, +1.921301f},
    {-0.006492f, +2.464531f}, {-1.308323f, +1.928008f}, {-2.202196f, +0.801181f}, {-2.229151f, -0.889488f},
    {-1.152257f, -2.075637f}, {+0.200840f, -2.368705f}, {+1.524886f, -1.809442f}, {+2.240060f, -0.632572f},
};

// Optimal 128-centroid BF16-exact 4D Antipodal Gaussian VQ codebook (256 points via {+-1}, 2.0 bpw).
__constant__ float4 kDh4dAnti20Cb[128] = {
    {-0.94140625f, -1.5625f, +0.15234375f, -2.015625f}, {-1.4609375f, +0.296875f, -0.38671875f, -1.09375f},
    {+1.1015625f, -0.118164062f, +1.1875f, +0.408203125f}, {-0.233398438f, +1.734375f, -0.72265625f, -0.244140625f},
    {-1.1171875f, +0.80859375f, -0.26953125f, +0.7734375f}, {+1.625f, +1.6328125f, -0.34765625f, -1.453125f},
    {-0.232421875f, -0.65234375f, +0.357421875f, -0.80859375f}, {+1.28125f, +0.0245361328f, -0.55859375f, +2.25f},
    {-1.4375f, +0.00115203857f, +0.58984375f, +2.0625f}, {+0.359375f, -0.193359375f, +0.435546875f, -2.5f},
    {+0.423828125f, -0.298828125f, -1.8515625f, -1.75f}, {-1.15625f, -2.125f, -1.0625f, -0.78515625f},
    {+1.953125f, -0.578125f, +2.140625f, -0.56640625f}, {+1.234375f, +0.247070312f, +0.396484375f, -0.2421875f},
    {-0.32421875f, +0.139648438f, -0.88671875f, -2.4375f}, {-1.390625f, +1.1640625f, -0.0471191406f, -1.8203125f},
    {+0.408203125f, +0.486328125f, +0.95703125f, +1.296875f}, {-1.2421875f, +1.4921875f, +1.8984375f, +0.298828125f},
    {-1.5f, -1.6953125f, -1.375f, +1.140625f}, {-1.0859375f, -0.7578125f, -1.921875f, -1.109375f},
    {+2.375f, +0.0207519531f, +1.1328125f, +0.5625f}, {-0.1640625f, -0.9765625f, +1.8828125f, +0.84375f},
    {-1.4453125f, -0.80859375f, -0.6953125f, -0.62109375f}, {+0.150390625f, -0.71875f, +0.178710938f, +0.16015625f},
    {-0.6875f, -0.53125f, -1.9453125f, +1.3984375f}, {-0.2265625f, -0.21875f, -0.143554688f, +0.8828125f},
    {-2.09375f, -0.72265625f, +0.93359375f, -0.439453125f}, {+0.92578125f, +0.6484375f, -1.59375f, +0.0246582031f},
    {+0.3125f, -0.041015625f, -0.9296875f, -0.231445312f}, {+0.93359375f, +0.466796875f, -1.203125f, -0.95703125f},
    {+0.48828125f, +1.578125f, -0.20703125f, +0.478515625f}, {+1.3125f, -0.9140625f, -0.59765625f, -0.200195312f},
    {-0.00817871094f, -1.4453125f, -1.6796875f, -0.97265625f}, {+0.154296875f, -1.9375f, +1.359375f, -0.8984375f},
    {+0.89453125f, -0.5546875f, -0.9609375f, +1.2421875f}, {+1.1015625f, +1.4609375f, +0.48828125f, -0.17578125f},
    {+1.0234375f, -0.365234375f, +1.3359375f, -0.58984375f}, {+0.421875f, -1.09375f, +0.65625f, -0.609375f},
    {+0.466796875f, +0.068359375f, -1.8046875f, +2.0625f}, {-0.63671875f, +0.63671875f, +1.28125f, +0.87109375f},
    {-0.0402832031f, -1.2265625f, +0.8671875f, -1.6484375f}, {+1.8125f, -1.484375f, +0.64453125f, +0.80859375f},
    {-0.7578125f, +0.2734375f, -2.328125f, -0.103027344f}, {-0.107910156f, +0.71875f, -0.9375f, -0.44921875f},
    {-1.0859375f, +1.34375f, -0.174804688f, +1.984375f}, {+1.046875f, -0.60546875f, -0.213867188f, +0.54296875f},
    {+0.0776367188f, +1.40625f, +0.92578125f, +2.09375f}, {-0.137695312f, +2.25f, +0.88671875f, +0.0981445312f},
    {+1.1328125f, -2.0625f, -0.6328125f, -0.828125f}, {-0.196289062f, -0.0270996094f, -1.5859375f, -0.71484375f},
    {+1.515625f, +2.046875f, -0.578125f, +0.62890625f}, {+2.1875f, -0.62109375f, -0.68359375f, +0.984375f},
    {-1.2265625f, -0.0512695312f, +0.75390625f, -0.0112304688f}, {+0.54296875f, -0.318359375f, +0.0554199219f, +1.5234375f},
    {-1.21875f, +0.64453125f, -1.953125f, -1.4140625f}, {-1.8203125f, +0.69921875f, -0.82421875f, +0.151367188f},
    {+0.111816406f, -1.0703125f, -1.203125f, +0.71875f}, {+0.116210938f, -0.50390625f, -0.34765625f, -0.5859375f},
    {+0.83203125f, +0.6484375f, +0.0324707031f, +1.40625f}, {-0.95703125f, +0.275390625f, +0.515625f, +0.91015625f},
    {-0.490234375f, -0.188476562f, +1.046875f, -0.58203125f}, {+0.16796875f, -0.271484375f, +1.25f, -1.1328125f},
    {+0.8125f, -2.5625f, +0.168945312f, +0.46875f}, {-1.6171875f, +1.0390625f, +0.89453125f, +1.3828125f},
    {+0.6328125f, -0.703125f, +1.0234375f, +1.2265625f}, {+0.93359375f, -1.5078125f, +1.6484375f, +0.25390625f},
    {-1.8359375f, -1.078125f, +1.78125f, +0.7734375f}, {+0.388671875f, -1.1171875f, +0.108398438f, +0.9921875f},
    {-2.09375f, -0.76171875f, +0.119140625f, -1.296875f}, {+0.76171875f, -0.333984375f, -0.0546875f, -0.221679688f},
    {-0.2421875f, +0.52734375f, +0.59375f, -0.5625f}, {-0.859375f, +0.0947265625f, -0.451171875f, +1.296875f},
    {-0.5078125f, +1.84375f, -0.953125f, -1.515625f}, {+1.5f, +0.82421875f, +1.640625f, -0.145507812f},
    {+0.671875f, +0.58203125f, -0.33203125f, -0.451171875f}, {+1.4140625f, +0.482421875f, +0.96484375f, +1.859375f},
    {+1.1015625f, -0.8671875f, +1.40625f, -1.609375f}, {-0.20703125f, -1.0546875f, -2.34375f, +0.0834960938f},
    {-0.8203125f, +1.3671875f, +1.796875f, -1.3359375f}, {-0.3203125f, -0.119628906f, +0.208007812f, -0.176757812f},
    {-0.466796875f, -0.62109375f, +0.318359375f, +1.6328125f}, {+0.431640625f, -1.3046875f, -0.3203125f, -1.03125f},
    {+0.122070312f, +0.7734375f, +0.83984375f, -1.6484375f}, {+0.8984375f, -0.58984375f, -1.3984375f, +0.216796875f},
    {-0.171875f, -1.09375f, +2.375f, -0.765625f}, {+0.0368652344f, +1.9453125f, +0.64453125f, -1.375f},
    {+0.11328125f, +0.208984375f, -0.7421875f, -1.1015625f}, {+0.5546875f, +0.73828125f, +1.1171875f, +0.255859375f},
    {-0.97265625f, +0.87890625f, -0.65625f, -0.215820312f}, {-1.1171875f, +1.6171875f, +0.7109375f, -0.71484375f},
    {+0.828125f, +1.078125f, +0.30078125f, -1.109375f}, {-0.243164062f, -0.184570312f, +0.66015625f, -1.5546875f},
    {+0.5f, +1.390625f, -0.609375f, -0.921875f}, {+1.859375f, -0.0688476562f, +0.0291748047f, +0.177734375f},
    {+0.155273438f, +0.22265625f, +1.5234375f, -0.2265625f}, {-0.7109375f, +1.4609375f, +0.0164794922f, +0.0825195312f},
    {+1.421875f, +1.21875f, -0.57421875f, -0.30859375f}, {+0.76953125f, +0.62890625f, +1.0390625f, -0.7578125f},
    {+0.54296875f, +1.0546875f, -0.8203125f, +0.182617188f}, {-0.640625f, -0.0151977539f, -0.396484375f, -0.61328125f},
    {-0.6953125f, +0.0302734375f, +2.484375f, +0.470703125f}, {+1.546875f, +0.5078125f, +0.8125f, -2.015625f},
    {+1.171875f, +0.235351562f, -0.625f, +0.9375f}, {+0.41015625f, -0.145507812f, +0.70703125f, -0.275390625f},
    {+0.124511719f, +1.4140625f, +0.341796875f, -0.341796875f}, {-0.640625f, -1.9375f, +1.5234375f, +0.287109375f},
    {+0.188476562f, -0.84375f, +1.4765625f, -0.27734375f}, {-1.0625f, -1.0625f, +1.3125f, -1.2265625f},
    {-0.0593261719f, +2.34375f, -0.174804688f, +1.0234375f}, {+1.859375f, +0.43359375f, +0.8671875f, -0.7890625f},
    {+2.5625f, +0.244140625f, -0.48046875f, -0.75390625f}, {+2.25f, +1.3671875f, +0.40625f, -0.0986328125f},
    {-0.2578125f, +0.53515625f, +0.192382812f, +1.6015625f}, {+0.47265625f, +0.671875f, +0.3671875f, -0.0368652344f},
    {+0.33203125f, -1.0625f, -0.87109375f, -0.216796875f}, {-0.51171875f, -1.765625f, -1.4765625f, +0.35546875f},
    {+0.392578125f, -0.114746094f, -2.21875f, +0.76953125f}, {+1.8203125f, -0.271484375f, -1.4609375f, -0.61328125f},
    {+0.54296875f, +1.0859375f, -1.4609375f, -1.9375f}, {+0.112304688f, +1.3203125f, +0.162109375f, -2.390625f},
    {-1.71875f, +0.0717773438f, +1.8828125f, -0.8515625f}, {+1.515625f, +0.427734375f, -0.259765625f, -0.91015625f},
    {-1.4296875f, +1.8515625f, -0.78515625f, +0.60546875f}, {-0.515625f, -1.296875f, -0.5546875f, -0.83984375f},
    {-0.67578125f, -2.46875f, +0.08203125f, +0.46875f}, {-2.078125f, +0.5703125f, -0.37109375f, +1.171875f},
    {+2.25f, -1.2265625f, -0.306640625f, -0.112304688f}, {+1.125f, +0.74609375f, -0.157226562f, +0.421875f},
};

// Optimal 256-centroid BF16-exact 4D Antipodal Gaussian VQ codebook (512 points via {+-1}, 2.25 bpw).
__constant__ float4 kDh4dAnti225Cb[256] = {
    {+1.703125f, -0.45703125f, -0.00127410889f, +0.87109375f}, {+0.46875f, -1.0859375f, -2.015625f, +0.201171875f},
    {-0.07421875f, -0.139648438f, -0.412109375f, -0.94140625f}, {-1.3984375f, -0.149414062f, -0.255859375f, -0.28515625f},
    {+0.9140625f, +0.494140625f, +1.1484375f, -1.015625f}, {-1.3203125f, +1.9296875f, +0.6015625f, -0.2734375f},
    {-1.4140625f, +0.45703125f, -0.91015625f, -0.6328125f}, {+0.9921875f, -0.5703125f, -0.484375f, -1.65625f},
    {-1.7734375f, +1.5078125f, -1.734375f, +0.0444335938f}, {-0.20703125f, +0.8046875f, +0.392578125f, +1.296875f},
    {-0.3359375f, -1.640625f, -0.0299072266f, +1.6796875f}, {-0.37109375f, -1.171875f, -0.3515625f, -0.188476562f},
    {+2.65625f, -0.453125f, +0.5859375f, -1.015625f}, {+0.0354003906f, -0.69921875f, +0.287109375f, -1.2578125f},
    {+0.7109375f, +0.104492188f, -0.9453125f, -0.0693359375f}, {+0.245117188f, +0.65625f, +1.0078125f, -2.515625f},
    {+0.49609375f, -1.3671875f, +1.453125f, -0.1484375f}, {-1.453125f, +2.5f, -0.209960938f, -0.5703125f},
    {-0.625f, -0.52734375f, -0.104003906f, +0.84765625f}, {+1.7265625f, +0.546875f, -0.82421875f, +2.203125f},
    {-1.234375f, +2.09375f, +0.5234375f, +0.71875f}, {-0.9765625f, +0.0966796875f, +1.34375f, +1.3125f},
    {+1.53125f, +0.91796875f, -0.78125f, +0.9140625f}, {-1.1640625f, +1.2578125f, -0.98046875f, +0.486328125f},
    {-1.0625f, +0.26171875f, -1.109375f, +0.349609375f}, {+1.75f, +0.143554688f, -0.466796875f, -1.6796875f},
    {+2.4375f, -0.228515625f, +0.443359375f, +0.267578125f}, {+1.2890625f, -1.3203125f, +0.165039062f, -0.306640625f},
    {-0.486328125f, -0.0238037109f, +0.243164062f, -1.0546875f}, {+1.890625f, +1.8359375f, +0.0576171875f, -0.0524902344f},
    {-0.4609375f, +1.1875f, +0.416015625f, -0.064453125f}, {+0.0830078125f, +2.015625f, -0.143554688f, +0.515625f},
    {-0.734375f, -0.78125f, -0.66796875f, +0.41015625f}, {-1.875f, +0.56640625f, +0.197265625f, +0.0427246094f},
    {-0.15234375f, -2.5625f, +0.451171875f, +1.0078125f}, {+1.0078125f, -0.00318908691f, +0.3359375f, -1.1953125f},
    {-0.32421875f, +1.9296875f, -1.078125f, -2.421875f}, {-0.76953125f, +1.1171875f, +0.60546875f, +0.734375f},
    {-0.94921875f, -1.421875f, -1.109375f, +0.283203125f}, {+0.69140625f, -1.6484375f, +0.6484375f, +1.3046875f},
    {-1.328125f, +0.455078125f, +0.671875f, +0.87890625f}, {-1.8046875f, -0.984375f, -0.1953125f, +1.3671875f},
    {-1.4765625f, -0.921875f, +0.302734375f, +0.168945312f}, {-0.40625f, -0.83984375f, +0.6484375f, +0.73046875f},
    {+0.9921875f, -0.95703125f, +1.4609375f, -2.234375f}, {+0.52734375f, +0.03515625f, -1.828125f, +1.953125f},
    {+0.91796875f, +1.515625f, -0.30859375f, +0.375f}, {-0.0505371094f, +0.443359375f, -0.0593261719f, -0.87109375f},
    {+0.87890625f, +0.609375f, -1.2890625f, +1.3515625f}, {+0.5234375f, -1.328125f, -0.92578125f, +0.875f},
    {+0.0908203125f, +0.88671875f, -0.46875f, -1.546875f}, {-0.2578125f, -0.9140625f, +0.7109375f, -2.640625f},
    {-2.015625f, -1.8359375f, +0.127929688f, -1.140625f}, {+0.875f, -0.208984375f, +0.2578125f, +0.86328125f},
    {+0.44140625f, -1.0390625f, +0.10546875f, +2.390625f}, {+0.59765625f, -0.190429688f, +0.890625f, +1.1953125f},
    {+0.7265625f, +0.62890625f, +0.466796875f, +1.3046875f}, {-1.046875f, -2.78125f, +0.18359375f, -0.133789062f},
    {-0.640625f, +0.259765625f, +0.188476562f, +0.73046875f}, {+0.82421875f, -1.2265625f, -1.1171875f, -0.0366210938f},
    {-1.265625f, -1.3046875f, -1.2734375f, -0.58984375f}, {+1.21875f, +1.3515625f, +1.296875f, -1.5703125f},
    {-0.33984375f, +0.0368652344f, -0.263671875f, -0.337890625f}, {-0.39453125f, -1.2109375f, -1.1640625f, +1.1171875f},
    {-1.4453125f, -0.369140625f, -1.0859375f, -0.1328125f}, {+0.51171875f, +1.4296875f, +1.3828125f, +1.9609375f},
    {-1.0f, +0.84375f, -2.046875f, +0.87890625f}, {+0.462890625f, -0.8828125f, +0.145507812f, -0.388671875f},
    {+0.88671875f, +0.267578125f, -0.0727539062f, -0.0620117188f}, {+2.21875f, +0.231445312f, +1.6484375f, +0.9765625f},
    {-0.453125f, +0.4375f, -1.640625f, +0.13671875f}, {+0.0242919922f, +1.0859375f, +0.30859375f, +2.09375f},
    {-1.9921875f, -0.44921875f, +0.35546875f, -0.490234375f}, {+0.0278320312f, -0.298828125f, -0.7734375f, -0.0607910156f},
    {+0.0145874023f, +0.73828125f, -1.5859375f, +0.72265625f}, {+0.94921875f, +0.99609375f, -1.8828125f, +0.3203125f},
    {-0.96484375f, -0.86328125f, +1.203125f, +1.5625f}, {+1.515625f, +0.263671875f, -1.15625f, -0.48828125f},
    {+1.9921875f, +0.2578125f, -0.0524902344f, -0.52734375f}, {+0.421875f, +0.0515136719f, +0.443359375f, +2.8125f},
    {-0.369140625f, -1.8671875f, -0.0561523438f, +0.34375f}, {-1.578125f, +0.248046875f, -2.640625f, -0.0747070312f},
    {-0.143554688f, -0.86328125f, +1.8203125f, +0.41796875f}, {-0.0747070312f, -0.33203125f, -1.0546875f, -1.765625f},
    {-0.8359375f, +0.451171875f, +0.5390625f, +0.153320312f}, {+0.875f, +0.9453125f, -1.15625f, -0.423828125f},
    {+0.00720214844f, -0.263671875f, +1.1796875f, +0.40625f}, {-1.265625f, +1.703125f, +0.236328125f, -1.609375f},
    {-2.21875f, -0.46484375f, -1.46875f, +1.3515625f}, {+0.0498046875f, -0.431640625f, -0.213867188f, +1.6171875f},
    {+2.875f, +1.0703125f, -0.423828125f, -0.365234375f}, {-0.55078125f, +0.79296875f, -0.8671875f, +0.76171875f},
    {+0.3984375f, +2.25f, +0.0849609375f, +1.59375f}, {-2.171875f, +1.2421875f, -1.0234375f, -1.1015625f},
    {-1.9140625f, -0.151367188f, +1.9140625f, +1.1796875f}, {-0.56640625f, +0.0844726562f, +0.84765625f, +0.70703125f},
    {-1.4765625f, -0.41015625f, -0.57421875f, +0.68359375f}, {+0.217773438f, -0.3828125f, +0.703125f, -0.212890625f},
    {+0.046875f, +0.84375f, -2.375f, +1.296875f}, {-0.3203125f, -0.392578125f, +2.328125f, +1.6328125f},
    {+1.6171875f, -0.151367188f, -0.78125f, +0.037109375f}, {-0.0240478516f, +1.4609375f, -1.0078125f, -0.294921875f},
    {+0.314453125f, -0.171875f, +1.6875f, +1.0234375f}, {+1.359375f, -1.6484375f, -1.453125f, +1.203125f},
    {-0.6328125f, -0.88671875f, -1.78125f, +0.255859375f}, {-0.64453125f, -0.64453125f, +2.71875f, +0.0568847656f},
    {-0.8828125f, -0.72265625f, +0.296875f, +2.4375f}, {+1.3046875f, -1.4375f, +0.8125f, +0.53125f},
    {+0.0490722656f, +0.85546875f, -1.0234375f, +1.6484375f}, {+0.7578125f, +0.044921875f, +0.56640625f, -0.30859375f},
    {-0.279296875f, -0.291015625f, +0.42578125f, +0.349609375f}, {+0.77734375f, +0.52734375f, +2.453125f, +1.3046875f},
    {-0.9765625f, +0.78125f, -1.546875f, -0.419921875f}, {+0.180664062f, +2.390625f, -1.1015625f, +0.453125f},
    {+1.1484375f, -1.0859375f, -1.5546875f, -0.8203125f}, {+1.9765625f, -1.3671875f, -1.5390625f, -0.0478515625f},
    {-0.88671875f, -1.8828125f, -0.53515625f, -0.3984375f}, {-0.80859375f, +2.40625f, -0.48046875f, +1.2265625f},
    {+1.421875f, +1.0703125f, +0.205078125f, +0.8046875f}, {-1.5703125f, -0.00219726562f, -0.73828125f, +2.0f},
    {+1.1875f, +0.192382812f, +1.1171875f, +2.015625f}, {+0.287109375f, -1.140625f, -2.140625f, +1.4375f},
    {+0.443359375f, -0.5703125f, -1.296875f, -0.2890625f}, {+0.181640625f, +1.515625f, +2.28125f, +0.6171875f},
    {-1.796875f, +1.609375f, -0.70703125f, +1.2890625f}, {+0.9453125f, -0.7890625f, +0.19921875f, -1.0234375f},
    {+0.0815429688f, +1.828125f, -2.109375f, +0.14453125f}, {+0.5078125f, -0.609375f, -0.58203125f, +0.6015625f},
    {-1.3359375f, +0.84765625f, +1.015625f, -2.484375f}, {-0.98046875f, -0.6484375f, +0.65625f, -0.1484375f},
    {-1.8203125f, +0.54296875f, -0.326171875f, -2.140625f}, {-1.546875f, -0.287109375f, -0.69921875f, -1.0390625f},
    {+0.243164062f, +2.015625f, +1.7734375f, -0.62890625f}, {+0.88671875f, -1.984375f, +1.7421875f, +1.0390625f},
    {+0.30078125f, +1.703125f, -1.515625f, -1.0625f}, {-0.76171875f, -0.30859375f, -1.984375f, +1.7421875f},
    {+0.251953125f, +0.421875f, -1.1171875f, -1.1328125f}, {+0.8046875f, +1.640625f, -1.7265625f, +1.6015625f},
    {-1.328125f, +0.53515625f, +2.5f, -1.03125f}, {+0.53125f, -0.345703125f, +1.3046875f, -1.2109375f},
    {-0.73828125f, +1.1328125f, -0.51171875f, +1.78125f}, {+0.4453125f, -1.453125f, +0.194335938f, -0.93359375f},
    {+0.388671875f, -0.00121307373f, -0.44921875f, +0.392578125f}, {+1.5546875f, +1.3828125f, +2.140625f, +0.09375f},
    {+1.2421875f, -0.71875f, +1.6640625f, +1.4921875f}, {-1.015625f, -0.255859375f, -1.84375f, -0.51171875f},
    {+0.243164062f, -0.142578125f, +0.365234375f, -0.90234375f}, {+0.169921875f, +1.1328125f, +0.287109375f, -0.91015625f},
    {-1.9921875f, -0.63671875f, +1.8671875f, -1.1171875f}, {+0.451171875f, +1.359375f, +0.75390625f, +1.109375f},
    {+0.283203125f, +0.0874023438f, +2.515625f, -0.5859375f}, {-1.4296875f, -0.241210938f, +0.118652344f, -1.4453125f},
    {-0.0329589844f, +0.390625f, +1.625f, -1.0390625f}, {+1.1484375f, -0.0258789062f, -0.5859375f, +0.71875f},
    {+0.240234375f, +1.4609375f, -0.86328125f, +1.0390625f}, {+0.470703125f, -0.0600585938f, +0.306640625f, +1.6953125f},
    {+0.9140625f, -0.73046875f, -2.765625f, -0.66015625f}, {-0.00619506836f, -0.84765625f, -1.1171875f, +0.333984375f},
    {+1.203125f, -0.31640625f, +0.181640625f, -0.4453125f}, {-0.5390625f, +0.9921875f, +0.07421875f, -1.1171875f},
    {+0.169921875f, +0.734375f, -0.98828125f, +0.0397949219f}, {+0.73828125f, -0.8359375f, +0.6953125f, +0.0717773438f},
    {+0.67578125f, +0.71875f, -0.40625f, +1.453125f}, {-1.078125f, +0.703125f, +0.38671875f, -1.53125f},
    {-2.421875f, +0.50390625f, +0.90625f, -0.5859375f}, {+0.75390625f, +0.163085938f, -1.796875f, -0.62109375f},
    {-0.79296875f, -0.141601562f, +2.015625f, -0.75390625f}, {+1.2578125f, -0.91015625f, -0.71875f, +0.6171875f},
    {-2.125f, +0.30078125f, -1.453125f, +0.263671875f}, {-0.71484375f, -0.53515625f, -0.4765625f, -0.4765625f},
    {-0.625f, +1.8046875f, -0.359375f, +0.00946044922f}, {+0.90234375f, -0.435546875f, +0.0121459961f, +0.232421875f},
    {-0.64453125f, +0.9296875f, -2.671875f, -0.6953125f}, {-1.875f, -1.0703125f, +0.7890625f, +0.98828125f},
    {+0.37890625f, -1.0859375f, +1.3125f, +1.0078125f}, {+0.189453125f, +0.3984375f, +1.09375f, +0.796875f},
    {+0.9765625f, +1.1484375f, -0.328125f, -1.0703125f}, {+0.8671875f, -1.0859375f, -1.609375f, -2.09375f},
    {-0.9609375f, -1.5234375f, +0.384765625f, -1.5546875f}, {-2.171875f, +1.2734375f, +0.178710938f, -0.87890625f},
    {-0.244140625f, +1.8125f, +0.60546875f, +0.4453125f}, {+0.0712890625f, +1.28125f, -0.55859375f, +0.29296875f},
    {-0.87109375f, -1.171875f, -2.59375f, +0.83203125f}, {-0.375f, +0.765625f, -0.65625f, -0.69921875f},
    {+1.7578125f, +1.390625f, +0.9609375f, +1.4609375f}, {-1.53125f, -2.0625f, -0.435546875f, +2.15625f},
    {+1.0859375f, -1.140625f, +0.0771484375f, +0.54296875f}, {+0.0346679688f, +0.1796875f, +2.265625f, +0.3203125f},
    {+0.45703125f, -0.80859375f, -1.078125f, -1.1640625f}, {+1.2109375f, +0.361328125f, -0.369140625f, -0.7421875f},
    {+1.7578125f, -0.75f, -0.0388183594f, -1.1015625f}, {+0.76171875f, +2.546875f, +1.3203125f, +0.6796875f},
    {-0.7421875f, +0.0456542969f, +1.21875f, +2.4375f}, {-1.0703125f, +0.76171875f, -0.578125f, -1.4296875f},
    {-1.140625f, -1.5546875f, -0.34375f, +0.7578125f}, {-0.0278320312f, -1.375f, -1.2890625f, -0.41796875f},
    {+0.24609375f, +0.62109375f, -0.174804688f, +0.404296875f}, {+1.1796875f, +0.298828125f, -1.296875f, +0.455078125f},
    {+1.609375f, -0.400390625f, -1.15625f, +1.265625f}, {-0.61328125f, -1.046875f, +0.170898438f, +0.244140625f},
    {+1.359375f, +2.015625f, -0.69921875f, -1.5078125f}, {-1.9765625f, +1.328125f, +1.046875f, +1.4453125f},
    {+0.314453125f, -0.00340270996f, -1.53125f, +0.225585938f}, {+1.4921875f, -0.546875f, +0.96484375f, -1.0859375f},
    {-1.28125f, -0.94921875f, -0.416015625f, -0.00988769531f}, {+0.50390625f, -0.0849609375f, +0.828125f, -1.796875f},
    {+0.6328125f, +0.0294189453f, -0.5f, +2.0625f}, {+0.9296875f, +0.61328125f, +1.3046875f, +1.1015625f},
    {+1.8671875f, +1.1796875f, +1.1640625f, -0.447265625f}, {-0.125976562f, -0.50390625f, -0.76171875f, +1.171875f},
    {+0.117675781f, -0.71875f, -1.8046875f, -1.015625f}, {-1.90625f, -1.1171875f, +1.28125f, -0.100585938f},
    {-0.451171875f, +0.8515625f, +1.046875f, -1.6796875f}, {-1.171875f, +0.48828125f, +1.6796875f, +0.0117797852f},
    {+0.72265625f, +1.2421875f, -1.125f, +0.4375f}, {-0.255859375f, +1.9765625f, +0.88671875f, -1.6953125f},
    {-0.3359375f, -1.2578125f, -0.0164794922f, -0.953125f}, {+2.453125f, -0.412109375f, -0.8125f, -0.6953125f},
    {-0.72265625f, +0.080078125f, -0.94921875f, -0.400390625f}, {+2.03125f, +0.104492188f, -2.0f, -0.0419921875f},
    {+0.168945312f, -1.015625f, +0.122558594f, +0.3203125f}, {-1.40625f, +1.0f, +0.5859375f, +0.233398438f},
    {+0.5390625f, -0.111328125f, -0.97265625f, +1.2890625f}, {-1.2109375f, +0.8828125f, +0.080078125f, +2.796875f},
    {-2.34375f, -0.7421875f, -0.609375f, -0.241210938f}, {+0.404296875f, -0.209960938f, +0.0296630859f, -1.8671875f},
    {+2.34375f, -1.484375f, +0.1015625f, -0.2421875f}, {-0.211914062f, -0.091796875f, -0.9921875f, +0.6484375f},
    {+0.146484375f, +0.5f, +0.28515625f, -0.271484375f}, {+0.546875f, -1.671875f, +1.359375f, -1.1796875f},
    {+2.59375f, +0.404296875f, -0.140625f, +1.3984375f}, {+0.161132812f, +1.671875f, +0.83984375f, -0.361328125f},
    {+0.57421875f, +0.859375f, +1.0234375f, +0.337890625f}, {-0.40234375f, +2.015625f, +1.2265625f, +1.2890625f},
    {-1.703125f, +0.6328125f, -0.64453125f, +0.12890625f}, {+0.84375f, +0.6484375f, -0.172851562f, +0.6953125f},
    {+1.28125f, +2.09375f, -1.1171875f, +0.6796875f}, {+0.25f, -0.247070312f, -0.000326156616f, -0.184570312f},
    {+0.11328125f, -0.734375f, +1.2734375f, +2.015625f}, {+0.83203125f, +0.7890625f, +0.455078125f, -1.6171875f},
    {+1.1796875f, +1.5078125f, -2.078125f, -0.921875f}, {-0.365234375f, -0.4609375f, +0.90625f, -0.78515625f},
    {-0.447265625f, -0.302734375f, -1.3984375f, +0.0329589844f}, {-0.95703125f, +1.578125f, +0.33984375f, +1.5859375f},
    {-1.015625f, -1.90625f, +0.87109375f, +0.39453125f}, {-0.640625f, -0.208984375f, +0.478515625f, +1.3828125f},
    {-0.8125f, +0.52734375f, +1.3125f, -0.7421875f}, {-0.7265625f, +2.234375f, +1.578125f, -0.0947265625f},
    {-0.12109375f, +0.76953125f, +0.5078125f, +0.5234375f}, {-0.890625f, -2.453125f, -0.86328125f, +0.83984375f},
    {+0.4921875f, -1.8671875f, -0.223632812f, +0.6875f}, {+0.8203125f, -2.453125f, +1.1953125f, +0.0161132812f},
    {-1.1484375f, -0.95703125f, -0.241210938f, -2.1875f}, {+1.25f, +0.177734375f, +1.78125f, -0.58984375f},
    {-0.26953125f, +2.921875f, +0.42578125f, +0.0434570312f}, {+0.0603027344f, +1.5625f, -0.50390625f, -0.86328125f},
};

__device__ __forceinline__ uint32_t dh_quant_nf3_sym(float u) {
  if (u < 0.0f) {
    if (u < -0.999951f) return (u < -1.747951f) ? 0u : 1u;
    return (u < -0.500550f) ? 2u : 3u;
  } else {
    if (u < +0.999951f) return (u < +0.500550f) ? 4u : 5u;
    return (u < +1.747951f) ? 6u : 7u;
  }
}

__device__ __forceinline__ uint32_t dh_quant_vq25(float u0, float u1) {
  uint32_t best_i = 0u;
  float best_d = 1e30f;
#pragma unroll
  for (uint32_t i = 0; i < 32u; ++i) {
    const float2 c = kDhVq25Cb[i];
    const float dx = u0 - c.x;
    const float dy = u1 - c.y;
    const float d = __fmaf_rn(dx, dx, dy * dy);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }
  return best_i;
}

template <int kK>
__device__ __forceinline__ uint32_t dh_quant_4d_anti_smem(
    float u0, float u1, float u2, float u3,
    const float4* __restrict__ s_cb, const float* __restrict__ s_hcn2) {
  uint32_t best_code = 0u;
  float best_score = -INFINITY;
#pragma unroll 4
  for (uint32_t k = 0; k < static_cast<uint32_t>(kK); ++k) {
    const float4 c = s_cb[k];
    const float dot = __fmaf_rn(u0, c.x, __fmaf_rn(u1, c.y, __fmaf_rn(u2, c.z, u3 * c.w)));
    const float sc = fabsf(dot) - s_hcn2[k];
    if (sc > best_score) {
      best_score = sc;
      best_code = k | ((dot < 0.0f) ? static_cast<uint32_t>(kK) : 0u);
    }
  }
  return best_code;
}

__device__ __forceinline__ uint32_t dh_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float dh_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

__device__ __forceinline__ uint32_t dh_f32_to_ord_u32(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) return 0u;  // NaN
  const uint32_t k = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
  return (k == 0u) ? 1u : k;
}

__device__ __forceinline__ void dh_top4_insert_u64(uint64_t (&top)[4], uint64_t cand) {
  if (cand <= top[3]) return;
  if (cand > top[0]) {
    top[3] = top[2]; top[2] = top[1]; top[1] = top[0]; top[0] = cand;
  } else if (cand > top[1]) {
    top[3] = top[2]; top[2] = top[1]; top[1] = cand;
  } else if (cand > top[2]) {
    top[3] = top[2]; top[2] = cand;
  } else {
    top[3] = cand;
  }
}

__device__ __forceinline__ uint64_t dh_warp_max_u64(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t hi = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v >> 32), o);
    const uint32_t lo = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v), o);
    const uint64_t ov = (static_cast<uint64_t>(hi) << 32) | lo;
    if (ov > v) v = ov;
  }
  return v;
}

__host__ __device__ __forceinline__ size_t dh_wave_tile_offset(
    int cta_tile, int kg, int num_cta_tiles, int num_kg, int grid_ctas, size_t step_bytes) {
  const int super_iter = cta_tile / grid_ctas;
  const int cta_id = cta_tile - super_iter * grid_ctas;
  const int rem = num_cta_tiles - super_iter * grid_ctas;
  const int active = (rem < grid_ctas) ? rem : grid_ctas;
  return (static_cast<size_t>(super_iter * grid_ctas) * num_kg +
          static_cast<size_t>(kg) * active + cta_id) * step_bytes;
}

// Streaming GPU packer: dequantizes NVFP4 rows on the fly (0 MB temporary buffer!),
// applies golden-ratio coprime row permutation + Rademacher + H_64/8 rotation + 4-iter OLS,
// and writes directly into wave-interleaved persistent CTA layout.
template <int kMode>
__global__ void __launch_bounds__(256) draft_head_pack_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int64_t* __restrict__ vocab_map,
    uint8_t* __restrict__ out,
    int N, int K, int grid_ctas, uint32_t perm_stride) {
  __shared__ float s_w[8][128];
  __shared__ __nv_bfloat16 s_sc[8];
  constexpr int kCbSize = (kMode == 2) ? 128 : ((kMode == 3) ? 256 : 1);
  __shared__ uint16_t s_code[(kMode >= 2) ? 8 : 1][(kMode >= 2) ? 32 : 1];
  __shared__ float4 s_cb[kCbSize];
  __shared__ float s_hcn2[kCbSize];

  const int tid = threadIdx.x;
  if constexpr (kMode == 2) {
    if (tid < 128) {
      const float4 c = kDh4dAnti20Cb[tid];
      s_cb[tid] = c;
      s_hcn2[tid] = 0.5f * (c.x * c.x + c.y * c.y + c.z * c.z + c.w * c.w);
    }
    __syncthreads();
  } else if constexpr (kMode == 3) {
    if (tid < 256) {
      const float4 c = kDh4dAnti225Cb[tid];
      s_cb[tid] = c;
      s_hcn2[tid] = 0.5f * (c.x * c.x + c.y * c.y + c.z * c.z + c.w * c.w);
    }
    __syncthreads();
  }

  const int sub_n = blockIdx.x;  // 0 .. (N/8)-1
  const int kg = blockIdx.y;     // 0 .. (K/128)-1
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int static_row = sub_n * 8 + warp;
  const uint32_t perm_row = static_cast<uint32_t>(
      (static_cast<uint64_t>(static_row) * perm_stride) % static_cast<uint32_t>(N));
  const int64_t orig_id = vocab_map[perm_row];
  const int k_base = kg * 128;
  const int num_g16 = K >> 4;
  const float gs = *global_scale;

  const uint8_t* w_row = raw_w + static_cast<size_t>(orig_id) * (K >> 1);
  const uint8_t* s_row = raw_s + static_cast<size_t>(orig_id) * num_g16;

  float rw[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int col0 = k_base + h * 64 + 2 * lane;
    const uint8_t packed_byte = w_row[col0 >> 1];
    const float sc_half = dh_fp8_e4m3_to_f32(s_row[col0 >> 4]) * 0.5f * gs;
    float v0 = dh_e2m1_x2_to_f32(packed_byte & 0xFu) * sc_half * dh_rad_sign(col0 + 0);
    float v1 = dh_e2m1_x2_to_f32(packed_byte >> 4) * sc_half * dh_rad_sign(col0 + 1);
    dh_warp_fwht64(v0, v1, lane);
    rw[2 * h + 0] = v0;
    rw[2 * h + 1] = v1;
    s_w[warp][h * 64 + 2 * lane + 0] = v0;
    s_w[warp][h * 64 + 2 * lane + 1] = v1;
  }

  // For 4D VQ modes, align each lane's 4-vector with the exact m16n8k16 B-operand 4-tuple.
  if constexpr (kMode >= 2) {
    __syncwarp();
    const int kb4 = (lane >> 2) * 16 + 2 * (lane & 3);
    rw[0] = s_w[warp][kb4 + 0];
    rw[1] = s_w[warp][kb4 + 1];
    rw[2] = s_w[warp][kb4 + 8];
    rw[3] = s_w[warp][kb4 + 9];
  }

  float lsq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) lsq = __fmaf_rn(rw[i], rw[i], lsq);
  const float rms = sqrtf(fmaxf(dh_warp_sum(lsq) * (1.0f / 128.0f), 1e-30f));

  __nv_bfloat16 best_sc_bf = __float2bfloat16_rn(rms);
  float best_err = INFINITY;
  uint32_t best_q4 = 0u;

  auto eval_sc = [&](float sc_cand) {
    const __nv_bfloat16 sc_b = __float2bfloat16_rn(fmaxf(sc_cand, 1e-30f));
    const float sc_f = __bfloat162float(sc_b);
    const float inv_sc = (sc_f > 1e-30f) ? (1.0f / sc_f) : 0.0f;
    float lerr = 0.0f;
    uint32_t my_q = 0u;
    if constexpr (kMode == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t q = dh_quant_nf3_sym(rw[i] * inv_sc);
        const float q_bf = __bfloat162float(__float2bfloat16_rn(kDhNf3Cb[q]));
        const float wh = __bfloat162float(__float2bfloat16_rn(q_bf * sc_f));
        const float d = rw[i] - wh;
        lerr = __fmaf_rn(d, d, lerr);
      }
    } else if constexpr (kMode == 1) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q = dh_quant_vq25(rw[2 * h + 0] * inv_sc, rw[2 * h + 1] * inv_sc);
        const float2 c = kDhVq25Cb[q];
        const float c0 = __bfloat162float(__float2bfloat16_rn(c.x));
        const float c1 = __bfloat162float(__float2bfloat16_rn(c.y));
        const float wh0 = __bfloat162float(__float2bfloat16_rn(c0 * sc_f));
        const float wh1 = __bfloat162float(__float2bfloat16_rn(c1 * sc_f));
        const float d0 = rw[2 * h + 0] - wh0;
        const float d1 = rw[2 * h + 1] - wh1;
        lerr = __fmaf_rn(d0, d0, __fmaf_rn(d1, d1, lerr));
      }
    } else {
      constexpr int kK = (kMode == 2) ? 128 : 256;
      my_q = dh_quant_4d_anti_smem<kK>(
          rw[0] * inv_sc, rw[1] * inv_sc, rw[2] * inv_sc, rw[3] * inv_sc, s_cb, s_hcn2);
      const float4 c = s_cb[my_q & (kK - 1)];
      const float sc_eff = (my_q & kK) ? -sc_f : sc_f;
      const float d0 = rw[0] - __bfloat162float(__float2bfloat16_rn(c.x * sc_eff));
      const float d1 = rw[1] - __bfloat162float(__float2bfloat16_rn(c.y * sc_eff));
      const float d2 = rw[2] - __bfloat162float(__float2bfloat16_rn(c.z * sc_eff));
      const float d3 = rw[3] - __bfloat162float(__float2bfloat16_rn(c.w * sc_eff));
      lerr = __fmaf_rn(d0, d0, __fmaf_rn(d1, d1, __fmaf_rn(d2, d2, d3 * d3)));
    }
    const float tot_err = dh_warp_sum(lerr);
    if (tot_err < best_err) {
      best_err = tot_err;
      best_sc_bf = sc_b;
      best_q4 = my_q;
    }
  };

  constexpr float kInitMults[5] = {0.85f, 0.95f, 1.02f, 1.10f, 1.20f};
#pragma unroll
  for (int m = 0; m < 5; ++m) {
    eval_sc(rms * kInitMults[m]);
  }

  float cur_sc = __bfloat162float(best_sc_bf);
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const float inv_sc = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
    float l_wq = 0.0f, l_qq = 0.0f;
    if constexpr (kMode == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t q = dh_quant_nf3_sym(rw[i] * inv_sc);
        const float qf = __bfloat162float(__float2bfloat16_rn(kDhNf3Cb[q]));
        l_wq = __fmaf_rn(rw[i], qf, l_wq);
        l_qq = __fmaf_rn(qf, qf, l_qq);
      }
    } else if constexpr (kMode == 1) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q = dh_quant_vq25(rw[2 * h + 0] * inv_sc, rw[2 * h + 1] * inv_sc);
        const float2 c = kDhVq25Cb[q];
        const float c0 = __bfloat162float(__float2bfloat16_rn(c.x));
        const float c1 = __bfloat162float(__float2bfloat16_rn(c.y));
        l_wq = __fmaf_rn(rw[2 * h + 0], c0, __fmaf_rn(rw[2 * h + 1], c1, l_wq));
        l_qq = __fmaf_rn(c0, c0, __fmaf_rn(c1, c1, l_qq));
      }
    } else {
      constexpr int kK = (kMode == 2) ? 128 : 256;
      const uint32_t q = best_q4;
      const float4 c = s_cb[q & (kK - 1)];
      const float sgn = (q & kK) ? -1.0f : 1.0f;
      l_wq = sgn * __fmaf_rn(rw[0], c.x, __fmaf_rn(rw[1], c.y, __fmaf_rn(rw[2], c.z, rw[3] * c.w)));
      l_qq = 2.0f * s_hcn2[q & (kK - 1)];
    }
    const float swq = dh_warp_sum(l_wq);
    const float sqq = dh_warp_sum(l_qq);
    if (sqq > 1e-20f) {
      cur_sc = fmaxf(swq / sqq, 1e-30f);
      eval_sc(cur_sc);
      cur_sc = __bfloat162float(best_sc_bf);
    }
  }

  if (lane == 0) {
    s_sc[warp] = best_sc_bf;
  }
  if constexpr (kMode >= 2) {
    s_code[warp][lane] = static_cast<uint16_t>(best_q4);
  }
  __syncthreads();

  if (warp != 0) return;

  const int r = lane >> 2;
  const int t = lane & 3;
  const float sc_f = __bfloat162float(s_sc[r]);
  const float inv_sc = (sc_f > 1e-30f) ? (1.0f / sc_f) : 0.0f;

  const int cta_tile = sub_n >> 3;
  const int sub_in_cta = sub_n & 7;
  const int num_cta_tiles = N / kDhRowsPerCTA;
  const int num_kg = K / kDhBK;
  constexpr size_t kStepBytes = dh_step_bytes<kMode>();
  constexpr size_t kSubBytes = dh_sub_bytes<kMode>();

  uint8_t* dst = out + dh_wave_tile_offset(cta_tile, kg, num_cta_tiles, num_kg, grid_ctas, kStepBytes) +
                 static_cast<size_t>(sub_in_cta) * kSubBytes;

  if constexpr (kMode == 0) {
    uint32_t c12[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int kb = s * 16 + 2 * t;
      const uint32_t q0 = dh_quant_nf3_sym(s_w[r][kb + 0] * inv_sc);
      const uint32_t q1 = dh_quant_nf3_sym(s_w[r][kb + 1] * inv_sc);
      const uint32_t q2 = dh_quant_nf3_sym(s_w[r][kb + 8] * inv_sc);
      const uint32_t q3 = dh_quant_nf3_sym(s_w[r][kb + 9] * inv_sc);
      c12[s] = q0 | (q1 << 3) | (q2 << 6) | (q3 << 9);
    }
    const uint32_t w0 = c12[0] | (c12[1] << 12) | (c12[2] << 24);
    const uint32_t w1 = (c12[2] >> 8) | (c12[3] << 4) | (c12[4] << 16) | (c12[5] << 28);
    const uint32_t w2 = (c12[5] >> 4) | (c12[6] << 8) | (c12[7] << 20);
    reinterpret_cast<uint32_t*>(dst + 0)[lane] = w0;
    reinterpret_cast<uint32_t*>(dst + 128)[lane] = w1;
    reinterpret_cast<uint32_t*>(dst + 256)[lane] = w2;
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 384)[r] = s_sc[r];
    }
  } else if constexpr (kMode == 1) {
    uint32_t c10[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int kb = s * 16 + 2 * t;
      const uint32_t p01 = dh_quant_vq25(s_w[r][kb + 0] * inv_sc, s_w[r][kb + 1] * inv_sc);
      const uint32_t p23 = dh_quant_vq25(s_w[r][kb + 8] * inv_sc, s_w[r][kb + 9] * inv_sc);
      c10[s] = p01 | (p23 << 5);
    }
    const uint32_t w0 = c10[0] | (c10[1] << 10) | (c10[2] << 20) | (c10[3] << 30);
    const uint32_t w1 = (c10[3] >> 2) | (c10[4] << 8) | (c10[5] << 18) | (c10[6] << 28);
    const uint16_t w2 = static_cast<uint16_t>((c10[6] >> 4) | (c10[7] << 6));
    reinterpret_cast<uint32_t*>(dst + 0)[lane] = w0;
    reinterpret_cast<uint32_t*>(dst + 128)[lane] = w1;
    reinterpret_cast<uint16_t*>(dst + 256)[lane] = w2;
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 320)[r] = s_sc[r];
    }
  } else if constexpr (kMode == 2) {
    uint32_t c8[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) c8[s] = s_code[r][s * 4 + t];
    reinterpret_cast<uint32_t*>(dst + 0)[lane] = c8[0] | (c8[1] << 8) | (c8[2] << 16) | (c8[3] << 24);
    reinterpret_cast<uint32_t*>(dst + 128)[lane] = c8[4] | (c8[5] << 8) | (c8[6] << 16) | (c8[7] << 24);
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 256)[r] = s_sc[r];
    }
  } else {
    uint32_t c9[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) c9[s] = s_code[r][s * 4 + t];
    reinterpret_cast<uint32_t*>(dst + 0)[lane] =
        (c9[0] & 0xFFu) | ((c9[1] & 0xFFu) << 8) | ((c9[2] & 0xFFu) << 16) | ((c9[3] & 0xFFu) << 24);
    reinterpret_cast<uint32_t*>(dst + 128)[lane] =
        (c9[4] & 0xFFu) | ((c9[5] & 0xFFu) << 8) | ((c9[6] & 0xFFu) << 16) | ((c9[7] & 0xFFu) << 24);
    uint32_t sgn = 0u;
#pragma unroll
    for (int s = 0; s < 8; ++s) sgn |= ((c9[s] >> 8) & 1u) << s;
    dst[256 + lane] = static_cast<uint8_t>(sgn);
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 288)[r] = s_sc[r];
    }
  }
}

// Prologue kernel (~1.8 us): rotates A [M, K] BF16 -> A_rot [num_kg, 2, 8, 64] BF16 (XOR-swizzled for ldmatrix).
__global__ void __launch_bounds__(256) dhead_rotate_a_kernel(
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16* __restrict__ A_rot,
    int M, int K, int pdl_mode) {
  const int blk64 = blockIdx.x;  // 0 .. (K/64)-1
  const int m = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int k_base = blk64 * 64;
  const int col8 = lane >> 2;
  const int phys_col8 = col8 ^ m;
  const int elem_off = phys_col8 * 8 + (lane & 3) * 2;
  __nv_bfloat16* dst_row = A_rot + (static_cast<size_t>(blk64) * 8 + m) * 64;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (m < M) {
    const int col0 = k_base + 2 * lane;
    const __nv_bfloat162 raw2 = *reinterpret_cast<const __nv_bfloat162*>(A + static_cast<size_t>(m) * K + col0);
    const float2 rf = __bfloat1622float2(raw2);
    float v0 = rf.x * dh_rad_sign(col0 + 0);
    float v1 = rf.y * dh_rad_sign(col0 + 1);
    dh_warp_fwht64(v0, v1, lane);
    *reinterpret_cast<__nv_bfloat162*>(dst_row + elem_off) = __floats2bfloat162_rn(v0, v1);
  } else {
    *reinterpret_cast<uint32_t*>(dst_row + elem_off) = 0u;
  }

  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

__device__ __forceinline__ void dh_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dh_ldmatrix_x2(uint32_t& a0, uint32_t& a2, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(a0), "=r"(a2)
      : "r"(addr));
}

__device__ __forceinline__ uint64_t dh_pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <int kMode>
struct alignas(128) DhSmem {
  static constexpr int kStepW = dh_step_bytes<kMode>();
  union {
    struct {
      alignas(128) uint8_t w[kDhStages][kStepW];
      alignas(128) __nv_bfloat16 a[kDhStages][2][8][64];  // 2048 B / stage
    } pipe;
    uint64_t epilogue_top[kDhMaxM][64];                   // 4096 B (reused at kernel exit!)
  };
  alignas(8) uint32_t lut[(kMode < 2) ? 64 : 1];          // 256 B (modes 0/1)
  alignas(8) uint2 lut4[(kMode >= 2) ? 256 : 1];          // 2048 B (modes 2/3)
  uint64_t mbar[kDhStages];
};

template <int kMode, int kTopPerCta, bool kWriteFullLogits = false>
__global__ void __launch_bounds__(128, 3) draft_head_fast_kernel(
    const __nv_bfloat16* __restrict__ A_rot,
    const uint8_t* __restrict__ W_packed,
    const int64_t* __restrict__ vocab_map,
    uint64_t* __restrict__ cand_out,
    __nv_bfloat16* __restrict__ full_logits_out,
    int M, int N, int K, int grid_ctas, uint32_t perm_stride, int pdl_mode) {
  __shared__ DhSmem<kMode> smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7 -> batch row m!
  const int t = lane & 3;   // 0..3 -> column pair!

  if constexpr (kMode == 0) {
    if (tid < 64) {
      smem.lut[tid] = dh_bf16x2_u32(kDhNf3Cb[tid & 7], kDhNf3Cb[(tid >> 3) & 7]);
    }
  } else if constexpr (kMode == 1) {
    if (tid < 32) {
      const float2 c = kDhVq25Cb[tid];
      smem.lut[tid] = dh_bf16x2_u32(c.x, c.y);
    }
  } else if constexpr (kMode == 2) {
    const float4 c = kDh4dAnti20Cb[tid];
    const uint2 pos = make_uint2(dh_bf16x2_u32(c.x, c.y), dh_bf16x2_u32(c.z, c.w));
    smem.lut4[tid] = pos;
    smem.lut4[tid + 128] = make_uint2(pos.x ^ 0x80008000u, pos.y ^ 0x80008000u);
  } else {
    for (int i = tid; i < 256; i += 128) {
      const float4 c = kDh4dAnti225Cb[i];
      smem.lut4[i] = make_uint2(dh_bf16x2_u32(c.x, c.y), dh_bf16x2_u32(c.z, c.w));
    }
  }

  const int cta_id = blockIdx.x;
  const int num_cta_tiles = N / kDhRowsPerCTA;
  const int num_kg = K / kDhBK;
  int my_tiles = 0;
  if (cta_id < num_cta_tiles) {
    my_tiles = (num_cta_tiles - 1 - cta_id) / grid_ctas + 1;
  }
  const int total_steps = my_tiles * num_kg;

  constexpr size_t kStepW = DhSmem<kMode>::kStepW;
  constexpr size_t kSubW = dh_sub_bytes<kMode>();

  auto w_stage_ptr = [&](int step) -> const uint8_t* {
    const int super_iter = step / num_kg;
    const int kg = step - super_iter * num_kg;
    const int cta_tile = super_iter * grid_ctas + cta_id;
    return W_packed + dh_wave_tile_offset(cta_tile, kg, num_cta_tiles, num_kg, grid_ctas, kStepW);
  };

  auto a_stage_ptr = [&](int kg) -> const __nv_bfloat16* {
    return A_rot + static_cast<size_t>(kg) * 1024;
  };

  auto mbar_addr = [&](int st) -> uint32_t {
    return static_cast<uint32_t>(__cvta_generic_to_shared(&smem.mbar[st]));
  };

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kDhStages; ++s) {
      const uint32_t b = mbar_addr(s);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  constexpr int kPreload = kDhStages - 1;  // 3
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;

  // Preload first 3 weight stages from LPDDR5x BEFORE griddepcontrol.wait to overlap A_rot & final_norm!
  if (tid == 0) {
    const uint64_t pol = dh_pol_evict_first();
    for (int p = 0; p < preload; ++p) {
      const uint32_t b = mbar_addr(p);
      const uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.w[p]));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   ::"r"(b), "r"(static_cast<int>(kStepW))
                   : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
          ::"r"(dst_w), "l"(w_stage_ptr(p)), "r"(static_cast<int>(kStepW)), "r"(b), "l"(pol)
          : "memory");
    }
  }
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  for (int p = 0; p < preload; ++p) {
    const int kg = p % num_kg;
    reinterpret_cast<uint4*>(smem.pipe.a[p])[tid] =
        reinterpret_cast<const uint4*>(a_stage_ptr(kg))[tid];
  }
  __syncthreads();

  uint64_t my_top[4] = {0ull, 0ull, 0ull, 0ull};
  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;

  const int ldm_row = lane & 7;
  const int ldm_col_off = (lane >> 3) & 1;
  const int trig_step = ((pdl_mode & 2) != 0 && total_steps > 0)
                            ? ((total_steps > 8) ? (total_steps - 8) : 0)
                            : -1;

  int super_iter = 0;
  int kg = 0;
  for (int step = 0; step < total_steps; ++step) {
    if (step == trig_step) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int st = step & (kDhStages - 1);
    const int next_step = step + kPreload;

    if (next_step < total_steps && tid == 0) {
      const int nst = next_step & (kDhStages - 1);
      const int nkg = next_step % num_kg;
      const uint32_t b = mbar_addr(nst);
      const uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.w[nst]));
      const uint32_t dst_a = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.a[nst]));
      const uint64_t pol = dh_pol_evict_first();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   ::"r"(b), "r"(static_cast<int>(kStepW + 2048))
                   : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
          ::"r"(dst_w), "l"(w_stage_ptr(next_step)), "r"(static_cast<int>(kStepW)), "r"(b), "l"(pol)
          : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
          ::"r"(dst_a), "l"(a_stage_ptr(nkg)), "r"(2048), "r"(b)
          : "memory");
    }

    const uint32_t cur_bar = mbar_addr(st);
    const uint32_t parity = static_cast<uint32_t>((step >> 2) & 1);
    asm volatile(
        "{\n\t.reg .pred P1;\n\tDH_MBAR_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@!P1 bra DH_MBAR_WAIT_%=;\n\t}"
        ::"r"(cur_bar), "r"(parity)
        : "memory");

    const uint8_t* wt0 = &smem.pipe.w[st][(warp * 2 + 0) * kSubW];
    const uint8_t* wt1 = &smem.pipe.w[st][(warp * 2 + 1) * kSubW];

    if constexpr (kMode == 0) {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const uint32_t w2_0 = reinterpret_cast<const uint32_t*>(wt0 + 256)[lane];
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 384)[g]);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const uint32_t w2_1 = reinterpret_cast<const uint32_t*>(wt1 + 256)[lane];
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 384)[g]);

      const uint32_t c0_arr[8] = {
          w0_0 & 0xFFFu,
          (w0_0 >> 12) & 0xFFFu,
          __funnelshift_r(w0_0, w1_0, 24) & 0xFFFu,
          (w1_0 >> 4) & 0xFFFu,
          (w1_0 >> 16) & 0xFFFu,
          __funnelshift_r(w1_0, w2_0, 28) & 0xFFFu,
          (w2_0 >> 8) & 0xFFFu,
          w2_0 >> 20,
      };
      const uint32_t c1_arr[8] = {
          w0_1 & 0xFFFu,
          (w0_1 >> 12) & 0xFFFu,
          __funnelshift_r(w0_1, w1_1, 24) & 0xFFFu,
          (w1_1 >> 4) & 0xFFFu,
          (w1_1 >> 16) & 0xFFFu,
          __funnelshift_r(w1_1, w2_1, 28) & 0xFFFu,
          (w2_1 >> 8) & 0xFFFu,
          w2_1 >> 20,
      };

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t q01_0 = smem.lut[c0_arr[s] & 63u];
        const uint32_t q23_0 = smem.lut[c0_arr[s] >> 6];
        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_0), sc0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_0), sc0);

        const uint32_t q01_1 = smem.lut[c1_arr[s] & 63u];
        const uint32_t q23_1 = smem.lut[c1_arr[s] >> 6];
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_1), sc1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_1), sc1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    } else if constexpr (kMode == 1) {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const uint32_t w2_0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 256)[lane]);
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 320)[g]);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const uint32_t w2_1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 256)[lane]);
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 320)[g]);

      const uint32_t c0_arr[8] = {
          w0_0 & 0x3FFu,
          (w0_0 >> 10) & 0x3FFu,
          (w0_0 >> 20) & 0x3FFu,
          __funnelshift_r(w0_0, w1_0, 30) & 0x3FFu,
          (w1_0 >> 8) & 0x3FFu,
          (w1_0 >> 18) & 0x3FFu,
          __funnelshift_r(w1_0, w2_0, 28) & 0x3FFu,
          (w2_0 >> 6) & 0x3FFu,
      };
      const uint32_t c1_arr[8] = {
          w0_1 & 0x3FFu,
          (w0_1 >> 10) & 0x3FFu,
          (w0_1 >> 20) & 0x3FFu,
          __funnelshift_r(w0_1, w1_1, 30) & 0x3FFu,
          (w1_1 >> 8) & 0x3FFu,
          (w1_1 >> 18) & 0x3FFu,
          __funnelshift_r(w1_1, w2_1, 28) & 0x3FFu,
          (w2_1 >> 6) & 0x3FFu,
      };

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t q01_0 = smem.lut[c0_arr[s] & 31u];
        const uint32_t q23_0 = smem.lut[c0_arr[s] >> 5];
        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_0), sc0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_0), sc0);

        const uint32_t q01_1 = smem.lut[c1_arr[s] & 31u];
        const uint32_t q23_1 = smem.lut[c1_arr[s] >> 5];
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_1), sc1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_1), sc1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    } else if constexpr (kMode == 2) {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 256)[g]);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 256)[g]);

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t idx0 = (((s < 4) ? w0_0 : w1_0) >> ((s & 3) * 8)) & 0xFFu;
        const uint32_t idx1 = (((s < 4) ? w0_1 : w1_1) >> ((s & 3) * 8)) & 0xFFu;
        const uint2 q0 = smem.lut4[idx0];
        const uint2 q1 = smem.lut4[idx1];

        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q0.x), sc0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q0.y), sc0);
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q1.x), sc1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q1.y), sc1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    } else {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const uint32_t sgn0 = static_cast<uint32_t>(wt0[256 + lane]);
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 288)[g]);
      const uint32_t sc0_nu = (*reinterpret_cast<const uint32_t*>(&sc0)) ^ 0x80008000u;
      const __nv_bfloat162 sc0_neg = *reinterpret_cast<const __nv_bfloat162*>(&sc0_nu);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const uint32_t sgn1 = static_cast<uint32_t>(wt1[256 + lane]);
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 288)[g]);
      const uint32_t sc1_nu = (*reinterpret_cast<const uint32_t*>(&sc1)) ^ 0x80008000u;
      const __nv_bfloat162 sc1_neg = *reinterpret_cast<const __nv_bfloat162*>(&sc1_nu);

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t idx0 = (((s < 4) ? w0_0 : w1_0) >> ((s & 3) * 8)) & 0xFFu;
        const uint32_t idx1 = (((s < 4) ? w0_1 : w1_1) >> ((s & 3) * 8)) & 0xFFu;
        const uint2 q0 = smem.lut4[idx0];
        const uint2 q1 = smem.lut4[idx1];
        const __nv_bfloat162 se0 = (sgn0 & (1u << s)) ? sc0_neg : sc0;
        const __nv_bfloat162 se1 = (sgn1 & (1u << s)) ? sc1_neg : sc1;

        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q0.x), se0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q0.y), se0);
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q1.x), se1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q1.y), se1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    }

    if (kg + 1 == num_kg) {
      const int cta_tile = super_iter * grid_ctas + cta_id;
      const uint32_t col0 = static_cast<uint32_t>(cta_tile * kDhRowsPerCTA + (warp * 2 + 0) * 8 + 2 * t);
      const uint32_t col1 = static_cast<uint32_t>(cta_tile * kDhRowsPerCTA + (warp * 2 + 1) * 8 + 2 * t);
      if (g < M) {
        if constexpr (kWriteFullLogits) {
          const uint32_t un = static_cast<uint32_t>(N);
          const uint32_t p00 = static_cast<uint32_t>((static_cast<uint64_t>(col0 + 0u) * perm_stride) % un);
          const uint32_t p01 = static_cast<uint32_t>((static_cast<uint64_t>(col0 + 1u) * perm_stride) % un);
          const uint32_t p10 = static_cast<uint32_t>((static_cast<uint64_t>(col1 + 0u) * perm_stride) % un);
          const uint32_t p11 = static_cast<uint32_t>((static_cast<uint64_t>(col1 + 1u) * perm_stride) % un);
          __nv_bfloat16* row_out = full_logits_out + static_cast<size_t>(g) * N;
          row_out[p00] = __float2bfloat16_rn(d0_0);
          row_out[p01] = __float2bfloat16_rn(d1_0);
          row_out[p10] = __float2bfloat16_rn(d0_1);
          row_out[p11] = __float2bfloat16_rn(d1_1);
        } else {
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d0_0)) << 32) | (col0 + 0u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d1_0)) << 32) | (col0 + 1u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d0_1)) << 32) | (col1 + 0u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d1_1)) << 32) | (col1 + 1u));
        }
      }
      d0_0 = d1_0 = d2_0 = d3_0 = 0.0f;
      d0_1 = d1_1 = d2_1 = d3_1 = 0.0f;
      ++super_iter;
      kg = 0;
    } else {
      ++kg;
    }

    __syncthreads();
  }

  if (total_steps == 0 && (pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if constexpr (kWriteFullLogits) {
    return;
  }

  __syncthreads();
  if (g < M) {
    const int base_idx = warp * 16 + t * 4;
    smem.epilogue_top[g][base_idx + 0] = my_top[0];
    smem.epilogue_top[g][base_idx + 1] = my_top[1];
    smem.epilogue_top[g][base_idx + 2] = my_top[2];
    smem.epilogue_top[g][base_idx + 3] = my_top[3];
  }
  __syncthreads();

  const int num_cands = grid_ctas * kTopPerCta;
  const uint32_t un = static_cast<uint32_t>(N);
  for (int m = warp; m < M; m += kDhWarps) {
    const uint64_t u0 = smem.epilogue_top[m][2 * lane + 0];
    const uint64_t u1 = smem.epilogue_top[m][2 * lane + 1];
    uint64_t cur = (u0 >= u1) ? u0 : u1;
    uint64_t nxt = (u0 >= u1) ? u1 : u0;
    uint64_t picked = 0ull;

#pragma unroll
    for (int r = 0; r < kTopPerCta; ++r) {
      const uint64_t best = dh_warp_max_u64(cur);
      if (lane == r) picked = best;
      if (best != 0ull) {
        const uint32_t msk = __ballot_sync(0xffffffffu, cur == best);
        const int win = __ffs(msk) - 1;
        if (lane == win) {
          cur = nxt;
          nxt = 0ull;
        }
      }
    }

    if (lane < kTopPerCta) {
      uint64_t out_val = 0ull;
      if (picked != 0ull) {
        const uint32_t scol = static_cast<uint32_t>(picked & 0xffffffffull);
        const uint32_t perm_col = static_cast<uint32_t>((static_cast<uint64_t>(scol) * perm_stride) % un);
        const uint32_t orig_id = static_cast<uint32_t>(__ldg(&vocab_map[perm_col]));
        out_val = (picked & 0xffffffff00000000ull) | (0xffffffffu - orig_id);
      }
      cand_out[static_cast<size_t>(m) * num_cands + cta_id * kTopPerCta + lane] = out_val;
    }
  }
}

struct DraftHeadFastState {
  torch::Tensor a_rot;
  torch::Tensor cand_buf;
  torch::Tensor sentinel;
  int64_t a_rot_elems = 0;
  int last_num_cands = 384;
  int last_rescore_k = 64;
  c10::DeviceIndex dev = -1;
};

DraftHeadFastState& get_dh_state(torch::Device device, int K) {
  static DraftHeadFastState st;
  const int64_t need_arot = static_cast<int64_t>(K / 64) * 8 * 64;
  if (!st.sentinel.defined() || st.dev != device.index() || st.a_rot_elems < need_arot) {
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    st.a_rot = torch::zeros({need_arot}, opt_bf16);
    st.cand_buf = torch::zeros({kDhMaxM, kDhMaxCandPool}, opt_i64);
    st.sentinel = torch::zeros({kDhMaxM, kDhSentinelCols}, opt_bf16);
    st.a_rot_elems = need_arot;
    st.dev = device.index();
  }
  return st;
}

template <int kMode>
static void set_dh_carveout_one() {
  cudaFuncSetAttribute(draft_head_fast_kernel<kMode, 8, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kMode, 4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kMode, 2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kMode, 4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

static void ensure_dh_carveout() {
  static bool done = false;
  if (!done) {
    done = true;
    set_dh_carveout_one<0>();
    set_dh_carveout_one<1>();
    set_dh_carveout_one<2>();
    set_dh_carveout_one<3>();
  }
}

}  // namespace

bool draft_head_try_get_cands(
    const torch::Tensor& static_logits,
    const uint64_t*& out_cand_ptr,
    int& out_num_cands,
    int& out_rescore_k) {
  if (!static_logits.is_cuda() || static_logits.size(1) != kDhSentinelCols) {
    return false;
  }
  auto& st = get_dh_state(static_logits.device(), 6656);
  if (!st.sentinel.defined() || static_logits.data_ptr() != st.sentinel.data_ptr()) {
    return false;
  }
  out_cand_ptr = reinterpret_cast<const uint64_t*>(st.cand_buf.data_ptr());
  out_num_cands = st.last_num_cands;
  out_rescore_k = st.last_rescore_k;
  return true;
}

torch::Tensor draft_head_pack(
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    torch::Tensor vocab_map,
    int64_t grid_ctas = 96,
    int64_t mode = 0) {
  TORCH_CHECK(raw_w.is_cuda() && raw_w.scalar_type() == torch::kUInt8 && raw_w.dim() == 2 && raw_w.is_contiguous());
  TORCH_CHECK(raw_s.is_cuda() && raw_s.dim() == 2 && raw_s.is_contiguous());
  TORCH_CHECK(vocab_map.is_cuda() && vocab_map.scalar_type() == torch::kInt64 && vocab_map.is_contiguous());
  const int N = static_cast<int>(vocab_map.numel());
  const int K = static_cast<int>(raw_w.size(1) * 2);
  TORCH_CHECK(N >= 64 && (N % kDhRowsPerCTA) == 0 && K >= 128 && (K % kDhBK) == 0);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192,
              "draft_head_pack grid_ctas must be 48, 96, or 192");
  TORCH_CHECK(mode >= 0 && mode <= 3, "draft_head_pack mode must be 0, 1, 2, or 3");

  (void)get_dh_state(raw_w.device(), K);
  ensure_dh_carveout();

  const uint32_t pstride = dh_perm_stride(N);
  const int64_t sub_bytes = dh_sub_bytes_rt(static_cast<int>(mode));
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / kDhBK) * sub_bytes;
  auto out = torch::empty({total_bytes}, raw_w.options().dtype(torch::kUInt8));

  dim3 grid(static_cast<unsigned>(N / 8), static_cast<unsigned>(K / kDhBK), 1u);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto launch_pack = [&](auto mode_tag) {
    constexpr int KMODE = decltype(mode_tag)::value;
    draft_head_pack_kernel<KMODE><<<grid, 256, 0, stream>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        vocab_map.data_ptr<int64_t>(),
        out.data_ptr<uint8_t>(),
        N, K, static_cast<int>(grid_ctas), pstride);
  };
  if (mode == 0) launch_pack(std::integral_constant<int, 0>{});
  else if (mode == 1) launch_pack(std::integral_constant<int, 1>{});
  else if (mode == 2) launch_pack(std::integral_constant<int, 2>{});
  else launch_pack(std::integral_constant<int, 3>{});
  return out;
}

torch::Tensor draft_head_topc(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor vocab_map,
    int64_t N,
    int64_t K,
    int64_t grid_ctas = 96,
    int64_t mode = 0,
    int64_t rescore_k = 0) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= kDhMaxM && A.size(1) == K);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192);
  TORCH_CHECK(mode >= 0 && mode <= 3, "draft_head_topc mode must be 0, 1, 2, or 3");

  ensure_dh_carveout();
  auto& st = get_dh_state(A.device(), static_cast<int>(K));
  const int top_per_cta = (grid_ctas == 48) ? 8 : ((grid_ctas == 96) ? 4 : 2);
  st.last_num_cands = static_cast<int>(grid_ctas) * top_per_cta;  // always 384!
  st.last_rescore_k = (rescore_k > 0) ? static_cast<int>(rescore_k) : dh_rescore_k_cfg();

  const int pdl = dh_pdl_mode();
  const uint32_t pstride = dh_perm_stride(static_cast<int>(N));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  dh_launch_pdl(
      dhead_rotate_a_kernel,
      dim3(static_cast<unsigned>(K / 64), 1u, 1u), 256, stream, pdl,
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(st.a_rot.data_ptr()),
      M, static_cast<int>(K), pdl);

  auto launch_main = [&](auto mode_tag, auto top_tag) {
    constexpr int KMODE = decltype(mode_tag)::value;
    constexpr int KTOP = decltype(top_tag)::value;
    dh_launch_pdl(
        draft_head_fast_kernel<KMODE, KTOP, false>,
        dim3(static_cast<unsigned>(grid_ctas), 1u, 1u), kDhThreads, stream, pdl,
        reinterpret_cast<const __nv_bfloat16*>(st.a_rot.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        vocab_map.data_ptr<int64_t>(),
        reinterpret_cast<uint64_t*>(st.cand_buf.data_ptr()),
        static_cast<__nv_bfloat16*>(nullptr),
        M, static_cast<int>(N), static_cast<int>(K), static_cast<int>(grid_ctas), pstride, pdl);
  };

  auto dispatch_top = [&](auto mode_tag) {
    if (top_per_cta == 8) launch_main(mode_tag, std::integral_constant<int, 8>{});
    else if (top_per_cta == 4) launch_main(mode_tag, std::integral_constant<int, 4>{});
    else launch_main(mode_tag, std::integral_constant<int, 2>{});
  };

  if (mode == 0) dispatch_top(std::integral_constant<int, 0>{});
  else if (mode == 1) dispatch_top(std::integral_constant<int, 1>{});
  else if (mode == 2) dispatch_top(std::integral_constant<int, 2>{});
  else dispatch_top(std::integral_constant<int, 3>{});

  return st.sentinel.narrow(0, 0, M);
}

torch::Tensor draft_head_gemm_full(
    torch::Tensor A,
    torch::Tensor W_packed,
    int64_t N,
    int64_t K,
    int64_t grid_ctas = 96,
    int64_t mode = 0) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  const int total_m = static_cast<int>(A.size(0));
  TORCH_CHECK(total_m >= 1 && A.size(1) == K);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192);
  TORCH_CHECK(mode >= 0 && mode <= 3, "draft_head_gemm_full mode must be 0, 1, 2, or 3");

  ensure_dh_carveout();
  auto& st = get_dh_state(A.device(), static_cast<int>(K));
  auto out = torch::empty({total_m, N}, A.options());
  const int pdl = dh_pdl_mode();
  const uint32_t pstride = dh_perm_stride(static_cast<int>(N));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  auto launch_full = [&](auto mode_tag, int m0, int cm) {
    constexpr int KMODE = decltype(mode_tag)::value;
    dh_launch_pdl(
        draft_head_fast_kernel<KMODE, 4, true>,
        dim3(static_cast<unsigned>(grid_ctas), 1u, 1u), kDhThreads, stream, pdl,
        reinterpret_cast<const __nv_bfloat16*>(st.a_rot.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        static_cast<const int64_t*>(nullptr),
        static_cast<uint64_t*>(nullptr),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) + static_cast<size_t>(m0) * N,
        cm, static_cast<int>(N), static_cast<int>(K), static_cast<int>(grid_ctas), pstride, pdl);
  };

  for (int m0 = 0; m0 < total_m; m0 += kDhMaxM) {
    const int cm = std::min(kDhMaxM, total_m - m0);
    dh_launch_pdl(
        dhead_rotate_a_kernel,
        dim3(static_cast<unsigned>(K / 64), 1u, 1u), 256, stream, pdl,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()) + static_cast<size_t>(m0) * K,
        reinterpret_cast<__nv_bfloat16*>(st.a_rot.data_ptr()),
        cm, static_cast<int>(K), pdl);
    if (mode == 0) launch_full(std::integral_constant<int, 0>{}, m0, cm);
    else if (mode == 1) launch_full(std::integral_constant<int, 1>{}, m0, cm);
    else if (mode == 2) launch_full(std::integral_constant<int, 2>{}, m0, cm);
    else launch_full(std::integral_constant<int, 3>{}, m0, cm);
  }
  return out;
}

// ============================================================================
// 1.500 bpw 4D-64 Lloyd-Max Gaussian VQ + 8-stage cross-wave cp.async Draft Head
// (also supports mode=158 ternary 1.625 bpw & mode=2 odd 2.000 bpw)
// ============================================================================

namespace dhf4d {

constexpr int kMaxM = 8;
constexpr int kWarpsPerCTA = 4;
constexpr int kThreads = kWarpsPerCTA * 32;                    // 128
constexpr int kNTilesPerWarp = 2;                              // 16 N-rows/warp -> 2x A reuse
constexpr int kNTilesPerCTA = kWarpsPerCTA * kNTilesPerWarp;   // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;                 // 64
constexpr int kWaveCTAs = 192;                                 // 4 CTAs/SM * 48 SMs on GB10
constexpr int kStages = 8;                                     // 8-stage continuous cross-wave cp.async (<=19.5 KiB SMEM)
constexpr int kCtaTopK = 8;                                    // per-CTA shortlist per row m
constexpr int kMaxCand = 512;                                  // max Stage-2 shortlist per row m (supports topk=256..512)
constexpr int kS2CTAsPerRow = 16;                              // 96 CTAs at M=6 (exact 2 CTAs/SM * 48 SMs single wave!)
constexpr int kS2Warps = 8;
constexpr int kS2MaxGroups = 512;                              // K <= 8192 (16 KiB SMEM)

constexpr int kTileBytes100 = 64;                              // 32 * uint16_t (2x 8-bit 8D-256 antipodal codes), 16B aligned
constexpr int kStepWBytes100 = kNTilesPerCTA * kTileBytes100;  // 512 B = 4 * 128 B = 32 * uint4 (1.000 bpw!)
constexpr int kStepWU4_100 = kStepWBytes100 / 16;              // 32

constexpr int kTileBytes150 = 96;                              // 24 * uint32_t (32 * 24 bits), 16B aligned
constexpr int kStepWBytes150 = kNTilesPerCTA * kTileBytes150;  // 768 B = 6 * 128 B = 48 * uint4 (100% cacheline aligned)
constexpr int kStepWU4_150 = kStepWBytes150 / 16;              // 48

constexpr int kTileBytes158 = 104;                             // 26 * uint32_t (32 * 26 bits)
constexpr int kStepWBytes158 = kNTilesPerCTA * kTileBytes158;  // 832 B = 52 * uint4
constexpr int kStepWU4_158 = kStepWBytes158 / 16;              // 52

constexpr int kTileBytes2 = 128;                               // 32 * uint32_t (32 * 32 bits)
constexpr int kStepWBytes2 = kNTilesPerCTA * kTileBytes2;      // 1024 B = 64 * uint4
constexpr int kStepWU4_2 = kStepWBytes2 / 16;                  // 64

// Pure-math N(0, I_4) 64-centroid Lloyd-Max Gaussian sphere-packing codebook (BF16-exact, zero model weights).
// 6 bits per 4 weights = 1.500 bpw (rho = 0.903 on N(0, I_4)).
__device__ __constant__ float kVq4D64[64][4] = {
  {-0.6640625f, -0.5195312f, -0.3886719f, -0.3593750f}, //  0
  {-0.7695312f, -1.3359375f, -0.6757812f, -1.2812500f}, //  1
  {0.3945312f, -0.4335938f, -0.7500000f, -0.8125000f}, //  2
  {1.0703125f, -1.4453125f, -0.8828125f, -1.2031250f}, //  3
  {-0.5781250f, 0.5117188f, -0.5898438f, -0.7382812f}, //  4
  {-1.2187500f, 1.1640625f, -0.8828125f, -1.4296875f}, //  5
  {0.5156250f, 0.4531250f, -0.3144531f, -0.4511719f}, //  6
  {1.0859375f, 0.7851562f, -1.3515625f, -1.0468750f}, //  7
  {-0.5117188f, -0.5781250f, 0.7382812f, -0.5898438f}, //  8
  {-1.1640625f, -1.2187500f, 1.4296875f, -0.8828125f}, //  9
  {0.5195312f, -0.6640625f, 0.3593750f, -0.3886719f}, // 10
  {1.3359375f, -0.7695312f, 1.2812500f, -0.6757812f}, // 11
  {-0.4531250f, 0.5156250f, 0.4511719f, -0.3144531f}, // 12
  {-0.7851562f, 1.0859375f, 1.0468750f, -1.3515625f}, // 13
  {0.4335938f, 0.3945312f, 0.8125000f, -0.7500000f}, // 14
  {1.4453125f, 1.0703125f, 1.2031250f, -0.8828125f}, // 15
  {-0.4335938f, -0.3945312f, -0.8125000f, 0.7500000f}, // 16
  {-1.4453125f, -1.0703125f, -1.2031250f, 0.8828125f}, // 17
  {0.4531250f, -0.5156250f, -0.4511719f, 0.3144531f}, // 18
  {0.7851562f, -1.0859375f, -1.0468750f, 1.3515625f}, // 19
  {-0.5195312f, 0.6640625f, -0.3593750f, 0.3886719f}, // 20
  {-1.3359375f, 0.7695312f, -1.2812500f, 0.6757812f}, // 21
  {0.5117188f, 0.5781250f, -0.7382812f, 0.5898438f}, // 22
  {1.1640625f, 1.2187500f, -1.4296875f, 0.8828125f}, // 23
  {-0.5156250f, -0.4531250f, 0.3144531f, 0.4511719f}, // 24
  {-1.0859375f, -0.7851562f, 1.3515625f, 1.0468750f}, // 25
  {0.5781250f, -0.5117188f, 0.5898438f, 0.7382812f}, // 26
  {1.2187500f, -1.1640625f, 0.8828125f, 1.4296875f}, // 27
  {-0.3945312f, 0.4335938f, 0.7500000f, 0.8125000f}, // 28
  {-1.0703125f, 1.4453125f, 0.8828125f, 1.2031250f}, // 29
  {0.6640625f, 0.5195312f, 0.3886719f, 0.3593750f}, // 30
  {0.7695312f, 1.3359375f, 0.6757812f, 1.2812500f}, // 31
  {-1.6640625f, -1.1640625f, 0.1640625f, 0.0781250f}, // 32
  {-1.1640625f, 1.6640625f, 0.0781250f, -0.1640625f}, // 33
  {1.1640625f, -1.6640625f, -0.0781250f, 0.1640625f}, // 34
  {1.6640625f, 1.1640625f, -0.1640625f, -0.0781250f}, // 35
  {-1.6406250f, -0.2177734f, -1.1953125f, -0.4589844f}, // 36
  {-1.3750000f, 0.3691406f, 1.4453125f, -0.0593262f}, // 37
  {1.3750000f, -0.3691406f, -1.4453125f, 0.0593262f}, // 38
  {1.6406250f, 0.2177734f, 1.1953125f, 0.4589844f}, // 39
  {-1.6015625f, -0.1269531f, 0.2773438f, -1.2812500f}, // 40
  {-1.3046875f, 0.1269531f, -0.0766602f, 1.5703125f}, // 41
  {1.3046875f, -0.1269531f, 0.0766602f, -1.5703125f}, // 42
  {1.6015625f, 0.1269531f, -0.2773438f, 1.2812500f}, // 43
  {-0.1269531f, -1.3046875f, -1.5703125f, -0.0766602f}, // 44
  {0.1269531f, -1.6015625f, 1.2812500f, 0.2773438f}, // 45
  {-0.1269531f, 1.6015625f, -1.2812500f, -0.2773438f}, // 46
  {0.1269531f, 1.3046875f, 1.5703125f, 0.0766602f}, // 47
  {0.2177734f, -1.6406250f, 0.4589844f, -1.1953125f}, // 48
  {-0.3691406f, -1.3750000f, 0.0593262f, 1.4453125f}, // 49
  {0.3691406f, 1.3750000f, -0.0593262f, -1.4453125f}, // 50
  {-0.2177734f, 1.6406250f, -0.4589844f, 1.1953125f}, // 51
  {-0.1835938f, -0.0849609f, -1.4843750f, -1.7109375f}, // 52
  {-0.0849609f, 0.1835938f, -1.7109375f, 1.4843750f}, // 53
  {0.0849609f, -0.1835938f, 1.7109375f, -1.4843750f}, // 54
  {0.1835938f, 0.0849609f, 1.4843750f, 1.7109375f}, // 55
  {-1.9843750f, 0.3261719f, 0.1503906f, 0.1118164f}, // 56
  {1.9843750f, -0.3261719f, -0.1503906f, -0.1118164f}, // 57
  {-0.3261719f, -1.9843750f, -0.1118164f, 0.1503906f}, // 58
  {0.3261719f, 1.9843750f, 0.1118164f, -0.1503906f}, // 59
  {-0.0493164f, 0.1425781f, -1.9531250f, -0.1220703f}, // 60
  {0.0493164f, -0.1425781f, 1.9531250f, 0.1220703f}, // 61
  {-0.1425781f, -0.0493164f, 0.1220703f, -1.9531250f}, // 62
  {0.1425781f, 0.0493164f, -0.1220703f, 1.9531250f}, // 63
};

__device__ __forceinline__ uint32_t dhf_vq4d64_encode(float x0, float x1, float x2, float x3) {
  float best_s = -1e30f;
  uint32_t best_i = 0u;
#pragma unroll 8
  for (uint32_t c = 0u; c < 64u; ++c) {
    const float c0 = kVq4D64[c][0];
    const float c1 = kVq4D64[c][1];
    const float c2 = kVq4D64[c][2];
    const float c3 = kVq4D64[c][3];
    const float half_n2 = 0.5f * (__fmaf_rn(c0, c0, __fmaf_rn(c1, c1, __fmaf_rn(c2, c2, c3 * c3))));
    const float s = __fmaf_rn(x0, c0, __fmaf_rn(x1, c1, __fmaf_rn(x2, c2, __fmaf_rn(x3, c3, -half_n2))));
    if (s > best_s) {
      best_s = s;
      best_i = c;
    }
  }
  return best_i;
}

int dhf_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

bool dhf_pair_waves_enabled() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  if (dyn) {
    const char* s = std::getenv("MUSE_DHF_PAIR_WAVES");
    return (s && *s) ? (std::atoi(s) != 0) : false;
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DHF_PAIR_WAVES");
    return (s && *s) ? (std::atoi(s) != 0) : false;
  }();
  return en;
}

bool dhf_tma100_enabled() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  if (dyn) {
    const char* s = std::getenv("MUSE_DHF_TMA100");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DHF_TMA100");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  return en;
}

bool dhf_tma100_pair_enabled() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  if (dyn) {
    const char* s = std::getenv("MUSE_DHF_TMA100_PAIR");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DHF_TMA100_PAIR");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  return en;
}

bool dhf_fuse_select_enabled() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  if (dyn) {
    const char* s = std::getenv("MUSE_DHF_FUSE_SELECT");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DHF_FUSE_SELECT");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  return en;
}

template <typename... KArgs, typename... Args>
static inline void dhf_launch_pdl(
    void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = grid;
  lcfg.blockDim = dim3(static_cast<unsigned>(block), 1u, 1u);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = st;
  cudaLaunchAttribute attr[1];
  if ((pdl_mode & 1) != 0) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
  }
  cudaError_t err = cudaLaunchKernelEx(&lcfg, kernel, static_cast<KArgs>(args)...);
  TORCH_CHECK(err == cudaSuccess, "draft_head_fast kernel launch failed: ", cudaGetErrorString(err));
}

__device__ __host__ __forceinline__ size_t dhf_wave_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  return static_cast<size_t>(wave_base) * num_kg + static_cast<size_t>(kg) * w_size + slot;
}

__device__ __host__ __forceinline__ size_t dhf_wave_bk128_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  const int kg2 = kg >> 1;
  const int kh = kg & 1;
  return ((static_cast<size_t>(wave_base) * (num_kg >> 1) +
           static_cast<size_t>(kg2) * w_size + slot)
          << 1) |
         static_cast<size_t>(kh);
}

__device__ __host__ __forceinline__ size_t dhf_wave_pair_bk128_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int full_pairs = n_blocks / (2 * kWaveCTAs);
  const int pair_limit = full_pairs * (2 * kWaveCTAs);
  const int num_kg2 = num_kg >> 1;
  const int kg2 = kg >> 1;
  const int kh = kg & 1;
  if (cta_n < pair_limit) {
    const int wave = cta_n / kWaveCTAs;
    const int slot = cta_n - wave * kWaveCTAs;
    const int wp = wave >> 1;
    const int wh = wave & 1;
    return (((static_cast<size_t>(wp) * num_kg2 + kg2) * kWaveCTAs + slot) << 2) |
           static_cast<size_t>((kh << 1) | wh);
  }
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  return ((static_cast<size_t>(wave_base) * num_kg2 +
           static_cast<size_t>(kg2) * w_size + slot)
          << 1) |
         static_cast<size_t>(kh);
}

__device__ __host__ __forceinline__ size_t dhf_wave_pair_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wp = wave >> 1;
  const int wh = wave & 1;
  const int pair_base = wp * (2 * kWaveCTAs);
  if (pair_base + 2 * kWaveCTAs <= n_blocks) {
    return static_cast<size_t>(pair_base) * num_kg +
           (static_cast<size_t>(kg) * kWaveCTAs + slot) * 2 + wh;
  }
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  return static_cast<size_t>(wave_base) * num_kg + static_cast<size_t>(kg) * w_size + slot;
}

__device__ __forceinline__ float dhf_e2m1_x2_to_f32(uint32_t nib) {
  const uint32_t mag_u = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
  const float mag = __uint_as_float(0x41800000u | (mag_u << 19)) - 16.0f;
  return __uint_as_float(__float_as_uint(mag) ^ ((nib & 8u) << 28));
}

__device__ __forceinline__ float dhf_fp8_e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint32_t dhf_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void dhf_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t dhf_pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <bool kEvictFirst = true>
__device__ __forceinline__ void dhf_cp_async16_cg(uint32_t sd, const void* gs, uint64_t pol = 0ull) {
  if constexpr (kEvictFirst) {
    asm volatile("cp.async.cg.shared.global.L2::256B.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "l"(pol));
  } else {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
  }
}

__device__ __forceinline__ void dhf_mbar_init(uint64_t* mbar, int count) {
  for (int s = 0; s < count; ++s) {
    const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[s]));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

template <bool kEvictFirst = true>
__device__ __forceinline__ void dhf_tma_bulk_w(
    void* smem_dst, const void* gmem_src, int bytes, uint64_t* mbar, uint64_t pol) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
  if constexpr (kEvictFirst) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        ::"r"(d), "l"(gmem_src), "r"(bytes), "r"(b), "l"(pol)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        ::"r"(d), "l"(gmem_src), "r"(bytes), "r"(b)
        : "memory");
  }
}

__device__ __forceinline__ void dhf_mbar_wait(uint64_t* mbar, uint32_t parity) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n\t.reg .pred P1;\n\tDHF_MBAR_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra DHF_MBAR_WAIT_%=;\n\t}"
      ::"r"(b), "r"(parity)
      : "memory");
}

__device__ __forceinline__ float dhf_ldcg_f(const float* ptr) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ int32_t dhf_ldcg_i(const int32_t* ptr) {
  int32_t v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t dhf_vq8d128_encode(
    const float (&x)[8], const float4* __restrict__ s_cb0, const float4* __restrict__ s_cb1,
    const float* __restrict__ s_hcn2) {
  uint32_t best_code = 0u;
  float best_sc = -1e30f;
#pragma unroll 4
  for (uint32_t k = 0u; k < 128u; ++k) {
    const float4 c0 = s_cb0[k];
    const float4 c1 = s_cb1[k];
    const float dot =
        __fmaf_rn(x[0], c0.x, __fmaf_rn(x[1], c0.y, __fmaf_rn(x[2], c0.z, __fmaf_rn(x[3], c0.w,
        __fmaf_rn(x[4], c1.x, __fmaf_rn(x[5], c1.y, __fmaf_rn(x[6], c1.z, x[7] * c1.w)))))));
    const float sc = fabsf(dot) - s_hcn2[k];
    if (sc > best_sc) {
      best_sc = sc;
      best_code = k | ((dot < 0.0f) ? 128u : 0u);
    }
  }
  return best_code;
}

__device__ __forceinline__ void dhf_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void dhf_ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ bool dhf_better(float v1, int i1, float v2, int i2) {
  return (v1 > v2) || (v1 == v2 && i1 < i2);
}

__device__ __forceinline__ void dhf_top2_insert(
    float& v1, int& i1, float& v2, int& i2, float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v2 = cand_v; i2 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top4_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v4 = cand_v; i4 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top6_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 ||
      cand_i == i4 || cand_i == i5 || cand_i == i6) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
    v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
    v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = cand_v; i4 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v5, i5)) {
    v6 = v5; i6 = i5; v5 = cand_v; i5 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v6, i6)) {
    v6 = cand_v; i6 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top8_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4 ||
      cand_i == i5 || cand_i == i6 || cand_i == i7 || cand_i == i8) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = cand_v; i4 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v5, i5)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = cand_v; i5 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v6, i6)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = cand_v; i6 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v7, i7)) {
    v8 = v7; i8 = i7; v7 = cand_v; i7 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v8, i8)) {
    v8 = cand_v; i8 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top8_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6,
    float ov7, int oi7, float ov8, int oi8) {
  if (ov1 >= v8) {
    dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov1, oi1);
    if (ov2 >= v8) {
      dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov2, oi2);
      if (ov3 >= v8) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov3, oi3);
        if (ov4 >= v8) {
          dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov4, oi4);
          if (ov5 >= v8) {
            dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov5, oi5);
            if (ov6 >= v8) {
              dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov6, oi6);
              if (ov7 >= v8) {
                dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov7, oi7);
                if (ov8 >= v8) {
                  dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov8, oi8);
                }
              }
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dhf_top6_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6) {
  if (ov1 >= v6) {
    dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov1, oi1);
    if (ov2 >= v6) {
      dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov2, oi2);
      if (ov3 >= v6) {
        dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov3, oi3);
        if (ov4 >= v6) {
          dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov4, oi4);
          if (ov5 >= v6) {
            dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov5, oi5);
            if (ov6 >= v6) {
              dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov6, oi6);
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dhf_top4_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4) {
  if (ov1 >= v4) {
    dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov1, oi1);
    if (ov2 >= v4) {
      dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov2, oi2);
      if (ov3 >= v4) {
        dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov3, oi3);
        if (ov4 >= v4) {
          dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov4, oi4);
        }
      }
    }
  }
}

__device__ __forceinline__ float dhf_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

// ============================================================================
// Boot-time GPU Packer Kernels (~15 ms total for 98,304 x 6,656)
// ============================================================================

// Pass 1: Compute per-column mean and std across all V_static rows directly from raw NVFP4.
// Grid: num_g = K / 16 blocks, Block: 256 threads (16 threads per group column? No: 256 threads
// stride over V_static rows, each thread accumulates 16 columns in registers -> 1 block reduction).
__global__ void dhf_col_stats_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ col_mean,
    float* __restrict__ col_std,
    int V_static, int K) {
  const int g = blockIdx.x;  // 0 .. K/16 - 1
  const int tid = threadIdx.x;
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const float gs_half = __ldg(global_scale) * 0.5f;

  float s1[16];
  float s2[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    s1[j] = 0.0f;
    s2[j] = 0.0f;
  }

  for (int r = tid; r < V_static; r += blockDim.x) {
    const int tok = __ldg(&packed_tok_ids[r]);
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&raw_s[static_cast<size_t>(tok) * num_g + g])) * gs_half;
    const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(
        raw_w + static_cast<size_t>(tok) * K2 + g * 8));
    uint32_t wlo = static_cast<uint32_t>(w8);
    uint32_t whi = static_cast<uint32_t>(w8 >> 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc;
      s1[2 * j + 0] += v0; s2[2 * j + 0] = __fmaf_rn(v0, v0, s2[2 * j + 0]);
      s1[2 * j + 1] += v1; s2[2 * j + 1] = __fmaf_rn(v1, v1, s2[2 * j + 1]);
      wlo >>= 8;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(whi & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc;
      s1[8 + 2 * j + 0] += v0; s2[8 + 2 * j + 0] = __fmaf_rn(v0, v0, s2[8 + 2 * j + 0]);
      s1[8 + 2 * j + 1] += v1; s2[8 + 2 * j + 1] = __fmaf_rn(v1, v1, s2[8 + 2 * j + 1]);
      whi >>= 8;
    }
  }

  __shared__ float sm_s1[8][16];
  __shared__ float sm_s2[8][16];
  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float w1 = dhf_warp_sum(s1[j]);
    const float w2 = dhf_warp_sum(s2[j]);
    if (lane == 0) {
      sm_s1[warp][j] = w1;
      sm_s2[warp][j] = w2;
    }
  }
  __syncthreads();

  if (tid < 16) {
    float tot1 = 0.0f;
    float tot2 = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      tot1 += sm_s1[w][tid];
      tot2 += sm_s2[w][tid];
    }
    const float inv_v = 1.0f / static_cast<float>(V_static);
    const float mu = tot1 * inv_v;
    const float var = fmaxf(tot2 * inv_v - mu * mu, 1e-12f);
    col_mean[g * 16 + tid] = mu;
    col_std[g * 16 + tid] = sqrtf(var);
  }
}

__global__ void dhf_extract_8d_samples_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_std,
    float* __restrict__ out_samples,
    int V_static, int K) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x;
  const int pack_row = static_cast<int>((static_cast<int64_t>(b) * V_static) / 512);
  const int tok = __ldg(&packed_tok_ids[pack_row]);
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const float gs_half = __ldg(global_scale) * 0.5f;
  const uint8_t* w_row = raw_w + static_cast<size_t>(tok) * K2;
  const uint8_t* s_row = raw_s + static_cast<size_t>(tok) * num_g;

  float u2 = 0.0f;
  for (int g = lane; g < num_g; g += 32) {
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
    const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
    uint32_t wlo = static_cast<uint32_t>(w8);
    uint32_t whi = static_cast<uint32_t>(w8 >> 32);
    const int kb = g << 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float u0 = (dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc - __ldg(&col_mean[kb + 2 * j + 0])) / __ldg(&col_std[kb + 2 * j + 0]);
      const float u1 = (dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc - __ldg(&col_mean[kb + 2 * j + 1])) / __ldg(&col_std[kb + 2 * j + 1]);
      const float u2v = (dhf_e2m1_x2_to_f32(whi & 0xFu) * sc - __ldg(&col_mean[kb + 8 + 2 * j + 0])) / __ldg(&col_std[kb + 8 + 2 * j + 0]);
      const float u3v = (dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc - __ldg(&col_mean[kb + 8 + 2 * j + 1])) / __ldg(&col_std[kb + 8 + 2 * j + 1]);
      u2 = __fmaf_rn(u0, u0, __fmaf_rn(u1, u1, __fmaf_rn(u2v, u2v, __fmaf_rn(u3v, u3v, u2))));
      wlo >>= 8;
      whi >>= 8;
    }
  }
  const float inv_std = rsqrtf(fmaxf(dhf_warp_sum(u2) / static_cast<float>(K), 1e-12f));

  const int num_g2 = num_g >> 1;
  const int g2 = static_cast<int>((static_cast<int64_t>(lane) * num_g2) / 32);
  const int t = (b + lane) & 3;
  const int g0 = g2 << 1;
  const int g1 = g0 | 1;
  const int kb0 = (g0 << 4) + 2 * t;
  const int kb1 = (g1 << 4) + 2 * t;
  const int ks[8] = {kb0 + 0, kb0 + 1, kb0 + 8, kb0 + 9, kb1 + 0, kb1 + 1, kb1 + 8, kb1 + 9};
  float* dst = out_samples + (static_cast<size_t>(b) * 32 + lane) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int kk = ks[i];
    const int gg = kk >> 4;
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[gg])) * gs_half;
    const uint8_t bt = __ldg(&w_row[kk >> 1]);
    const uint32_t nib = (kk & 1) ? (bt >> 4) : (bt & 0xFu);
    dst[i] = ((dhf_e2m1_x2_to_f32(nib) * sc - __ldg(&col_mean[kk])) / __ldg(&col_std[kk])) * inv_std;
  }
}

// Pass 2: Compute weighted-LS optimal per-row scale & threshold in centered logit space.
// Grid: V_static / 8 blocks, Block: 256 threads (1 warp per packed row).
template <int kMode>
__global__ void dhf_row_scales_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_std,
    float* __restrict__ row_scales,
    float* __restrict__ row_thresh,
    int V_static, int K) {
  __shared__ float4 s_cb0[(kMode == 100) ? 128 : 1];
  __shared__ float4 s_cb1[(kMode == 100) ? 128 : 1];
  __shared__ float s_hcn2[(kMode == 100) ? 128 : 1];
  if constexpr (kMode == 100) {
    if (threadIdx.x < 128) {
      const float* cb = col_std + K + threadIdx.x * 8;
      const float4 c0 = make_float4(cb[0], cb[1], cb[2], cb[3]);
      const float4 c1 = make_float4(cb[4], cb[5], cb[6], cb[7]);
      s_cb0[threadIdx.x] = c0;
      s_cb1[threadIdx.x] = c1;
      s_hcn2[threadIdx.x] = 0.5f * (c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                                    c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w);
    }
    __syncthreads();
  }
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= V_static) return;

  const int tok = __ldg(&packed_tok_ids[warp_id]);
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const float gs_half = __ldg(global_scale) * 0.5f;
  const uint8_t* w_row = raw_w + static_cast<size_t>(tok) * K2;
  const uint8_t* s_row = raw_s + static_cast<size_t>(tok) * num_g;

  // Step 1: Compute RMS of whitened centered row u[k] = (w[k] - mu[k]) / cs[k].
  float u2 = 0.0f;
  for (int g = lane; g < num_g; g += 32) {
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
    const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
    uint32_t wlo = static_cast<uint32_t>(w8);
    uint32_t whi = static_cast<uint32_t>(w8 >> 32);
    const int kb = g << 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc;
      const float u0 = (v0 - __ldg(&col_mean[kb + 2 * j + 0])) / __ldg(&col_std[kb + 2 * j + 0]);
      const float u1 = (v1 - __ldg(&col_mean[kb + 2 * j + 1])) / __ldg(&col_std[kb + 2 * j + 1]);
      u2 = __fmaf_rn(u0, u0, u2);
      u2 = __fmaf_rn(u1, u1, u2);
      wlo >>= 8;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(whi & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc;
      const float u0 = (v0 - __ldg(&col_mean[kb + 8 + 2 * j + 0])) / __ldg(&col_std[kb + 8 + 2 * j + 0]);
      const float u1 = (v1 - __ldg(&col_mean[kb + 8 + 2 * j + 1])) / __ldg(&col_std[kb + 8 + 2 * j + 1]);
      u2 = __fmaf_rn(u0, u0, u2);
      u2 = __fmaf_rn(u1, u1, u2);
      whi >>= 8;
    }
  }
  const float std_u = sqrtf(fmaxf(dhf_warp_sum(u2) / static_cast<float>(K), 1e-12f));

  if constexpr (kMode == 100) {
    const float inv_std = 1.0f / std_u;
    float num = 0.0f;
    float den = 0.0f;
    const int num_g2 = num_g >> 1;
    for (int g2 = lane; g2 < num_g2; g2 += 32) {
      const int g0 = g2 << 1;
      const int g1 = g0 | 1;
      const float sc0 = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g0])) * gs_half;
      const float sc1 = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g1])) * gs_half;
      const uint64_t w8_0 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g0 * 8));
      const uint64_t w8_1 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g1 * 8));
      uint32_t wlo0 = static_cast<uint32_t>(w8_0), whi0 = static_cast<uint32_t>(w8_0 >> 32);
      uint32_t wlo1 = static_cast<uint32_t>(w8_1), whi1 = static_cast<uint32_t>(w8_1 >> 32);
      const int kb0 = g0 << 4;
      const int kb1 = g1 << 4;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int ks[8] = {
            kb0 + 2 * t + 0, kb0 + 2 * t + 1, kb0 + 8 + 2 * t + 0, kb0 + 8 + 2 * t + 1,
            kb1 + 2 * t + 0, kb1 + 2 * t + 1, kb1 + 8 + 2 * t + 0, kb1 + 8 + 2 * t + 1};
        const float raw_vals[8] = {
            dhf_e2m1_x2_to_f32(wlo0 & 0xFu) * sc0, dhf_e2m1_x2_to_f32((wlo0 >> 4) & 0xFu) * sc0,
            dhf_e2m1_x2_to_f32(whi0 & 0xFu) * sc0, dhf_e2m1_x2_to_f32((whi0 >> 4) & 0xFu) * sc0,
            dhf_e2m1_x2_to_f32(wlo1 & 0xFu) * sc1, dhf_e2m1_x2_to_f32((wlo1 >> 4) & 0xFu) * sc1,
            dhf_e2m1_x2_to_f32(whi1 & 0xFu) * sc1, dhf_e2m1_x2_to_f32((whi1 >> 4) & 0xFu) * sc1};
        float cs[8], wc[8], x[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          cs[i] = __ldg(&col_std[ks[i]]);
          wc[i] = raw_vals[i] - __ldg(&col_mean[ks[i]]);
          x[i] = (wc[i] / cs[i]) * inv_std;
        }
        const uint32_t code = dhf_vq8d128_encode(x, s_cb0, s_cb1, s_hcn2);
        const float4 c0 = s_cb0[code & 127u];
        const float4 c1 = s_cb1[code & 127u];
        const float sgn = (code & 128u) ? -1.0f : 1.0f;
        const float q[8] = {sgn * c0.x, sgn * c0.y, sgn * c0.z, sgn * c0.w,
                            sgn * c1.x, sgn * c1.y, sgn * c1.z, sgn * c1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          num = __fmaf_rn(wc[i] * cs[i], q[i], num);
          den = __fmaf_rn(cs[i] * cs[i], q[i] * q[i], den);
        }
        wlo0 >>= 8; whi0 >>= 8;
        wlo1 >>= 8; whi1 >>= 8;
      }
    }
    const float g_num = dhf_warp_sum(num);
    const float g_den = dhf_warp_sum(den);
    const float alpha = (g_den > 1e-20f) ? (g_num / g_den) : std_u;
    if (lane == 0) {
      row_scales[warp_id] = alpha;
      row_thresh[warp_id] = inv_std;
    }
    return;
  }

  if constexpr (kMode == 150) {
    // 4D-64 Lloyd-Max Gaussian VQ on exact MMA quads {kb+0, kb+1, kb+8, kb+9}.
    const float inv_std = 1.0f / std_u;
    float num = 0.0f;
    float den = 0.0f;
    for (int g = lane; g < num_g; g += 32) {
      const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      uint32_t wlo = static_cast<uint32_t>(w8);
      uint32_t whi = static_cast<uint32_t>(w8 >> 32);
      const int kb = g << 4;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int k0 = kb + 2 * t + 0;
        const int k1 = kb + 2 * t + 1;
        const int k2 = kb + 8 + 2 * t + 0;
        const int k3 = kb + 8 + 2 * t + 1;
        const float cs0 = __ldg(&col_std[k0]);
        const float cs1 = __ldg(&col_std[k1]);
        const float cs2 = __ldg(&col_std[k2]);
        const float cs3 = __ldg(&col_std[k3]);
        const float wc0 = dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc - __ldg(&col_mean[k0]);
        const float wc1 = dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc - __ldg(&col_mean[k1]);
        const float wc2 = dhf_e2m1_x2_to_f32(whi & 0xFu) * sc - __ldg(&col_mean[k2]);
        const float wc3 = dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc - __ldg(&col_mean[k3]);
        const uint32_t bi = dhf_vq4d64_encode(
            (wc0 / cs0) * inv_std, (wc1 / cs1) * inv_std,
            (wc2 / cs2) * inv_std, (wc3 / cs3) * inv_std);
        const float q0 = kVq4D64[bi][0];
        const float q1 = kVq4D64[bi][1];
        const float q2 = kVq4D64[bi][2];
        const float q3 = kVq4D64[bi][3];
        num = __fmaf_rn(wc0 * cs0, q0, __fmaf_rn(wc1 * cs1, q1, __fmaf_rn(wc2 * cs2, q2, __fmaf_rn(wc3 * cs3, q3, num))));
        den = __fmaf_rn(cs0 * cs0, q0 * q0, __fmaf_rn(cs1 * cs1, q1 * q1, __fmaf_rn(cs2 * cs2, q2 * q2, __fmaf_rn(cs3 * cs3, q3 * q3, den))));
        wlo >>= 8;
        whi >>= 8;
      }
    }
    const float g_num = dhf_warp_sum(num);
    const float g_den = dhf_warp_sum(den);
    const float alpha = (g_den > 1e-20f) ? (g_num / g_den) : std_u;
    if (lane == 0) {
      row_scales[warp_id] = alpha;
      row_thresh[warp_id] = inv_std;
    }
    return;
  }

  float thresh = (kMode == 158) ? (0.6120f * std_u) : (0.9957f * std_u);
  float alpha = std_u;

  // Step 2: 2-iteration weighted least-squares refinement in true centered logit space.
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    float num = 0.0f;
    float den = 0.0f;
    for (int g = lane; g < num_g; g += 32) {
      const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      uint32_t wlo = static_cast<uint32_t>(w8);
      uint32_t whi = static_cast<uint32_t>(w8 >> 32);
      const int kb = g << 4;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t ww = (h == 0) ? wlo : whi;
        const int ko = kb + (h << 3);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float v0 = dhf_e2m1_x2_to_f32(ww & 0xFu) * sc;
          const float v1 = dhf_e2m1_x2_to_f32((ww >> 4) & 0xFu) * sc;
          const float c0 = __ldg(&col_std[ko + 2 * j + 0]);
          const float c1 = __ldg(&col_std[ko + 2 * j + 1]);
          const float wc0 = v0 - __ldg(&col_mean[ko + 2 * j + 0]);
          const float wc1 = v1 - __ldg(&col_mean[ko + 2 * j + 1]);
          const float u0 = wc0 / c0;
          const float u1 = wc1 / c1;
          float q0, q1;
          if constexpr (kMode == 158) {
            q0 = (u0 > thresh) ? 1.0f : ((u0 < -thresh) ? -1.0f : 0.0f);
            q1 = (u1 > thresh) ? 1.0f : ((u1 < -thresh) ? -1.0f : 0.0f);
          } else {
            q0 = (u0 >= thresh) ? 3.0f : ((u0 >= 0.0f) ? 1.0f : ((u0 > -thresh) ? -1.0f : -3.0f));
            q1 = (u1 >= thresh) ? 3.0f : ((u1 >= 0.0f) ? 1.0f : ((u1 > -thresh) ? -1.0f : -3.0f));
          }
          num = __fmaf_rn(wc0 * c0, q0, num);
          num = __fmaf_rn(wc1 * c1, q1, num);
          den = __fmaf_rn(c0 * c0, q0 * q0, den);
          den = __fmaf_rn(c1 * c1, q1 * q1, den);
          ww >>= 8;
        }
      }
    }
    const float g_num = dhf_warp_sum(num);
    const float g_den = dhf_warp_sum(den);
    if (g_den > 1e-20f) {
      alpha = g_num / g_den;
      if (it == 0) {
        thresh = (kMode == 158) ? (0.5f * fabsf(alpha)) : (2.0f * fabsf(alpha));
      }
    }
  }

  if (lane == 0) {
    row_scales[warp_id] = alpha;
    row_thresh[warp_id] = thresh;
  }
}

// Helper: decode single NVFP4 element at (tok, k) directly from raw_w/raw_s.
__device__ __forceinline__ float dhf_decode_nvfp4_elem(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    float gs_half, int tok, int k, int K2, int num_g) {
  const int g = k >> 4;
  const float sc = dhf_fp8_e4m3_to_f32(__ldg(&raw_s[static_cast<size_t>(tok) * num_g + g])) * gs_half;
  const uint8_t byte = __ldg(&raw_w[static_cast<size_t>(tok) * K2 + (k >> 1)]);
  const uint32_t nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
  return dhf_e2m1_x2_to_f32(nib) * sc;
}

// Pass 3: Pack wave-interleaved 8x64 tiles for mode=158 (104 B/tile) or mode=2 (128 B/tile).
// Grid: (V_static / 8, K / 64), Block: 32 threads (1 warp per 8x64 tile).
template <int kMode, int kLayout = 0>
__global__ void dhf_pack_tiles_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_std,
    const float* __restrict__ row_thresh,
    uint8_t* __restrict__ out,
    int V_static, int K) {
  const int n_tile = blockIdx.x;  // 0 .. V_static/8 - 1
  const int kg = blockIdx.y;      // 0 .. K/64 - 1
  const int lane = threadIdx.x;   // 0 .. 31
  const int r = lane >> 2;        // 0 .. 7
  const int t = lane & 3;         // 0 .. 3
  const int pack_row = n_tile * 8 + r;
  const int tok = __ldg(&packed_tok_ids[pack_row]);
  const float thresh = __ldg(&row_thresh[pack_row]);
  const float gs_half = __ldg(global_scale) * 0.5f;
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const int k_base = kg * 64;

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int n_blocks = V_static / kRowsPerCTA;
  const int num_kg = K >> 6;
  const size_t step_slot =
      (kMode == 100 && kLayout == 3)
          ? dhf_wave_pair_bk128_step_idx(cta_n, kg, n_blocks, num_kg)
          : ((kMode == 100 && kLayout == 2)
                 ? dhf_wave_bk128_step_idx(cta_n, kg, n_blocks, num_kg)
                 : ((kMode == 100 && kLayout == 1)
                        ? dhf_wave_pair_step_idx(cta_n, kg, n_blocks, num_kg)
                        : dhf_wave_step_idx(cta_n, kg, n_blocks, num_kg)));
  const size_t tile_idx = step_slot * kNTilesPerCTA + sub_tile;

  __shared__ float4 s_cb0[(kMode == 100) ? 128 : 1];
  __shared__ float4 s_cb1[(kMode == 100) ? 128 : 1];
  __shared__ float s_hcn2[(kMode == 100) ? 128 : 1];
  if constexpr (kMode == 100) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) {
      const float* cb = col_std + K + idx * 8;
      const float4 c0 = make_float4(cb[0], cb[1], cb[2], cb[3]);
      const float4 c1 = make_float4(cb[4], cb[5], cb[6], cb[7]);
      s_cb0[idx] = c0;
      s_cb1[idx] = c1;
      s_hcn2[idx] = 0.5f * (c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                            c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w);
    }
    __syncthreads();
  }

  if constexpr (kMode == 100) {
    const float inv_std = thresh;
    uint32_t c16 = 0u;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int kb0 = k_base + (2 * pair + 0) * 16 + 2 * t;
      const int kb1 = k_base + (2 * pair + 1) * 16 + 2 * t;
      const int ks[8] = {kb0 + 0, kb0 + 1, kb0 + 8, kb0 + 9, kb1 + 0, kb1 + 1, kb1 + 8, kb1 + 9};
      float x[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int kk = ks[i];
        x[i] = ((dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, kk, K2, num_g) - __ldg(&col_mean[kk])) /
                __ldg(&col_std[kk])) * inv_std;
      }
      const uint32_t code = dhf_vq8d128_encode(x, s_cb0, s_cb1, s_hcn2);
      c16 |= (code << (8 * pair));
    }
    reinterpret_cast<uint16_t*>(out + tile_idx * kTileBytes100)[lane] = static_cast<uint16_t>(c16);
  } else if constexpr (kMode == 150) {
    const float inv_std = thresh;  // row_thresh holds inv_std in mode 150
    uint32_t c24 = 0u;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kb = k_base + s * 16 + 2 * t;
      const int k0 = kb + 0, k1 = kb + 1, k2 = kb + 8, k3 = kb + 9;
      const float x0 = ((dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, k0, K2, num_g) - __ldg(&col_mean[k0])) / __ldg(&col_std[k0])) * inv_std;
      const float x1 = ((dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, k1, K2, num_g) - __ldg(&col_mean[k1])) / __ldg(&col_std[k1])) * inv_std;
      const float x2 = ((dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, k2, K2, num_g) - __ldg(&col_mean[k2])) / __ldg(&col_std[k2])) * inv_std;
      const float x3 = ((dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, k3, K2, num_g) - __ldg(&col_mean[k3])) / __ldg(&col_std[k3])) * inv_std;
      const uint32_t bi = dhf_vq4d64_encode(x0, x1, x2, x3);
      c24 |= (bi << (6 * s));
    }
    // Pack 32 x 24-bit values into 24 x uint32_t (96 bytes = 6 x 16B) via full-warp __shfl_sync.
    const int w_safe = (lane < 24) ? lane : 0;
    const int grp = w_safe / 3;
    const int rem = w_safe - grp * 3;
    const int base_l = grp * 4;
    const uint32_t v0 = __shfl_sync(0xffffffffu, c24, base_l + 0);
    const uint32_t v1 = __shfl_sync(0xffffffffu, c24, base_l + 1);
    const uint32_t v2 = __shfl_sync(0xffffffffu, c24, base_l + 2);
    const uint32_t v3 = __shfl_sync(0xffffffffu, c24, base_l + 3);
    const uint32_t word =
        (rem == 0) ? (v0 | (v1 << 24))
                   : ((rem == 1) ? ((v1 >> 8) | (v2 << 16))
                                 : ((v2 >> 16) | (v3 << 8)));
    if (lane < 24) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(out + tile_idx * kTileBytes150);
      dst[lane] = word;
    }
  } else if constexpr (kMode == 158) {
    uint32_t code[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kb = k_base + s * 16 + 2 * t;
      const int ks[4] = {kb + 0, kb + 1, kb + 8, kb + 9};
      uint32_t tr[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kk = ks[j];
        const float w = dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, kk, K2, num_g);
        const float u = (w - __ldg(&col_mean[kk])) / __ldg(&col_std[kk]);
        tr[j] = (u > thresh) ? 2u : ((u < -thresh) ? 0u : 1u);
      }
      code[s] = tr[0] + 3u * tr[1] + 9u * tr[2] + 27u * tr[3];
    }
    const uint32_t c_lo = code[0] + 81u * code[1];
    const uint32_t c_hi = code[2] + 81u * code[3];
    const uint32_t c26 = c_lo | (c_hi << 13);

    // Pack 32 x 26-bit values into 26 x uint32_t (104 bytes) via full-warp __shfl_sync.
    const int w_safe = (lane < 26) ? lane : 0;
    const int bit_start = w_safe * 32;
    const int l0 = bit_start / 26;
    const int o0 = bit_start - 26 * l0;
    const int n0 = 26 - o0;  // 1..26
    const int l1 = (l0 + 1 < 32) ? (l0 + 1) : 31;
    const int l2 = (l0 + 2 < 32) ? (l0 + 2) : 31;
    const uint32_t v0 = __shfl_sync(0xffffffffu, c26, l0);
    const uint32_t v1 = __shfl_sync(0xffffffffu, c26, l1);
    const uint32_t v2 = __shfl_sync(0xffffffffu, c26, l2);
    uint32_t word = (v0 >> o0) | (v1 << n0);
    if (n0 < 6) {
      word |= (v2 << (n0 + 26));
    }
    if (lane < 26) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(out + tile_idx * kTileBytes158);
      dst[lane] = word;
    }
  } else {
    uint32_t q32 = 0u;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kb = k_base + s * 16 + 2 * t;
      const int ks[4] = {kb + 0, kb + 1, kb + 8, kb + 9};
      uint32_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kk = ks[j];
        const float w = dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, kk, K2, num_g);
        const float u = (w - __ldg(&col_mean[kk])) / __ldg(&col_std[kk]);
        q[j] = (u >= thresh) ? 3u : ((u >= 0.0f) ? 2u : ((u > -thresh) ? 1u : 0u));
      }
      // Marlin-interleaved 2-bit layout:
      //   q[0] at [4*s+1 : 4*s], q[2] at [4*s+3 : 4*s+2] (low 16)
      //   q[1] at [16+4*s+1 : 16+4*s], q[3] at [16+4*s+3 : 16+4*s+2] (high 16)
      q32 |= (q[0] << (4 * s)) | (q[2] << (4 * s + 2)) |
             (q[1] << (16 + 4 * s)) | (q[3] << (16 + 4 * s + 2));
    }
    uint32_t* dst = reinterpret_cast<uint32_t*>(out + tile_idx * kTileBytes2);
    dst[lane] = q32;
  }
}

// ============================================================================
// Inference Kernels (100% CUDA-Graph Safe, PDL-Chained)
// ============================================================================

// Kernel 0: Scale input A [M, K] by col_scales [K] -> A_scaled [M, K] (L2-hot, ~2 us).
__global__ void dhf_prep_a_kernel(
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ col_scales,
    __nv_bfloat16* __restrict__ A_scaled,
    int M, int K, int pdl_mode) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const __nv_bfloat16* a_in = A + static_cast<size_t>(m) * K;
  __nv_bfloat16* a_out = A_scaled + static_cast<size_t>(m) * K;
  const int k2 = K >> 1;
  for (int i = threadIdx.x; i < k2; i += blockDim.x) {
    const __nv_bfloat162 av = reinterpret_cast<const __nv_bfloat162*>(a_in)[i];
    const float2 af = __bfloat1622float2(av);
    const float2 cs = reinterpret_cast<const float2*>(col_scales)[i];
    reinterpret_cast<__nv_bfloat162*>(a_out)[i] = __floats2bfloat162_rn(af.x * cs.x, af.y * cs.y);
  }
  if ((pdl_mode & 2) != 0 && threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

// Native BF16x2 2-bit odd-integer dequant: q in {0,1,2,3} -> {-3,-1,+1,+3} via 1 LOP3 + 1 HFMA2.
template <int kStep>
__device__ __forceinline__ void dhf_deq2_k16(uint32_t q32, uint32_t& b0, uint32_t& b1) {
  uint32_t t0, t1;
  if constexpr (kStep == 0) {
    t0 = ((q32 << 4) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 << 2) & 0x00300030u) | 0x41004100u;
  } else if constexpr (kStep == 1) {
    t0 = (q32 & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 2) & 0x00300030u) | 0x41004100u;
  } else if constexpr (kStep == 2) {
    t0 = ((q32 >> 4) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 6) & 0x00300030u) | 0x41004100u;
  } else {
    t0 = ((q32 >> 8) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 10) & 0x00300030u) | 0x41004100u;
  }
  const uint32_t two_u = 0x40004000u;    // bf16x2(2.0, 2.0)
  const uint32_t neg19_u = 0xC198C198u;  // bf16x2(-19.0, -19.0)
  const __nv_bfloat162 two = *reinterpret_cast<const __nv_bfloat162*>(&two_u);
  const __nv_bfloat162 neg19 = *reinterpret_cast<const __nv_bfloat162*>(&neg19_u);
  const __nv_bfloat162 w0 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&t0), two, neg19);
  const __nv_bfloat162 w1 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&t1), two, neg19);
  b0 = *reinterpret_cast<const uint32_t*>(&w0);
  b1 = *reinterpret_cast<const uint32_t*>(&w1);
}

template <int kStepWBytes, int kMode>
struct alignas(16) DhfStageSmem {
  uint8_t w[kStages][kStepWBytes];
  __nv_bfloat16 a[kStages][kMaxM][64];
  union {
    uint2 lut[81];
    uint4 lut8[(kMode == 100) ? 256 : 1];
  } u;
  float s_scores[kMaxM][kRowsPerCTA];
  int32_t s_toks[kRowsPerCTA];
  float s_topv[kMaxM][kCtaTopK];
  int32_t s_topi[kMaxM][kCtaTopK];
};

constexpr int kPairStages100 = 6;
constexpr int kPairStepWBytes100 = 2 * kStepWBytes100;  // 1024 B = 64 * uint4

struct alignas(16) DhfPair100Smem {
  uint8_t w[kPairStages100][kPairStepWBytes100];  // 6 * 1024 = 6144 B
  __nv_bfloat16 a[kPairStages100][kMaxM][64];     // 6 * 1024 = 6144 B
  uint4 lut8[256];                                // 4096 B (1 x LDS.128 antipodal 8D BF16x2 LUT)
  float s_scores[kMaxM][2 * kRowsPerCTA];         // 4096 B
  int32_t s_toks[2 * kRowsPerCTA];                // 512 B
  float s_topv[kMaxM][kCtaTopK];                  // 256 B
  int32_t s_topi[kMaxM][kCtaTopK];                // 256 B -> 21,504 B (21.0 KiB -> 4 CTAs/SM!)
};

// Paired-wave 128-row/CTA Stage-1 kernel for mode=100 (8D-256 1.000 bpw):
// Halves __syncthreads(), cp.async.wait_group, A_scaled cp.async, and ldmatrix.x2 (832 -> 416 steps),
// utilizes 100% of 128 threads on cp.async (64 A + 64 W = 1024B contiguous W/CTA), eliminates /192 divs,
// and uses single-instruction LDS.128 lut8[256] — 100.0000% 0-ULP bit-identical to dhf_stage1_kernel<100>!
template <bool kEvictFirst = true>
__global__ void __launch_bounds__(128, 4) dhf_stage1_mode100_pair_kernel(
    const __nv_bfloat16* __restrict__ A_scaled,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ row_scales,
    const float* __restrict__ col_scales,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ cta_vals,
    int32_t* __restrict__ cta_ids,
    int M, int V_static, int K, int pdl_mode) {
  __shared__ DhfPair100Smem smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int n_blocks = V_static / kRowsPerCTA;
  const int bid = blockIdx.x;

  {
    const float* cb = col_scales + K + tid * 8;
    const uint4 pos = make_uint4(
        dhf_bf16x2_u32(cb[0], cb[1]),
        dhf_bf16x2_u32(cb[2], cb[3]),
        dhf_bf16x2_u32(cb[4], cb[5]),
        dhf_bf16x2_u32(cb[6], cb[7]));
    smem.lut8[tid] = pos;
    smem.lut8[tid + 128] = make_uint4(
        pos.x ^ 0x80008000u,
        pos.y ^ 0x80008000u,
        pos.z ^ 0x80008000u,
        pos.w ^ 0x80008000u);
  }

  if (tid < kMaxM * kCtaTopK) {
    smem.s_topv[tid >> 3][tid & 7] = -1e38f;
    smem.s_topi[tid >> 3][tid & 7] = -1;
  }

  if (bid == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg * kStepWBytes100;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (tid < 64) && (a_row < M);
  const int w_u4_idx = tid - 64;

  uint32_t ldm_off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
    ldm_off[s] = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[0][lane & 7][ldm_col8 * 8]) -
        __cvta_generic_to_shared(&smem.a[0][0][0]));
  }

  const int num_pairs = n_blocks / (2 * kWaveCTAs);
  const int total_steps = num_pairs * num_kg;
  const uint64_t w_pol = kEvictFirst ? dhf_pol_evict_first() : 0ull;

  auto w_ptr_for = [&](int wp, int kg) -> const uint8_t* {
    return W_packed +
           (static_cast<size_t>(wp * num_kg + kg) * kWaveCTAs + bid) * kPairStepWBytes100;
  };

  auto issue_w_only = [&](int wp, int kg, int st) {
    if (tid >= 64) {
      const uint8_t* w_src = w_ptr_for(wp, kg);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
      dhf_cp_async16_cg<kEvictFirst>(sd, w_src + w_u4_idx * 16, w_pol);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int wp, int kg, int st) {
    if (tid >= 64) {
      const uint8_t* w_src = w_ptr_for(wp, kg);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
      dhf_cp_async16_cg<kEvictFirst>(sd, w_src + w_u4_idx * 16, w_pol);
    } else {
      const uint32_t sd = static_cast<uint32_t>(
          __cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs =
          a_valid ? (A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A_scaled;
      dhf_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kPairStages100 - 1;  // 6
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;
  int pf_wp = 0, pf_kg = 0;
  auto adv_pf = [&]() {
    if (++pf_kg == num_kg) {
      pf_kg = 0;
      ++pf_wp;
    }
  };

  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(pf_wp, pf_kg, i);
      adv_pf();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int akg = 0;
    for (int i = 0; i < preload; ++i) {
      load_a_sync(akg, i);
      if (++akg == num_kg) akg = 0;
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_stage(pf_wp, pf_kg, i);
      adv_pf();
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  float e0_0 = 0.0f, e1_0 = 0.0f, e2_0 = 0.0f, e3_0 = 0.0f;
  float e0_1 = 0.0f, e1_1 = 0.0f, e2_1 = 0.0f, e3_1 = 0.0f;
  const int trig_gs = ((pdl_mode & 2) != 0) ? ((total_steps > 4) ? (total_steps - 4) : 0) : -1;

  int cur_wp = 0;
  int cur_kg = 0;
  int st = 0;
  for (int gs = 0; gs < total_steps; ++gs) {
    if (gs == trig_gs) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = total_steps - 1 - gs;
    if (rem >= 4) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    if (gs + kPreload < total_steps) {
      const int next_st = (st == 0) ? (kPairStages100 - 1) : (st - 1);
      issue_stage(pf_wp, pf_kg, next_st);
      adv_pf();
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes100];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes100];
    const uint8_t* wt2 = &smem.w[st][kStepWBytes100 + (warp * 2 + 0) * kTileBytes100];
    const uint8_t* wt3 = &smem.w[st][kStepWBytes100 + (warp * 2 + 1) * kTileBytes100];
    const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][0][0]));

    const uint32_t c0 = reinterpret_cast<const uint16_t*>(wt0)[lane];
    const uint32_t c1 = reinterpret_cast<const uint16_t*>(wt1)[lane];
    const uint32_t c2 = reinterpret_cast<const uint16_t*>(wt2)[lane];
    const uint32_t c3 = reinterpret_cast<const uint16_t*>(wt3)[lane];

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint4 q0 = smem.lut8[(half == 0) ? (c0 & 0xFFu) : (c0 >> 8)];
      const uint4 q1 = smem.lut8[(half == 0) ? (c1 & 0xFFu) : (c1 >> 8)];
      const uint4 q2 = smem.lut8[(half == 0) ? (c2 & 0xFFu) : (c2 >> 8)];
      const uint4 q3 = smem.lut8[(half == 0) ? (c3 & 0xFFu) : (c3 >> 8)];
      {
        uint32_t a0, a2;
        dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 0]);
        dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.x, q0.y);
        dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.x, q1.y);
        dhf_mma_m16n8k16(e0_0, e1_0, e2_0, e3_0, a0, a2, q2.x, q2.y);
        dhf_mma_m16n8k16(e0_1, e1_1, e2_1, e3_1, a0, a2, q3.x, q3.y);
      }
      {
        uint32_t a0, a2;
        dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 1]);
        dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.z, q0.w);
        dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.z, q1.w);
        dhf_mma_m16n8k16(e0_0, e1_0, e2_0, e3_0, a0, a2, q2.z, q2.w);
        dhf_mma_m16n8k16(e0_1, e1_1, e2_1, e3_1, a0, a2, q3.z, q3.w);
      }
    }

    st = (st + 1 == kPairStages100) ? 0 : (st + 1);

    if (++cur_kg == num_kg) {
      const int cta_n0 = bid + (2 * cur_wp) * kWaveCTAs;
      const int row_base0 = cta_n0 * kRowsPerCTA;
      const int row_base1 = (cta_n0 + kWaveCTAs) * kRowsPerCTA;
      smem.s_toks[tid] = __ldg(&packed_tok_ids[(tid < 64) ? (row_base0 + tid) : (row_base1 + tid - 64)]);
      if (g < M) {
        const int col0 = (warp * 2 + 0) * 8 + 2 * t;
        const int col1 = (warp * 2 + 1) * 8 + 2 * t;
        smem.s_scores[g][col0 + 0] = d0_0 * __ldg(&row_scales[row_base0 + col0 + 0]);
        smem.s_scores[g][col0 + 1] = d1_0 * __ldg(&row_scales[row_base0 + col0 + 1]);
        smem.s_scores[g][col1 + 0] = d0_1 * __ldg(&row_scales[row_base0 + col1 + 0]);
        smem.s_scores[g][col1 + 1] = d1_1 * __ldg(&row_scales[row_base0 + col1 + 1]);
        smem.s_scores[g][64 + col0 + 0] = e0_0 * __ldg(&row_scales[row_base1 + col0 + 0]);
        smem.s_scores[g][64 + col0 + 1] = e1_0 * __ldg(&row_scales[row_base1 + col0 + 1]);
        smem.s_scores[g][64 + col1 + 0] = e0_1 * __ldg(&row_scales[row_base1 + col1 + 0]);
        smem.s_scores[g][64 + col1 + 1] = e1_1 * __ldg(&row_scales[row_base1 + col1 + 1]);
      }
      __syncthreads();

      for (int m = warp; m < M; m += kWarpsPerCTA) {
        const float thresh0 = smem.s_topv[m][7];
        const float v0 = smem.s_scores[m][lane];
        const float v1 = smem.s_scores[m][lane + 32];
        const float v2 = smem.s_scores[m][lane + 64];
        const float v3 = smem.s_scores[m][lane + 96];
        uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh0);
        uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh0);
        uint32_t m2 = __ballot_sync(0xffffffffu, v2 >= thresh0);
        uint32_t m3 = __ballot_sync(0xffffffffu, v3 >= thresh0);
        if (lane == 0 && ((m0 | m1 | m2 | m3) != 0u)) {
          float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
          float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
          float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
          float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
          int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
          int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
          int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
          int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
          while (m0 != 0u) {
            const int j = __ffs(m0) - 1;
            m0 &= m0 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          while (m1 != 0u) {
            const int j = 32 + __ffs(m1) - 1;
            m1 &= m1 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          while (m2 != 0u) {
            const int j = 64 + __ffs(m2) - 1;
            m2 &= m2 - 1u;
            if (smem.s_scores[m][j] >= tv8) {
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
          }
          while (m3 != 0u) {
            const int j = 96 + __ffs(m3) - 1;
            m3 &= m3 - 1u;
            if (smem.s_scores[m][j] >= tv8) {
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
          }
          smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
          smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
          smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
          smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
          smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
          smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
          smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
          smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
        }
      }
      __syncthreads();
      d0_0 = 0.0f; d1_0 = 0.0f; d2_0 = 0.0f; d3_0 = 0.0f;
      d0_1 = 0.0f; d1_1 = 0.0f; d2_1 = 0.0f; d3_1 = 0.0f;
      e0_0 = 0.0f; e1_0 = 0.0f; e2_0 = 0.0f; e3_0 = 0.0f;
      e0_1 = 0.0f; e1_1 = 0.0f; e2_1 = 0.0f; e3_1 = 0.0f;
      cur_kg = 0;
      ++cur_wp;
    }
  }

  const int row_stride = kWaveCTAs * kCtaTopK;
  if (tid < M * kCtaTopK) {
    const int m = tid >> 3;
    const int k = tid & 7;
    cta_vals[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topv[m][k];
    cta_ids[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topi[m][k];
  }
}

constexpr int kTma100Stages = 5;
constexpr int kBk128StepWBytes100 = 2 * kStepWBytes100;  // 1024 B = 8 * 128 B

struct alignas(128) DhfTma100Bk128Smem {
  alignas(128) uint8_t w[kTma100Stages][kBk128StepWBytes100];  // 5 * 1024 = 5120 B
  alignas(128) __nv_bfloat16 a[kTma100Stages][2][kMaxM][64];   // 5 * 2048 = 10240 B
  uint4 lut8[256];                                             // 4096 B
  float s_scores[kMaxM][kRowsPerCTA];                          // 2048 B
  int32_t s_toks[kRowsPerCTA];                                 // 256 B
  float s_topv[kMaxM][kCtaTopK];                               // 256 B
  int32_t s_topi[kMaxM][kCtaTopK];                             // 256 B
  alignas(8) uint64_t mbar[kTma100Stages];                     // 40 B -> 22,400 B (4 CTAs/SM!)
};

// BK=128 Hardware-TMA (1024B cp.async.bulk + L2::evict_first) persistent Stage-1 kernel for mode=100:
// Halves __syncthreads() barriers (832 -> 416), utilizes 100% of 128 threads on A cp.async while
// dedicated SM121 TMA streams contiguous 1024B weight blocks — 100.0000% 0-ULP bit-identical to dhf_stage1_kernel<100>!
template <bool kEvictFirst = true>
__global__ void __launch_bounds__(128, 4) dhf_stage1_tma100_bk128_kernel(
    const __nv_bfloat16* __restrict__ A_scaled,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ row_scales,
    const float* __restrict__ col_scales,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ cta_vals,
    int32_t* __restrict__ cta_ids,
    int M, int V_static, int K, int pdl_mode) {
  __shared__ DhfTma100Bk128Smem smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg2 = K >> 7;
  const int n_blocks = V_static / kRowsPerCTA;
  const int bid = blockIdx.x;
  const int grid_ctas = gridDim.x;
  if (bid >= n_blocks) return;

  {
    const float* cb = col_scales + K + tid * 8;
    const uint4 pos = make_uint4(
        dhf_bf16x2_u32(cb[0], cb[1]),
        dhf_bf16x2_u32(cb[2], cb[3]),
        dhf_bf16x2_u32(cb[4], cb[5]),
        dhf_bf16x2_u32(cb[6], cb[7]));
    smem.lut8[tid] = pos;
    smem.lut8[tid + 128] = make_uint4(
        pos.x ^ 0x80008000u,
        pos.y ^ 0x80008000u,
        pos.z ^ 0x80008000u,
        pos.w ^ 0x80008000u);
  }
  if (tid < kMaxM * kCtaTopK) {
    smem.s_topv[tid >> 3][tid & 7] = -1e38f;
    smem.s_topi[tid >> 3][tid & 7] = -1;
  }
  if (tid == 0) {
    dhf_mbar_init(smem.mbar, kTma100Stages);
  }
  __syncthreads();

  if (bid == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg2 * kBk128StepWBytes100;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_kh = tid >> 6;          // 0 for tid<64, 1 for tid>=64
  const int a_row = (tid >> 3) & 7;   // 0..7
  const int a_col8 = tid & 7;         // 0..7
  const int a_swiz8 = a_col8 ^ a_row;
  const bool a_valid = (a_row < M);
  const int a_safe_row = a_valid ? a_row : 0;
  const int a_src_bytes = a_valid ? 16 : 0;

  uint32_t ldm_off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
    ldm_off[s] = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[0][0][lane & 7][ldm_col8 * 8]) -
        __cvta_generic_to_shared(&smem.a[0][0][0][0]));
  }

  const int num_waves = (n_blocks - bid + grid_ctas - 1) / grid_ctas;
  const int total_steps = num_waves * num_kg2;
  const uint64_t w_pol = kEvictFirst ? dhf_pol_evict_first() : 0ull;

  auto w_ptr_for = [&](int w_iter, int kg2) -> const uint8_t* {
    const int cta_n = bid + w_iter * grid_ctas;
    const int wave = cta_n / kWaveCTAs;
    const int slot = cta_n - wave * kWaveCTAs;
    const int wave_base = wave * kWaveCTAs;
    const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
    return W_packed +
           (static_cast<size_t>(wave_base) * num_kg2 + static_cast<size_t>(kg2) * w_size + slot) *
               kBk128StepWBytes100;
  };

  auto issue_w_tma = [&](int w_iter, int kg2, int st, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      dhf_tma_bulk_w<kEvictFirst>(
          &smem.w[st][0], w_ptr_for(w_iter, kg2), kBk128StepWBytes100, &smem.mbar[st], w_pol);
    }
  };

  auto issue_a_async = [&](int kg2, int st) {
    const uint32_t sd = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[st][a_kh][a_row][a_swiz8 * 8]));
    const __nv_bfloat16* gs =
        A_scaled + static_cast<size_t>(a_safe_row) * K + (kg2 * 128 + a_kh * 64) + a_col8 * 8;
    dhf_cp_async16_ca_z(sd, gs, a_src_bytes);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kTma100Stages - 1;  // 4
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;
  int pf_w = 0, pf_kg2 = 0;
  auto adv_pf = [&]() {
    if (++pf_kg2 == num_kg2) {
      pf_kg2 = 0;
      ++pf_w;
    }
  };

  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(pf_w, pf_kg2, i, false);
      adv_pf();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int akg2 = 0;
    for (int i = 0; i < preload; ++i) {
      issue_a_async(akg2, i);
      if (++akg2 == num_kg2) akg2 = 0;
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(pf_w, pf_kg2, i, false);
      issue_a_async(pf_kg2, i);
      adv_pf();
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const int trig_gs = ((pdl_mode & 2) != 0) ? ((total_steps > 4) ? (total_steps - 4) : 0) : -1;

  int cur_w = 0;
  int cur_kg2 = 0;
  int st = 0;
  uint32_t mbar_phase = 0u;
  for (int gs = 0; gs < total_steps; ++gs) {
    if (gs == trig_gs) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = total_steps - 1 - gs;
    if (rem >= 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    dhf_mbar_wait(&smem.mbar[st], mbar_phase);
    __syncthreads();

    if (gs + kPreload < total_steps) {
      const int next_st = (st == 0) ? (kTma100Stages - 1) : (st - 1);
      issue_w_tma(pf_w, pf_kg2, next_st, true);
      issue_a_async(pf_kg2, next_st);
      adv_pf();
    }

#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      const uint8_t* w_kh = &smem.w[st][kh * kStepWBytes100];
      const uint16_t c0 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 0) * kTileBytes100)[lane];
      const uint16_t c1 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 1) * kTileBytes100)[lane];
      const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][kh][0][0]));
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4 q0 = smem.lut8[(half == 0) ? (c0 & 0xFFu) : (c0 >> 8)];
        const uint4 q1 = smem.lut8[(half == 0) ? (c1 & 0xFFu) : (c1 >> 8)];
        {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 0]);
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.x, q0.y);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.x, q1.y);
        }
        {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 1]);
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.z, q0.w);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.z, q1.w);
        }
      }
    }

    if (++st == kTma100Stages) {
      st = 0;
      mbar_phase ^= 1u;
    }

    if (++cur_kg2 == num_kg2) {
      const int cta_n = bid + cur_w * grid_ctas;
      const int row_base = cta_n * kRowsPerCTA;
      if (tid < kRowsPerCTA) {
        smem.s_toks[tid] = __ldg(&packed_tok_ids[row_base + tid]);
      }
      if (g < M) {
        const int c0 = (warp * 2 + 0) * 8 + 2 * t;
        const int c1 = (warp * 2 + 1) * 8 + 2 * t;
        smem.s_scores[g][c0 + 0] = d0_0 * __ldg(&row_scales[row_base + c0 + 0]);
        smem.s_scores[g][c0 + 1] = d1_0 * __ldg(&row_scales[row_base + c0 + 1]);
        smem.s_scores[g][c1 + 0] = d0_1 * __ldg(&row_scales[row_base + c1 + 0]);
        smem.s_scores[g][c1 + 1] = d1_1 * __ldg(&row_scales[row_base + c1 + 1]);
      }
      __syncthreads();

      for (int m = warp; m < M; m += kWarpsPerCTA) {
        const float thresh = smem.s_topv[m][7];
        const float v0 = smem.s_scores[m][lane];
        const float v1 = smem.s_scores[m][lane + 32];
        uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh);
        uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh);
        if (lane == 0 && ((m0 | m1) != 0u)) {
          float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
          float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
          float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
          float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
          int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
          int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
          int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
          int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
          while (m0 != 0u) {
            const int j = __ffs(m0) - 1;
            m0 &= m0 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          while (m1 != 0u) {
            const int j = 32 + __ffs(m1) - 1;
            m1 &= m1 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
          smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
          smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
          smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
          smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
          smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
          smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
          smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
        }
      }
      __syncthreads();
      d0_0 = 0.0f; d1_0 = 0.0f; d2_0 = 0.0f, d3_0 = 0.0f;
      d0_1 = 0.0f; d1_1 = 0.0f; d2_1 = 0.0f, d3_1 = 0.0f;
      cur_kg2 = 0;
      ++cur_w;
    }
  }

  const int row_stride = grid_ctas * kCtaTopK;
  if (tid < M * kCtaTopK) {
    const int m = tid >> 3;
    const int k = tid & 7;
    cta_vals[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topv[m][k];
    cta_ids[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topi[m][k];
  }
}

constexpr int kTma100PairWStages = 3;
constexpr int kPairBk128StepWBytes100 = 4 * kStepWBytes100;  // 2048 B = 16 * 128 B

struct alignas(128) DhfTma100PairBk128Smem {
  alignas(128) uint8_t w[kTma100PairWStages][kPairBk128StepWBytes100];  // 3 * 2048 = 6144 B
  alignas(128) __nv_bfloat16 a[kTma100PairWStages][2][kMaxM][64];       // 3 * 2048 = 6144 B
  uint4 lut8[256];                                                      // 4096 B
  float s_scores[kMaxM][2 * kRowsPerCTA];                               // 4096 B
  int32_t s_toks[2 * kRowsPerCTA];                                      // 512 B
  float s_topv[kMaxM][kCtaTopK];                                        // 256 B
  int32_t s_topi[kMaxM][kCtaTopK];                                      // 256 B
  alignas(8) uint64_t mbar[kTma100PairWStages];                         // 24 B -> 21,632 B (4 CTAs/SM!)
};

// Paired-wave (128 vocab rows/CTA) + BK=128 2048B Hardware-TMA Stage-1 kernel for mode=100:
// Halves total_steps, barriers, A_scaled cp.async, and ldmatrix.x2 vs dhf_stage1_tma100_bk128_kernel,
// works for ALL V_static (64*k), and is 100.0000% 0-ULP bit-identical!
template <bool kEvictFirst = true>
__global__ void __launch_bounds__(128, 4) dhf_stage1_tma100_pair_bk128_kernel(
    const __nv_bfloat16* __restrict__ A_scaled,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ row_scales,
    const float* __restrict__ col_scales,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ cta_vals,
    int32_t* __restrict__ cta_ids,
    int M, int V_static, int K, int pdl_mode) {
  __shared__ DhfTma100PairBk128Smem smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg2 = K >> 7;
  const int n_blocks = V_static / kRowsPerCTA;
  const int bid = blockIdx.x;
  const int grid_ctas = gridDim.x;
  if (bid >= n_blocks) return;

  {
    const float* cb = col_scales + K + tid * 8;
    const uint4 pos = make_uint4(
        dhf_bf16x2_u32(cb[0], cb[1]),
        dhf_bf16x2_u32(cb[2], cb[3]),
        dhf_bf16x2_u32(cb[4], cb[5]),
        dhf_bf16x2_u32(cb[6], cb[7]));
    smem.lut8[tid] = pos;
    smem.lut8[tid + 128] = make_uint4(
        pos.x ^ 0x80008000u,
        pos.y ^ 0x80008000u,
        pos.z ^ 0x80008000u,
        pos.w ^ 0x80008000u);
  }
  if (tid < kMaxM * kCtaTopK) {
    smem.s_topv[tid >> 3][tid & 7] = -1e38f;
    smem.s_topi[tid >> 3][tid & 7] = -1;
  }
  if (tid == 0) {
    dhf_mbar_init(smem.mbar, kTma100PairWStages);
  }
  __syncthreads();

  if (bid == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg2 * kBk128StepWBytes100;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_kh = tid >> 6;
  const int a_row = (tid >> 3) & 7;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ a_row;
  const bool a_valid = (a_row < M);
  const int a_safe_row = a_valid ? a_row : 0;
  const int a_src_bytes = a_valid ? 16 : 0;

  uint32_t ldm_off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
    ldm_off[s] = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[0][0][lane & 7][ldm_col8 * 8]) -
        __cvta_generic_to_shared(&smem.a[0][0][0][0]));
  }

  const int full_pairs = n_blocks / (2 * kWaveCTAs);
  const int rem_waves = (n_blocks - (full_pairs * 2 * kWaveCTAs) - bid + kWaveCTAs - 1);
  const int tail_waves = (rem_waves > 0) ? (rem_waves / kWaveCTAs) : 0;
  const int total_LogicalWaves = full_pairs + tail_waves;
  const int total_steps = total_LogicalWaves * num_kg2;
  const uint64_t w_pol = kEvictFirst ? dhf_pol_evict_first() : 0ull;

  auto issue_w_tma = [&](int lw, int kg2, int st, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      if (lw < full_pairs) {
        const uint8_t* wp = W_packed +
            (static_cast<size_t>(lw * num_kg2 + kg2) * kWaveCTAs + bid) * kPairBk128StepWBytes100;
        dhf_tma_bulk_w<kEvictFirst>(&smem.w[st][0], wp, kPairBk128StepWBytes100, &smem.mbar[st], w_pol);
      } else {
        const int wave = full_pairs * 2 + (lw - full_pairs);
        const int wave_base = wave * kWaveCTAs;
        const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
        const uint8_t* wp = W_packed +
            (static_cast<size_t>(wave_base) * num_kg2 + static_cast<size_t>(kg2) * w_size + bid) * kBk128StepWBytes100;
        dhf_tma_bulk_w<kEvictFirst>(&smem.w[st][0], wp, kBk128StepWBytes100, &smem.mbar[st], w_pol);
      }
    }
  };

  auto issue_a_async = [&](int kg2, int st) {
    const uint32_t sd = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[st][a_kh][a_row][a_swiz8 * 8]));
    const __nv_bfloat16* gs =
        A_scaled + static_cast<size_t>(a_safe_row) * K + (kg2 * 128 + a_kh * 64) + a_col8 * 8;
    dhf_cp_async16_ca_z(sd, gs, a_src_bytes);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kTma100PairWStages - 1;  // 2
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;
  int pf_lw = 0, pf_kg2 = 0;
  auto adv_pf = [&]() {
    if (++pf_kg2 == num_kg2) {
      pf_kg2 = 0;
      ++pf_lw;
    }
  };

  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(pf_lw, pf_kg2, i, false);
      adv_pf();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int akg2 = 0;
    for (int i = 0; i < preload; ++i) {
      issue_a_async(akg2, i);
      if (++akg2 == num_kg2) akg2 = 0;
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(pf_lw, pf_kg2, i, false);
      issue_a_async(pf_kg2, i);
      adv_pf();
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  float e0_0 = 0.0f, e1_0 = 0.0f, e2_0 = 0.0f, e3_0 = 0.0f;
  float e0_1 = 0.0f, e1_1 = 0.0f, e2_1 = 0.0f, e3_1 = 0.0f;
  const int trig_gs = ((pdl_mode & 2) != 0) ? ((total_steps > 3) ? (total_steps - 3) : 0) : -1;

  int cur_lw = 0;
  int cur_kg2 = 0;
  int st = 0;
  uint32_t mbar_phase = 0u;
  for (int gs = 0; gs < total_steps; ++gs) {
    if (gs == trig_gs) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    dhf_mbar_wait(&smem.mbar[st], mbar_phase);
    const int rem = total_steps - 1 - gs;
    if (rem >= 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    if (gs + kPreload < total_steps) {
      const int next_st = (st == 0) ? (kTma100PairWStages - 1) : (st - 1);
      issue_w_tma(pf_lw, pf_kg2, next_st, true);
      issue_a_async(pf_kg2, next_st);
      adv_pf();
    }

    const bool is_pair = (cur_lw < full_pairs);
    if (is_pair) {
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        const uint8_t* w_kh = &smem.w[st][kh * 1024];
        const uint16_t c0 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 0) * kTileBytes100)[lane];
        const uint16_t c1 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 1) * kTileBytes100)[lane];
        const uint16_t c2 = reinterpret_cast<const uint16_t*>(w_kh + 512 + (warp * 2 + 0) * kTileBytes100)[lane];
        const uint16_t c3 = reinterpret_cast<const uint16_t*>(w_kh + 512 + (warp * 2 + 1) * kTileBytes100)[lane];
        const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][kh][0][0]));
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const uint4 q0 = smem.lut8[(half == 0) ? (c0 & 0xFFu) : (c0 >> 8)];
          const uint4 q1 = smem.lut8[(half == 0) ? (c1 & 0xFFu) : (c1 >> 8)];
          const uint4 q2 = smem.lut8[(half == 0) ? (c2 & 0xFFu) : (c2 >> 8)];
          const uint4 q3 = smem.lut8[(half == 0) ? (c3 & 0xFFu) : (c3 >> 8)];
          {
            uint32_t a0, a2;
            dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 0]);
            dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.x, q0.y);
            dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.x, q1.y);
            dhf_mma_m16n8k16(e0_0, e1_0, e2_0, e3_0, a0, a2, q2.x, q2.y);
            dhf_mma_m16n8k16(e0_1, e1_1, e2_1, e3_1, a0, a2, q3.x, q3.y);
          }
          {
            uint32_t a0, a2;
            dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 1]);
            dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.z, q0.w);
            dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.z, q1.w);
            dhf_mma_m16n8k16(e0_0, e1_0, e2_0, e3_0, a0, a2, q2.z, q2.w);
            dhf_mma_m16n8k16(e0_1, e1_1, e2_1, e3_1, a0, a2, q3.z, q3.w);
          }
        }
      }
    } else {
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        const uint8_t* w_kh = &smem.w[st][kh * kStepWBytes100];
        const uint16_t c0 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 0) * kTileBytes100)[lane];
        const uint16_t c1 = reinterpret_cast<const uint16_t*>(w_kh + (warp * 2 + 1) * kTileBytes100)[lane];
        const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][kh][0][0]));
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const uint4 q0 = smem.lut8[(half == 0) ? (c0 & 0xFFu) : (c0 >> 8)];
          const uint4 q1 = smem.lut8[(half == 0) ? (c1 & 0xFFu) : (c1 >> 8)];
          {
            uint32_t a0, a2;
            dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 0]);
            dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.x, q0.y);
            dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.x, q1.y);
          }
          {
            uint32_t a0, a2;
            dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 1]);
            dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.z, q0.w);
            dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.z, q1.w);
          }
        }
      }
    }

    if (++st == kTma100PairWStages) {
      st = 0;
      mbar_phase ^= 1u;
    }

    if (++cur_kg2 == num_kg2) {
      if (is_pair) {
        const int cta_n0 = bid + (2 * cur_lw) * kWaveCTAs;
        const int row_base0 = cta_n0 * kRowsPerCTA;
        const int row_base1 = (cta_n0 + kWaveCTAs) * kRowsPerCTA;
        smem.s_toks[tid] = __ldg(&packed_tok_ids[(tid < 64) ? (row_base0 + tid) : (row_base1 + tid - 64)]);
        if (g < M) {
          const int col0 = (warp * 2 + 0) * 8 + 2 * t;
          const int col1 = (warp * 2 + 1) * 8 + 2 * t;
          smem.s_scores[g][col0 + 0] = d0_0 * __ldg(&row_scales[row_base0 + col0 + 0]);
          smem.s_scores[g][col0 + 1] = d1_0 * __ldg(&row_scales[row_base0 + col0 + 1]);
          smem.s_scores[g][col1 + 0] = d0_1 * __ldg(&row_scales[row_base0 + col1 + 0]);
          smem.s_scores[g][col1 + 1] = d1_1 * __ldg(&row_scales[row_base0 + col1 + 1]);
          smem.s_scores[g][64 + col0 + 0] = e0_0 * __ldg(&row_scales[row_base1 + col0 + 0]);
          smem.s_scores[g][64 + col0 + 1] = e1_0 * __ldg(&row_scales[row_base1 + col0 + 1]);
          smem.s_scores[g][64 + col1 + 0] = e0_1 * __ldg(&row_scales[row_base1 + col1 + 0]);
          smem.s_scores[g][64 + col1 + 1] = e1_1 * __ldg(&row_scales[row_base1 + col1 + 1]);
        }
        __syncthreads();

        for (int m = warp; m < M; m += kWarpsPerCTA) {
          const float thresh0 = smem.s_topv[m][7];
          const float v0 = smem.s_scores[m][lane];
          const float v1 = smem.s_scores[m][lane + 32];
          const float v2 = smem.s_scores[m][lane + 64];
          const float v3 = smem.s_scores[m][lane + 96];
          uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh0);
          uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh0);
          uint32_t m2 = __ballot_sync(0xffffffffu, v2 >= thresh0);
          uint32_t m3 = __ballot_sync(0xffffffffu, v3 >= thresh0);
          if (lane == 0 && ((m0 | m1 | m2 | m3) != 0u)) {
            float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
            float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
            float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
            float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
            int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
            int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
            int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
            int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
            while (m0 != 0u) {
              const int j = __ffs(m0) - 1;
              m0 &= m0 - 1u;
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
            while (m1 != 0u) {
              const int j = 32 + __ffs(m1) - 1;
              m1 &= m1 - 1u;
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
            while (m2 != 0u) {
              const int j = 64 + __ffs(m2) - 1;
              m2 &= m2 - 1u;
              if (smem.s_scores[m][j] >= tv8) {
                dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                                tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                                smem.s_scores[m][j], smem.s_toks[j]);
              }
            }
            while (m3 != 0u) {
              const int j = 96 + __ffs(m3) - 1;
              m3 &= m3 - 1u;
              if (smem.s_scores[m][j] >= tv8) {
                dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                                tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                                smem.s_scores[m][j], smem.s_toks[j]);
              }
            }
            smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
            smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
            smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
            smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
            smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
            smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
            smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
            smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
          }
        }
      } else {
        const int wave = full_pairs * 2 + (cur_lw - full_pairs);
        const int cta_n = bid + wave * kWaveCTAs;
        const int row_base = cta_n * kRowsPerCTA;
        if (tid < kRowsPerCTA) {
          smem.s_toks[tid] = __ldg(&packed_tok_ids[row_base + tid]);
        }
        if (g < M) {
          const int c0 = (warp * 2 + 0) * 8 + 2 * t;
          const int c1 = (warp * 2 + 1) * 8 + 2 * t;
          smem.s_scores[g][c0 + 0] = d0_0 * __ldg(&row_scales[row_base + c0 + 0]);
          smem.s_scores[g][c0 + 1] = d1_0 * __ldg(&row_scales[row_base + c0 + 1]);
          smem.s_scores[g][c1 + 0] = d0_1 * __ldg(&row_scales[row_base + c1 + 0]);
          smem.s_scores[g][c1 + 1] = d1_1 * __ldg(&row_scales[row_base + c1 + 1]);
        }
        __syncthreads();

        for (int m = warp; m < M; m += kWarpsPerCTA) {
          const float thresh = smem.s_topv[m][7];
          const float v0 = smem.s_scores[m][lane];
          const float v1 = smem.s_scores[m][lane + 32];
          uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh);
          uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh);
          if (lane == 0 && ((m0 | m1) != 0u)) {
            float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
            float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
            float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
            float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
            int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
            int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
            int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
            int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
            while (m0 != 0u) {
              const int j = __ffs(m0) - 1;
              m0 &= m0 - 1u;
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
            while (m1 != 0u) {
              const int j = 32 + __ffs(m1) - 1;
              m1 &= m1 - 1u;
              dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                              tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                              smem.s_scores[m][j], smem.s_toks[j]);
            }
            smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
            smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
            smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
            smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
            smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
            smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
            smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
            smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
          }
        }
      }
      __syncthreads();
      d0_0 = 0.0f; d1_0 = 0.0f; d2_0 = 0.0f; d3_0 = 0.0f;
      d0_1 = 0.0f; d1_1 = 0.0f; d2_1 = 0.0f; d3_1 = 0.0f;
      e0_0 = 0.0f; e1_0 = 0.0f; e2_0 = 0.0f; e3_0 = 0.0f;
      e0_1 = 0.0f; e1_1 = 0.0f; e2_1 = 0.0f; e3_1 = 0.0f;
      cur_kg2 = 0;
      ++cur_lw;
    }
  }

  const int row_stride = grid_ctas * kCtaTopK;
  if (tid < M * kCtaTopK) {
    const int m = tid >> 3;
    const int k = tid & 7;
    cta_vals[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topv[m][k];
    cta_ids[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topi[m][k];
  }
}

// Kernel 1: Persistent 192-CTA fused Stage-1 compressed MMA + in-CTA top-8 shortlist extraction
// with continuous 8-stage cross-wave cp.async.cg.L2::256B + L2::evict_first pipeline (zero wave drains).
template <int kMode, bool kEvictFirst = true>
__global__ void __launch_bounds__(128, 4) dhf_stage1_kernel(
    const __nv_bfloat16* __restrict__ A_scaled,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ row_scales,
    const float* __restrict__ col_scales,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ cta_vals,
    int32_t* __restrict__ cta_ids,
    int M, int V_static, int K, int pdl_mode) {
  constexpr int kTileBytes = (kMode == 100) ? kTileBytes100 : ((kMode == 150) ? kTileBytes150 : ((kMode == 158) ? kTileBytes158 : kTileBytes2));
  constexpr int kStepWBytes = (kMode == 100) ? kStepWBytes100 : ((kMode == 150) ? kStepWBytes150 : ((kMode == 158) ? kStepWBytes158 : kStepWBytes2));
  constexpr int kStepWU4 = (kMode == 100) ? kStepWU4_100 : ((kMode == 150) ? kStepWU4_150 : ((kMode == 158) ? kStepWU4_158 : kStepWU4_2));
  __shared__ DhfStageSmem<kStepWBytes, kMode> smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int n_blocks = V_static / kRowsPerCTA;
  const int bid = blockIdx.x;
  const int grid_ctas = gridDim.x;
  if (bid >= n_blocks) return;

  // Populate SMEM BF16x2 LUT (mode 100: 256-entry 8D antipodal VQ; mode 150: 64-entry 4D VQ; mode 158: 81-entry 4-trit).
  if constexpr (kMode == 100) {
    if (tid < 128) {
      const float* cb = col_scales + K + tid * 8;
      const uint4 pos = make_uint4(
          dhf_bf16x2_u32(cb[0], cb[1]),
          dhf_bf16x2_u32(cb[2], cb[3]),
          dhf_bf16x2_u32(cb[4], cb[5]),
          dhf_bf16x2_u32(cb[6], cb[7]));
      smem.u.lut8[tid] = pos;
      smem.u.lut8[tid + 128] = make_uint4(
          pos.x ^ 0x80008000u,
          pos.y ^ 0x80008000u,
          pos.z ^ 0x80008000u,
          pos.w ^ 0x80008000u);
    }
  } else if constexpr (kMode == 150) {
    if (tid < 64) {
      smem.u.lut[tid] = make_uint2(
          dhf_bf16x2_u32(kVq4D64[tid][0], kVq4D64[tid][1]),
          dhf_bf16x2_u32(kVq4D64[tid][2], kVq4D64[tid][3]));
    }
  } else if constexpr (kMode == 158) {
    if (tid < 81) {
      const int c = tid;
      const int t0 = c % 3;
      const int t1 = (c / 3) % 3;
      const int t2 = (c / 9) % 3;
      const int t3 = c / 27;
      smem.u.lut[c] = make_uint2(
          dhf_bf16x2_u32(static_cast<float>(t0 - 1), static_cast<float>(t1 - 1)),
          dhf_bf16x2_u32(static_cast<float>(t2 - 1), static_cast<float>(t3 - 1)));
    }
  }
  if (tid < kMaxM * kCtaTopK) {
    smem.s_topv[tid >> 3][tid & 7] = -1e38f;
    smem.s_topi[tid >> 3][tid & 7] = -1;
  }

  // 2 MiB TLB prefetch from block 0 (immutable W_packed: safe before PDL wait).
  if (bid == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg * kStepWBytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  // Thread mapping for single-pass cp.async (tid 0..63 -> A, tid 64..64+kStepWU4-1 -> W).
  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (tid < 64) && (a_row < M);
  const int w_u4_idx = tid - 64;
  const bool w_loader = (tid >= 64) && (w_u4_idx < kStepWU4);

  // Precompute bit-unpack indices for mode 150 (24-bit) and mode 158 (26-bit).
  const int grp150 = lane >> 2;
  const int sub150 = lane & 3;
  const int w_idx150 = grp150 * 3 + ((sub150 * 3) >> 2);
  const int b_off150 = (sub150 * 24) & 31;
  const bool need_hi150 = (b_off150 > 8);

  const int bit158 = lane * 26;
  const int w_idx158 = bit158 >> 5;
  const int b_off158 = bit158 & 31;
  const bool need_hi158 = (b_off158 > 6);

  // Precompute ldmatrix SMEM offsets for the 4 k16 steps.
  uint32_t ldm_off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
    ldm_off[s] = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[0][lane & 7][ldm_col8 * 8]) -
        __cvta_generic_to_shared(&smem.a[0][0][0]));
  }

  const int num_waves = (n_blocks - bid + grid_ctas - 1) / grid_ctas;
  const int total_steps = num_waves * num_kg;

  auto w_ptr_for = [&](int w_iter, int kg) -> const uint8_t* {
    const int cta_n = bid + w_iter * grid_ctas;
    const int wave = cta_n / kWaveCTAs;
    const int slot = cta_n - wave * kWaveCTAs;
    const int wave_base = wave * kWaveCTAs;
    const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
    return W_packed +
           (static_cast<size_t>(wave_base) * num_kg + static_cast<size_t>(kg) * w_size + slot) * kStepWBytes;
  };

  const uint64_t w_pol = kEvictFirst ? dhf_pol_evict_first() : 0ull;

  auto issue_w_only = [&](int w_iter, int kg, int st) {
    if (w_loader) {
      const uint8_t* w_src = w_ptr_for(w_iter, kg);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
      dhf_cp_async16_cg<kEvictFirst>(sd, w_src + w_u4_idx * 16, w_pol);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int w_iter, int kg, int st) {
    if (w_loader) {
      const uint8_t* w_src = w_ptr_for(w_iter, kg);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
      dhf_cp_async16_cg<kEvictFirst>(sd, w_src + w_u4_idx * 16, w_pol);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(
          __cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs =
          a_valid ? (A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A_scaled;
      dhf_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 7
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;
  int pf_w = 0, pf_kg = 0;
  auto adv_pf = [&]() {
    if (++pf_kg == num_kg) {
      pf_kg = 0;
      ++pf_w;
    }
  };

  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(pf_w, pf_kg, i);
      adv_pf();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int akg = 0;
    for (int i = 0; i < preload; ++i) {
      load_a_sync(akg, i);
      if (++akg == num_kg) akg = 0;
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_stage(pf_w, pf_kg, i);
      adv_pf();
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const int trig_gs = ((pdl_mode & 2) != 0) ? ((total_steps > 6) ? (total_steps - 6) : 0) : -1;

  int cur_w = 0;
  int cur_kg = 0;
  int st = 0;
  for (int gs = 0; gs < total_steps; ++gs) {
    if (gs == trig_gs) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = total_steps - 1 - gs;
    if (rem >= 6) {
      asm volatile("cp.async.wait_group 6;\n" ::: "memory");
    } else if (rem == 5) {
      asm volatile("cp.async.wait_group 5;\n" ::: "memory");
    } else if (rem == 4) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    if (gs + kPreload < total_steps) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(pf_w, pf_kg, next_st);
      adv_pf();
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][0][0]));

    if constexpr (kMode == 100) {
      const uint16_t c0 = reinterpret_cast<const uint16_t*>(wt0)[lane];
      const uint16_t c1 = reinterpret_cast<const uint16_t*>(wt1)[lane];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4 q0 = smem.u.lut8[(half == 0) ? (c0 & 0xFFu) : (c0 >> 8)];
        const uint4 q1 = smem.u.lut8[(half == 0) ? (c1 & 0xFFu) : (c1 >> 8)];
        {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 0]);
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.x, q0.y);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.x, q1.y);
        }
        {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[2 * half + 1]);
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, q0.z, q0.w);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, q1.z, q1.w);
        }
      }
    } else if constexpr (kMode == 150) {
      const uint32_t* u0 = reinterpret_cast<const uint32_t*>(wt0);
      const uint32_t* u1 = reinterpret_cast<const uint32_t*>(wt1);
      const uint32_t c24_0 = __funnelshift_r(u0[w_idx150], need_hi150 ? u0[w_idx150 + 1] : 0u, b_off150) & 0x00FFFFFFu;
      const uint32_t c24_1 = __funnelshift_r(u1[w_idx150], need_hi150 ? u1[w_idx150 + 1] : 0u, b_off150) & 0x00FFFFFFu;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t a0, a2;
        dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[s]);
        const uint2 b = smem.u.lut[(c24_0 >> (6 * s)) & 63u];
        const uint2 c = smem.u.lut[(c24_1 >> (6 * s)) & 63u];
        dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b.x, b.y);
        dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c.x, c.y);
      }
    } else if constexpr (kMode == 158) {
      const uint32_t* u0 = reinterpret_cast<const uint32_t*>(wt0);
      const uint32_t* u1 = reinterpret_cast<const uint32_t*>(wt1);
      const uint32_t c26_0 = __funnelshift_r(u0[w_idx158], need_hi158 ? u0[w_idx158 + 1] : 0u, b_off158) & 0x03FFFFFFu;
      const uint32_t c26_1 = __funnelshift_r(u1[w_idx158], need_hi158 ? u1[w_idx158 + 1] : 0u, b_off158) & 0x03FFFFFFu;

      const uint32_t lo0 = c26_0 & 0x1FFFu;
      const uint32_t hi0 = c26_0 >> 13;
      const uint32_t q0_1 = __umulhi(lo0, 0x03291620u);
      const uint32_t q0_0 = lo0 - q0_1 * 81u;
      const uint32_t q0_3 = __umulhi(hi0, 0x03291620u);
      const uint32_t q0_2 = hi0 - q0_3 * 81u;

      const uint32_t lo1 = c26_1 & 0x1FFFu;
      const uint32_t hi1 = c26_1 >> 13;
      const uint32_t q1_1 = __umulhi(lo1, 0x03291620u);
      const uint32_t q1_0 = lo1 - q1_1 * 81u;
      const uint32_t q1_3 = __umulhi(hi1, 0x03291620u);
      const uint32_t q1_2 = hi1 - q1_3 * 81u;

      const uint32_t idx0[4] = {q0_0, q0_1, q0_2, q0_3};
      const uint32_t idx1[4] = {q1_0, q1_1, q1_2, q1_3};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t a0, a2;
        dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[s]);
        const uint2 b = smem.u.lut[idx0[s]];
        const uint2 c = smem.u.lut[idx1[s]];
        dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b.x, b.y);
        dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c.x, c.y);
      }
    } else {
      const uint32_t q32_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
      const uint32_t q32_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t a0, a2;
        dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[s]);
        uint32_t b0, b1, c0, c1;
        if (s == 0) {
          dhf_deq2_k16<0>(q32_0, b0, b1); dhf_deq2_k16<0>(q32_1, c0, c1);
        } else if (s == 1) {
          dhf_deq2_k16<1>(q32_0, b0, b1); dhf_deq2_k16<1>(q32_1, c0, c1);
        } else if (s == 2) {
          dhf_deq2_k16<2>(q32_0, b0, b1); dhf_deq2_k16<2>(q32_1, c0, c1);
        } else {
          dhf_deq2_k16<3>(q32_0, b0, b1); dhf_deq2_k16<3>(q32_1, c0, c1);
        }
        dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);

    if (++cur_kg == num_kg) {
      // Wave boundary epilogue (runs while next wave's 7 cp.async stages stream in!).
      const int cta_n = bid + cur_w * grid_ctas;
      const int row_base = cta_n * kRowsPerCTA;
      if (tid < kRowsPerCTA) {
        smem.s_toks[tid] = __ldg(&packed_tok_ids[row_base + tid]);
      }
      if (g < M) {
        const int c0 = (warp * 2 + 0) * 8 + 2 * t;
        const int c1 = (warp * 2 + 1) * 8 + 2 * t;
        smem.s_scores[g][c0 + 0] = d0_0 * __ldg(&row_scales[row_base + c0 + 0]);
        smem.s_scores[g][c0 + 1] = d1_0 * __ldg(&row_scales[row_base + c0 + 1]);
        smem.s_scores[g][c1 + 0] = d0_1 * __ldg(&row_scales[row_base + c1 + 0]);
        smem.s_scores[g][c1 + 1] = d1_1 * __ldg(&row_scales[row_base + c1 + 1]);
      }
      __syncthreads();

      for (int m = warp; m < M; m += kWarpsPerCTA) {
        const float thresh = smem.s_topv[m][7];
        const float v0 = smem.s_scores[m][lane];
        const float v1 = smem.s_scores[m][lane + 32];
        uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh);
        uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh);
        if (lane == 0 && ((m0 | m1) != 0u)) {
          float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
          float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
          float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
          float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
          int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
          int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
          int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
          int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
          while (m0 != 0u) {
            const int j = __ffs(m0) - 1;
            m0 &= m0 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          while (m1 != 0u) {
            const int j = 32 + __ffs(m1) - 1;
            m1 &= m1 - 1u;
            dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                            tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                            smem.s_scores[m][j], smem.s_toks[j]);
          }
          smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
          smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
          smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
          smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
          smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
          smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
          smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
          smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
        }
      }
      __syncthreads();
      d0_0 = 0.0f; d1_0 = 0.0f; d2_0 = 0.0f; d3_0 = 0.0f;
      d0_1 = 0.0f; d1_1 = 0.0f; d2_1 = 0.0f; d3_1 = 0.0f;
      cur_kg = 0;
      ++cur_w;
    }
  }

  // Emit per-CTA top-8 to L2-resident scratch.
  const int row_stride = grid_ctas * kCtaTopK;
  if (tid < M * kCtaTopK) {
    const int m = tid >> 3;
    const int k = tid & 7;
    cta_vals[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topv[m][k];
    cta_ids[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topi[m][k];
  }
}

__device__ __forceinline__ uint32_t dhf_f32_to_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

// Kernel 2: Exact 3-pass 8-bit MSB radix select from N_in (<=1536) CTA shortlist -> K_cand (<=256).
__device__ __forceinline__ void dhf_radix_scan_warp(
    const volatile uint32_t* hist_perm, uint32_t pref, uint32_t shift,
    uint32_t* s_prefix_p, uint32_t* s_rem_k_p) {
  const int lane = threadIdx.x & 31;
  const uint32_t need = *s_rem_k_p;
  const int g = 31 - lane;
  uint32_t c;
  const uint32_t c0 = hist_perm[g];
  const uint32_t c1 = hist_perm[g + 32];
  const uint32_t c2 = hist_perm[g + 64];
  const uint32_t c3 = hist_perm[g + 96];
  const uint32_t c4 = hist_perm[g + 128];
  const uint32_t c5 = hist_perm[g + 160];
  const uint32_t c6 = hist_perm[g + 192];
  const uint32_t c7 = hist_perm[g + 224];
  c = c0 + c1; c = c + c2; c = c + c3; c = c + c4; c = c + c5; c = c + c6; c = c + c7;
  uint32_t incl = c;
  incl += __shfl_up_sync(0xffffffffu, incl, 1);
  incl += __shfl_up_sync(0xffffffffu, incl, 2);
  incl += __shfl_up_sync(0xffffffffu, incl, 4);
  incl += __shfl_up_sync(0xffffffffu, incl, 8);
  incl += __shfl_up_sync(0xffffffffu, incl, 16);
  const uint32_t excl = incl - c;
  const bool is_cross = (incl >= need) && ((excl < need) || (need == 0u && lane == 0));
  const uint32_t cm = __ballot_sync(0xffffffffu, is_cross);
  uint32_t chosen = 0u;
  uint32_t rem_out = need;
  bool rem_written = false;
  if (cm != 0u) {
    if (is_cross) {
      uint32_t acc = excl;
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t cc = hist_perm[g + (j << 5)];
        if (!rem_written && acc + cc >= need) {
          chosen = static_cast<uint32_t>(8 * g + (7 - j));
          rem_out = need - acc;
          rem_written = true;
        }
        acc += cc;
      }
    }
    const int cl = __ffs(cm) - 1;
    chosen = __shfl_sync(0xffffffffu, chosen, cl);
    rem_out = __shfl_sync(0xffffffffu, rem_out, cl);
    rem_written = static_cast<bool>(__shfl_sync(0xffffffffu, rem_written ? 1u : 0u, cl));
    if (lane == 0) {
      *s_prefix_p = pref | (chosen << shift);
      if (rem_written) {
        *s_rem_k_p = rem_out;
      }
    }
  } else {
    if (lane == 0) {
      *s_prefix_p = pref;
    }
  }
}

// Grid: M blocks, Block: 256 threads (~3 us, L2-resident).
__global__ void dhf_select_topk_kernel(
    const float* __restrict__ cta_vals,
    const int32_t* __restrict__ cta_ids,
    int32_t* __restrict__ cand_ids,
    int N_in, int K_cand, int pdl_mode) {
  const int m = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ uint32_t s_keys[kWaveCTAs * kCtaTopK];
  __shared__ int32_t s_ids[kWaveCTAs * kCtaTopK];
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_rem_k;
  __shared__ uint32_t s_out_cnt;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  const float* v_in = cta_vals + static_cast<size_t>(m) * N_in;
  const int32_t* i_in = cta_ids + static_cast<size_t>(m) * N_in;
  int32_t* out = cand_ids + static_cast<size_t>(m) * K_cand;

  for (int i = tid; i < N_in; i += 256) {
    float v = v_in[i];
    int32_t id = i_in[i];
    if (isnan(v) || id < 0) {
      v = -1e38f;
      id = -1;
    }
    s_keys[i] = dhf_f32_to_key(v);
    s_ids[i] = id;
  }
  if (tid == 0) {
    s_prefix = 0u;
    s_rem_k = static_cast<uint32_t>((K_cand < N_in) ? K_cand : N_in);
    s_out_cnt = 0u;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const uint32_t shift = static_cast<uint32_t>(24 - pass * 8);
    const uint32_t mask = (pass == 0) ? 0u : (~0u << (32 - pass * 8));
    const uint32_t pref = s_prefix;
    s_hist[tid] = 0u;
    __syncthreads();

    for (int i = tid; i < N_in; i += 256) {
      const uint32_t k = s_keys[i];
      if ((k & mask) == pref) {
        const uint32_t bkt = (k >> shift) & 0xFFu;
        atomicAdd(&s_hist[(bkt >> 3) + ((bkt & 7u) << 5)], 1u);
      }
    }
    __syncthreads();

    if (tid < 32) {
      dhf_radix_scan_warp(s_hist, pref, shift, &s_prefix, &s_rem_k);
    }
    __syncthreads();
  }

  const uint32_t thresh = s_prefix;  // exact 32-bit float key threshold
  // Deterministically compact strictly-greater items, then exact tie-bucket items on warp 0.
  if (tid < 32) {
    uint32_t cur = 0u;
    const uint32_t cap = static_cast<uint32_t>(K_cand);
    for (int base = 0; base < N_in && cur < cap; base += 32) {
      const int i = base + tid;
      const bool hit = (i < N_in) && (s_keys[i] > thresh) && (s_ids[i] >= 0);
      const uint32_t bmask = __ballot_sync(0xffffffffu, hit);
      if (bmask != 0u) {
        const uint32_t slot = cur + static_cast<uint32_t>(__popc(bmask & ((1u << tid) - 1u)));
        if (hit && slot < cap) {
          out[slot] = s_ids[i];
        }
        cur += static_cast<uint32_t>(__popc(bmask));
      }
    }
    for (int base = 0; base < N_in && cur < cap; base += 32) {
      const int i = base + tid;
      const bool hit = (i < N_in) && (s_keys[i] == thresh) && (s_ids[i] >= 0);
      const uint32_t bmask = __ballot_sync(0xffffffffu, hit);
      if (bmask != 0u) {
        const uint32_t slot = cur + static_cast<uint32_t>(__popc(bmask & ((1u << tid) - 1u)));
        if (hit && slot < cap) {
          out[slot] = s_ids[i];
        }
        cur += static_cast<uint32_t>(__popc(bmask));
      }
    }
    if (tid == 0) {
      s_out_cnt = (cur < cap) ? cur : cap;
    }
  }
  __syncthreads();

  const uint32_t final_cnt = s_out_cnt;
  for (int i = static_cast<int>(final_cnt) + tid; i < K_cand; i += 256) {
    out[i] = -1;
  }
  if ((pdl_mode & 2) != 0 && tid == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

// Exact warp-level NVFP4 dot product (bit-identical to draft_vocab_tail.cu's dvt_warp_nvfp4_dot).
__device__ __forceinline__ float dhf_nvfp4_group_dot(
    uint32_t wlo, uint32_t whi, const uint4& a0, const uint4& a1) {
  const __nv_bfloat162* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const __nv_bfloat162* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float gsum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap0[i]);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32(wlo & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu), af.y, gsum);
    wlo >>= 8;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap1[i]);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32(whi & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu), af.y, gsum);
    whi >>= 8;
  }
  return gsum;
}

__device__ __forceinline__ float dhf_warp_nvfp4_dot(
    const __nv_bfloat16* __restrict__ a_row,
    const uint4* __restrict__ s_a0,
    const uint4* __restrict__ s_a1,
    bool use_smem_a,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    int v, int K, int num_g, int lane, float total_scale) {
  const uint8_t* w_row = raw_w + static_cast<size_t>(v) * (K >> 1);
  const uint8_t* s_row = raw_s + static_cast<size_t>(v) * num_g;
  float acc = 0.0f;
  if (((num_g & 1) == 0) && ((reinterpret_cast<uintptr_t>(w_row) & 15u) == 0u) &&
      ((reinterpret_cast<uintptr_t>(s_row) & 1u) == 0u)) {
    const int num_g2 = num_g >> 1;
    const uint4* w16_row = reinterpret_cast<const uint4*>(w_row);
    const uint16_t* s2_row = reinterpret_cast<const uint16_t*>(s_row);
    if (num_g2 == 208 && use_smem_a) {
#pragma unroll 2
      for (int g2 = lane; g2 < 208; g2 += 32) {
        const uint16_t s2 = __ldg(&s2_row[g2]);
        const uint4 w16 = __ldg(&w16_row[g2]);
        const int g0 = g2 << 1;
        const int g1 = g0 | 1;
        const float sc0 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 & 0xFFu)) * 0.5f;
        const float sc1 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 >> 8)) * 0.5f;
        acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.x, w16.y, s_a0[g0], s_a1[g0]), sc0, acc);
        acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.z, w16.w, s_a0[g1], s_a1[g1]), sc1, acc);
      }
    } else {
      for (int g2 = lane; g2 < num_g2; g2 += 32) {
        const uint16_t s2 = __ldg(&s2_row[g2]);
        const uint4 w16 = __ldg(&w16_row[g2]);
        const int g0 = g2 << 1;
        const int g1 = g0 | 1;
        const float sc0 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 & 0xFFu)) * 0.5f;
        const float sc1 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 >> 8)) * 0.5f;
        const uint4 a00 = use_smem_a ? s_a0[g0] : *(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 0);
        const uint4 a01 = use_smem_a ? s_a1[g0] : *(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 1);
        const uint4 a10 = use_smem_a ? s_a0[g1] : *(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 0);
        const uint4 a11 = use_smem_a ? s_a1[g1] : *(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 1);
        acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.x, w16.y, a00, a01), sc0, acc);
        acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.z, w16.w, a10, a11), sc1, acc);
      }
    }
  } else {
    for (int g = lane; g < num_g; g += 32) {
      const float sc_half = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * 0.5f;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      const uint4 a0 = use_smem_a ? s_a0[g] : *(reinterpret_cast<const uint4*>(a_row) + g * 2 + 0);
      const uint4 a1 = use_smem_a ? s_a1[g] : *(reinterpret_cast<const uint4*>(a_row) + g * 2 + 1);
      acc = __fmaf_rn(
          dhf_nvfp4_group_dot(static_cast<uint32_t>(w8), static_cast<uint32_t>(w8 >> 32), a0, a1),
          sc_half, acc);
    }
  }
  acc = dhf_warp_sum(acc);
  return __bfloat162float(__float2bfloat16_rn(acc * total_scale));
}

// Kernel 3: Fused 1-wave Stage-2 exact NVFP4 rescore + deterministic last-CTA final top-8 merge.
// Grid: (8, M) = 48 CTAs at M=6 (exact 1 CTA/SM on GB10), Block: 256 threads (8 warps).
__global__ void __launch_bounds__(256, 2) dhf_stage2_rescore_kernel(
    const __nv_bfloat16* __restrict__ A,
    const int32_t* __restrict__ cand_ids,
    const int32_t* __restrict__ tail_ids,
    const int32_t* __restrict__ tail_count,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    float* __restrict__ s2_vals,
    int32_t* __restrict__ s2_ids,
    int32_t* __restrict__ s2_done,
    int64_t* __restrict__ out,
    int64_t* __restrict__ out_top2_row0,
    int top2_rows,
    float* __restrict__ out_margins,
    int margin_rows,
    int64_t* __restrict__ out_tree_top,
    float* __restrict__ out_tree_vals,
    int tree_top_rows,
    int tree_k,
    int K_cand, int K, int max_tail, float logit_scale, int pdl_mode) {
  const int bid = blockIdx.x;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int num_g = K >> 4;
  const bool use_smem_a = (num_g <= kS2MaxGroups);

  __shared__ uint4 s_a0[kS2MaxGroups];
  __shared__ uint4 s_a1[kS2MaxGroups];
  __shared__ float wv[kS2Warps][8];
  __shared__ int32_t wi[kS2Warps][8];
  __shared__ int s_is_last;

  const __nv_bfloat16* a_row = A + static_cast<size_t>(m) * K;
  if (use_smem_a) {
    const uint4* a_u4 = reinterpret_cast<const uint4*>(a_row);
    for (int g = tid; g < num_g; g += 256) {
      s_a0[g] = a_u4[g * 2 + 0];
      s_a1[g] = a_u4[g * 2 + 1];
    }
    __syncthreads();
  }

  float v1 = -1e38f, v2 = -1e38f, v3 = -1e38f, v4 = -1e38f;
  float v5 = -1e38f, v6 = -1e38f, v7 = -1e38f, v8 = -1e38f;
  int i1 = 0x7fffffff, i2 = 0x7fffffff, i3 = 0x7fffffff, i4 = 0x7fffffff;
  int i5 = 0x7fffffff, i6 = 0x7fffffff, i7 = 0x7fffffff, i8 = 0x7fffffff;

  const float total_scale = __ldg(global_scale) * logit_scale;
  const int stride_w = kS2CTAsPerRow * kS2Warps;  // 64

  // Phase A (before PDL wait): score dynamic prompt-tail OOV rows (already ready in memory!).
  int tcnt = tail_count[0];
  if (tcnt < 0) tcnt = 0;
  if (tcnt > max_tail) tcnt = max_tail;
  for (int idx = bid * kS2Warps + warp; idx < tcnt; idx += stride_w) {
    const int tok = tail_ids[idx];
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  // Wait for Kernel 2 (cand_ids) via SM121 PDL.
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  // Phase B: score static shortlist candidates against exact raw NVFP4 weights.
  const int32_t* my_cands = cand_ids + static_cast<size_t>(m) * K_cand;
  for (int idx = bid * kS2Warps + warp; idx < K_cand; idx += stride_w) {
    const int tok = my_cands[idx];
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  if (lane == 0) {
    wv[warp][0] = v1; wi[warp][0] = i1;
    wv[warp][1] = v2; wi[warp][1] = i2;
    wv[warp][2] = v3; wi[warp][2] = i3;
    wv[warp][3] = v4; wi[warp][3] = i4;
    wv[warp][4] = v5; wi[warp][4] = i5;
    wv[warp][5] = v6; wi[warp][5] = i6;
    wv[warp][6] = v7; wi[warp][6] = i7;
    wv[warp][7] = v8; wi[warp][7] = i8;
  }
  __syncthreads();

  if (tid < 32) {
    float bv1 = (tid < kS2Warps) ? wv[tid][0] : -1e38f;
    float bv2 = (tid < kS2Warps) ? wv[tid][1] : -1e38f;
    float bv3 = (tid < kS2Warps) ? wv[tid][2] : -1e38f;
    float bv4 = (tid < kS2Warps) ? wv[tid][3] : -1e38f;
    float bv5 = (tid < kS2Warps) ? wv[tid][4] : -1e38f;
    float bv6 = (tid < kS2Warps) ? wv[tid][5] : -1e38f;
    float bv7 = (tid < kS2Warps) ? wv[tid][6] : -1e38f;
    float bv8 = (tid < kS2Warps) ? wv[tid][7] : -1e38f;
    int bi1 = (tid < kS2Warps) ? wi[tid][0] : 0x7fffffff;
    int bi2 = (tid < kS2Warps) ? wi[tid][1] : 0x7fffffff;
    int bi3 = (tid < kS2Warps) ? wi[tid][2] : 0x7fffffff;
    int bi4 = (tid < kS2Warps) ? wi[tid][3] : 0x7fffffff;
    int bi5 = (tid < kS2Warps) ? wi[tid][4] : 0x7fffffff;
    int bi6 = (tid < kS2Warps) ? wi[tid][5] : 0x7fffffff;
    int bi7 = (tid < kS2Warps) ? wi[tid][6] : 0x7fffffff;
    int bi8 = (tid < kS2Warps) ? wi[tid][7] : 0x7fffffff;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      const float ov1 = __shfl_down_sync(0xffffffffu, bv1, o);
      const int oi1 = __shfl_down_sync(0xffffffffu, bi1, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, bv2, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, bi2, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, bv3, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, bi3, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, bv4, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, bi4, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, bv5, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, bi5, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, bv6, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, bi6, o);
      const float ov7 = __shfl_down_sync(0xffffffffu, bv7, o);
      const int oi7 = __shfl_down_sync(0xffffffffu, bi7, o);
      const float ov8 = __shfl_down_sync(0xffffffffu, bv8, o);
      const int oi8 = __shfl_down_sync(0xffffffffu, bi8, o);
      dhf_top8_merge(
          bv1, bi1, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
          ov1, oi1, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
    }
    if (tid == 0) {
      const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + bid) * 8;
      s2_vals[base + 0] = bv1; s2_ids[base + 0] = bi1;
      s2_vals[base + 1] = bv2; s2_ids[base + 1] = bi2;
      s2_vals[base + 2] = bv3; s2_ids[base + 2] = bi3;
      s2_vals[base + 3] = bv4; s2_ids[base + 3] = bi4;
      s2_vals[base + 4] = bv5; s2_ids[base + 4] = bi5;
      s2_vals[base + 5] = bv6; s2_ids[base + 5] = bi6;
      s2_vals[base + 6] = bv7; s2_ids[base + 6] = bi7;
      s2_vals[base + 7] = bv8; s2_ids[base + 7] = bi8;
      __threadfence();
      s_is_last = (atomicAdd(&s2_done[m], 1) + 1 == kS2CTAsPerRow) ? 1 : 0;
    }
  }
  __syncthreads();

  if (s_is_last && tid < 32) {
    float fv1 = -1e38f, fv2 = -1e38f, fv3 = -1e38f, fv4 = -1e38f;
    float fv5 = -1e38f, fv6 = -1e38f, fv7 = -1e38f, fv8 = -1e38f;
    int fi1 = 0x7fffffff, fi2 = 0x7fffffff, fi3 = 0x7fffffff, fi4 = 0x7fffffff;
    int fi5 = 0x7fffffff, fi6 = 0x7fffffff, fi7 = 0x7fffffff, fi8 = 0x7fffffff;
    if (tid < kS2CTAsPerRow) {
      const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + tid) * 8;
      fv1 = dhf_ldcg_f(&s2_vals[base + 0]); fi1 = dhf_ldcg_i(&s2_ids[base + 0]);
      fv2 = dhf_ldcg_f(&s2_vals[base + 1]); fi2 = dhf_ldcg_i(&s2_ids[base + 1]);
      fv3 = dhf_ldcg_f(&s2_vals[base + 2]); fi3 = dhf_ldcg_i(&s2_ids[base + 2]);
      fv4 = dhf_ldcg_f(&s2_vals[base + 3]); fi4 = dhf_ldcg_i(&s2_ids[base + 3]);
      fv5 = dhf_ldcg_f(&s2_vals[base + 4]); fi5 = dhf_ldcg_i(&s2_ids[base + 4]);
      fv6 = dhf_ldcg_f(&s2_vals[base + 5]); fi6 = dhf_ldcg_i(&s2_ids[base + 5]);
      fv7 = dhf_ldcg_f(&s2_vals[base + 6]); fi7 = dhf_ldcg_i(&s2_ids[base + 6]);
      fv8 = dhf_ldcg_f(&s2_vals[base + 7]); fi8 = dhf_ldcg_i(&s2_ids[base + 7]);
    }
#pragma unroll
    for (int o = (kS2CTAsPerRow >> 1); o > 0; o >>= 1) {
      const float ov1 = __shfl_down_sync(0xffffffffu, fv1, o);
      const int oi1 = __shfl_down_sync(0xffffffffu, fi1, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, fv2, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, fi2, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, fv3, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, fi3, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, fv4, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, fi4, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, fv5, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, fi5, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, fv6, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, fi6, o);
      const float ov7 = __shfl_down_sync(0xffffffffu, fv7, o);
      const int oi7 = __shfl_down_sync(0xffffffffu, fi7, o);
      const float ov8 = __shfl_down_sync(0xffffffffu, fv8, o);
      const int oi8 = __shfl_down_sync(0xffffffffu, fi8, o);
      dhf_top8_merge(
          fv1, fi1, fv2, fi2, fv3, fi3, fv4, fi4, fv5, fi5, fv6, fi6, fv7, fi7, fv8, fi8,
          ov1, oi1, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
    }
    if (tid == 0) {
      atomicExch(&s2_done[m], 0);
      out[m] = static_cast<int64_t>(fi1);
      if (out_top2_row0 != nullptr && m < top2_rows) {
        out_top2_row0[2 * m + 0] = static_cast<int64_t>(fi1);
        out_top2_row0[2 * m + 1] = (fi2 == 0x7fffffff) ? static_cast<int64_t>(fi1) : static_cast<int64_t>(fi2);
      }
      if (out_margins != nullptr && m < margin_rows) {
        out_margins[m] = (fi2 == 0x7fffffff) ? 100.0f : (fv1 - fv2);
      }
      if (out_tree_top != nullptr && m < tree_top_rows) {
        if (tree_k >= 8) {
          out_tree_top[8 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[8 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[8 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[8 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
          out_tree_top[8 * m + 4] = (fi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi5);
          out_tree_top[8 * m + 5] = (fi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi6);
          out_tree_top[8 * m + 6] = (fi7 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi7);
          out_tree_top[8 * m + 7] = (fi8 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi8);
          if (out_tree_vals != nullptr) {
            out_tree_vals[8 * m + 0] = (fi1 == 0x7fffffff) ? -1e30f : fv1;
            out_tree_vals[8 * m + 1] = (fi2 == 0x7fffffff) ? -1e30f : fv2;
            out_tree_vals[8 * m + 2] = (fi3 == 0x7fffffff) ? -1e30f : fv3;
            out_tree_vals[8 * m + 3] = (fi4 == 0x7fffffff) ? -1e30f : fv4;
            out_tree_vals[8 * m + 4] = (fi5 == 0x7fffffff) ? -1e30f : fv5;
            out_tree_vals[8 * m + 5] = (fi6 == 0x7fffffff) ? -1e30f : fv6;
            out_tree_vals[8 * m + 6] = (fi7 == 0x7fffffff) ? -1e30f : fv7;
            out_tree_vals[8 * m + 7] = (fi8 == 0x7fffffff) ? -1e30f : fv8;
          }
        } else if (tree_k >= 6) {
          out_tree_top[6 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[6 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[6 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[6 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
          out_tree_top[6 * m + 4] = (fi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi5);
          out_tree_top[6 * m + 5] = (fi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi6);
          if (out_tree_vals != nullptr) {
            out_tree_vals[6 * m + 0] = (fi1 == 0x7fffffff) ? -1e30f : fv1;
            out_tree_vals[6 * m + 1] = (fi2 == 0x7fffffff) ? -1e30f : fv2;
            out_tree_vals[6 * m + 2] = (fi3 == 0x7fffffff) ? -1e30f : fv3;
            out_tree_vals[6 * m + 3] = (fi4 == 0x7fffffff) ? -1e30f : fv4;
            out_tree_vals[6 * m + 4] = (fi5 == 0x7fffffff) ? -1e30f : fv5;
            out_tree_vals[6 * m + 5] = (fi6 == 0x7fffffff) ? -1e30f : fv6;
          }
        } else {
          out_tree_top[4 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[4 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[4 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[4 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
        }
      }
      if ((pdl_mode & 2) != 0) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }
  }
}

// Fused 1-wave Stage-2 kernel (96 CTAs = 2 CTAs/SM on GB10):
// Phase A (A -> SMEM + OOV tail NVFP4 rescoring) overlaps Stage-1 tail BEFORE griddepcontrol.wait,
// then runs exact 4-pass radix select in SMEM (eliminating dhf_select_topk_kernel + global cand_ids),
// then rescores static shortlist candidates and performs deterministic last-CTA top-8 merge!
__global__ void __launch_bounds__(256, 2) dhf_stage2_rescore_fused_kernel(
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ cta_vals,
    const int32_t* __restrict__ cta_ids,
    const int32_t* __restrict__ tail_ids,
    const int32_t* __restrict__ tail_count,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    float* __restrict__ s2_vals,
    int32_t* __restrict__ s2_ids,
    int32_t* __restrict__ s2_done,
    int64_t* __restrict__ out,
    int64_t* __restrict__ out_top2_row0,
    int top2_rows,
    float* __restrict__ out_margins,
    int margin_rows,
    int64_t* __restrict__ out_tree_top,
    float* __restrict__ out_tree_vals,
    int tree_top_rows,
    int tree_k,
    int N_in, int K_cand, int K, int max_tail, float logit_scale, int pdl_mode) {
  const int bid = blockIdx.x;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int num_g = K >> 4;
  const bool use_smem_a = (num_g <= kS2MaxGroups);

  __shared__ uint4 s_a0[kS2MaxGroups];
  __shared__ uint4 s_a1[kS2MaxGroups];
  __shared__ uint32_t s_keys[kWaveCTAs * kCtaTopK];
  __shared__ int32_t s_ids[kWaveCTAs * kCtaTopK];
  __shared__ uint32_t s_hist[256];
  __shared__ int32_t s_cand[kMaxCand];
  __shared__ float wv[kS2Warps][8];
  __shared__ int32_t wi[kS2Warps][8];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_rem_k;
  __shared__ uint32_t s_out_cnt;
  __shared__ int s_is_last;

  const __nv_bfloat16* a_row = A + static_cast<size_t>(m) * K;
  if (use_smem_a) {
    const uint4* a_u4 = reinterpret_cast<const uint4*>(a_row);
    for (int g = tid; g < num_g; g += 256) {
      s_a0[g] = a_u4[g * 2 + 0];
      s_a1[g] = a_u4[g * 2 + 1];
    }
    __syncthreads();
  }

  float v1 = -1e38f, v2 = -1e38f, v3 = -1e38f, v4 = -1e38f;
  float v5 = -1e38f, v6 = -1e38f, v7 = -1e38f, v8 = -1e38f;
  int i1 = 0x7fffffff, i2 = 0x7fffffff, i3 = 0x7fffffff, i4 = 0x7fffffff;
  int i5 = 0x7fffffff, i6 = 0x7fffffff, i7 = 0x7fffffff, i8 = 0x7fffffff;

  const float total_scale = __ldg(global_scale) * logit_scale;
  const int stride_w = kS2CTAsPerRow * kS2Warps;

  // Phase A (before PDL wait, overlapping Stage-1 tail): score dynamic prompt-tail OOV rows.
  int tcnt = tail_count[0];
  if (tcnt < 0) tcnt = 0;
  if (tcnt > max_tail) tcnt = max_tail;
  for (int idx = bid * kS2Warps + warp; idx < tcnt; idx += stride_w) {
    const int tok = tail_ids[idx];
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  // Wait for Stage-1 (cta_vals, cta_ids) via SM121 PDL.
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  // In-SMEM exact 4-pass MSB radix select (100% bit-identical to dhf_select_topk_kernel).
  const float* v_in = cta_vals + static_cast<size_t>(m) * N_in;
  const int32_t* i_in = cta_ids + static_cast<size_t>(m) * N_in;
  for (int i = tid; i < N_in; i += 256) {
    float v = v_in[i];
    int32_t id = i_in[i];
    if (isnan(v) || id < 0) {
      v = -1e38f;
      id = -1;
    }
    s_keys[i] = dhf_f32_to_key(v);
    s_ids[i] = id;
  }
  if (tid == 0) {
    s_prefix = 0u;
    s_rem_k = static_cast<uint32_t>((K_cand < N_in) ? K_cand : N_in);
    s_out_cnt = 0u;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const uint32_t shift = static_cast<uint32_t>(24 - pass * 8);
    const uint32_t mask = (pass == 0) ? 0u : (~0u << (32 - pass * 8));
    const uint32_t pref = s_prefix;
    s_hist[tid] = 0u;
    __syncthreads();

    for (int i = tid; i < N_in; i += 256) {
      const uint32_t k = s_keys[i];
      if ((k & mask) == pref) {
        const uint32_t bkt = (k >> shift) & 0xFFu;
        atomicAdd(&s_hist[(bkt >> 3) + ((bkt & 7u) << 5)], 1u);
      }
    }
    __syncthreads();

    if (tid < 32) {
      dhf_radix_scan_warp(s_hist, pref, shift, &s_prefix, &s_rem_k);
    }
    __syncthreads();
  }

  const uint32_t thresh = s_prefix;
  if (tid < 32) {
    uint32_t cur = 0u;
    const uint32_t cap = static_cast<uint32_t>(K_cand);
    for (int base = 0; base < N_in && cur < cap; base += 32) {
      const int i = base + tid;
      const bool hit = (i < N_in) && (s_keys[i] > thresh) && (s_ids[i] >= 0);
      const uint32_t bmask = __ballot_sync(0xffffffffu, hit);
      if (bmask != 0u) {
        const uint32_t slot = cur + static_cast<uint32_t>(__popc(bmask & ((1u << tid) - 1u)));
        if (hit && slot < cap) {
          s_cand[slot] = s_ids[i];
        }
        cur += static_cast<uint32_t>(__popc(bmask));
      }
    }
    for (int base = 0; base < N_in && cur < cap; base += 32) {
      const int i = base + tid;
      const bool hit = (i < N_in) && (s_keys[i] == thresh) && (s_ids[i] >= 0);
      const uint32_t bmask = __ballot_sync(0xffffffffu, hit);
      if (bmask != 0u) {
        const uint32_t slot = cur + static_cast<uint32_t>(__popc(bmask & ((1u << tid) - 1u)));
        if (hit && slot < cap) {
          s_cand[slot] = s_ids[i];
        }
        cur += static_cast<uint32_t>(__popc(bmask));
      }
    }
    if (tid == 0) {
      s_out_cnt = (cur < cap) ? cur : cap;
    }
  }
  __syncthreads();

  // Phase B: score static shortlist candidates against exact raw NVFP4 weights.
  const int cand_lim = static_cast<int>(s_out_cnt);
  for (int idx = bid * kS2Warps + warp; idx < cand_lim; idx += stride_w) {
    const int tok = s_cand[idx];
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  if (lane == 0) {
    wv[warp][0] = v1; wi[warp][0] = i1;
    wv[warp][1] = v2; wi[warp][1] = i2;
    wv[warp][2] = v3; wi[warp][2] = i3;
    wv[warp][3] = v4; wi[warp][3] = i4;
    wv[warp][4] = v5; wi[warp][4] = i5;
    wv[warp][5] = v6; wi[warp][5] = i6;
    wv[warp][6] = v7; wi[warp][6] = i7;
    wv[warp][7] = v8; wi[warp][7] = i8;
  }
  __syncthreads();

  if (tid < 32) {
    float bv1 = (tid < kS2Warps) ? wv[tid][0] : -1e38f;
    float bv2 = (tid < kS2Warps) ? wv[tid][1] : -1e38f;
    float bv3 = (tid < kS2Warps) ? wv[tid][2] : -1e38f;
    float bv4 = (tid < kS2Warps) ? wv[tid][3] : -1e38f;
    float bv5 = (tid < kS2Warps) ? wv[tid][4] : -1e38f;
    float bv6 = (tid < kS2Warps) ? wv[tid][5] : -1e38f;
    float bv7 = (tid < kS2Warps) ? wv[tid][6] : -1e38f;
    float bv8 = (tid < kS2Warps) ? wv[tid][7] : -1e38f;
    int bi1 = (tid < kS2Warps) ? wi[tid][0] : 0x7fffffff;
    int bi2 = (tid < kS2Warps) ? wi[tid][1] : 0x7fffffff;
    int bi3 = (tid < kS2Warps) ? wi[tid][2] : 0x7fffffff;
    int bi4 = (tid < kS2Warps) ? wi[tid][3] : 0x7fffffff;
    int bi5 = (tid < kS2Warps) ? wi[tid][4] : 0x7fffffff;
    int bi6 = (tid < kS2Warps) ? wi[tid][5] : 0x7fffffff;
    int bi7 = (tid < kS2Warps) ? wi[tid][6] : 0x7fffffff;
    int bi8 = (tid < kS2Warps) ? wi[tid][7] : 0x7fffffff;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      const float ov1 = __shfl_down_sync(0xffffffffu, bv1, o);
      const int oi1 = __shfl_down_sync(0xffffffffu, bi1, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, bv2, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, bi2, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, bv3, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, bi3, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, bv4, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, bi4, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, bv5, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, bi5, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, bv6, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, bi6, o);
      const float ov7 = __shfl_down_sync(0xffffffffu, bv7, o);
      const int oi7 = __shfl_down_sync(0xffffffffu, bi7, o);
      const float ov8 = __shfl_down_sync(0xffffffffu, bv8, o);
      const int oi8 = __shfl_down_sync(0xffffffffu, bi8, o);
      dhf_top8_merge(
          bv1, bi1, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
          ov1, oi1, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
    }
    if (tid == 0) {
      const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + bid) * 8;
      s2_vals[base + 0] = bv1; s2_ids[base + 0] = bi1;
      s2_vals[base + 1] = bv2; s2_ids[base + 1] = bi2;
      s2_vals[base + 2] = bv3; s2_ids[base + 2] = bi3;
      s2_vals[base + 3] = bv4; s2_ids[base + 3] = bi4;
      s2_vals[base + 4] = bv5; s2_ids[base + 4] = bi5;
      s2_vals[base + 5] = bv6; s2_ids[base + 5] = bi6;
      s2_vals[base + 6] = bv7; s2_ids[base + 6] = bi7;
      s2_vals[base + 7] = bv8; s2_ids[base + 7] = bi8;
      __threadfence();
      s_is_last = (atomicAdd(&s2_done[m], 1) + 1 == kS2CTAsPerRow) ? 1 : 0;
    }
  }
  __syncthreads();

  if (s_is_last && tid < 32) {
    float fv1 = -1e38f, fv2 = -1e38f, fv3 = -1e38f, fv4 = -1e38f;
    float fv5 = -1e38f, fv6 = -1e38f, fv7 = -1e38f, fv8 = -1e38f;
    int fi1 = 0x7fffffff, fi2 = 0x7fffffff, fi3 = 0x7fffffff, fi4 = 0x7fffffff;
    int fi5 = 0x7fffffff, fi6 = 0x7fffffff, fi7 = 0x7fffffff, fi8 = 0x7fffffff;
    if (tid < kS2CTAsPerRow) {
      const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + tid) * 8;
      fv1 = dhf_ldcg_f(&s2_vals[base + 0]); fi1 = dhf_ldcg_i(&s2_ids[base + 0]);
      fv2 = dhf_ldcg_f(&s2_vals[base + 1]); fi2 = dhf_ldcg_i(&s2_ids[base + 1]);
      fv3 = dhf_ldcg_f(&s2_vals[base + 2]); fi3 = dhf_ldcg_i(&s2_ids[base + 2]);
      fv4 = dhf_ldcg_f(&s2_vals[base + 3]); fi4 = dhf_ldcg_i(&s2_ids[base + 3]);
      fv5 = dhf_ldcg_f(&s2_vals[base + 4]); fi5 = dhf_ldcg_i(&s2_ids[base + 4]);
      fv6 = dhf_ldcg_f(&s2_vals[base + 5]); fi6 = dhf_ldcg_i(&s2_ids[base + 5]);
      fv7 = dhf_ldcg_f(&s2_vals[base + 6]); fi7 = dhf_ldcg_i(&s2_ids[base + 6]);
      fv8 = dhf_ldcg_f(&s2_vals[base + 7]); fi8 = dhf_ldcg_i(&s2_ids[base + 7]);
    }
#pragma unroll
    for (int o = (kS2CTAsPerRow >> 1); o > 0; o >>= 1) {
      const float ov1 = __shfl_down_sync(0xffffffffu, fv1, o);
      const int oi1 = __shfl_down_sync(0xffffffffu, fi1, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, fv2, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, fi2, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, fv3, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, fi3, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, fv4, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, fi4, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, fv5, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, fi5, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, fv6, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, fi6, o);
      const float ov7 = __shfl_down_sync(0xffffffffu, fv7, o);
      const int oi7 = __shfl_down_sync(0xffffffffu, fi7, o);
      const float ov8 = __shfl_down_sync(0xffffffffu, fv8, o);
      const int oi8 = __shfl_down_sync(0xffffffffu, fi8, o);
      dhf_top8_merge(
          fv1, fi1, fv2, fi2, fv3, fi3, fv4, fi4, fv5, fi5, fv6, fi6, fv7, fi7, fv8, fi8,
          ov1, oi1, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
    }
    if (tid == 0) {
      atomicExch(&s2_done[m], 0);
      out[m] = static_cast<int64_t>(fi1);
      if (out_top2_row0 != nullptr && m < top2_rows) {
        out_top2_row0[2 * m + 0] = static_cast<int64_t>(fi1);
        out_top2_row0[2 * m + 1] = (fi2 == 0x7fffffff) ? static_cast<int64_t>(fi1) : static_cast<int64_t>(fi2);
      }
      if (out_margins != nullptr && m < margin_rows) {
        out_margins[m] = (fi2 == 0x7fffffff) ? 100.0f : (fv1 - fv2);
      }
      if (out_tree_top != nullptr && m < tree_top_rows) {
        if (tree_k >= 8) {
          out_tree_top[8 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[8 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[8 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[8 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
          out_tree_top[8 * m + 4] = (fi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi5);
          out_tree_top[8 * m + 5] = (fi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi6);
          out_tree_top[8 * m + 6] = (fi7 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi7);
          out_tree_top[8 * m + 7] = (fi8 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi8);
          if (out_tree_vals != nullptr) {
            out_tree_vals[8 * m + 0] = (fi1 == 0x7fffffff) ? -1e30f : fv1;
            out_tree_vals[8 * m + 1] = (fi2 == 0x7fffffff) ? -1e30f : fv2;
            out_tree_vals[8 * m + 2] = (fi3 == 0x7fffffff) ? -1e30f : fv3;
            out_tree_vals[8 * m + 3] = (fi4 == 0x7fffffff) ? -1e30f : fv4;
            out_tree_vals[8 * m + 4] = (fi5 == 0x7fffffff) ? -1e30f : fv5;
            out_tree_vals[8 * m + 5] = (fi6 == 0x7fffffff) ? -1e30f : fv6;
            out_tree_vals[8 * m + 6] = (fi7 == 0x7fffffff) ? -1e30f : fv7;
            out_tree_vals[8 * m + 7] = (fi8 == 0x7fffffff) ? -1e30f : fv8;
          }
        } else if (tree_k >= 6) {
          out_tree_top[6 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[6 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[6 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[6 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
          out_tree_top[6 * m + 4] = (fi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi5);
          out_tree_top[6 * m + 5] = (fi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi6);
          if (out_tree_vals != nullptr) {
            out_tree_vals[6 * m + 0] = (fi1 == 0x7fffffff) ? -1e30f : fv1;
            out_tree_vals[6 * m + 1] = (fi2 == 0x7fffffff) ? -1e30f : fv2;
            out_tree_vals[6 * m + 2] = (fi3 == 0x7fffffff) ? -1e30f : fv3;
            out_tree_vals[6 * m + 3] = (fi4 == 0x7fffffff) ? -1e30f : fv4;
            out_tree_vals[6 * m + 4] = (fi5 == 0x7fffffff) ? -1e30f : fv5;
            out_tree_vals[6 * m + 5] = (fi6 == 0x7fffffff) ? -1e30f : fv6;
          }
        } else {
          out_tree_top[4 * m + 0] = static_cast<int64_t>(fi1);
          out_tree_top[4 * m + 1] = (fi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi2);
          out_tree_top[4 * m + 2] = (fi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi3);
          out_tree_top[4 * m + 3] = (fi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(fi4);
        }
      }
      if ((pdl_mode & 2) != 0) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }
  }
}

struct DHeadFastScratch {
  torch::Tensor a_scaled;
  torch::Tensor cta_vals;
  torch::Tensor cta_ids;
  torch::Tensor cand_ids;
  torch::Tensor s2_vals;
  torch::Tensor s2_ids;
  torch::Tensor s2_done;
  int cap_k = 0;
  c10::DeviceIndex dev = -1;
};

DHeadFastScratch& get_dhf_scratch(torch::Device device, int K) {
  static DHeadFastScratch sc;
  if (!sc.cta_vals.defined() || sc.dev != device.index() || sc.cap_k < K) {
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    sc.a_scaled = torch::empty({kMaxM, K}, opt_bf16);
    sc.cta_vals = torch::empty({kMaxM, kWaveCTAs * kCtaTopK}, opt_f32);
    sc.cta_ids = torch::empty({kMaxM, kWaveCTAs * kCtaTopK}, opt_i32);
    sc.cand_ids = torch::empty({kMaxM, kMaxCand}, opt_i32);
    sc.s2_vals = torch::empty({kMaxM, kS2CTAsPerRow, 8}, opt_f32);
    sc.s2_ids = torch::empty({kMaxM, kS2CTAsPerRow, 8}, opt_i32);
    sc.s2_done = torch::zeros({kMaxM}, opt_i32);
    sc.cap_k = K;
    sc.dev = device.index();
  }
  return sc;
}

int pick_coprime_stride(int V_static) {
  static constexpr int kPrimes[] = {1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061};
  for (int p : kPrimes) {
    if (std::gcd(p, V_static) == 1) return p;
  }
  for (int p = 1063; ; p += 2) {
    if (std::gcd(p, V_static) == 1) return p;
  }
}

}  // namespace dhf4d

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dhead_fast_pack(
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    torch::Tensor vocab_map,
    int64_t mode) {
  vq4d_reset_norm_rot_table();
  using namespace dhf4d;
  TORCH_CHECK(raw_w.is_cuda() && raw_w.scalar_type() == torch::kUInt8 && raw_w.dim() == 2 && raw_w.is_contiguous());
  TORCH_CHECK(raw_s.is_cuda() && raw_s.dim() == 2 && raw_s.is_contiguous());
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(vocab_map.is_cuda() && vocab_map.scalar_type() == torch::kInt64 && vocab_map.dim() == 1);
  TORCH_CHECK(mode == 100 || mode == 150 || mode == 158 || mode == 2,
              "dhead_fast_pack: mode must be 100, 150, 158, or 2, got ", mode);

  const int V_static = static_cast<int>(vocab_map.numel());
  const int K = static_cast<int>(raw_w.size(1) * 2);
  TORCH_CHECK(V_static >= 64 && (V_static % kRowsPerCTA) == 0,
              "dhead_fast_pack: V_static must be multiple of 64, got ", V_static);
  TORCH_CHECK(K >= 64 && (K % 64) == 0, "dhead_fast_pack: K must be multiple of 64, got ", K);

  auto dev = raw_w.device();
  auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

  // Coprime bijection disperses high-frequency BPE tokens uniformly across all 192 CTAs.
  const int stride = pick_coprime_stride(V_static);
  auto perm = (torch::arange(V_static, vocab_map.options()) * stride).remainder(V_static);
  auto packed_tok_ids = vocab_map.index_select(0, perm).to(torch::kInt32).contiguous();

  auto col_mean = torch::empty({K}, opt_f32);
  auto col_std = torch::empty({(mode == 100) ? (K + 1024) : K}, opt_f32);
  auto row_scales = torch::empty({V_static}, opt_f32);
  auto row_thresh = torch::empty({V_static}, opt_f32);

  const int64_t tile_bytes =
      (mode == 100) ? kTileBytes100 : ((mode == 150) ? kTileBytes150 : ((mode == 158) ? kTileBytes158 : kTileBytes2));
  const int64_t num_tiles = static_cast<int64_t>(V_static / 8) * (K / 64);
  auto blob = torch::empty({num_tiles * tile_bytes}, opt_u8);

  auto st = c10::cuda::getCurrentCUDAStream().stream();

  // Pass 1: column mean & std
  dhf_col_stats_kernel<<<K >> 4, 256, 0, st>>>(
      raw_w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
      global_scale.data_ptr<float>(),
      packed_tok_ids.data_ptr<int32_t>(),
      col_mean.data_ptr<float>(),
      col_std.data_ptr<float>(),
      V_static, K);

  if (mode == 100) {
    // Fit 128 antipodal 8D centroids directly to empirical whitened lm_head 8-tuples via 15-iter Voronoi Lloyd-Max (~8 ms).
    auto X = torch::empty({512 * 32, 8}, opt_f32);
    dhf_extract_8d_samples_kernel<<<512, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        X.data_ptr<float>(),
        V_static, K);
    auto init_idx = (torch::arange(128, vocab_map.options()) * 127).remainder(X.size(0));
    auto C = X.index_select(0, init_idx).contiguous();
    auto ones_n = torch::ones({X.size(0)}, opt_f32);
    for (int it = 0; it < 15; ++it) {
      auto dots = torch::mm(X, C.t());
      auto cn2 = 0.5f * (C * C).sum(1, true).t();
      auto best_idx = std::get<1>((dots.abs() - cn2).max(1));
      auto picked = dots.gather(1, best_idx.unsqueeze(1));
      auto sgn = torch::where(picked >= 0.0f, torch::ones_like(picked), -torch::ones_like(picked));
      auto sum_x = torch::zeros_like(C);
      sum_x.index_add_(0, best_idx, X * sgn);
      auto cnt = torch::zeros({128}, opt_f32);
      cnt.index_add_(0, best_idx, ones_n);
      auto new_c = sum_x / cnt.clamp_min(1.0f).unsqueeze(1);
      C = torch::where((cnt > 0.5f).unsqueeze(1), new_c, C);
    }
    C = C.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
    col_std.narrow(0, K, 1024).copy_(C.reshape({1024}));
  }

  // Pass 2: weighted-LS optimal row scales & thresholds / inv_std
  const int row_blocks = (V_static + 7) / 8;
  if (mode == 100) {
    dhf_row_scales_kernel<100><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  } else if (mode == 150) {
    dhf_row_scales_kernel<150><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  } else if (mode == 158) {
    dhf_row_scales_kernel<158><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  } else {
    dhf_row_scales_kernel<2><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  }

  // Pass 3: pack wave-interleaved 8x64 tiles
  dim3 pack_grid(V_static / 8, K / 64);
  if (mode == 100) {
    const int n_blocks = V_static / kRowsPerCTA;
    const bool use_pair = dhf_pair_waves_enabled() && ((n_blocks % (2 * kWaveCTAs)) == 0);
    const bool use_tma_pair_bk128 =
        !use_pair && dhf_tma100_enabled() && dhf_tma100_pair_enabled() && ((K & 127) == 0) &&
        (n_blocks >= 8 * kWaveCTAs) && ((reinterpret_cast<uintptr_t>(blob.data_ptr()) & 15u) == 0u);
    const bool use_tma_bk128 =
        !use_pair && !use_tma_pair_bk128 && dhf_tma100_enabled() && ((K & 127) == 0) &&
        ((reinterpret_cast<uintptr_t>(blob.data_ptr()) & 15u) == 0u);
    if (use_pair) {
      dhf_pack_tiles_kernel<100, 1><<<pack_grid, 32, 0, st>>>(
          raw_w.data_ptr<uint8_t>(),
          reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
          global_scale.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          col_mean.data_ptr<float>(),
          col_std.data_ptr<float>(),
          row_thresh.data_ptr<float>(),
          blob.data_ptr<uint8_t>(),
          V_static, K);
    } else if (use_tma_pair_bk128) {
      dhf_pack_tiles_kernel<100, 3><<<pack_grid, 32, 0, st>>>(
          raw_w.data_ptr<uint8_t>(),
          reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
          global_scale.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          col_mean.data_ptr<float>(),
          col_std.data_ptr<float>(),
          row_thresh.data_ptr<float>(),
          blob.data_ptr<uint8_t>(),
          V_static, K);
    } else if (use_tma_bk128) {
      dhf_pack_tiles_kernel<100, 2><<<pack_grid, 32, 0, st>>>(
          raw_w.data_ptr<uint8_t>(),
          reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
          global_scale.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          col_mean.data_ptr<float>(),
          col_std.data_ptr<float>(),
          row_thresh.data_ptr<float>(),
          blob.data_ptr<uint8_t>(),
          V_static, K);
    } else {
      dhf_pack_tiles_kernel<100, 0><<<pack_grid, 32, 0, st>>>(
          raw_w.data_ptr<uint8_t>(),
          reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
          global_scale.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          col_mean.data_ptr<float>(),
          col_std.data_ptr<float>(),
          row_thresh.data_ptr<float>(),
          blob.data_ptr<uint8_t>(),
          V_static, K);
    }
  } else if (mode == 150) {
    dhf_pack_tiles_kernel<150><<<pack_grid, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        blob.data_ptr<uint8_t>(),
        V_static, K);
  } else if (mode == 158) {
    dhf_pack_tiles_kernel<158><<<pack_grid, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        blob.data_ptr<uint8_t>(),
        V_static, K);
  } else {
    dhf_pack_tiles_kernel<2><<<pack_grid, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        blob.data_ptr<uint8_t>(),
        V_static, K);
  }

  return std::make_tuple(blob, row_scales, col_std, packed_tok_ids);
}

torch::Tensor dhead_fast_argmax(
    torch::Tensor A,
    torch::Tensor blob,
    torch::Tensor row_scales,
    torch::Tensor col_scales,
    torch::Tensor packed_tok_ids,
    torch::Tensor tail_ids,
    torch::Tensor tail_count,
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    double logit_scale,
    int64_t mode,
    int64_t topk,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt) {
  vq4d_clear_all_prerot();
  using namespace dhf4d;
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == torch::kUInt8 && blob.is_contiguous());
  TORCH_CHECK(mode == 100 || mode == 150 || mode == 158 || mode == 2);
  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int V_static = static_cast<int>(packed_tok_ids.numel());
  const int max_tail = static_cast<int>(tail_ids.numel());
  TORCH_CHECK(M >= 1 && M <= kMaxM && (K % 64) == 0 && (V_static % kRowsPerCTA) == 0);
  TORCH_CHECK(K == raw_w.size(1) * 2);

  int64_t* top2_ptr = nullptr;
  int top2_rows = 0;
  if (out_top2_row0.has_value() && out_top2_row0.value().defined() && out_top2_row0.value().numel() >= 2) {
    TORCH_CHECK(out_top2_row0.value().is_cuda() && out_top2_row0.value().scalar_type() == torch::kInt64);
    top2_ptr = out_top2_row0.value().data_ptr<int64_t>();
    top2_rows = (out_top2_row0.value().numel() >= 4 && M >= 2) ? 2 : 1;
  }
  float* margin_ptr = nullptr;
  int margin_rows = 0;
  if (out_margins.has_value() && out_margins.value().defined() && out_margins.value().numel() >= 1) {
    TORCH_CHECK(out_margins.value().is_cuda() && out_margins.value().scalar_type() == torch::kFloat32);
    margin_ptr = out_margins.value().data_ptr<float>();
    margin_rows = std::min(static_cast<int>(out_margins.value().numel()), top2_rows);
  }
  int64_t* tree_top_ptr = nullptr;
  int tree_top_rows = 0;
  int tree_k = 0;
  if (out_tree_top.has_value() && out_tree_top.value().defined() && out_tree_top.value().numel() >= 4) {
    TORCH_CHECK(out_tree_top.value().is_cuda() && out_tree_top.value().scalar_type() == torch::kInt64);
    tree_top_ptr = out_tree_top.value().data_ptr<int64_t>();
    if (out_tree_top.value().numel() >= 48) {
      tree_k = 8;
      tree_top_rows = std::min(6, std::min(M, static_cast<int>(out_tree_top.value().numel() / 8)));
    } else if (out_tree_top.value().numel() >= 36) {
      tree_k = 6;
      tree_top_rows = std::min(6, std::min(M, static_cast<int>(out_tree_top.value().numel() / 6)));
    } else {
      tree_k = 4;
      tree_top_rows = std::min(5, std::min(M, static_cast<int>(out_tree_top.value().numel() / 4)));
    }
  }
  float* tree_vals_ptr = nullptr;
  if (tree_k >= 6 && tree_top_rows > 0 && out_tree_vals.has_value() && out_tree_vals.value().defined() &&
      out_tree_vals.value().numel() >= tree_top_rows * tree_k) {
    TORCH_CHECK(out_tree_vals.value().is_cuda() && out_tree_vals.value().scalar_type() == torch::kFloat32);
    tree_vals_ptr = out_tree_vals.value().data_ptr<float>();
  }

  auto& sc = get_dhf_scratch(A.device(), K);
  auto out = torch::empty({M}, A.options().dtype(torch::kInt64));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl_mode = dhf_pdl_mode();

  const int n_blocks = V_static / kRowsPerCTA;
  const int grid_ctas = std::min(kWaveCTAs, n_blocks);
  const int N_in = grid_ctas * kCtaTopK;
  int K_cand = static_cast<int>(topk);
  if (K_cand < 16) K_cand = 16;
  if (K_cand > kMaxCand) K_cand = kMaxCand;
  if (K_cand > N_in) K_cand = N_in;

  // Kernel 0: Scale A by col_scales (does not wait on PDL; triggers Stage 1).
  dhf_launch_pdl(
      &dhf_prep_a_kernel, dim3(M, 1, 1), 256, st, (pdl_mode & 2),
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      col_scales.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(sc.a_scaled.data_ptr()),
      M, K, (pdl_mode & 2));

  // Kernel 1: Persistent Stage 1 compressed MMA + in-CTA top-8 shortlist.
  static bool carveout_set = false;
  if (!carveout_set) {
    carveout_set = true;
    cudaFuncSetAttribute(dhf_stage1_mode100_pair_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_mode100_pair_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_tma100_pair_bk128_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_tma100_pair_bk128_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_tma100_bk128_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_tma100_bk128_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_kernel<100, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_kernel<100, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_kernel<150, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage1_kernel<150, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage2_rescore_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(dhf_stage2_rescore_fused_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }
  static const bool dyn_ev = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  static const bool evict_first_cached = []() {
    const char* s = std::getenv("MUSE_DHF_EVICT_FIRST");
    return !s || !*s || std::atoi(s) != 0;
  }();
  const bool evict_first = dyn_ev ? ([]() {
    const char* s = std::getenv("MUSE_DHF_EVICT_FIRST");
    return !s || !*s || std::atoi(s) != 0;
  }()) : evict_first_cached;

  auto launch_s1 = [&](auto mode_tag) {
    constexpr int KM = decltype(mode_tag)::value;
    if constexpr (KM == 100) {
      const bool use_pair = dhf_pair_waves_enabled() && ((n_blocks % (2 * kWaveCTAs)) == 0);
      if (use_pair) {
        if (evict_first) {
          dhf_launch_pdl(
              &dhf_stage1_mode100_pair_kernel<true>, dim3(kWaveCTAs, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        } else {
          dhf_launch_pdl(
              &dhf_stage1_mode100_pair_kernel<false>, dim3(kWaveCTAs, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        }
        return;
      }
      if (dhf_tma100_enabled() && dhf_tma100_pair_enabled() && ((K & 127) == 0) &&
          (n_blocks >= 8 * kWaveCTAs) && ((reinterpret_cast<uintptr_t>(blob.data_ptr()) & 15u) == 0u)) {
        if (evict_first) {
          dhf_launch_pdl(
              &dhf_stage1_tma100_pair_bk128_kernel<true>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        } else {
          dhf_launch_pdl(
              &dhf_stage1_tma100_pair_bk128_kernel<false>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        }
        return;
      }
      if (dhf_tma100_enabled() && ((K & 127) == 0) &&
          ((reinterpret_cast<uintptr_t>(blob.data_ptr()) & 15u) == 0u)) {
        if (evict_first) {
          dhf_launch_pdl(
              &dhf_stage1_tma100_bk128_kernel<true>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        } else {
          dhf_launch_pdl(
              &dhf_stage1_tma100_bk128_kernel<false>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
              reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
              blob.data_ptr<uint8_t>(),
              row_scales.data_ptr<float>(),
              col_scales.data_ptr<float>(),
              packed_tok_ids.data_ptr<int32_t>(),
              sc.cta_vals.data_ptr<float>(),
              sc.cta_ids.data_ptr<int32_t>(),
              M, V_static, K, pdl_mode);
        }
        return;
      }
    }
    if (evict_first) {
      dhf_launch_pdl(
          &dhf_stage1_kernel<KM, true>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
          blob.data_ptr<uint8_t>(),
          row_scales.data_ptr<float>(),
          col_scales.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          sc.cta_vals.data_ptr<float>(),
          sc.cta_ids.data_ptr<int32_t>(),
          M, V_static, K, pdl_mode);
    } else {
      dhf_launch_pdl(
          &dhf_stage1_kernel<KM, false>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
          blob.data_ptr<uint8_t>(),
          row_scales.data_ptr<float>(),
          col_scales.data_ptr<float>(),
          packed_tok_ids.data_ptr<int32_t>(),
          sc.cta_vals.data_ptr<float>(),
          sc.cta_ids.data_ptr<int32_t>(),
          M, V_static, K, pdl_mode);
    }
  };
  if (mode == 100) launch_s1(std::integral_constant<int, 100>{});
  else if (mode == 150) launch_s1(std::integral_constant<int, 150>{});
  else if (mode == 158) launch_s1(std::integral_constant<int, 158>{});
  else launch_s1(std::integral_constant<int, 2>{});

  if (dhf_fuse_select_enabled()) {
    dhf_launch_pdl(
        &dhf_stage2_rescore_fused_kernel, dim3(kS2CTAsPerRow, M, 1), 256, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const float*>(sc.cta_vals.data_ptr<float>()),
        static_cast<const int32_t*>(sc.cta_ids.data_ptr<int32_t>()),
        tail_ids.data_ptr<int32_t>(),
        tail_count.data_ptr<int32_t>(),
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        sc.s2_vals.data_ptr<float>(),
        sc.s2_ids.data_ptr<int32_t>(),
        sc.s2_done.data_ptr<int32_t>(),
        out.data_ptr<int64_t>(),
        top2_ptr, top2_rows,
        margin_ptr, margin_rows,
        tree_top_ptr, tree_vals_ptr, tree_top_rows, tree_k,
        N_in, K_cand, K, max_tail, static_cast<float>(logit_scale), pdl_mode);
    return out;
  }

  // Kernel 2: Exact 3-pass radix select -> top K_cand target token IDs per row m.
  dhf_launch_pdl(
      &dhf_select_topk_kernel, dim3(M, 1, 1), 256, st, pdl_mode,
      static_cast<const float*>(sc.cta_vals.data_ptr<float>()),
      static_cast<const int32_t*>(sc.cta_ids.data_ptr<int32_t>()),
      sc.cand_ids.data_ptr<int32_t>(),
      N_in, K_cand, pdl_mode);

  // Kernel 3: Fused 1-wave Stage 2 exact NVFP4 rescore + deterministic last-CTA final top-8 reduction.
  dhf_launch_pdl(
      &dhf_stage2_rescore_kernel, dim3(kS2CTAsPerRow, M, 1), 256, st, pdl_mode,
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const int32_t*>(sc.cand_ids.data_ptr<int32_t>()),
      tail_ids.data_ptr<int32_t>(),
      tail_count.data_ptr<int32_t>(),
      raw_w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
      global_scale.data_ptr<float>(),
      sc.s2_vals.data_ptr<float>(),
      sc.s2_ids.data_ptr<int32_t>(),
      sc.s2_done.data_ptr<int32_t>(),
      out.data_ptr<int64_t>(),
      top2_ptr, top2_rows,
      margin_ptr, margin_rows,
      tree_top_ptr, tree_vals_ptr, tree_top_rows, tree_k,
      K_cand, K, max_tail, static_cast<float>(logit_scale), pdl_mode);

  return out;
}