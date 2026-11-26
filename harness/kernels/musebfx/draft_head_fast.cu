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
        __ldg(&reinterpret_cast<const uint4*>(a_stage_ptr(kg))[tid]);
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
