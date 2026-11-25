// museleaf: fused per-layer bookkeeping for the one-leaf DFlash verify (see vllm .../dflash/muse_leaf.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

// kv: [blocks, 2, bs, 256] bf16 (strided). One CTA, 256 threads.
//  1) if hit_prev: kv[fix_rb, h, fix_n, :] = kv[sb, h, fix_n, :]          (commit last step's leaf K/V)
//  2) if mode:     kv[sb, h, s, :] = kv[rb, h, s, :] for s < n            (scratch = real block below p+1)
//  3) q2[0] = q[0]; q2[1] = q[leaf_row]                                   (32 x 128 bf16 each)
__global__ void leaf_prep_kernel(__nv_bfloat16* kv, int64_t sB, int64_t sH, int64_t sS, int64_t sb,
                                 const int64_t* rb, const int64_t* n, const int64_t* fix_rb, const int64_t* fix_n,
                                 const bool* hit_prev, const bool* mode,
                                 const __nv_bfloat16* q, int64_t q_row_stride, int leaf_row, __nv_bfloat16* q2) {
  const int t = threadIdx.x;
  if (hit_prev[0]) {
    const int64_t fr = fix_rb[0], fn = fix_n[0];
    for (int i = t; i < 2 * 256; i += blockDim.x) {
      const int h = i >> 8, c = i & 255;
      kv[fr * sB + h * sH + fn * sS + c] = kv[sb * sB + h * sH + fn * sS + c];
    }
  }
  __syncthreads();
  if (mode[0]) {
    const int64_t r = rb[0], nn = n[0];
    for (int64_t i = t; i < 2 * nn * 256; i += blockDim.x) {
      const int64_t c = i & 255, hs = i >> 8, h = hs / nn, s = hs - h * nn;
      kv[sb * sB + h * sH + s * sS + c] = kv[r * sB + h * sH + s * sS + c];
    }
  }
  const uint4* q0 = reinterpret_cast<const uint4*>(q);
  const uint4* q6 = reinterpret_cast<const uint4*>(q + (int64_t)leaf_row * q_row_stride);
  uint4* o = reinterpret_cast<uint4*>(q2);
  for (int i = t; i < 512; i += blockDim.x) {  // 32*128 bf16 = 8 KB = 512 uint4 per row
    o[i] = q0[i];
    o[512 + i] = q6[i];
  }
}

__global__ void leaf_select_kernel(const bool* mode, const __nv_bfloat16* o2, __nv_bfloat16* out_row) {
  if (!mode[0]) return;
  const uint4* s = reinterpret_cast<const uint4*>(o2) + 512;  // row 1 of o2
  uint4* d = reinterpret_cast<uint4*>(out_row);
  for (int i = threadIdx.x; i < 512; i += blockDim.x) d[i] = s[i];
}

void leaf_prep(torch::Tensor kv, int64_t sb, torch::Tensor rb, torch::Tensor n, torch::Tensor fix_rb, torch::Tensor fix_n,
               torch::Tensor hit_prev, torch::Tensor mode, torch::Tensor q, int64_t leaf_row, torch::Tensor q2) {
  TORCH_CHECK(kv.dim() == 4 && kv.size(3) == 256 && kv.stride(3) == 1 && kv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(1) * q.size(2) == 4096 && q.stride(2) == 1 && q.stride(1) == 128);
  leaf_prep_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      (__nv_bfloat16*)kv.data_ptr(), kv.stride(0), kv.stride(1), kv.stride(2), sb,
      rb.data_ptr<int64_t>(), n.data_ptr<int64_t>(), fix_rb.data_ptr<int64_t>(), fix_n.data_ptr<int64_t>(),
      hit_prev.data_ptr<bool>(), mode.data_ptr<bool>(), (const __nv_bfloat16*)q.data_ptr(), q.stride(0), (int)leaf_row,
      (__nv_bfloat16*)q2.data_ptr());
}

void leaf_select(torch::Tensor mode, torch::Tensor o2, torch::Tensor out_row) {
  TORCH_CHECK(out_row.is_contiguous() && o2.is_contiguous());
  leaf_select_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      mode.data_ptr<bool>(), (const __nv_bfloat16*)o2.data_ptr(), (__nv_bfloat16*)out_row.data_ptr());
}


// ---- fused per-step setup: all kv-cache groups in one launch ----
// gstate int64 [G, 8]: 0 rb1, 1 n1, 2 fix_rb, 3 fix_n (x2 dup at 2,3 used by prep as [2]), 4 next_fix_rb, 5 next_fix_n
__global__ void leaf_setup_kernel(int64_t* positions, int32_t* input_ids, int64_t* slot_map, int64_t slot_stride,
                                  const int64_t* bt_ptrs, int32_t* gbt2, int W, int32_t* gseq, int64_t* gstate,
                                  int G, int bs, int64_t sb_top, const int64_t* alt, const int64_t* prev_ns, int low,
                                  bool* mode_out, int leaf_row) {
  __shared__ int64_t s_pos1;
  __shared__ bool s_mode;
  const int t = threadIdx.x;
  if (t == 0) {
    const int64_t pos1 = positions[1];
    const bool m = (prev_ns[0] <= low) && (alt[1] != alt[0]);
    s_pos1 = pos1; s_mode = m; mode_out[0] = m;
    if (m) { input_ids[leaf_row] = (int32_t)alt[1]; positions[leaf_row] = pos1; }
  }
  __syncthreads();
  const int64_t pos1 = s_pos1;
  const int64_t blk = pos1 / bs, n = pos1 - blk * bs;
  for (int g = 0; g < G; g++) {
    const int32_t* bt = reinterpret_cast<const int32_t*>(bt_ptrs[g]);
    const int64_t sb = sb_top - g;
    for (int i = t; i < W; i += blockDim.x) gbt2[(int64_t)g * W + i] = (i == blk) ? (int32_t)sb : bt[i];
    if (t == 0) {
      const int64_t rb = bt[blk];
      int64_t* st = gstate + (int64_t)g * 8;
      st[0] = rb; st[1] = n; st[4] = rb; st[5] = n;
      gseq[g] = (int32_t)(pos1 + 1);
      if (s_mode) slot_map[(int64_t)g * slot_stride + leaf_row] = sb * bs + n;
    }
  }
}

// ---- fused commit after rejection sampling ----
template <typename TS, typename TN>
__global__ void leaf_commit_kernel(TS* sampled, TN* num_sampled, TN* num_rejected, const int64_t* u, const int64_t* alt,
                                   const bool* mode, int64_t* prev_ns, bool* hit, bool* hit_prev, int64_t* gstate, int G,
                                   int64_t* stats, int nohit) {
  if (threadIdx.x != 0) return;
  const bool m = mode[0];
  TN ns = num_sampled[0], nr = num_rejected[0];
  const bool clamp = m && ns == 7;
  if (clamp) { ns -= 1; nr += 1; }
  bool h = m && ns == 1 && (int64_t)sampled[0] == alt[1] && !nohit;
  if (h) { sampled[1] = (TS)u[0]; ns += 1; nr -= 1; }
  num_sampled[0] = ns; num_rejected[0] = nr;
  hit[0] = h; hit_prev[0] = h; prev_ns[0] = (int64_t)ns;
  for (int g = 0; g < G; g++) {
    int64_t* st = gstate + (int64_t)g * 8;
    st[2] = st[4]; st[3] = st[5];
  }
  stats[0] += 1; stats[1] += h; stats[2] += m; stats[3] += clamp;
}

void leaf_setup(torch::Tensor positions, torch::Tensor input_ids, torch::Tensor slot_map, torch::Tensor bt_ptrs,
                torch::Tensor gbt2, torch::Tensor gseq, torch::Tensor gstate, int64_t G, int64_t bs, int64_t sb_top,
                torch::Tensor alt, torch::Tensor prev_ns, int64_t low, torch::Tensor mode, int64_t leaf_row) {
  TORCH_CHECK(input_ids.scalar_type() == torch::kInt32 && positions.scalar_type() == torch::kInt64);
  TORCH_CHECK(slot_map.scalar_type() == torch::kInt64 && slot_map.dim() == 2);
  leaf_setup_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      positions.data_ptr<int64_t>(), input_ids.data_ptr<int32_t>(), slot_map.data_ptr<int64_t>(), slot_map.stride(0),
      bt_ptrs.data_ptr<int64_t>(), gbt2.data_ptr<int32_t>(), (int)gbt2.size(1), gseq.data_ptr<int32_t>(),
      gstate.data_ptr<int64_t>(), (int)G, (int)bs, sb_top, alt.data_ptr<int64_t>(), prev_ns.data_ptr<int64_t>(),
      (int)low, mode.data_ptr<bool>(), (int)leaf_row);
}

void leaf_commit(torch::Tensor sampled, torch::Tensor num_sampled, torch::Tensor num_rejected, torch::Tensor u,
                 torch::Tensor alt, torch::Tensor mode, torch::Tensor prev_ns, torch::Tensor hit, torch::Tensor hit_prev,
                 torch::Tensor gstate, int64_t G, torch::Tensor stats, int64_t nohit) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(num_sampled.scalar_type() == num_rejected.scalar_type());
#define LC(TS_, TN_) leaf_commit_kernel<TS_, TN_><<<1, 32, 0, st>>>(sampled.data_ptr<TS_>(), num_sampled.data_ptr<TN_>(), \
      num_rejected.data_ptr<TN_>(), u.data_ptr<int64_t>(), alt.data_ptr<int64_t>(), mode.data_ptr<bool>(), \
      prev_ns.data_ptr<int64_t>(), hit.data_ptr<bool>(), hit_prev.data_ptr<bool>(), gstate.data_ptr<int64_t>(), (int)G, \
      stats.data_ptr<int64_t>(), (int)nohit)
  const bool s64 = sampled.scalar_type() == torch::kInt64, n64 = num_sampled.scalar_type() == torch::kInt64;
  if (s64 && n64) LC(int64_t, int64_t); else if (s64) LC(int64_t, int32_t); else if (n64) LC(int32_t, int64_t); else LC(int32_t, int32_t);
#undef LC
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &leaf_prep);
  m.def("select", &leaf_select);
  m.def("setup", &leaf_setup);
  m.def("commit", &leaf_commit);
}
