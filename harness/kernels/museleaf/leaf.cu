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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &leaf_prep);
  m.def("select", &leaf_select);
}
