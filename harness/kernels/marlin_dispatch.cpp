#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

void launch_muse_marlin(const void*, const void*, const void*, const float*,
                        void*, void*, int*, int, int, int, bool, cudaStream_t);

at::Tensor decode(const at::Tensor& a, const at::Tensor& weight,
                  const at::Tensor& scales, const at::Tensor& global_scale,
                  at::Tensor& workspace, int64_t n, int64_t k, bool expanded) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 &&
                  a.dim() == 2 && a.size(0) == 1 && a.size(1) == k &&
                  a.is_contiguous(),
              "Marlin decode requires one contiguous BF16 input row");
  TORCH_CHECK(n > 0 && n % 64 == 0 && k > 0 && k % 128 == 0,
              "Marlin decode dimensions must match the stock 128-K tiles");
  TORCH_CHECK(weight.device() == a.device() && weight.is_contiguous() &&
                  weight.scalar_type() == at::kInt && weight.dim() == 2 &&
                  weight.size(0) == k / 16 && weight.size(1) == n * 2,
              "Marlin decode requires the stock packed NVFP4 layout");
  TORCH_CHECK(scales.device() == a.device() && scales.is_contiguous() &&
                  scales.scalar_type() == at::kFloat8_e4m3fn &&
                  scales.dim() == 2 && scales.size(0) == k / 16 &&
                  scales.size(1) == n,
              "Marlin decode requires the stock group-16 scale layout");
  TORCH_CHECK(global_scale.device() == a.device() &&
                  global_scale.scalar_type() == at::kFloat &&
                  global_scale.numel() == 1,
              "Marlin decode requires one FP32 global scale");
  c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp props;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, a.get_device()));
  TORCH_CHECK(props.major == 12 && props.minor == 1,
              "This Marlin decode kernel is compiled for SM121");
  const int sms = props.multiProcessorCount;
  TORCH_CHECK(workspace.device() == a.device() && workspace.is_contiguous() &&
                  workspace.scalar_type() == at::kInt &&
                  workspace.numel() >= sms,
              "Marlin decode requires the stock lock workspace");
  auto output = at::empty({1, n}, a.options());
  auto scratch = at::empty({sms * 16 * 128}, a.options().dtype(at::kFloat));
  launch_muse_marlin(
      a.const_data_ptr(), weight.const_data_ptr(), scales.const_data_ptr(),
      global_scale.const_data_ptr<float>(), output.mutable_data_ptr(),
      scratch.mutable_data_ptr(), workspace.mutable_data_ptr<int>(), n, k, sms,
      expanded, c10::cuda::getCurrentCUDAStream(a.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

TORCH_LIBRARY(muse_marlin, m) {
  m.def("decode(Tensor a, Tensor weight, Tensor scales, Tensor global_scale, "
        "Tensor(a!) workspace, int n, int k, bool expanded=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(muse_marlin, CUDA, m) { m.impl("decode", &decode); }
