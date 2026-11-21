"""Decode (M <= 16) FP8 w8a8 GEMMs through glimmer_fp8_ext (harness/kernels/glimmer_fp8): CUTLASS sm120 FP8 kernels whose
epilogue is vLLM's ScaledEpilogue arithmetic, D = bf16(scale_a * fp32(scale_b * acc)), in a 16x32x128 tile instead of vLLM's 16x64x128. The per-element
k order is the same for every tile shape (no split-K), so the output is bit-identical to ops.cutlass_scaled_mm. Only
per-tensor scales (both scalars), no bias, bf16 output and the shapes listed in _CFG take this path; everything else, and
every prefill, calls ops.cutlass_scaled_mm unchanged. The shape test is inside an opaque custom op."""
import torch

from vllm import _custom_ops as ops

try:
    import glimmer_fp8_ext as _ext
except Exception:
    _ext = None

_CFG: dict[tuple[int, int], int] = {
    (4608, 6656): 2,
    (4096, 6656): 2,
    (6656, 4096): 2,
    (39936, 6656): 2,
    (6656, 19968): 2,
}


@torch.library.custom_op("glimmer::fp8_scaled_mm", mutates_args=())
def fp8_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if (_ext is not None and a.size(0) <= 16 and scale_a.numel() == 1 and scale_b.numel() == 1
            and a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
            and a.is_contiguous() and b.stride(0) == 1 and b.stride(1) == b.size(0)):
        cfg = _CFG.get((b.size(1), b.size(0)))
        if cfg is not None:
            return _ext.gemm(a, b, scale_a.reshape(1), scale_b.reshape(1), cfg)
    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None)


@fp8_scaled_mm.register_fake
def _(a, b, scale_a, scale_b):
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.bfloat16)
