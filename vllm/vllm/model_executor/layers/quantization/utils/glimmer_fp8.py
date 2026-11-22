# SPDX-License-Identifier: Apache-2.0
"""Decode (M <= 16) FP8 w8a8 GEMMs through glimmer_fp8_ext (harness/kernels/glimmer_fp8): CUTLASS sm120 FP8 kernels whose
epilogue is vLLM's ScaledEpilogue arithmetic, D = bf16(scale_a * fp32(scale_b * acc)), in a 16x32x128 tile instead of vLLM's 16x64x128. The per-element
k order is the same for every tile shape (no split-K), so the output is bit-identical to ops.cutlass_scaled_mm. Only
per-tensor scales (both scalars), no bias, bf16 output and the shapes listed in _CFG take this path; everything else, and
every prefill, calls ops.cutlass_scaled_mm unchanged. The shape test is inside an opaque custom op."""
import torch

from vllm import _custom_ops as ops

try:
    import glimmer_fp8_ext as _ext  # built into KERNELS_OUT
except Exception:
    _ext = None

# (N, K) -> fp8x config id (0 = vLLM's own M16 config; 1..4 alternatives), chosen by quiet-GPU timing on GB10.
_CFG: dict[tuple[int, int], int] = {
    (4608, 6656): 2,   # qkv
    (4096, 6656): 2,   # attention output gate
    (6656, 4096): 2,   # o_proj
    (39936, 6656): 2,  # gate_up
    (6656, 19968): 2,  # down
}


_fp8z = []


def _fp8z_lookup(a, b, scale_a, scale_b):
    # LOCAL PATCH (glimmer dec2): one-row calls of layers with a verified fp8z
    # code (kernels/linear/scaled_mm/fp8z.py) read the 7-bit code instead of B.
    # Imported on first use: the scaled_mm package imports this module.
    if not _fp8z:
        try:
            from vllm.model_executor.kernels.linear.scaled_mm import fp8z
            _fp8z.append(fp8z.lookup)
        except Exception:  # noqa: BLE001
            _fp8z.append(None)
    return None if _fp8z[0] is None else _fp8z[0](a, b, scale_a, scale_b)


@torch.library.custom_op("glimmer::fp8_scaled_mm", mutates_args=())
def fp8_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    comp = _fp8z_lookup(a, b, scale_a, scale_b) if a.size(0) == 1 else None
    if comp is not None:
        return torch.ops.vllm.fp8z_gemm.default(a, comp, scale_a, scale_b, b.size(1), b.size(0))
    if (_ext is not None and 0 < a.size(0) <= 16
            and a.is_cuda and a.device == b.device == scale_a.device == scale_b.device
            and scale_a.numel() == scale_b.numel() == 1
            and scale_a.dtype == scale_b.dtype == torch.float32
            and a.dtype == b.dtype == torch.float8_e4m3fn
            and a.is_contiguous() and b.stride(0) == 1 and b.stride(1) == b.size(0)):
        cfg = _CFG.get((b.size(1), b.size(0)))
        if cfg is not None:
            return _ext.gemm(a, b, scale_a.reshape(1), scale_b.reshape(1), cfg)
    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None)


@fp8_scaled_mm.register_fake
def _(a, b, scale_a, scale_b):
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.bfloat16)
