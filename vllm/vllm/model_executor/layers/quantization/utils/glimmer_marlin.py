# SPDX-License-Identifier: Apache-2.0
"""Prefill NVFP4 W4A16 GEMM through glimmer_marlin_ext (harness/kernels/glimmer_marlin): vLLM's own Marlin kernel with an
exact tile-order remap when M is a multiple of 64 and >= 128 (every output tile is computed whole by one block, in the
same k order, so results are bit-identical to stock Marlin; only the order in which blocks visit tiles changes, which
lets the eight 64-row slabs of a 512-token prompt share each weight tile through L2). Every other shape (decode M = 1,
odd M) goes to stock ops.marlin_gemm. The shape test lives INSIDE this opaque custom op, so torch.compile never bakes
a branch on M into the graph."""
import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

try:
    import glimmer_marlin_ext as _ext  # built into KERNELS_OUT (first on the resident's PYTHONPATH after the overlay)
except Exception:  # extension absent: stock path everywhere
    _ext = None


@torch.library.custom_op("glimmer::nvfp4_marlin_gemm", mutates_args=())
def nvfp4_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor, b_scales: torch.Tensor,
                      global_scale: torch.Tensor, workspace: torch.Tensor, size_m: int, size_n: int,
                      size_k: int, use_atomic_add: bool, use_fp32_reduce: bool) -> torch.Tensor:
    if (_ext is not None and size_m >= 128 and size_m % 64 == 0 and a.dtype == torch.bfloat16
            and not use_atomic_add and use_fp32_reduce):
        return _ext.gemm(a, b_q_weight, b_scales, global_scale, workspace, size_m, size_n, size_k, use_fp32_reduce)
    return ops.marlin_gemm(a=a, c=None, b_q_weight=b_q_weight, b_bias=None, b_scales=b_scales, a_scales=None,
                           global_scale=global_scale, b_zeros=None, g_idx=None, perm=None, workspace=workspace,
                           b_q_type=scalar_types.float4_e2m1f, size_m=size_m, size_n=size_n, size_k=size_k,
                           use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce)


@nvfp4_marlin_gemm.register_fake
def _(a, b_q_weight, b_scales, global_scale, workspace, size_m, size_n, size_k, use_atomic_add, use_fp32_reduce):
    return a.new_empty((size_m, size_n))
