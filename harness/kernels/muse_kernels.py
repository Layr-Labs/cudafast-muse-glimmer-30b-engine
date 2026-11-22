"""Loader for custom bit-exact SM121 CUDA kernels compiled by harness/kernels/build.sh."""

from __future__ import annotations

import os
import torch

_LOADED = False
_AVAILABLE = False


def load_muse_kernels() -> bool:
    """Load _muse_kernels.so and register FakeTensor implementations once."""
    global _LOADED, _AVAILABLE
    if _LOADED:
        return _AVAILABLE
    _LOADED = True
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_muse_kernels.so")
    if not os.path.isfile(so_path):
        _AVAILABLE = False
        return False
    torch.ops.load_library(so_path)

    @torch.library.register_fake("muse_fast::nvfp4_down_m1")
    def _fake_down(a, b_q, b_s, g_scale):
        return torch.empty((a.shape[0], 6656), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_lmhead_m1")
    def _fake_lmhead(a, b_q, b_s, g_scale):
        return torch.empty((a.shape[0], 202048), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_prefill_bm64_bn128")
    def _fake_prefill(a, b_q, b_s, g_scale, n):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::repack_fp8_mma")
    def _fake_repack_fp8(b_colmajor):
        k, n = b_colmajor.shape
        return torch.empty(((n >> 3) * (k >> 6) * 32 * 4,), dtype=torch.int32, device=b_colmajor.device)

    @torch.library.register_fake("muse_fast::fp8_mma_repacked_m1")
    def _fake_fp8_m1(a, b_pack, scale_a, scale_b, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::init_decode_loop")
    def _fake_init_decode_loop(
        first_token,
        start_pos,
        count,
        step_idx,
        step_limit,
        input_ids,
        positions,
        seq_lens,
        query_start_loc,
        input_block_table_ptrs,
        block_sizes,
        slot_mappings,
    ):
        return None

    @torch.library.register_fake("muse_fast::greedy_argmax_step_update")
    def _fake_greedy_argmax_step_update(
        logits,
        out_tokens,
        step_idx,
        step_limit,
        input_ids,
        positions,
        seq_lens,
        input_block_table_ptrs,
        block_sizes,
        slot_mappings,
    ):
        return None

    _AVAILABLE = True
    return True
