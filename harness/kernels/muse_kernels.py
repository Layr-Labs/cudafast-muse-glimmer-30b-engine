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
    def _fake_down(a, b_q, b_s, g_scale, partials):
        return torch.empty((a.shape[0], 6656), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_gate_up_silu_m1")
    def _fake_gate_up_silu_m1(a, b_q, b_s, g_scale):
        return torch.empty((a.shape[0], 19968), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::silu_mul_mode0")
    def _fake_silu_mul_mode0(gate_up):
        return torch.empty((gate_up.shape[0], gate_up.shape[1] // 2), dtype=gate_up.dtype, device=gate_up.device)

    @torch.library.register_fake("muse_fast::nvfp4_lmhead_m1")
    def _fake_lmhead(a, b_q, b_s, g_scale):
        return torch.empty((a.shape[0], 202048), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_prefill_bm64_bn128")
    def _fake_prefill(a, b_q, b_s, g_scale, n):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_gate_up_silu_m512")
    def _fake_gate_up_silu_m512(a, b_q, b_s, g_scale, half_n):
        return torch.empty((a.shape[0], half_n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_gate_up_silu_fp8quant_m512")
    def _fake_gate_up_silu_fp8quant_m512(a, b_q, b_s, g_scale, out_scale_a, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.float8_e4m3fn, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_down_m512_lpt")
    def _fake_down_m512_lpt(a, b_q, b_s, g_scale, task_list, reduce_list, partials, n):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_lmhead_m1_argmax_step_update")
    def _fake_lmhead_argmax_step(
        a,
        b_q,
        b_s,
        g_scale,
        output_multiplier,
        softcap,
        blk_max_val,
        blk_max_idx,
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

    @torch.library.register_fake("muse_fast::repack_nvfp4_m1")
    def _fake_repack_nvfp4_m1(b_q, b_s, n, k):
        num_n64 = n // 64
        k_pairs = k // 256
        return (
            torch.empty((num_n64 * k_pairs * 4 * 128 * 4,), dtype=torch.int32, device=b_q.device),
            torch.empty((num_n64 * k_pairs * 4 * 16 * 4,), dtype=torch.int32, device=b_q.device),
            torch.empty((num_n64 * k_pairs * 4,), dtype=torch.uint8, device=b_q.device),
            torch.empty((num_n64,), dtype=torch.int64, device=b_q.device),
            torch.empty((num_n64,), dtype=torch.uint8, device=b_q.device),
        )

    @torch.library.register_fake("muse_fast::nvfp4_tm_linear_m1")
    def _fake_nvfp4_tm_linear_m1(a, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode, g_scale, n, n_grp, p1_cols, p2_iters, grid, mode_mask):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_tm_gate_up_silu_m1")
    def _fake_nvfp4_tm_gate_up_silu_m1(a, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode, g_scale, mode_mask):
        return torch.empty((a.shape[0], 19968), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_tm_lmhead_m1_argmax_step_update")
    def _fake_nvfp4_tm_lmhead_argmax_step(
        a,
        bq_tm,
        bs_tm,
        b_base,
        tile_s_off,
        tile_s_mode,
        g_scale,
        output_multiplier,
        softcap,
        blk_max_val,
        blk_max_idx,
        out_tokens,
        step_idx,
        step_limit,
        input_ids,
        positions,
        seq_lens,
        input_block_table_ptrs,
        block_sizes,
        slot_mappings,
        mode_mask,
    ):
        return None

    @torch.library.register_fake("muse_fast::repack_nvfp4_rans")
    def _fake_repack_nvfp4_rans(b_q, b_s, n, k, n_grp, p1_cols, p2_iters):
        use_w16 = (n == 39936 and k == 6656 and n_grp == 64 and p1_cols == 576 and p2_iters == 26)
        num_streams = 1248 if use_w16 else ((n >> 6) * 4)
        bq_steps = 104 if use_w16 else (k >> 7)
        return (
            torch.empty((num_streams * bq_steps * 64 * 4,), dtype=torch.int32, device=b_q.device),
            torch.empty((num_streams * 256,), dtype=torch.int16, device=b_q.device),
            torch.empty((num_streams + 1,), dtype=torch.int64, device=b_q.device),
            torch.empty((num_streams, 128), dtype=torch.int32, device=b_q.device),
            torch.empty((1024,), dtype=torch.int32, device=b_q.device),
        )

    @torch.library.register_fake("muse_fast::nvfp4_rans_linear_m1")
    def _fake_nvfp4_rans_linear_m1(a, bq_tm, sc_stream, stream_chunk_off, init_states, lut, g_scale, n, n_grp, p1_cols, p2_iters, grid):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::repack_nvfp4_rans_splitk96")
    def _fake_repack_nvfp4_rans_splitk96(b_q, b_s, n, k, iters):
        if iters <= 64:
            num_streams = 192
            n_pairs = (iters + 1) >> 1
            bq_len = n_pairs * 192 * 256 * 4
        else:
            num_streams = 192 * 4
            bq_len = num_streams * iters * 32 * 4
        return (
            torch.empty((bq_len,), dtype=torch.int32, device=b_q.device),
            torch.empty((num_streams * 256,), dtype=torch.int16, device=b_q.device),
            torch.empty((num_streams + 1,), dtype=torch.int64, device=b_q.device),
            torch.empty((num_streams, 128), dtype=torch.int32, device=b_q.device),
            torch.empty((1024,), dtype=torch.int32, device=b_q.device),
        )

    @torch.library.register_fake("muse_fast::nvfp4_rans_splitk96_m1")
    def _fake_nvfp4_rans_splitk96_m1(a, bq_tm, sc_stream, stream_chunk_off, init_states, lut, g_scale, partials, n, iters):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_rans_gate_up_silu_m1")
    def _fake_nvfp4_rans_gate_up_silu_m1(a, bq_tm, sc_stream, stream_chunk_off, init_states, lut, g_scale):
        return torch.empty((a.shape[0], 19968), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_rans_gate_up_silu_fp8quant_m1")
    def _fake_nvfp4_rans_gate_up_silu_fp8quant_m1(a, bq_tm, sc_stream, stream_chunk_off, init_states, lut, g_scale, out_scale_a):
        return torch.empty((a.shape[0], 19968), dtype=torch.float8_e4m3fn, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_rans_lmhead_m1_argmax_step_update")
    def _fake_nvfp4_rans_lmhead_argmax_step(
        a,
        bq_tm,
        sc_stream,
        stream_chunk_off,
        init_states,
        lut,
        g_scale,
        output_multiplier,
        softcap,
        blk_max_val,
        blk_max_idx,
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

    @torch.library.register_fake("muse_fast::repack_fp8_mma")
    def _fake_repack_fp8(b_colmajor):
        k, n = b_colmajor.shape
        return torch.empty(((n >> 3) * (k >> 6) * 32 * 4,), dtype=torch.int32, device=b_colmajor.device)

    @torch.library.register_fake("muse_fast::fp8_mma_repacked_m1")
    def _fake_fp8_m1(a, b_pack, scale_a, scale_b, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::repack_fp8_rans")
    def _fake_repack_fp8_rans(b_colmajor):
        k, n = b_colmajor.shape
        num_n8 = n >> 3
        k_pairs = k >> 7
        sm_tiles = (96 * 9) if (k >= 16384 and n == 6656) else num_n8
        return (
            torch.empty((sm_tiles * k_pairs * 32 * 4,), dtype=torch.int32, device=b_colmajor.device),
            torch.empty((num_n8 * 256,), dtype=torch.int16, device=b_colmajor.device),
            torch.empty((num_n8 + 1,), dtype=torch.int64, device=b_colmajor.device),
            torch.empty((num_n8, 128), dtype=torch.int32, device=b_colmajor.device),
            torch.empty((2048,), dtype=torch.int32, device=b_colmajor.device),
        )

    @torch.library.register_fake("muse_fast::fp8_rans_m1")
    def _fake_fp8_rans_m1(a, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::fp8_rans_gate_up_silu_m1")
    def _fake_fp8_rans_gate_up_silu_m1(a, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::fp8_rans_gate_up_silu_fp8quant_m1")
    def _fake_fp8_rans_gate_up_silu_fp8quant_m1(a, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, out_scale_a, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.float8_e4m3fn, device=a.device)

    @torch.library.register_fake("muse_fast::fp8_rans_sigmoid_mul_m1")
    def _fake_fp8_rans_sigmoid_mul_m1(a, g, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::sigmoid_mul_fp8quant_exact")
    def _fake_sigmoid_mul_fp8quant_exact(attn_out, gate, scale_a):
        return torch.empty_like(attn_out, dtype=torch.float8_e4m3fn)

    @torch.library.register_fake("muse_fast::repack_bf16_rans")
    def _fake_repack_bf16_rans(b_rowmajor):
        n, k = b_rowmajor.shape
        num_n4 = n >> 2
        k_steps = k >> 9
        return (
            torch.empty((num_n4 * 4 * k_steps * 32 * 4,), dtype=torch.int32, device=b_rowmajor.device),
            torch.empty((num_n4 * 128,), dtype=torch.int16, device=b_rowmajor.device),
            torch.empty((num_n4 + 1,), dtype=torch.int64, device=b_rowmajor.device),
            torch.empty((num_n4, 128), dtype=torch.int32, device=b_rowmajor.device),
            torch.empty((1024,), dtype=torch.int32, device=b_rowmajor.device),
        )

    @torch.library.register_fake("muse_fast::bf16_rans_m1")
    def _fake_bf16_rans_m1(a, sm_pack, exp_stream, tile_chunk_off, init_states, lut, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::quant_fp8_triton_exact")
    def _fake_quant_fp8_triton_exact(x, scale_a):
        return torch.empty_like(x, dtype=torch.float8_e4m3fn)

    @torch.library.register_fake("muse_fast::nvfp4_rans_lmhead_m1_argmax_only")
    def _fake_nvfp4_rans_lmhead_m1_argmax_only(
        a, bq_tm, sc_stream, stream_chunk_off, init_states, lut, g_scale,
        output_multiplier, softcap, blk_max_val, blk_max_idx, out_token,
    ):
        return None

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

    @torch.library.register_fake("muse_fast::silu_mul_fp8quant_mode0")
    def _fake_silu_mul_fp8quant_mode0(gate_up, out_scale_a):
        return torch.empty((gate_up.shape[0], gate_up.shape[1] // 2), dtype=torch.float8_e4m3fn, device=gate_up.device)

    @torch.library.register_fake("muse_fast::repack_nvfp4_m512_contig")
    def _fake_repack_nvfp4_m512_contig(b_q, b_s, n, k, is_paired):
        return torch.empty(((n // 128) * (k // 64) * 320 * 4,), dtype=torch.int32, device=b_q.device)

    @torch.library.register_fake("muse_fast::nvfp4_gate_up_silu_contig_m512")
    def _fake_nvfp4_gate_up_silu_contig_m512(a, b_pack, g_scale, half_n):
        return torch.empty((a.shape[0], half_n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_gate_up_silu_fp8quant_contig_m512")
    def _fake_nvfp4_gate_up_silu_fp8quant_contig_m512(a, b_pack, g_scale, out_scale_a, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.float8_e4m3fn, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_lpt_contig_m512")
    def _fake_nvfp4_lpt_contig_m512(a, b_pack, g_scale, task_list, reduce_list, partials, n):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::nvfp4_prefill_contig_bm64_bn128")
    def _fake_nvfp4_prefill_contig_bm64_bn128(a, b_pack, g_scale, n):
        return torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)

    @torch.library.register_fake("muse_fast::repack_fp8_m512_contig")
    def _fake_repack_fp8_m512_contig(b_colmajor, is_paired):
        k, n = b_colmajor.shape
        return torch.empty(((n // 128) * (k // 64) * 512 * 4,), dtype=torch.int32, device=b_colmajor.device)

    @torch.library.register_fake("muse_fast::fp8_gate_up_silu_contig_m512")
    def _fake_fp8_gate_up_silu_contig_m512(a, b_pack, scale_a, scale_b, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::fp8_gate_up_silu_fp8quant_contig_m512")
    def _fake_fp8_gate_up_silu_fp8quant_contig_m512(a, b_pack, scale_a, scale_b, out_scale_a, half_n):
        return torch.empty((a.shape[0], half_n), dtype=torch.float8_e4m3fn, device=a.device)

    @torch.library.register_fake("muse_fast::fp8_linear_contig_m512")
    def _fake_fp8_linear_contig_m512(a, b_pack, scale_a, scale_b, n):
        return torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)

    @torch.library.register_fake("muse_fast::gb10_probe_bw_u4")
    def _fake_gb10_probe_bw_u4(buf, sink):
        return None

    @torch.library.register_fake("muse_fast::l2_prefetch_rans_startup")
    def _fake_l2_prefetch_rans_startup(bq_or_sm, stream_buf, chunk_off, init_states, lut, num_active, bq_stride_u4, bq_first_u4):
        return None

    @torch.library.register_fake("muse_fast::sw_rmsnorm_bf16_exact")
    def _fake_sw_rmsnorm_bf16_exact(x, res, w_post, w_pre, eps_post, eps_pre, mode):
        return torch.empty_like(x), torch.empty_like(res)

    @torch.library.register_fake("muse_fast::final_rmsnorm_bf16_exact")
    def _fake_final_rmsnorm_bf16_exact(x, res, w_post, w_final, eps_post, eps_final, mode):
        return torch.empty_like(x)

    @torch.library.register_fake("muse_fast::sw_rmsnorm_fp8_exact")
    def _fake_sw_rmsnorm_fp8_exact(x, res, w_post, w_pre, scale_a, eps_post, eps_pre, mode):
        return torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty_like(res)

    @torch.library.register_fake("muse_fast::qk_norm_rope_exact")
    def _fake_qk_norm_rope_exact(qkv, positions, cos_sin, scale_q, eps, use_rope, q_mode, k_mode):
        m = qkv.shape[0]
        return (
            torch.empty((m, 4096), dtype=qkv.dtype, device=qkv.device),
            torch.empty((m, 256), dtype=qkv.dtype, device=qkv.device),
        )

    _AVAILABLE = True
    return True
