# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# Registry of NVFP4 layers handled by custom bit-exact SM121 CUDA kernels:
#  - gate_up (N=39936, K=6656): streaming M=1 decode (exact part1/part2) and
#    column-major L2-resident BM=64, BN=256 prefill (exact zero-split-K when
#    M >= 256 and M % 256 == 0, with zero duplicate weight copies).
#  - down (N=6656, K=19968): chunked-SMEM 96-block M=1 decode (exact iters=169).
#  - lm_head (N=202048, K=6656): streaming M=1 decode (exact part1/part2).
_MUSE_KERNELS_READY = False
try:
    from muse_kernels import load_muse_kernels as _load_muse_kernels

    _MUSE_KERNELS_READY = _load_muse_kernels()
except Exception:
    _MUSE_KERNELS_READY = False

_FAST_GATE_UP_LAYERS: list[torch.nn.Module] = []
_FAST_DOWN_LAYERS: list[torch.nn.Module] = []
_FAST_LMHEAD_LAYERS: list[torch.nn.Module] = []
_FAST_QKV_LAYERS: list[torch.nn.Module] = []
_FAST_GATE_LAYERS: list[torch.nn.Module] = []
_FAST_OPROJ_LAYERS: list[torch.nn.Module] = []
_MUSE_SINGLE_REQ_P512_MODE: bool = False
_M512_LPT_SCHED_CACHE: dict[tuple[torch.device, int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_M512_LPT_M7_SCHED_CACHE: dict[tuple[torch.device, int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_m512_lpt_sched_m7(
    device: torch.device,
    n: int,
    k: int,
    p1: int,
    p2_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, n, k, p1, p2_iters)
    sched = _M512_LPT_M7_SCHED_CACHE.get(key)
    if sched is not None:
        return sched
    k_stages = k // 64
    n256_cnt = n // 256
    tasks: list[tuple[int, int, int, int, int, int]] = []
    rtasks: list[tuple[int, int, int, int]] = []
    p_idx = 0
    m_blk = 7
    for n_blk in range(n // 128):
        n256 = n_blk // 2
        tile_id = m_blk * n256_cnt + n256
        if tile_id < p1:
            tasks.append((0, n_blk, 0, k_stages, -1, 0))
        else:
            local = tile_id - p1
            cs = local * k_stages
            ce = cs + k_stages
            b0 = cs // p2_iters
            b1 = (ce - 1) // p2_iters
            if b0 == b1:
                tasks.append((0, n_blk, 0, k_stages, -1, 0))
            else:
                ps = p_idx
                for b in range(b0, b1 + 1):
                    tasks.append(
                        (
                            0,
                            n_blk,
                            max(cs, b * p2_iters) - cs,
                            min(ce, (b + 1) * p2_iters) - cs,
                            p_idx,
                            0,
                        )
                    )
                    p_idx += 1
                rtasks.append((0, n_blk, ps, b1 - b0 + 1))
    tasks.sort(key=lambda t: (-(t[3] - t[2]), t[1], t[0]))
    tt = torch.tensor([v for r in tasks for v in r], dtype=torch.int16, device=device)
    rt = torch.tensor([v for r in rtasks for v in r], dtype=torch.int16, device=device)
    partials = torch.empty((max(p_idx, 1), 8192), dtype=torch.float32, device=device)
    sched = (tt, rt, partials)
    _M512_LPT_M7_SCHED_CACHE[key] = sched
    return sched


def _get_m512_lpt_sched(
    device: torch.device,
    n: int,
    k: int,
    p1: int,
    p2_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, n, k, p1, p2_iters)
    sched = _M512_LPT_SCHED_CACHE.get(key)
    if sched is not None:
        return sched
    k_stages = k // 64
    n256_cnt = n // 256
    tasks: list[tuple[int, int, int, int, int, int]] = []
    rtasks: list[tuple[int, int, int, int]] = []
    p_idx = 0
    for n_blk in range(n // 128):
        n256 = n_blk // 2
        for m_blk in range(8):
            tile_id = m_blk * n256_cnt + n256
            if tile_id < p1:
                tasks.append((m_blk, n_blk, 0, k_stages, -1, 0))
            else:
                local = tile_id - p1
                cs = local * k_stages
                ce = cs + k_stages
                b0 = cs // p2_iters
                b1 = (ce - 1) // p2_iters
                if b0 == b1:
                    tasks.append((m_blk, n_blk, 0, k_stages, -1, 0))
                else:
                    ps = p_idx
                    for b in range(b0, b1 + 1):
                        tasks.append(
                            (
                                m_blk,
                                n_blk,
                                max(cs, b * p2_iters) - cs,
                                min(ce, (b + 1) * p2_iters) - cs,
                                p_idx,
                                0,
                            )
                        )
                        p_idx += 1
                    rtasks.append((m_blk, n_blk, ps, b1 - b0 + 1))
    tasks.sort(key=lambda t: (-(t[3] - t[2]), t[1], t[0]))
    tt = torch.tensor([v for r in tasks for v in r], dtype=torch.int16, device=device)
    rt = torch.tensor([v for r in rtasks for v in r], dtype=torch.int16, device=device)
    partials = torch.empty((max(p_idx, 1), 8192), dtype=torch.float32, device=device)
    sched = (tt, rt, partials)
    _M512_LPT_SCHED_CACHE[key] = sched
    return sched


_SK_PARTIALS_CACHE: dict[torch.device, torch.Tensor] = {}
_SK_GATE_PARTIALS_CACHE: dict[torch.device, torch.Tensor] = {}
_SLAB_PLACEMENT_DONE: bool = False


def run_gb10_cold_slab_placement(device: torch.device) -> None:
    global _SLAB_PLACEMENT_DONE
    if _SLAB_PLACEMENT_DONE or not _MUSE_KERNELS_READY:
        return
    _SLAB_PLACEMENT_DONE = True
    try:
        import sys
        from vllm.model_executor.kernels.linear.scaled_mm.cutlass import _FAST_FP8_PACKS

        evict_buf = torch.ones((12 * 1024 * 1024,), dtype=torch.int32, device=device)
        sink = torch.zeros((4,), dtype=torch.int32, device=device)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)

        def _probe_cold_ms(buf: torch.Tensor) -> float:
            best = 1e9
            for _ in range(2):
                torch.ops.muse_fast.gb10_probe_bw_u4(evict_buf, sink)
                ev0.record()
                torch.ops.muse_fast.gb10_probe_bw_u4(buf, sink)
                ev1.record()
                ev1.synchronize()
                t = ev0.elapsed_time(ev1)
                if t < best:
                    best = t
            return best

        # Warm probe kernel once
        _ = _probe_cold_ms(evict_buf[:1024])

        total_swapped = 0
        total_saved_ms = 0.0

        def _place_group(items: list[tuple[torch.Tensor, object]], n_spares: int) -> None:
            nonlocal total_swapped, total_saved_ms
            if not items or n_spares <= 0:
                return
            sample = items[0][0]
            spares = [torch.empty_like(sample) for _ in range(n_spares)]
            for sp in spares:
                sp.fill_(1)
            torch.cuda.synchronize(device)
            orig_times = [(_probe_cold_ms(buf), idx) for idx, (buf, _) in enumerate(items)]
            spare_times = [(_probe_cold_ms(sp), s_idx) for s_idx, sp in enumerate(spares)]
            orig_times.sort(key=lambda x: -x[0])
            spare_times.sort(key=lambda x: x[0])
            n_pair = min(len(orig_times), len(spare_times))
            for k_i in range(n_pair):
                o_ms, o_idx = orig_times[k_i]
                s_ms, s_idx = spare_times[k_i]
                if s_ms + 0.002 < o_ms:
                    orig_buf, setter = items[o_idx]
                    sp = spares[s_idx]
                    sp.copy_(orig_buf)
                    if torch.equal(sp, orig_buf):
                        setter(sp)
                        total_swapped += 1
                        total_saved_ms += o_ms - s_ms
            del spares
            torch.cuda.empty_cache()

        gu_items = []
        for lyr in _FAST_GATE_UP_LAYERS:
            rp = getattr(lyr, "_muse_rans", None)
            if rp is not None:
                gu_items.append((rp[0], lambda sp, l=lyr: setattr(l, "_muse_rans", (sp, *l._muse_rans[1:]))))
        _place_group(gu_items, 16)

        dn_items = []
        for lyr in _FAST_DOWN_LAYERS:
            sk = getattr(lyr, "_muse_sk_rans", None)
            if sk is not None:
                dn_items.append((sk[0], lambda sp, l=lyr: setattr(l, "_muse_sk_rans", (sp, *l._muse_sk_rans[1:]))))
        _place_group(dn_items, 16)

        lm_items = []
        for lyr in _FAST_LMHEAD_LAYERS:
            rp = getattr(lyr, "_muse_rans", None)
            if rp is not None:
                lm_items.append((rp[0], lambda sp, l=lyr: setattr(l, "_muse_rans", (sp, *l._muse_rans[1:]))))
        _place_group(lm_items, 3)

        fp8_gu_items = []
        fp8_dn_items = []
        for p_idx, entry in enumerate(_FAST_FP8_PACKS):
            orig_ptr, payload, n_out, is_rans, m512_p = entry
            if is_rans and isinstance(payload, tuple):
                sm_p = payload[0]
                if n_out == 39936:
                    fp8_gu_items.append((
                        sm_p,
                        lambda sp, i=p_idx: _FAST_FP8_PACKS.__setitem__(
                            i, (_FAST_FP8_PACKS[i][0], (sp, *_FAST_FP8_PACKS[i][1][1:]), _FAST_FP8_PACKS[i][2], True, _FAST_FP8_PACKS[i][4])
                        ),
                    ))
                elif n_out == 6656 and sm_p.numel() > 8_000_000:
                    fp8_dn_items.append((
                        sm_p,
                        lambda sp, i=p_idx: _FAST_FP8_PACKS.__setitem__(
                            i, (_FAST_FP8_PACKS[i][0], (sp, *_FAST_FP8_PACKS[i][1][1:]), _FAST_FP8_PACKS[i][2], True, _FAST_FP8_PACKS[i][4])
                        ),
                    ))
        _place_group(fp8_gu_items, 6)
        _place_group(fp8_dn_items, 8)

        del evict_buf, sink
        torch.cuda.empty_cache()
        print(
            f"resident: gb10 cold slab placement swapped {total_swapped} buffers (saved {total_saved_ms:.3f} ms raw)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        import sys
        print(f"resident: gb10 slab placement skipped: {exc}", file=sys.stderr, flush=True)


def _get_sk_partials(device: torch.device) -> torch.Tensor:
    buf = _SK_PARTIALS_CACHE.get(device)
    if buf is None:
        buf = torch.zeros((4, 6656), dtype=torch.float32, device=device)
        _SK_PARTIALS_CACHE[device] = buf
    return buf


def _get_sk_gate_partials(device: torch.device) -> torch.Tensor:
    buf = _SK_GATE_PARTIALS_CACHE.get(device)
    if buf is None:
        buf = torch.zeros((4, 6656), dtype=torch.float32, device=device)
        _SK_GATE_PARTIALS_CACHE[device] = buf
    return buf


def _repack_nvfp4_for_m1(
    layer: torch.nn.Module,
    n: int,
    k: int,
    n_grp: int,
    p1_cols: int,
    p2_iters: int,
    grid: int,
    use_rans: bool = False,
    use_splitk_rans: bool = False,
) -> None:
    if use_splitk_rans:
        sk_pack = torch.ops.muse_fast.repack_nvfp4_rans_splitk96(
            layer.weight, layer.weight_scale, n, k, p2_iters
        )
        gen = torch.Generator(device=layer.weight.device)
        gen.manual_seed(12345)
        probe_a = torch.randn(
            (1, k), dtype=torch.bfloat16, device=layer.weight.device, generator=gen
        )
        ref = apply_fp4_marlin_linear(
            input=probe_a,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            bias=None,
        )
        bq_tm, sc_stream, stream_chunk_off, init_states, lut = sk_pack
        out = torch.ops.muse_fast.nvfp4_rans_splitk96_m1(
            probe_a,
            bq_tm,
            sc_stream,
            stream_chunk_off,
            init_states,
            lut,
            layer.weight_global_scale,
            _get_sk_partials(layer.weight.device),
            n,
            p2_iters,
        )
        if torch.equal(out, ref):
            layer._muse_sk_rans = sk_pack
            return
    if use_rans:
        rans_pack = torch.ops.muse_fast.repack_nvfp4_rans(
            layer.weight, layer.weight_scale, n, k, n_grp, p1_cols, p2_iters
        )
        gen = torch.Generator(device=layer.weight.device)
        gen.manual_seed(12345)
        probe_a = torch.randn((1, k), dtype=torch.bfloat16, device=layer.weight.device, generator=gen)
        ref = apply_fp4_marlin_linear(
            input=probe_a,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            bias=None,
        )
        bq_tm, sc_stream, stream_chunk_off, init_states, lut = rans_pack
        if n == 39936 and k == 6656 and init_states.shape[0] in (1248, 2496):
            out = torch.ops.muse_fast.nvfp4_rans_gate_up_silu_m1(
                probe_a, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                layer.weight_global_scale,
            )
            ref_silu = torch.ops.muse_fast.silu_mul_mode0(ref)
            if torch.equal(out, ref_silu):
                layer._muse_rans = rans_pack
                return
        else:
            out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                probe_a, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                layer.weight_global_scale, n, n_grp, p1_cols, p2_iters, grid,
            )
            if torch.equal(out, ref):
                layer._muse_rans = rans_pack
                return
    tm = torch.ops.muse_fast.repack_nvfp4_m1(layer.weight, layer.weight_scale, n, k)
    modes = tm[4]
    mask = 0
    if bool((modes == 5).any().item()):
        mask |= 1
    if bool((modes == 6).any().item()):
        mask |= 2
    if bool((modes == 8).any().item()):
        mask |= 4
    layer._muse_tm = tm
    layer._muse_tm_mask = mask


def _muse_nvfp4_gate_up_silu_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_GATE_UP_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    out_shape = x.shape[:-1] + (19968,)
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if m == 1:
            if not reshaped_x.is_contiguous():
                reshaped_x = reshaped_x.contiguous()
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None:
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_gate_up_silu_m1(
                    reshaped_x,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    layer.weight_global_scale,
                )
            else:
                tm = getattr(layer, "_muse_tm", None)
                if tm is not None:
                    bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                    out = torch.ops.muse_fast.nvfp4_tm_gate_up_silu_m1(
                        reshaped_x,
                        bq_tm,
                        bs_tm,
                        b_base,
                        tile_s_off,
                        tile_s_mode,
                        layer.weight_global_scale,
                        getattr(layer, "_muse_tm_mask", 7),
                    )
                else:
                    out = torch.ops.muse_fast.nvfp4_gate_up_silu_m1(
                        reshaped_x,
                        layer.weight,
                        layer.weight_scale,
                        layer.weight_global_scale,
                    )
            return out.reshape(out_shape)
        if m >= 256 and m % 256 == 0:
            if not reshaped_x.is_contiguous():
                reshaped_x = reshaped_x.contiguous()
            m512_pack = getattr(layer, "_muse_m512_pack", None)
            if m512_pack is not None:
                if (
                    _MUSE_SINGLE_REQ_P512_MODE
                    and m == 512
                    and layer_id == len(_FAST_GATE_UP_LAYERS) - 1
                ):
                    out = torch.empty((512, 19968), dtype=reshaped_x.dtype, device=reshaped_x.device)
                    out[448:512] = torch.ops.muse_fast.nvfp4_gate_up_silu_contig_m512(
                        reshaped_x[448:512],
                        m512_pack,
                        layer.weight_global_scale,
                        19968,
                    )
                    return out.reshape(out_shape)
                out = torch.ops.muse_fast.nvfp4_gate_up_silu_contig_m512(
                    reshaped_x,
                    m512_pack,
                    layer.weight_global_scale,
                    19968,
                )
            else:
                out = torch.ops.muse_fast.nvfp4_gate_up_silu_m512(
                    reshaped_x,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    19968,
                )
            return out.reshape(out_shape)

    gate_up = apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=39936,
        size_k=6656,
        bias=None,
    )
    gu_2d = gate_up.reshape(-1, 39936)
    if _MUSE_KERNELS_READY and gu_2d.dtype == torch.bfloat16:
        if not gu_2d.is_contiguous():
            gu_2d = gu_2d.contiguous()
        return torch.ops.muse_fast.silu_mul_mode0(gu_2d).reshape(out_shape)
    g = gu_2d[:, :19968].float()
    u = gu_2d[:, 19968:].float()
    return (g / (1.0 + torch.exp(-g)) * u).to(gate_up.dtype).reshape(out_shape)


def _muse_nvfp4_gate_up_silu_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (19968,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_gate_up_silu_fp8quant_impl(
    x: torch.Tensor, layer_id: int, out_scale_a: torch.Tensor
) -> torch.Tensor:
    layer = _FAST_GATE_UP_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    out_shape = x.shape[:-1] + (19968,)
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if m == 1:
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_gate_up_silu_fp8quant_m1(
                    reshaped_x,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    layer.weight_global_scale,
                    out_scale_a,
                )
                return out.reshape(out_shape)
        if m >= 256 and m % 256 == 0:
            if not reshaped_x.is_contiguous():
                reshaped_x = reshaped_x.contiguous()
            m512_pack = getattr(layer, "_muse_m512_pack", None)
            if m512_pack is not None:
                out = torch.ops.muse_fast.nvfp4_gate_up_silu_fp8quant_contig_m512(
                    reshaped_x,
                    m512_pack,
                    layer.weight_global_scale,
                    out_scale_a,
                    19968,
                )
            else:
                out = torch.ops.muse_fast.nvfp4_gate_up_silu_fp8quant_m512(
                    reshaped_x,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    out_scale_a,
                    19968,
                )
            return out.reshape(out_shape)

    gate_up = apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=39936,
        size_k=6656,
        bias=None,
    )
    gu_2d = gate_up.reshape(-1, 39936)
    g = gu_2d[:, :19968].float()
    u = gu_2d[:, 19968:].float()
    prod = (g / (1.0 + torch.exp(-g))) * u
    q = (prod * (1.0 / out_scale_a.float())).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(out_shape)


def _muse_nvfp4_gate_up_silu_fp8quant_fake(
    x: torch.Tensor, layer_id: int, out_scale_a: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        x.shape[:-1] + (19968,), dtype=torch.float8_e4m3fn, device=x.device
    )


def _muse_nvfp4_gate_up_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_GATE_UP_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    out_shape = x.shape[:-1] + (39936,)
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if m == 1:
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None and rp[3].shape[0] == 2496:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                    reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                    layer.weight_global_scale, 39936, 64, 576, 26, 156,
                )
                return out.reshape(out_shape)
            tm = getattr(layer, "_muse_tm", None)
            if tm is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                    reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                    layer.weight_global_scale, 39936, 64, 576, 26, 156,
                    getattr(layer, "_muse_tm_mask", 7),
                )
                return out.reshape(out_shape)
        if m >= 256 and m % 256 == 0:
            if not reshaped_x.is_contiguous():
                reshaped_x = reshaped_x.contiguous()
            out = torch.ops.muse_fast.nvfp4_prefill_bm64_bn128(
                reshaped_x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                39936,
            )
            return out.reshape(out_shape)

    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=39936,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_gate_up_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (39936,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_qkv_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_QKV_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    out_shape = x.shape[:-1] + (4608,)
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if m == 1:
            sk = getattr(layer, "_muse_sk_rans", None)
            if sk is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = sk
                out = torch.ops.muse_fast.nvfp4_rans_splitk96_m1(
                    reshaped_x,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    layer.weight_global_scale,
                    _get_sk_partials(reshaped_x.device),
                    4608,
                    39,
                )
                return out.reshape(out_shape)
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                    reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                    layer.weight_global_scale, 4608, 128, 0, 39, 72,
                )
                return out.reshape(out_shape)
            tm = getattr(layer, "_muse_tm", None)
            if tm is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                    reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                    layer.weight_global_scale, 4608, 128, 0, 39, 72,
                    getattr(layer, "_muse_tm_mask", 7),
                )
                return out.reshape(out_shape)
        if m >= 512 and m % 512 == 0:
            if not reshaped_x.is_contiguous():
                reshaped_x = reshaped_x.contiguous()
            m512_pack = getattr(layer, "_muse_m512_pack", None)
            if m512_pack is not None:
                out = torch.ops.muse_fast.nvfp4_prefill_contig_bm64_bn128(
                    reshaped_x,
                    m512_pack,
                    layer.weight_global_scale,
                    4608,
                )
            else:
                out = torch.ops.muse_fast.nvfp4_prefill_bm64_bn128(
                    reshaped_x,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    4608,
                )
            return out.reshape(out_shape)

    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=4608,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_qkv_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (4608,), dtype=x.dtype, device=x.device)


_DOWN_M1_PARTIALS_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_down_m1_partials(device: torch.device) -> torch.Tensor:
    buf = _DOWN_M1_PARTIALS_CACHE.get(device)
    if buf is None:
        buf = torch.empty((2, 6656), dtype=torch.float32, device=device)
        _DOWN_M1_PARTIALS_CACHE[device] = buf
    return buf


def _muse_nvfp4_down_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_DOWN_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    m = reshaped_x.shape[0]
    if (
        m == 1
        and _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        sk = getattr(layer, "_muse_sk_rans", None)
        if sk is not None:
            bq_tm, sc_stream, stream_chunk_off, init_states, lut = sk
            out = torch.ops.muse_fast.nvfp4_rans_splitk96_m1(
                reshaped_x,
                bq_tm,
                sc_stream,
                stream_chunk_off,
                init_states,
                lut,
                layer.weight_global_scale,
                _get_sk_partials(reshaped_x.device),
                6656,
                169,
            )
            return out.reshape(x.shape[:-1] + (6656,))
        rp = getattr(layer, "_muse_rans", None)
        if rp is not None:
            bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
            out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                layer.weight_global_scale, 6656, 128, 0, 169, 104,
            )
            return out.reshape(x.shape[:-1] + (6656,))
        tm = getattr(layer, "_muse_tm", None)
        if tm is not None:
            bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
            out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                layer.weight_global_scale, 6656, 128, 0, 169, 104,
                getattr(layer, "_muse_tm_mask", 7),
            )
            return out.reshape(x.shape[:-1] + (6656,))
        out = torch.ops.muse_fast.nvfp4_down_m1(
            reshaped_x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            _get_down_m1_partials(reshaped_x.device),
        )
        return out.reshape(x.shape[:-1] + (6656,))
    if (
        m == 512
        and _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        m512_pack = getattr(layer, "_muse_m512_pack", None)
        if m512_pack is not None:
            if (
                _MUSE_SINGLE_REQ_P512_MODE
                and layer_id == len(_FAST_DOWN_LAYERS) - 1
            ):
                tt7, rt7, partials7 = _get_m512_lpt_sched_m7(
                    reshaped_x.device, 6656, 19968, 144, 416
                )
                out = torch.empty((512, 6656), dtype=reshaped_x.dtype, device=reshaped_x.device)
                out[448:512] = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                    reshaped_x[448:512],
                    m512_pack,
                    layer.weight_global_scale,
                    tt7,
                    rt7,
                    partials7,
                    6656,
                )
                return out.reshape(x.shape[:-1] + (6656,))
            tt, rt, partials = _get_m512_lpt_sched(reshaped_x.device, 6656, 19968, 144, 416)
            out = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                reshaped_x,
                m512_pack,
                layer.weight_global_scale,
                tt,
                rt,
                partials,
                6656,
            )
        else:
            tt, rt, partials = _get_m512_lpt_sched(reshaped_x.device, 6656, 19968, 144, 416)
            out = torch.ops.muse_fast.nvfp4_down_m512_lpt(
                reshaped_x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                tt,
                rt,
                partials,
                6656,
            )
        return out.reshape(x.shape[:-1] + (6656,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=6656,
        size_k=19968,
        bias=None,
    )


def _muse_nvfp4_gate_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_GATE_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if reshaped_x.shape[0] == 1:
            sk = getattr(layer, "_muse_sk_rans", None)
            if sk is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = sk
                out = torch.ops.muse_fast.nvfp4_rans_splitk96_m1(
                    reshaped_x,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    layer.weight_global_scale,
                    _get_sk_gate_partials(reshaped_x.device),
                    4096,
                    35,
                )
                return out.reshape(x.shape[:-1] + (4096,))
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                    reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                    layer.weight_global_scale, 4096, 128, 0, 35, 32,
                )
                return out.reshape(x.shape[:-1] + (4096,))
            tm = getattr(layer, "_muse_tm", None)
            if tm is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                    reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                    layer.weight_global_scale, 4096, 128, 0, 35, 64,
                    getattr(layer, "_muse_tm_mask", 7),
                )
                return out.reshape(x.shape[:-1] + (4096,))
        if reshaped_x.shape[0] == 512:
            m512_pack = getattr(layer, "_muse_m512_pack", None)
            if m512_pack is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                if (
                    _MUSE_SINGLE_REQ_P512_MODE
                    and layer_id == len(_FAST_GATE_LAYERS) - 1
                ):
                    tt7, rt7, partials7 = _get_m512_lpt_sched_m7(
                        reshaped_x.device, 4096, 6656, 96, 70
                    )
                    out = torch.empty((512, 4096), dtype=reshaped_x.dtype, device=reshaped_x.device)
                    out[448:512] = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                        reshaped_x[448:512],
                        m512_pack,
                        layer.weight_global_scale,
                        tt7,
                        rt7,
                        partials7,
                        4096,
                    )
                    return out.reshape(x.shape[:-1] + (4096,))
                tt, rt, partials = _get_m512_lpt_sched(reshaped_x.device, 4096, 6656, 96, 70)
                out = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                    reshaped_x,
                    m512_pack,
                    layer.weight_global_scale,
                    tt,
                    rt,
                    partials,
                    4096,
                )
                return out.reshape(x.shape[:-1] + (4096,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=4096,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_gate_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (4096,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_oproj_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_OPROJ_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    if _MUSE_KERNELS_READY and reshaped_x.dtype == torch.bfloat16:
        if reshaped_x.shape[0] == 1:
            sk = getattr(layer, "_muse_sk_rans", None)
            if sk is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = sk
                out = torch.ops.muse_fast.nvfp4_rans_splitk96_m1(
                    reshaped_x,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    layer.weight_global_scale,
                    _get_sk_partials(reshaped_x.device),
                    6656,
                    35,
                )
                return out.reshape(x.shape[:-1] + (6656,))
            rp = getattr(layer, "_muse_rans", None)
            if rp is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                    reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                    layer.weight_global_scale, 6656, 128, 0, 35, 104,
                )
                return out.reshape(x.shape[:-1] + (6656,))
            tm = getattr(layer, "_muse_tm", None)
            if tm is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                    reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                    layer.weight_global_scale, 6656, 128, 0, 35, 104,
                    getattr(layer, "_muse_tm_mask", 7),
                )
                return out.reshape(x.shape[:-1] + (6656,))
        if reshaped_x.shape[0] == 512:
            m512_pack = getattr(layer, "_muse_m512_pack", None)
            if m512_pack is not None:
                if not reshaped_x.is_contiguous():
                    reshaped_x = reshaped_x.contiguous()
                if (
                    _MUSE_SINGLE_REQ_P512_MODE
                    and layer_id == len(_FAST_OPROJ_LAYERS) - 1
                ):
                    tt7, rt7, partials7 = _get_m512_lpt_sched_m7(
                        reshaped_x.device, 6656, 4096, 144, 86
                    )
                    out = torch.empty((512, 6656), dtype=reshaped_x.dtype, device=reshaped_x.device)
                    out[448:512] = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                        reshaped_x[448:512],
                        m512_pack,
                        layer.weight_global_scale,
                        tt7,
                        rt7,
                        partials7,
                        6656,
                    )
                    return out.reshape(x.shape[:-1] + (6656,))
                tt, rt, partials = _get_m512_lpt_sched(reshaped_x.device, 6656, 4096, 144, 86)
                out = torch.ops.muse_fast.nvfp4_lpt_contig_m512(
                    reshaped_x,
                    m512_pack,
                    layer.weight_global_scale,
                    tt,
                    rt,
                    partials,
                    6656,
                )
                return out.reshape(x.shape[:-1] + (6656,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=6656,
        size_k=4096,
        bias=None,
    )


def _muse_nvfp4_oproj_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (6656,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_down_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (6656,), dtype=x.dtype, device=x.device)


def _muse_nvfp4_lmhead_impl(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    layer = _FAST_LMHEAD_LAYERS[layer_id]
    reshaped_x = x.reshape(-1, x.shape[-1])
    if (
        reshaped_x.shape[0] == 1
        and _MUSE_KERNELS_READY
        and reshaped_x.dtype == torch.bfloat16
    ):
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        rp = getattr(layer, "_muse_rans", None)
        if rp is not None:
            bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
            out = torch.ops.muse_fast.nvfp4_rans_linear_m1(
                reshaped_x, bq_tm, sc_stream, stream_chunk_off, init_states, lut,
                layer.weight_global_scale, 202048, 64, 3120, 41, 192,
            )
            return out.reshape(x.shape[:-1] + (202048,))
        tm = getattr(layer, "_muse_tm", None)
        if tm is not None:
            bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
            out = torch.ops.muse_fast.nvfp4_tm_linear_m1(
                reshaped_x, bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode,
                layer.weight_global_scale, 202048, 64, 3120, 41, 192,
                getattr(layer, "_muse_tm_mask", 7),
            )
            return out.reshape(x.shape[:-1] + (202048,))
        out = torch.ops.muse_fast.nvfp4_lmhead_m1(
            reshaped_x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
        )
        return out.reshape(x.shape[:-1] + (202048,))
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=202048,
        size_k=6656,
        bias=None,
    )


def _muse_nvfp4_lmhead_fake(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty(x.shape[:-1] + (202048,), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="muse_nvfp4_gate_up",
    op_func=_muse_nvfp4_gate_up_impl,
    fake_impl=_muse_nvfp4_gate_up_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_gate_up_silu",
    op_func=_muse_nvfp4_gate_up_silu_impl,
    fake_impl=_muse_nvfp4_gate_up_silu_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_gate_up_silu_fp8quant",
    op_func=_muse_nvfp4_gate_up_silu_fp8quant_impl,
    fake_impl=_muse_nvfp4_gate_up_silu_fp8quant_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_down",
    op_func=_muse_nvfp4_down_impl,
    fake_impl=_muse_nvfp4_down_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_lmhead",
    op_func=_muse_nvfp4_lmhead_impl,
    fake_impl=_muse_nvfp4_lmhead_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_qkv",
    op_func=_muse_nvfp4_qkv_impl,
    fake_impl=_muse_nvfp4_qkv_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_gate",
    op_func=_muse_nvfp4_gate_impl,
    fake_impl=_muse_nvfp4_gate_fake,
)
direct_register_custom_op(
    op_name="muse_nvfp4_oproj",
    op_func=_muse_nvfp4_oproj_impl,
    fake_impl=_muse_nvfp4_oproj_fake,
)


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        prepare_fp4_layer_for_marlin(layer)
        if not _MUSE_KERNELS_READY or getattr(layer, "bias", None) is not None:
            return
        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        if (
            n == 39936
            and k == 6656
            and layer.weight.shape == (416, 79872)
            and layer.weight_scale.shape == (416, 39936)
        ):
            idx = len(_FAST_GATE_UP_LAYERS)
            _FAST_GATE_UP_LAYERS.append(layer)
            layer._muse_fast_gate_up_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 64, 576, 26, 156, use_rans=True)
            layer._muse_m512_pack = torch.ops.muse_fast.repack_nvfp4_m512_contig(
                layer.weight, layer.weight_scale, n, k, True
            )
        elif (
            n == 6656
            and k == 19968
            and layer.weight.shape == (1248, 13312)
            and layer.weight_scale.shape == (1248, 6656)
        ):
            idx = len(_FAST_DOWN_LAYERS)
            _FAST_DOWN_LAYERS.append(layer)
            layer._muse_fast_down_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 128, 0, 169, 104, use_splitk_rans=True)
            layer._muse_m512_pack = torch.ops.muse_fast.repack_nvfp4_m512_contig(
                layer.weight, layer.weight_scale, n, k, False
            )
        elif (
            n == 202048
            and k == 6656
            and layer.weight.shape == (416, 404096)
            and layer.weight_scale.shape == (416, 202048)
        ):
            idx = len(_FAST_LMHEAD_LAYERS)
            _FAST_LMHEAD_LAYERS.append(layer)
            layer._muse_fast_lmhead_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 64, 3120, 41, 192, use_rans=True)
        elif (
            n == 4608
            and k == 6656
            and layer.weight.shape == (416, 9216)
            and layer.weight_scale.shape == (416, 4608)
        ):
            idx = len(_FAST_QKV_LAYERS)
            _FAST_QKV_LAYERS.append(layer)
            layer._muse_fast_qkv_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 128, 0, 39, 72, use_splitk_rans=True)
            layer._muse_m512_pack = torch.ops.muse_fast.repack_nvfp4_m512_contig(
                layer.weight, layer.weight_scale, n, k, False
            )
        elif (
            n == 4096
            and k == 6656
            and layer.weight.shape == (416, 8192)
            and layer.weight_scale.shape == (416, 4096)
        ):
            idx = len(_FAST_GATE_LAYERS)
            _FAST_GATE_LAYERS.append(layer)
            layer._muse_fast_gate_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 128, 0, 35, 64, use_splitk_rans=True)
            layer._muse_m512_pack = torch.ops.muse_fast.repack_nvfp4_m512_contig(
                layer.weight, layer.weight_scale, n, k, False
            )
        elif (
            n == 6656
            and k == 4096
            and layer.weight.shape == (256, 13312)
            and layer.weight_scale.shape == (256, 6656)
        ):
            idx = len(_FAST_OPROJ_LAYERS)
            _FAST_OPROJ_LAYERS.append(layer)
            layer._muse_fast_oproj_id = idx
            _repack_nvfp4_for_m1(layer, n, k, 128, 0, 35, 104, use_splitk_rans=True)
            layer._muse_m512_pack = torch.ops.muse_fast.repack_nvfp4_m512_contig(
                layer.weight, layer.weight_scale, n, k, False
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            gu_id = getattr(layer, "_muse_fast_gate_up_id", -1)
            if gu_id >= 0:
                return torch.ops.vllm.muse_nvfp4_gate_up(x, gu_id)
            down_id = getattr(layer, "_muse_fast_down_id", -1)
            if down_id >= 0:
                return torch.ops.vllm.muse_nvfp4_down(x, down_id)
            lmhead_id = getattr(layer, "_muse_fast_lmhead_id", -1)
            if lmhead_id >= 0:
                return torch.ops.vllm.muse_nvfp4_lmhead(x, lmhead_id)
            qkv_id = getattr(layer, "_muse_fast_qkv_id", -1)
            if qkv_id >= 0:
                return torch.ops.vllm.muse_nvfp4_qkv(x, qkv_id)
            gate_id = getattr(layer, "_muse_fast_gate_id", -1)
            if gate_id >= 0:
                return torch.ops.vllm.muse_nvfp4_gate(x, gate_id)
            oproj_id = getattr(layer, "_muse_fast_oproj_id", -1)
            if oproj_id >= 0:
                return torch.ops.vllm.muse_nvfp4_oproj(x, oproj_id)
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
