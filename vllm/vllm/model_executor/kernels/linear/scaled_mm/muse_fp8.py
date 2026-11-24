# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer: the decode (M == 1) FP8 w8a8 GEMMs through musefp8 (harness/kernels/musefp8).

For one token, `musefp8.gemv` issues the same chain of mma.sync m16n8k32 e4m3 instructions in the same k order as the
SM120 M16 cutlass kernel behind `ops.cutlass_scaled_mm`, and the same ScaledEpilogue arithmetic
(bf16_rn(scale_a * fp32(scale_b * acc))), so its output is bitwise identical; it reads a fragment-ordered copy of the
weight, made once at load, which streams faster. Every layer is admitted only after a load-time bitwise comparison with
`ops.cutlass_scaled_mm` on its own weight. Anything else (M > 1, a missing or disabled extension, unusual shapes or
scales) goes to the tree's small-row FP8 op (`glimmer_fp8.fp8_scaled_mm`: the G2 CUTLASS tile, else stock cutlass)
unchanged. The M test lives inside an opaque custom op, so torch.compile never bakes a branch on M into the graph.
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.glimmer_fp8 import fp8_scaled_mm as _small_row_fp8
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    import musefp8 as _mf  # built into KERNELS_OUT (on the resident's PYTHONPATH after the overlay)
except Exception:  # noqa: BLE001 - extension absent: stock cutlass everywhere
    _mf = None

try:
    # Lossless-compressed variant (harness/kernels/muserans): the same MMA chain and epilogue, fed by weights decoded
    # from an exponent-pair rANS stream (~0.84 of the bytes). MUSERANS=0 keeps the uncompressed musefp8 GEMV.
    import muse_rans_ext as _rans  # built into KERNELS_OUT

    if os.environ.get("MUSERANS", "1") == "0":
        _rans = None
except Exception:  # noqa: BLE001
    _rans = None

_stats = {"admitted": 0, "rejected": 0, "rans": 0, "logged": False}
_decode_gemm_operand = None
_consume_rans_prologue = None
_rans_meta: dict[tuple[int, int, int, int, int], tuple[int, int, int, int, int, float, float]] = {}
_boot_tuned_attn: dict[int, tuple[int, int]] = {}
_boot_tuned_gu: dict[int, int] = {}
_gate_blocks_cached: dict[int, int] = {}


def set_runtime_pdl(enabled: bool | None = None) -> None:
    if _rans is not None and hasattr(_rans, "set_runtime_pdl"):
        _rans.set_runtime_pdl(enabled)
    try:
        import muse_bf16_lane as _bf16

        if hasattr(_bf16, "set_runtime_pdl"):
            _bf16.set_runtime_pdl(enabled)
    except Exception:  # noqa: BLE001
        pass


def set_runtime_gate_blocks(blocks: int | None = None) -> None:
    if _rans is not None and hasattr(_rans, "set_runtime_gate_blocks"):
        _rans.set_runtime_gate_blocks(blocks)
    try:
        import muse_bf16_lane as _bf16

        if hasattr(_bf16, "set_runtime_gate_blocks"):
            _bf16.set_runtime_gate_blocks(blocks)
    except Exception:  # noqa: BLE001
        pass


def _sync_gate_blocks(device: torch.device) -> int:
    if (os.environ.get("MUSERANS_GATE_32_BLOCKS") or "0") != "0":
        blocks = 32
        dev_idx = 0 if device.index is None else int(device.index)
        if _gate_blocks_cached.get(dev_idx) != blocks:
            _gate_blocks_cached[dev_idx] = blocks
            if _rans is not None and hasattr(_rans, "set_runtime_gate_blocks"):
                _rans.set_runtime_gate_blocks(blocks)
        return blocks
    dev_idx = 0 if device.index is None else int(device.index)
    cached = _gate_blocks_cached.get(dev_idx)
    if cached is not None:
        return cached
    admitted: bool | None = None
    try:
        import muse_gqa16_attn as _gqa

        fn = getattr(_gqa, "is_admitted", None)
        if callable(fn):
            admitted = fn(device)
    except Exception:  # noqa: BLE001
        admitted = False
    if admitted is None:
        return 48
    blocks = 48 if admitted else 32
    _gate_blocks_cached[dev_idx] = blocks
    if _rans is not None and hasattr(_rans, "set_runtime_gate_blocks"):
        _rans.set_runtime_gate_blocks(blocks)
    return blocks


def _should_boot_tune(device: torch.device) -> bool:
    env = os.environ.get("MUSE_FP8_BOOT_TUNE")
    if env is not None and env != "":
        return env != "0"
    try:
        return int(torch.cuda.get_device_properties(device).multi_processor_count) == 48
    except Exception:  # noqa: BLE001
        return False


def _bench_rans_candidate(
    blob: torch.Tensor,
    tw: int,
    ns: int,
    wpb: int,
    rg: int,
    ilv: int,
    ws_flat: torch.Tensor,
    xs_flat: torch.Tensor,
    n: int,
    k: int,
) -> float:
    xq = torch.zeros((k,), dtype=torch.float8_e4m3fn, device=blob.device)
    out = torch.empty((1, n), dtype=torch.bfloat16, device=blob.device)
    flush = torch.empty((16 << 20,), dtype=torch.int32, device=blob.device)
    for _ in range(3):
        _rans.rans_gemv_cfg(blob, xq, ws_flat, xs_flat, out, n, k, tw, ns, wpb, rg, ilv)
    times: list[float] = []
    for _ in range(9):
        flush.zero_()
        st_ev = torch.cuda.Event(enable_timing=True)
        en_ev = torch.cuda.Event(enable_timing=True)
        st_ev.record()
        _rans.rans_gemv_cfg(blob, xq, ws_flat, xs_flat, out, n, k, tw, ns, wpb, rg, ilv)
        en_ev.record()
        en_ev.synchronize()
        times.append(float(st_ev.elapsed_time(en_ev)))
    del flush, xq, out
    times.sort()
    return times[len(times) // 2]


def _maybe_boot_tune_attn(
    w: torch.Tensor,
    w_u8: torch.Tensor,
    ws: torch.Tensor,
    xs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor | None:
    if _rans is None or not hasattr(_rans, "rans_pack_cfg") or not hasattr(_rans, "set_runtime_pack_cfg"):
        return None
    if os.environ.get("MUSERANS_ILV") or os.environ.get("MUSERANS_NS"):
        return None
    dev_idx = 0 if w.device.index is None else int(w.device.index)
    if dev_idx in _boot_tuned_attn or not _should_boot_tune(w.device):
        return None
    if not ((n in (4096, 4608, 6656)) and (k in (4096, 6656, 8192))):
        return None
    _boot_tuned_attn[dev_idx] = (9, 4)
    try:
        default_cfg = _rans.rans_cfg_info(n, k)
        base_tw, base_wpb = int(default_cfg[0]), int(default_cfg[1])
        ws_flat, xs_flat = ws.reshape(-1), xs.reshape(-1)
        candidates = [(9, 4, 4), (10, 3, 4), (9, 4, 2), (10, 3, 2)]
        best_score = float("inf")
        default_score = float("inf")
        best_cfg: tuple[int, int] | None = None
        best_blob: torch.Tensor | None = None
        for ilv, rg, ns in candidates:
            cand_blob = None
            try:
                cand_blob = _rans.rans_pack_cfg(w_u8, base_tw, ns, ilv)
                unpacked = _rans.rans_unpack(cand_blob, n, k)
                if not torch.equal(unpacked, w_u8):
                    del unpacked, cand_blob
                    continue
                del unpacked
                ok = _self_check(
                    lambda xq, out, _b=cand_blob, _tw=base_tw, _ns=ns, _wpb=base_wpb, _rg=rg, _ilv=ilv: (
                        _rans.rans_gemv_cfg(
                            _b, xq.view(-1), ws_flat, xs_flat, out, n, k, _tw, _ns, _wpb, _rg, _ilv
                        )
                    ),
                    w,
                    ws,
                    xs,
                )
                if not ok:
                    del cand_blob
                    continue
                ms = _bench_rans_candidate(cand_blob, base_tw, ns, base_wpb, rg, ilv, ws_flat, xs_flat, n, k)
                if ilv == 9 and ns == 4:
                    default_score = ms
                if ms < best_score:
                    best_score = ms
                    best_cfg = (ilv, ns)
                    if best_blob is not None:
                        del best_blob
                    best_blob = cand_blob
                else:
                    del cand_blob
            except Exception:  # noqa: BLE001
                if cand_blob is not None and cand_blob is not best_blob:
                    del cand_blob
        if best_cfg is not None:
            if best_cfg != (9, 4) and default_score < float("inf") and best_score > default_score * 0.992:
                best_cfg = (9, 4)
                if best_blob is not None:
                    del best_blob
                    best_blob = None
            _boot_tuned_attn[dev_idx] = best_cfg
            _rans.set_runtime_pack_cfg(best_cfg[0], best_cfg[1], -1)
            logger.info(
                "muserans boot tune (attn %dx%d): selected ilv=%d, ns=%d (%.4f ms vs default %.4f ms)",
                n,
                k,
                best_cfg[0],
                best_cfg[1],
                best_score,
                default_score,
            )
            return best_blob
    except Exception:  # noqa: BLE001
        pass
    return None


def _maybe_boot_tune_gu(
    w: torch.Tensor,
    w_u8: torch.Tensor,
    ws: torch.Tensor,
    xs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor | None:
    if _rans is None or not hasattr(_rans, "rans_pack_cfg") or not hasattr(_rans, "set_runtime_pack_cfg"):
        return None
    if os.environ.get("MUSERANS_GATE_UP_TW"):
        return None
    if n != 39936 or k != 6656:
        return None
    dev_idx = 0 if w.device.index is None else int(w.device.index)
    if dev_idx in _boot_tuned_gu or not _should_boot_tune(w.device):
        return None
    _boot_tuned_gu[dev_idx] = 624
    try:
        cur = _rans.rans_cfg_info(n, k)
        ns, rg, ilv = int(cur[2]), int(cur[3]), int(cur[4])
        ws_flat, xs_flat = ws.reshape(-1), xs.reshape(-1)
        best_ms = float("inf")
        default_ms = float("inf")
        best_tw = 624
        best_blob: torch.Tensor | None = None
        for tw in (624, 960, 1248, 384):
            wpb = tw // 48
            cand_blob = None
            try:
                cand_blob = _rans.rans_pack_cfg(w_u8, tw, ns, ilv)
                unpacked = _rans.rans_unpack(cand_blob, n, k)
                if not torch.equal(unpacked, w_u8):
                    del unpacked, cand_blob
                    continue
                del unpacked
                ok = _self_check(
                    lambda xq, out, _b=cand_blob, _tw=tw, _ns=ns, _wpb=wpb, _rg=rg, _ilv=ilv: (
                        _rans.rans_gemv_cfg(
                            _b, xq.view(-1), ws_flat, xs_flat, out, n, k, _tw, _ns, _wpb, _rg, _ilv
                        )
                    ),
                    w,
                    ws,
                    xs,
                )
                if not ok:
                    del cand_blob
                    continue
                ms = _bench_rans_candidate(cand_blob, tw, ns, wpb, rg, ilv, ws_flat, xs_flat, n, k)
                if tw == 624:
                    default_ms = ms
                if ms < best_ms:
                    best_ms = ms
                    best_tw = tw
                    if best_blob is not None:
                        del best_blob
                    best_blob = cand_blob
                else:
                    del cand_blob
            except Exception:  # noqa: BLE001
                if cand_blob is not None and cand_blob is not best_blob:
                    del cand_blob
        if best_tw != 624 and default_ms < float("inf") and best_ms > default_ms * 0.992:
            best_tw = 624
            if best_blob is not None:
                del best_blob
                best_blob = None
        _boot_tuned_gu[dev_idx] = best_tw
        _rans.set_runtime_pack_cfg(-1, -1, best_tw)
        logger.info(
            "muserans boot tune (gate_up %dx%d): selected gu_tw=%d (%.4f ms vs 624 %.4f ms)",
            n,
            k,
            best_tw,
            best_ms,
            default_ms,
        )
        return best_blob
    except Exception:  # noqa: BLE001
        pass
    return None


def _cutlass(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    return ops.cutlass_scaled_mm(A, B, out_dtype=torch.bfloat16, scale_a=As, scale_b=Bs, bias=None)


def _scales_ok(As: torch.Tensor | None, Bs: torch.Tensor | None, n: int) -> bool:
    return (
        As is not None
        and Bs is not None
        and As.dtype == torch.float32
        and Bs.dtype == torch.float32
        and As.is_cuda
        and Bs.is_cuda
        and As.is_contiguous()
        and Bs.is_contiguous()
        and As.numel() == 1
        and Bs.numel() in (1, n)
    )


def prepare(layer: torch.nn.Module, w_name: str, w_s_name: str, i_s_name: str) -> torch.Tensor | None:
    """Pack the layer's finished [K, N] column-major e4m3 weight for the M == 1 GEMV, check it, register it."""
    if _mf is None or _mf.lib() is None:
        return None
    w = getattr(layer, w_name, None)
    ws = getattr(layer, w_s_name, None)
    xs = getattr(layer, i_s_name, None)
    if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_cuda:
        return None
    k, n = w.shape
    if not _mf.supported(n, k) or not w.t().is_contiguous() or not _scales_ok(xs, ws, n):
        return None
    if _rans is not None:
        try:
            w_u8 = w.t().contiguous().view(torch.uint8)
            blob = _maybe_boot_tune_attn(w, w_u8, ws, xs, n, k)
            if blob is None:
                blob = _maybe_boot_tune_gu(w, w_u8, ws, xs, n, k)
            if blob is None:
                blob = _rans.rans_pack(w_u8)
            info = _rans.rans_blob_info(blob)
            tw, ns, ilv, gmin = int(info[2]), int(info[3]), int(info[5]), int(info[6])
            rg = 3 if ilv == 10 else 4
            sa_val = float(xs.reshape(-1)[0].item())
            sb_scalar = ws.numel() == 1
            sb_val = float(ws.reshape(-1)[0].item()) if sb_scalar else 0.0
            wpb = (16 if _sync_gate_blocks(w.device) == 32 else 11) if (n == 4096 and k == 6656) else ((tw + 47) // 48)
            def _run_self_check(xq: torch.Tensor, out: torch.Tensor) -> None:
                if hasattr(_rans, "rans_gemv_cfg_fast"):
                    _rans.rans_gemv_cfg_fast(
                        blob,
                        xq.view(-1),
                        ws.reshape(-1),
                        xs.reshape(-1),
                        out,
                        n,
                        k,
                        tw,
                        ns,
                        wpb,
                        rg,
                        ilv,
                        gmin,
                        sa_val,
                        sb_val,
                        sb_scalar,
                    )
                else:
                    _rans.rans_gemv_auto(blob, xq.view(-1), ws.reshape(-1), xs.reshape(-1), out, n, k)
            ok = _self_check(_run_self_check, w, ws, xs)
            if ok:
                _stats["admitted"] += 1
                _stats["rans"] += 1
                _rans_meta[(xs.data_ptr(), ws.data_ptr(), blob.numel(), n, k)] = (
                    tw,
                    ns,
                    rg,
                    ilv,
                    gmin,
                    sa_val,
                    sb_val,
                )
                if hasattr(_rans, "rans_register_fast_meta"):
                    _rans.rans_register_fast_meta(blob, ws.reshape(-1), xs.reshape(-1), n, k, sa_val, sb_val)
                # The magic byte 'A' at blob[0] marks the compressed format for the op (one uint8 buffer either way).
                layer.register_buffer("muse_fp8_packed", blob, persistent=False)
                return blob
        except Exception as e:  # noqa: BLE001
            logger.warning("muserans: packing or self-check failed (%s); trying musefp8", e)
    try:
        packed = _mf.pack(w.t().view(torch.uint8))
        ok = _mf.self_check(packed, w, ws, xs, _cutlass)
    except Exception as e:  # noqa: BLE001
        logger.warning("musefp8: packing or self-check failed (%s); using cutlass_scaled_mm", e)
        ok = False
    _stats["admitted" if ok else "rejected"] += 1
    if not ok:
        return None
    # A buffer, so the weight-range bookkeeping (l2pf) sees it as model memory.
    layer.register_buffer("muse_fp8_packed", packed, persistent=False)
    return packed


def _self_check(run, w_kn: torch.Tensor, sb: torch.Tensor, sa: torch.Tensor) -> bool:
    """run(xq [1,K] e4m3, out [1,N] bf16) vs cutlass_scaled_mm on this weight: large and cancelling activations, bitwise."""
    k, n = w_kn.shape
    gen = torch.Generator(device=w_kn.device)
    gen.manual_seed(n * 131 + k + 7)
    for trial in range(4):
        x = torch.randn(1, k, generator=gen, device=w_kn.device) * (0.05, 1.0, 8.0, 40.0)[trial]
        x[:, ::3] *= 16
        x[:, 1::2] = -x[:, ::2]
        xq = (x / sa.float().reshape(1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ref = _cutlass(xq, w_kn, sa, sb)
        got = torch.empty(1, n, dtype=torch.bfloat16, device=w_kn.device)
        run(xq, got)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    return True


def _muse_fp8_scaled_mm(
    A: torch.Tensor, B: torch.Tensor, packed: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor
) -> torch.Tensor:
    gate_blks = _sync_gate_blocks(A.device) if _rans is not None else (
        32 if (os.environ.get("MUSERANS_GATE_32_BLOCKS") or "0") != "0" else 48
    )
    if not _stats["logged"]:
        _stats["logged"] = True
        tpf_kb = _rans.rans_tpf_kb() if (_rans is not None and hasattr(_rans, "rans_tpf_kb")) else 0
        tpf_note = f", translation prefetch every {tpf_kb} KiB" if tpf_kb > 0 else ", translation prefetch off"
        logger.info(
            "musefp8 decode GEMV: %d FP8 layers admitted (%d of them rANS-compressed%s), %d rejected; "
            "attention gate GEMV on %d blocks",
            _stats["admitted"], _stats["rans"], tpf_note, _stats["rejected"], gate_blks,
        )
    if A.shape[0] <= 16 and _decode_gemm_operand is not None:
        A = _decode_gemm_operand(A)
    k, n = B.shape
    if A.shape[0] == 1 and packed.numel() > 0 and A.is_contiguous() and _scales_ok(As, Bs, n):
        if (
            packed.numel() != n * k
            and _consume_rans_prologue is not None
            and _rans is not None
            and hasattr(_rans, "rans_gemv_fused_auto")
        ):
            fused = _consume_rans_prologue(A, As, packed, n, k, Bs)
            if fused is not None:
                return fused
        out = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
        if packed.numel() == n * k:
            _mf.gemv(packed, A, Bs, As, out, n, k)
        else:
            meta = _rans_meta.get((As.data_ptr(), Bs.data_ptr(), packed.numel(), n, k))
            if meta is not None and hasattr(_rans, "rans_gemv_cfg_fast"):
                tw, ns, rg, ilv, gmin, sa_val, sb_val = meta
                wpb = (16 if gate_blks == 32 else 11) if (n == 4096 and k == 6656) else ((tw + 47) // 48)
                _rans.rans_gemv_cfg_fast(
                    packed,
                    A.view(-1),
                    Bs.reshape(-1),
                    As.reshape(-1),
                    out,
                    n,
                    k,
                    tw,
                    ns,
                    wpb,
                    rg,
                    ilv,
                    gmin,
                    sa_val,
                    sb_val,
                    Bs.numel() == 1,
                )
            else:
                _rans.rans_gemv_auto(packed, A.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k)
        return out
    return _small_row_fp8(A, B, As, Bs)


def _muse_fp8_scaled_mm_fake(
    A: torch.Tensor, B: torch.Tensor, packed: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor
) -> torch.Tensor:
    return torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16, device=A.device)


direct_register_custom_op(
    op_name="muse_fp8_scaled_mm",
    op_func=_muse_fp8_scaled_mm,
    mutates_args=[],
    fake_impl=_muse_fp8_scaled_mm_fake,
)
