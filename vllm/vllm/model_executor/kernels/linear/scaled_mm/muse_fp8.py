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

_rans_mod = None
try:
    # Lossless-compressed variant (harness/kernels/muserans): the same MMA chain and epilogue, fed by weights decoded
    # from an exponent-pair rANS stream (~0.84 of the bytes). MUSERANS=0 keeps the uncompressed musefp8 GEMV.
    import muse_rans_ext as _rans_mod  # built into KERNELS_OUT

    if os.environ.get("MUSERANS", "1") == "0":
        _rans = None
    else:
        _rans = _rans_mod
except Exception as _rans_err:  # noqa: BLE001
    # Deliberately non-fatal: the resident must still boot on the uncompressed musefp8 path when the
    # extension is absent. But it must not be SILENT. A staged .so that the sandbox user cannot read
    # raises PermissionError here, and the resident then boots perfectly happily without the compressed
    # path -- which is the entire rANS contribution to the score -- with no other symptom. Naming the
    # exception turns that into a log line instead of an inference: a PermissionError means the staged
    # .so is not world-readable, an ImportError means it was never staged. The uncompressed path also
    # logs why it admitted (see _log_uncompressed), so the two lines agree on the cause.
    _rans = None
    logger.warning(
        "muserans: muse_rans_ext unavailable (%s: %s); the compressed decode path is OFF",
        type(_rans_err).__name__, _rans_err,
    )

_stats = {"admitted": 0, "rejected": 0, "rans": 0, "logged": False}
# Per-shape byte accounting for the uncompressed path. This exists so "MUSERANS=0 did nothing" and
# "rANS is neutral" cannot be confused: the line is emitted only when the uncompressed path is the
# one actually admitted, and it carries both sizes so the compressed ratio is measured, not modelled.
_uncompressed_seen: set = set()
_decode_gemm_operand = None
_consume_rans_prologue = None

# --- per-layer output_gate launch configuration (opt-in) -------------------------------------
# The attention output_gate GEMV (4096x6656) runs beside the attention kernel on a side
# stream. RoPE sliding-window layers run a 2-CTA attention kernel and can afford a wider gate
# grid; NoPE full-attention layers run 16-32 CTAs and need those SMs back. The rANS blob is
# independent of the launch grid (only the warp-per-block split changes, each output column
# still accumulates in one chain), so one packed blob serves both configurations; each
# configuration is admitted only after a bitwise check on the layer's own weight.
_GATE_SHAPE = (4096, 6656)
# (tw, ns, wpb, rg, ilv) as taken by rans_gemv_cfg; wpb 12 -> 43 blocks, 16 -> 32 blocks.
_GATE_CFG_ROPE_SW = (512, 4, 12, 4, 9)
_GATE_CFG_NOPE = (512, 4, 16, 4, 9)
_gate_cfg_by_blob: dict[int, tuple[int, ...]] = {}
_gate_dispatch: bool | None = None


def _gate_dispatch_enabled() -> bool:
    global _gate_dispatch
    if _gate_dispatch is None:
        _gate_dispatch = os.environ.get(
            "MUSE_GATE_ROPE_SW_DISPATCH", "0"
        ).strip() not in ("", "0")
    return _gate_dispatch


def _register_gate_cfg(layer: torch.nn.Module, blob: torch.Tensor, w: torch.Tensor,
                       ws: torch.Tensor, xs: torch.Tensor, n: int, k: int) -> None:
    """Record this layer's gate GEMV launch config, after checking it bitwise."""
    if _rans is None or not _gate_dispatch_enabled() or (n, k) != _GATE_SHAPE:
        return
    rope_sw = getattr(layer, "_muse_rope_sw", None)
    if rope_sw is None:
        return
    cfg = _GATE_CFG_ROPE_SW if rope_sw else _GATE_CFG_NOPE
    try:
        ok = _self_check(
            lambda xq, out: _rans.rans_gemv_cfg(
                blob, xq.view(-1), ws.reshape(-1), xs.reshape(-1), out, n, k, *cfg
            ),
            w, ws, xs,
        )
    except Exception as e:  # noqa: BLE001 - keep the automatic config
        logger.warning("muserans: gate dispatch self-check failed (%s); keeping auto", e)
        ok = False
    if ok:
        _gate_cfg_by_blob[blob.data_ptr()] = cfg
        logger.info(
            "muserans: output_gate GEMV on %d blocks for layer %s (%s)",
            (cfg[0] + cfg[2] - 1) // cfg[2],
            getattr(layer, "prefix", "?"),
            "RoPE sliding-window" if rope_sw else "NoPE full attention",
        )


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
            blob = _rans.rans_pack(w.t().contiguous().view(torch.uint8))
            if _self_check(lambda xq, out: _rans.rans_gemv_auto(blob, xq.view(-1), ws.reshape(-1), xs.reshape(-1), out, n, k), w, ws, xs):
                _stats["admitted"] += 1
                _stats["rans"] += 1
                # The magic byte 'A' at blob[0] marks the compressed format for the op (one uint8 buffer either way).
                layer.register_buffer("muse_fp8_packed", blob, persistent=False)
                _register_gate_cfg(layer, blob, w, ws, xs, n, k)
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
    # Reached only when the uncompressed pack is what was admitted. That happens for three
    # different reasons and they must not be confused, so the reason is part of the line.
    if _rans is None:
        why = ("MUSERANS=0" if os.environ.get("MUSERANS", "1") == "0"
               else "muse_rans_ext unavailable")
    else:
        why = "rANS declined this shape"
    _log_uncompressed(w, packed, n, k, why)
    # A buffer, so the weight-range bookkeeping (l2pf) sees it as model memory. The name is
    # deliberately NOT `muse_fp8_packed`: that name means "this is a rANS stream", and
    # decode_placement builds its rANS placement groups from it. Under MUSERANS=0 the buffer holds
    # uncompressed fp8 weights, and handing those to the rANS decoder walks off the end of the
    # allocation -- an illegal memory access at load, not a rejection. Two names make the two
    # cases structurally distinguishable instead of depending on an env var read in two places.
    layer.register_buffer("muse_fp8_plain", packed, persistent=False)
    return packed


def _log_uncompressed(w: torch.Tensor, packed: torch.Tensor, n: int, k: int, why: str) -> None:
    """One line per shape, load time, only when the uncompressed path is the one admitted.

    Its presence is the proof that the uncompressed path took effect: if the compressed path had
    been taken, this never runs. The compressed size is measured here by packing once through the
    rANS extension even though dispatch is off, so the ratio is a logged number rather than a
    modelled one -- that is what removes the 2.7x uncertainty in the FP8 byte budget.
    """
    if (n, k) in _uncompressed_seen:
        return
    _uncompressed_seen.add((n, k))
    raw = n * k
    comp = None
    if _rans_mod is not None:
        # One shape at a time, and the blob is released before the next shape is packed: at ~0.83
        # of raw these are large (o_proj alone is ~0.4 GB), and holding all of them to print at the
        # end would reserve ~2.2 GB across the FP8 shapes for no reason. The allocator may keep the
        # blocks cached, which it reclaims under pressure; nothing here is retained by this module.
        blob = None
        try:
            blob = _rans_mod.rans_pack(w.t().contiguous().view(torch.uint8))
            comp = int(blob.numel())
        except Exception as e:  # noqa: BLE001 - the ratio is a nicety, not a gate
            logger.warning("muserans: probe pack of %dx%d failed (%s)", n, k, e)
        finally:
            blob = None
    if comp:
        logger.info(
            "muse_fp8: UNCOMPRESSED path admitted for %dx%d (%s): %d B packed, %d B raw, "
            "rANS would be %d B (%.4f of raw, uncompressed is %.1f%% more bytes)",
            n, k, why, int(packed.numel()), raw, comp, comp / raw,
            100.0 * (packed.numel() / comp - 1.0),
        )
    else:
        logger.info(
            "muse_fp8: UNCOMPRESSED path admitted for %dx%d (%s): %d B packed, %d B raw",
            n, k, why, int(packed.numel()), raw,
        )


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
    if not _stats["logged"]:
        _stats["logged"] = True
        tpf_kb = _rans.rans_tpf_kb() if (_rans is not None and hasattr(_rans, "rans_tpf_kb")) else 0
        tpf_note = f", translation prefetch every {tpf_kb} KiB" if tpf_kb > 0 else ", translation prefetch off"
        # Report the config the kernel will actually launch, not a re-derivation from one env var. The
        # gate's block count is chosen by rans_pick_cfg (MUSERANS_GATE_BLOCKS, then
        # MUSERANS_GATE_32_BLOCKS, then the wcap fall-through), and this line used to recompute it from
        # MUSERANS_GATE_32_BLOCKS alone -- so it printed "32 blocks" even with MUSERANS_GATE_BLOCKS=47
        # in effect. A log line that disagrees with the launch has cost this session three retractions,
        # so ask the kernel. rans_cfg_info runs the same rans_pick_cfg the GEMV does.
        gate_note = ""
        if _rans is not None and hasattr(_rans, "rans_cfg_info"):
            try:
                gate_tw, gate_wpb = _rans.rans_cfg_info(4096, 6656)[:2]
                if gate_tw > 0 and gate_wpb > 0:
                    gate_note = f"attention gate GEMV on {(gate_tw + gate_wpb - 1) // gate_wpb} blocks"
            except Exception:  # noqa: BLE001 - a log line must never fail the load
                gate_note = ""
        if not gate_note:
            gate_note = "attention gate GEMV on %d blocks" % (
                32 if os.environ.get("MUSERANS_GATE_32_BLOCKS", "1") != "0" else 47
            )
        if _gate_cfg_by_blob:
            # The per-layer dispatch overrides the auto config named above, so without this the
            # line reads as "the dispatch did not take" when it did: rans_cfg_info knows nothing
            # about the per-layer override. Report what was actually admitted, with counts.
            _blk: dict[int, int] = {}
            for _cfg in _gate_cfg_by_blob.values():
                _nb = (_cfg[0] + _cfg[2] - 1) // _cfg[2]
                _blk[_nb] = _blk.get(_nb, 0) + 1
            gate_note += "; per-layer dispatch overrides %d layers (%s)" % (
                len(_gate_cfg_by_blob),
                ", ".join("%d on %d blocks" % (c, b) for b, c in sorted(_blk.items())),
            )
        logger.info(
            "musefp8 decode GEMV: %d FP8 layers admitted (%d of them rANS-compressed%s), %d rejected; %s",
            _stats["admitted"], _stats["rans"], tpf_note, _stats["rejected"], gate_note,
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
            cfg = _gate_cfg_by_blob.get(packed.data_ptr()) if (n, k) == _GATE_SHAPE else None
            if cfg is not None:
                _rans.rans_gemv_cfg(
                    packed, A.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k, *cfg
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
