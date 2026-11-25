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
# Verify-step rows (M = 2..8, the drafter's verify batch): the rANS blob through the same per-row MMA chain and
# epilogue as the M <= 16 CUTLASS tile, admitted per (N, K) only when every layer of that shape matched
# _small_row_fp8 bitwise at load. MUSERANS_ROWS=0 keeps every M > 1 call on _small_row_fp8.
_ROWS_MAX = 8
_rows_ok: dict[tuple[int, int], bool] = {}
_ROWS = os.environ.get("MUSERANS_ROWS", "1") != "0"
_decode_gemm_operand = None
_consume_rans_prologue = None


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
                if _ROWS and hasattr(_rans, "rans_gemm_rows_auto"):
                    ok_rows = _rows_check(blob, w, ws, xs)
                    _rows_ok[(n, k)] = _rows_ok.get((n, k), True) and ok_rows
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


def _rows_check(blob: torch.Tensor, w_kn: torch.Tensor, sb: torch.Tensor, sa: torch.Tensor) -> bool:
    """rans_gemm_rows_auto vs _small_row_fp8 (the M <= 16 path it replaces) on this weight, bitwise, for M = 2, 7, 8
    with large and cancelling activations."""
    k, n = w_kn.shape
    gen = torch.Generator(device=w_kn.device)
    gen.manual_seed(n * 131 + k + 11)
    try:
        for m, scale in ((7, 0.05), (7, 1.0), (7, 40.0), (2, 8.0), (8, 8.0)):
            x = torch.randn(m, k, generator=gen, device=w_kn.device) * scale
            x[:, ::3] *= 16
            x[:, 1::2] = -x[:, ::2]
            xq = (x / sa.float().reshape(1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            ref = _small_row_fp8(xq, w_kn, sa, sb)
            got = torch.empty(m, n, dtype=torch.bfloat16, device=w_kn.device)
            _rans.rans_gemm_rows_auto(blob, xq, sb.reshape(-1), sa.reshape(-1), got, n, k)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return False
    except Exception as e:  # noqa: BLE001
        logger.warning("muserans rows: self-check failed (%s); M > 1 stays on the CUTLASS tile", e)
        return False
    return True


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
        gate_blks = 32 if os.environ.get("MUSERANS_GATE_32_BLOCKS", "1") != "0" else 47
        logger.info(
            "musefp8 decode GEMV: %d FP8 layers admitted (%d of them rANS-compressed%s), %d rejected; "
            "attention gate GEMV on %d blocks",
            _stats["admitted"], _stats["rans"], tpf_note, _stats["rejected"], gate_blks,
        )
        logger.info("muserans verify rows (M 2..%d): %s", _ROWS_MAX,
                    ", ".join(f"{n}x{k} {'on' if v else 'OFF'}" for (n, k), v in sorted(_rows_ok.items())) or "none")
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
            _rans.rans_gemv_auto(packed, A.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k)
        return out
    if (
        1 < A.shape[0] <= _ROWS_MAX
        and packed.numel() > 0
        and packed.numel() != n * k
        and _rows_ok.get((n, k), False)
        and A.is_contiguous()
        and _scales_ok(As, Bs, n)
    ):
        out = torch.empty((A.shape[0], n), dtype=torch.bfloat16, device=A.device)
        _rans.rans_gemm_rows_auto(packed, A, Bs.reshape(-1), As.reshape(-1), out, n, k)
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
