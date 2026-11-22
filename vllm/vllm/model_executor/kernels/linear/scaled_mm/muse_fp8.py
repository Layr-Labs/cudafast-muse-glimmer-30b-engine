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

try:
    # 512-row prompt GEMM of the FP8 down projections (harness/kernels/fp8m512; kernel from public submission
    # 98cd9fcd). MUSE_FP8_M512=0 keeps cutlass_scaled_mm.
    import fp8m512 as _m512
except Exception:  # noqa: BLE001 - extension absent: stock cutlass for the prompt pass
    _m512 = None

_stats = {"admitted": 0, "rejected": 0, "rans": 0, "logged": False, "m512": 0, "m512_rejected": 0}

# Prompt-pass shapes (N, K) and row count the 512-row kernel is used for: the FP8 down projection only. On the
# attention shapes it is not faster than cutlass_scaled_mm on GB10.
_M512_SHAPES = ((6656, 19968),)
_M512_ROWS = 512
# weight data pointer -> (the weight, its tile-ordered copy). The entry keeps the weight alive, so the address
# cannot come to name other bytes while it is in the table.
_m512_packs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


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


def _m512_inputs(k: int, sa: torch.Tensor, device: torch.device, seed: int):
    """e4m3 [512, K] activations for the admission check: quantized normal rows at four magnitudes with large and
    cancelling columns, zeros and negative zeros; uniformly random e4m3 bytes; rows mixed from all of them."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cases = []
    for scale in (0.05, 1.0, 8.0, 40.0):
        x = torch.randn(_M512_ROWS, k, generator=gen, device=device) * scale
        x[:, ::3] *= 16
        x[:, 1::2] = -x[:, ::2]
        x = torch.where(torch.rand(_M512_ROWS, k, generator=gen, device=device) < 0.05, torch.zeros_like(x), x)
        x = torch.where(torch.rand(_M512_ROWS, k, generator=gen, device=device) < 0.02, torch.full_like(x, -0.0), x)
        cases.append(x.clamp(-448, 448).to(torch.float8_e4m3fn))
    raw = torch.randint(0, 256, (_M512_ROWS, k), generator=gen, device=device, dtype=torch.int64).to(torch.uint8)
    raw = torch.where((raw & 0x7F) == 0x7F, raw - 1, raw)  # 0x7f and 0xff are NaN in e4m3fn
    cases.append(raw.view(torch.float8_e4m3fn))
    mix = cases[0].clone().view(torch.uint8)
    pick = torch.randint(0, len(cases), (_M512_ROWS,), generator=gen, device=device)
    for i, a in enumerate(cases):
        mix[pick == i] = a.view(torch.uint8)[pick == i]
    cases.append(mix.view(torch.float8_e4m3fn))
    return cases


def _prepare_m512(layer: torch.nn.Module, w: torch.Tensor | None, ws: torch.Tensor | None,
                  xs: torch.Tensor | None) -> None:
    """Admit the layer to the 512-row prompt GEMM: tile-ordered copy of its weight, kept only if the kernel
    equals cutlass_scaled_mm bitwise on this weight for every check input. Anything else keeps the stock call."""
    if _m512 is None or w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_cuda:
        return
    k, n = w.shape
    if (n, k) not in _M512_SHAPES or w.stride(0) != 1 or w.stride(1) != k:
        return
    if not _scales_ok(xs, ws, n) or ws.numel() != 1:
        return
    packed = None
    try:
        if _m512.lib() is None or not _m512.supported(n, k):
            return
        packed = _m512.pack(w)
        ok = True
        for a in _m512_inputs(k, xs, w.device, n * 131 + k + 512):
            ref = _cutlass(a, w, xs, ws)
            got = _m512.gemm(a, packed, xs.reshape(1), ws.reshape(1), n, k)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                ok = False
                break
    except Exception as e:  # noqa: BLE001
        logger.warning("fp8m512: packing or self-check failed (%s); using cutlass_scaled_mm", e)
        ok = False
    _stats["m512" if ok else "m512_rejected"] += 1
    if not ok:
        del packed
        return
    # A buffer, so the weight-range bookkeeping sees it as model memory.
    layer.register_buffer("muse_fp8_m512_packed", packed, persistent=False)
    _m512_packs[w.data_ptr()] = (w, packed)


def prepare(layer: torch.nn.Module, w_name: str, w_s_name: str, i_s_name: str) -> torch.Tensor | None:
    """Pack the layer's finished [K, N] column-major e4m3 weight for the M == 1 GEMV, check it, register it."""
    _prepare_m512(layer, getattr(layer, w_name, None), getattr(layer, w_s_name, None),
                  getattr(layer, i_s_name, None))
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
            w_nk = w.t().contiguous().view(torch.uint8)
            blob = _rans.rans_pack(w_nk)
            # Full read-back first: the blob must decode to every byte of the weight (rans_unpack also checks that each
            # rANS state ends where the encoder started), then the GEMV must match cutlass bitwise.
            if torch.equal(_rans.rans_unpack(blob, n, k), w_nk) and _self_check(
                    lambda xq, out: _rans.rans_gemv_auto(blob, xq.view(-1), ws.reshape(-1), xs.reshape(-1), out, n, k),
                    w, ws, xs):
                _stats["admitted"] += 1
                _stats["rans"] += 1
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
    if not _stats["logged"]:
        _stats["logged"] = True
        logger.info(
            "musefp8 decode GEMV: %d FP8 layers admitted (%d of them rANS-compressed), %d rejected",
            _stats["admitted"], _stats["rans"], _stats["rejected"],
        )
        logger.info(
            "fp8m512 prompt GEMM: %d FP8 down layers admitted at %d rows, %d rejected",
            _stats["m512"], _M512_ROWS, _stats["m512_rejected"],
        )
    k, n = B.shape
    if A.shape[0] == 1 and packed.numel() > 0 and A.is_contiguous() and _scales_ok(As, Bs, n):
        out = torch.empty((1, n), dtype=torch.bfloat16, device=A.device)
        if packed.numel() == n * k:
            _mf.gemv(packed, A, Bs, As, out, n, k)
        else:
            _rans.rans_gemv_auto(packed, A.view(-1), Bs.reshape(-1), As.reshape(-1), out, n, k)
        return out
    if A.shape[0] == _M512_ROWS and _m512_packs:
        entry = _m512_packs.get(B.data_ptr())
        if (entry is not None and entry[0].shape == B.shape and A.dtype == torch.float8_e4m3fn
                and A.is_contiguous() and A.shape[1] == k and _scales_ok(As, Bs, n) and Bs.numel() == 1):
            return _m512.gemm(A, entry[1], As.reshape(1), Bs.reshape(1), n, k)
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
