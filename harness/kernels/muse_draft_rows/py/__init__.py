"""muse_draft_rows: the DFlash drafter's BF16 linear layers for 1..8 query rows (csrc rows.cu).

The drafter only proposes tokens and the target verifies every one, so this GEMM is not required to match cuBLAS bit
for bit: a different fp32 summation order changes which drafts are accepted, never the emitted tokens. Each layer is
still checked at load against F.linear (relative error bound) so a broken build cannot silently degrade acceptance.

``install(model)`` swaps the quant method of every admitted unquantized BF16 linear layer of the drafter for one that
calls the opaque op ``torch.ops.vllm.muse_draft_rows_linear``; inside the op, 1..8 rows run this kernel and anything
else runs F.linear, so torch.compile never sees a branch on the row count. ``MUSE_DRAFT_ROWS=0`` disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch
import torch.nn.functional as F

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmuse_draft_rows.so")
ENABLED = os.environ.get("MUSE_DRAFT_ROWS", "1") != "0"
MAX_ROWS = 8

_lib = None
_tried = False


def lib():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if not ENABLED:
        return None
    try:
        h = ctypes.CDLL(LIB_PATH)
    except OSError:
        return None
    vp, ci = ctypes.c_void_p, ctypes.c_int
    h.muse_draft_rows.argtypes = [vp, vp, vp, ci, ci, ci, ci, vp]
    h.muse_draft_rows.restype = ci
    _lib = h
    return h


def _blocks(n: int) -> int:
    tiles = n // 16
    return max(1, min((tiles + 7) // 8, 48 * 2))


def _usable(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and w.dim() == 2
        and w.is_contiguous()
        and x.shape[1] == w.shape[1]
        and w.shape[0] % 16 == 0
        and w.shape[1] % 32 == 0
        and w.data_ptr() % 16 == 0
    )


def rows_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16 (M <= 8), w [N, K] bf16 -> x . w^T [M, N] bf16 through rows.cu; F.linear otherwise."""
    h = lib()
    if h is None or not _usable(x, w):
        return F.linear(x, w)
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.contiguous().clone()
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = h.muse_draft_rows(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, _blocks(n),
                           torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        return F.linear(x, w)
    return out


def _rows_linear_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w.shape[0]))


_registered = False


def _register() -> bool:
    global _registered
    if _registered:
        return True
    try:
        from vllm.utils.torch_utils import direct_register_custom_op

        direct_register_custom_op("muse_draft_rows_linear", rows_linear, fake_impl=_rows_linear_fake)
    except Exception:  # noqa: BLE001
        return False
    _registered = True
    return True


class _RowsMethod:
    """Wraps an UnquantizedLinearMethod: bias-free 2-D calls go through the opaque op, the rest unchanged."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply(self, layer, x, bias=None):
        if bias is None and x.dim() == 2:
            return torch.ops.vllm.muse_draft_rows_linear(x, layer.weight)
        return self.inner.apply(layer, x, bias)


def _check(w: torch.Tensor) -> bool:
    gen = torch.Generator(device=w.device)
    gen.manual_seed(w.shape[0] * 7 + w.shape[1])
    for m in (1, 7, 8):
        x = (torch.randn(m, w.shape[1], generator=gen, device=w.device) * 0.5).to(torch.bfloat16)
        ref = F.linear(x, w).float()
        got = rows_linear(x, w).float()
        torch.cuda.synchronize(w.device)
        tol = 2e-2 * float(ref.abs().max()) + 1e-3
        if got.shape != ref.shape or not torch.isfinite(got).all() or float((got - ref).abs().max()) > tol:
            return False
    return True


def install(model: torch.nn.Module, logger=None) -> int:
    """Route the drafter's admitted BF16 linear layers through the rows kernel; returns the number of layers."""
    if lib() is None or not _register():
        return 0
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    admitted = rejected = 0
    for module in model.modules():
        if not isinstance(module, LinearBase) or type(getattr(module, "quant_method", None)) is not UnquantizedLinearMethod:
            continue
        w = getattr(module, "weight", None)
        if w is None or getattr(module, "bias", None) is not None or not w.is_cuda or not _usable(
            torch.empty((1, w.shape[1] if w.dim() == 2 else 0), dtype=torch.bfloat16, device=w.device), w
        ):
            continue
        try:
            ok = _check(w)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            module.quant_method = _RowsMethod(module.quant_method)
            admitted += 1
        else:
            rejected += 1
    if logger is not None:
        logger.info("DFlash drafter BF16 rows GEMM: %d linear layers admitted, %d kept on the stock GEMM",
                    admitted, rejected)
    return admitted
