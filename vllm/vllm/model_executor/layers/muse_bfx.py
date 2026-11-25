# SPDX-License-Identifier: Apache-2.0
"""Lossless 12-bit coded BF16 weights for small-row GEMMs (the DFlash drafter at verify width).

A BF16 weight row keeps its sign+7-bit mantissa byte and a 4-bit exponent code relative to the row's
max exponent (window of 15); zeros and the rare out-of-window values code 0 and every nonzero one of
them is carried exactly in a per-row escape list. Decoding reproduces every weight bit for bit (checked
at load); the kernel (harness/kernels/musebfx) reads 12 bits per weight instead of 16. Rows > 8 use the
dense weight unchanged.
"""
import torch
import torch.nn.functional as F

try:
    import muse_bfx_ext as _ext
except ImportError:  # kernels not built: dense path
    _ext = None

CH = 512
MAX_M = 8


def _encode_rows(w: torch.Tensor):
    N, K = w.shape
    u = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    base = e.max(dim=1, keepdim=True).values.clamp_min(15) - 15
    e -= base
    inwin = (e >= 1) & (e <= 15)
    e.masked_fill_(~inwin, 0)
    lo = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    del u
    nib = (e[:, 0::2] | (e[:, 1::2] << 4)).to(torch.uint8)
    # bitwise reconstruction check of this chunk
    lo32 = lo.to(torch.int32)
    bits = ((lo32 & 0x80) << 8) | ((base + e) << 7) | (lo32 & 0x7F)
    del lo32
    bits.masked_fill_(e == 0, 0)
    del e
    esc = (~inwin) & ((w.view(torch.int16) & 0x7FFF) != 0)
    rows, cols = esc.nonzero(as_tuple=True)
    rec = bits.to(torch.int16)
    del bits
    rec[rows, cols] = w.view(torch.int16)[rows, cols]
    ok = torch.equal(rec, w.view(torch.int16))
    return ok, lo, nib, base.squeeze(1).to(torch.int32), rows, cols


def _encode(w: torch.Tensor) -> dict | None:
    if _ext is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % CH or not w.is_cuda:
        return None
    w = w.contiguous()
    N, K = w.shape
    lo = torch.empty(N, K, dtype=torch.uint8, device=w.device)
    nib = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    base = torch.empty(N, dtype=torch.int32, device=w.device)
    R, C = [], []
    step = max(1, (64 << 20) // K)
    for r0 in range(0, N, step):
        ok, l, nb, b, rr, cc = _encode_rows(w[r0:r0 + step])
        if not ok:
            return None
        lo[r0:r0 + step] = l; nib[r0:r0 + step] = nb; base[r0:r0 + step] = b
        R.append(rr + r0); C.append(cc)
        del l, nb, b
    rows = torch.cat(R); cols = torch.cat(C)
    rowptr = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    rowptr[1:] = torch.bincount(rows, minlength=N).cumsum(0).to(torch.int32)
    ecol = cols.to(torch.int32).contiguous()
    evals = w[rows, cols].contiguous()
    if ecol.numel() == 0:
        ecol = torch.zeros(1, dtype=torch.int32, device=w.device)
        evals = torch.zeros(1, dtype=torch.bfloat16, device=w.device)
    return dict(lo=lo, nib=nib, base=base, rowptr=rowptr, ecol=ecol, eval=evals)


@torch.library.custom_op("muse::bfx_linear", mutates_args=())
def bfx_linear(x: torch.Tensor, w: torch.Tensor, lo: torch.Tensor, nib: torch.Tensor, base: torch.Tensor,
               rowptr: torch.Tensor, ecol: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _ext.gemm(x2.contiguous(), lo, nib, base, rowptr, ecol, ev)
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], w.shape[0])


@bfx_linear.register_fake
def _(x, w, lo, nib, base, rowptr, ecol, ev):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def _register(mod: torch.nn.Module, prefix: str, enc: dict) -> None:
    for k, v in enc.items():
        mod.register_buffer(f"{prefix}_{k}", v, persistent=False)


def attach_linear(layer: torch.nn.Module) -> bool:
    w = getattr(layer, "weight", None)
    if w is None or getattr(layer, "bias", None) is not None or getattr(layer, "_bfx", False):
        return False
    enc = _encode(w.data)
    if enc is None:
        return False
    _register(layer, "_bfx", enc)
    layer._bfx = True
    return True


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.muse.bfx_linear(x, layer.weight, layer._bfx_lo, layer._bfx_nib, layer._bfx_base,
                                     layer._bfx_rowptr, layer._bfx_ecol, layer._bfx_eval)


def attach_tensor(owner: torch.nn.Module, name: str, w: torch.Tensor) -> bool:
    enc = _encode(w)
    if enc is None:
        return False
    _register(owner, f"_bfx_{name}", enc)
    return True


def tensor_apply(owner: torch.nn.Module, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p = f"_bfx_{name}_"
    if not hasattr(owner, p + "lo"):
        return F.linear(x, w)
    return torch.ops.muse.bfx_linear(x, w, getattr(owner, p + "lo"), getattr(owner, p + "nib"), getattr(owner, p + "base"),
                                     getattr(owner, p + "rowptr"), getattr(owner, p + "ecol"), getattr(owner, p + "eval"))


def attach_drafter(model: torch.nn.Module) -> int:
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc"):
            n += attach_linear(mod)
    return n
