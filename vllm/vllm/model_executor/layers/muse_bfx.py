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

try:
    import muse_bf16_lane as _lane  # rANS-coded BF16 (high byte) GEMV/GEMM, K == 6656
    if _lane._lib() is None or not hasattr(_lane._lib(), "muse_bf16_rans_gemm_m"):
        _lane = None
except Exception:  # noqa: BLE001
    _lane = None

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


@torch.library.custom_op("muse::bf16r_linear", mutates_args=())
def bf16r_linear(x: torch.Tensor, w: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m = x2.size(0)
    if 1 <= m <= MAX_M and x2.dtype == torch.bfloat16:
        x2 = x2.contiguous()
        out = x2.new_empty((m, w.shape[0]))
        xp = torch.empty_like(x2)
        err = _lane._lib().muse_bf16_rans_gemm_m(m, w.shape[0], blob.numel(), x2.data_ptr(), xp.data_ptr(), blob.data_ptr(),
                                                 out.data_ptr(), torch.cuda.current_stream(x2.device).cuda_stream)
        if err:
            raise RuntimeError(f"muse_bf16_rans_gemm_m returned {err}")
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], w.shape[0])


@bf16r_linear.register_fake
def _(x, w, blob):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def _try_lane(layer: torch.nn.Module) -> bool:
    w = layer.weight.data
    if _lane is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] != 6656 or w.shape[0] % 8:
        return False
    r = _lane._pack_rans(w.contiguous())
    if r is None:
        return False
    blob = r[0]
    # lossless check: the one-row decode must equal cuBLAS bit for bit on probes (same check as the target lane)
    n = w.shape[0]
    gen = torch.Generator(device=w.device).manual_seed(4242)
    for mag in (0.02, 1.0, 32.0):
        x = (torch.randn((1, 6656), generator=gen, device=w.device) * mag).to(torch.bfloat16)
        out = x.new_empty((1, n)); xp = torch.empty_like(x)
        if _lane._lib().muse_bf16_rans_gemm_m(1, n, blob.numel(), x.data_ptr(), xp.data_ptr(), blob.data_ptr(), out.data_ptr(),
                                              torch.cuda.current_stream(w.device).cuda_stream):
            return False
        if not torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)):
            return False
    layer.register_buffer("_bf16r_blob", blob, persistent=False)
    layer._bf16r = True
    return True


def _register(mod: torch.nn.Module, prefix: str, enc: dict) -> None:
    for k, v in enc.items():
        mod.register_buffer(f"{prefix}_{k}", v, persistent=False)


def attach_linear(layer: torch.nn.Module) -> bool:
    w = getattr(layer, "weight", None)
    if w is None or getattr(layer, "bias", None) is not None or getattr(layer, "_bfx", False):
        return False
    if _try_lane(layer):
        layer._bfx = True
        return True
    enc = _encode(w.data)
    if enc is None:
        return False
    _register(layer, "_bfx", enc)
    layer._bfx = True
    return True


def layer_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "_nf4", False):
        return torch.ops.muse.nf4_linear(x, layer.weight, layer._nf4_codes, layer._nf4_scales, layer._nf4_lut)
    if getattr(layer, "_bf16r", False):
        return torch.ops.muse.bf16r_linear(x, layer.weight, layer._bf16r_blob)
    return torch.ops.muse.bfx_linear(x, layer.weight, layer._bfx_lo, layer._bfx_nib, layer._bfx_base,
                                     layer._bfx_rowptr, layer._bfx_ecol, layer._bfx_eval)


def attach_tensor(owner: torch.nn.Module, name: str, w: torch.Tensor) -> bool:
    if DRAFT4 and _DRAFT4_SET == "all" and attach4_tensor(owner, name, w):
        return True
    enc = _encode(w)
    if enc is None:
        return False
    _register(owner, f"_bfx_{name}", enc)
    return True


def tensor_apply(owner: torch.nn.Module, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if hasattr(owner, f"_nf4_{name}_codes"):
        return torch.ops.muse.nf4_linear(x, w, getattr(owner, f"_nf4_{name}_codes"), getattr(owner, f"_nf4_{name}_scales"),
                                         getattr(owner, f"_nf4_{name}_lut"))
    p = f"_bfx_{name}_"
    if not hasattr(owner, p + "lo"):
        return F.linear(x, w)
    return torch.ops.muse.bfx_linear(x, w, getattr(owner, p + "lo"), getattr(owner, p + "nib"), getattr(owner, p + "base"),
                                     getattr(owner, p + "rowptr"), getattr(owner, p + "ecol"), getattr(owner, p + "eval"))


def attach_drafter(model: torch.nn.Module) -> int:
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc"):
            four = DRAFT4 and (_DRAFT4_SET == "all" or name.split(".")[-1] in ("gate_up_proj", "down_proj"))
            n += (four and attach4(mod)) or attach_linear(mod)
    return n


# ---- optional 4-bit drafter weights (NF4 codes, bf16 per-32 scale): drafter proposals only ----
_NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0]
DRAFT4 = __import__("os").environ.get("MUSE_DRAFT4", "1") != "0"
_DRAFT4_SET = __import__("os").environ.get("MUSE_DRAFT4_SET", "mlp")


def _encode4(w: torch.Tensor):
    N, K = w.shape
    lut = torch.tensor(_NF4, device=w.device)
    mids = (lut[1:] + lut[:-1]) / 2
    codes = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 32, dtype=torch.bfloat16, device=w.device)
    step = max(1, (2 << 20) // K)
    for r0 in range(0, N, step):
        g = w[r0:r0 + step].float().view(-1, K // 32, 32)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        best_err = best_c = best_s = None
        for f in (1.0, 0.95, 0.9, 0.85, 0.8):
            sc = (amax * f).to(torch.bfloat16).float()
            c = torch.bucketize((g / sc).clamp(-1, 1), mids)
            err = ((lut[c] * sc - g) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_c, best_s = err, c, sc
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_c = torch.where(m, c, best_c)
                best_s = torch.where(m, sc, best_s)
            del c, err
        c = best_c.view(g.shape[0], K).to(torch.uint8)
        codes[r0:r0 + step] = c[:, 0::2] | (c[:, 1::2] << 4)
        scales[r0:r0 + step] = best_s.squeeze(-1).to(torch.bfloat16)
        del g, amax, best_err, best_c, best_s, c
    return codes, scales, lut


@torch.library.custom_op("muse::nf4_linear", mutates_args=())
def nf4_linear(x: torch.Tensor, w: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if 1 <= x2.size(0) <= MAX_M and x2.dtype == torch.bfloat16:
        out = _ext.gemm4(x2.contiguous(), codes, scales, lut if lut.dtype == torch.float32 else lut.float())
    else:
        out = F.linear(x2, w)
    return out.reshape(*shape[:-1], w.shape[0])


@nf4_linear.register_fake
def _(x, w, codes, scales, lut):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def attach4(layer: torch.nn.Module) -> bool:
    w = getattr(layer, "weight", None)
    if (_ext is None or not hasattr(_ext, "gemm4") or w is None or getattr(layer, "bias", None) is not None
            or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % CH):
        return False
    codes, scales, lut = _encode4(w.data)
    layer.register_buffer("_nf4_codes", codes, persistent=False)
    layer.register_buffer("_nf4_scales", scales, persistent=False)
    layer.register_buffer("_nf4_lut", lut, persistent=False)
    layer._nf4 = True
    layer._bfx = True
    return True


def attach4_tensor(owner: torch.nn.Module, name: str, w: torch.Tensor) -> bool:
    if _ext is None or not hasattr(_ext, "gemm4") or w.shape[1] % CH:
        return False
    codes, scales, lut = _encode4(w)
    owner.register_buffer(f"_nf4_{name}_codes", codes, persistent=False)
    owner.register_buffer(f"_nf4_{name}_scales", scales, persistent=False)
    owner.register_buffer(f"_nf4_{name}_lut", lut, persistent=False)
    return True
