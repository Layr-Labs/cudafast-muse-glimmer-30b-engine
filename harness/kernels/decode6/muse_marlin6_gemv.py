# SPDX-License-Identifier: Apache-2.0
"""Decode (M == 1) NVFP4 GEMV with 6-bit block scales: the board's tiled Marlin GEMV (muse_marlin_gemv) reading each
layer's group-16 FP8 block scales as 6-bit codes into a 64-entry palette, 3/4 of the scale bytes. The decoded bytes are
the same bytes, in the same shared-memory fragment, so the kernel computes the same bits; every layer is admitted only
after a bitwise comparison with the uncompressed tiled GEMV on its own weight."""
import ctypes
import os
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
STATS = {"admitted": 0, "kept": 0, "fixed_grid": 0, "gate_up_srans": 0, "gate_up_srans_rejected": 0,
         "srans_scales": 0, "srans_bytes": 0, "logged": False}

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin6.so")))
_gemv6 = _lib.muse_marlin6_gemv
_gemv6.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv6.restype = ctypes.c_int
_gate_up = getattr(_lib, "muse_marlin6_gate_up", None)
if _gate_up is not None:
    _gate_up.argtypes = _gemv6.argtypes
    _gate_up.restype = ctypes.c_int
# The same fixed grid reading rANS-coded block scales (gemv6_srans.cu).
_gate_up_srans = getattr(_lib, "muse_marlin6_gate_up_srans", None)
if _gate_up_srans is not None:
    _gate_up_srans.argtypes = _gemv6.argtypes
    _gate_up_srans.restype = ctypes.c_int


GATE_UP_N, GATE_UP_K = 39936, 6656

# ---------------------------------------------------------------- rANS-coded gate/up block scales
# Symbols: the layer's palette indices of the tiled one-byte scales. One table per layer (10-bit probabilities),
# 32-bit states, 16-bit renormalisation. Every CTA slice of the fixed grid has 8 warps x 32 lanes: lane l of warp w
# owns bytes 128 w + l + 32 m (m = 0..3) of each 1 KB stage block of raw scales, in stage order; a lane that
# renormalises reads its warp's next 16-bit word for that stage in lane order. Buffer (uint8, rows of 16 bytes):
#   [decode table: u32[1024] = symbol | freq << 6 | (slot - cum) << 17]
#   [336 headers of 3,328 bytes: u32 states[256] (warp-major), u32 offsets[64] (block X >= 2 at the stage-block
#    region + 16 * offsets[X], offsets[iters] ends the last), block 0 and block 1, each padded to 1 KB]
#   [stage blocks X >= 2 of every CTA in CTA order: u8 words-per-warp[8], then the 16-bit words of warp 0..7,
#    padded to 16 bytes]
# A shape (rows, 16) uint8 identifies the coded form. Encoding is a pure function of the scale bytes.
SRANS_M_BITS, SRANS_M, SRANS_L = 10, 1024, 1 << 16
SRANS_HDR, SRANS_CAP = 3328, 1024


def _srans_tasks():
    t = [(i, 0, 52) for i in range(288)]
    t += [(288 + j, 26, 26) for j in range(24)]
    t += [(288 + j, 0, 26) for j in range(24)]
    return t


def _srans_table(counts: torch.Tensor):
    counts = counts.cpu().long()
    total = int(counts.sum())
    f = torch.zeros(64, dtype=torch.long)
    for s in range(64):
        c = int(counts[s])
        if c:
            f[s] = max(1, round(c * SRANS_M / total))
    diff = SRANS_M - int(f.sum())
    while diff:
        k = int(torch.argmax(f))
        step = diff if diff > 0 else max(diff, -(int(f[k]) - 1))
        if step == 0:
            raise ValueError("cannot normalise the scale frequencies")
        f[k] += step
        diff -= step
    cum = torch.cumsum(f, 0) - f
    lut = torch.zeros(SRANS_M, dtype=torch.long)
    for s in range(64):
        if f[s]:
            slots = torch.arange(int(f[s]))
            lut[int(cum[s]) + slots] = s | (int(f[s]) << 6) | (slots << 17)
    return f, cum, lut


def is_gate_up_srans(scales: torch.Tensor) -> bool:
    return scales.dim() == 2 and scales.shape[1] == 16 and scales.dtype == torch.uint8


def encode_gate_up_scales(tiled_scales: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Tiled one-byte gate/up scales [312, 416, 128] and the layer's 64-byte palette -> the coded buffer
    (new tensor; raises ValueError if a stage block exceeds SRANS_CAP)."""
    dev = tiled_scales.device
    raw = tiled_scales.contiguous().view(torch.uint8).reshape(312, 52, 1024)
    vals = torch.unique(raw)
    inv = torch.full((256,), -1, dtype=torch.long, device=dev)
    pal = palette.long()
    if not torch.equal(pal[:vals.numel()], vals.long()):
        raise ValueError("palette does not list the scale bytes in order")
    inv[pal[:vals.numel()]] = torch.arange(vals.numel(), device=dev)
    sym = inv[raw.long()]
    f, cum, lut = _srans_table(torch.bincount(sym.reshape(-1), minlength=64))
    F, C = f.to(dev), cum.to(dev)
    tasks = _srans_tasks()
    out = []
    for ids, iters in ((list(range(288)), 52), (list(range(288, 336)), 26)):
        G = len(ids)
        s = torch.stack([sym[tasks[t][0], tasks[t][1]:tasks[t][1] + iters] for t in ids])
        steps = iters * 4
        seq = s.view(G, iters, 8, 4, 32).permute(0, 2, 4, 1, 3).reshape(G, 8, 32, steps)
        x = torch.full((G, 8, 32), SRANS_L, dtype=torch.long, device=dev)
        emit = torch.empty((G, 8, 32, steps), dtype=torch.bool, device=dev)
        words = torch.empty((G, 8, 32, steps), dtype=torch.long, device=dev)
        for st in range(steps - 1, -1, -1):
            sy = seq[..., st]
            fs, cs = F[sy], C[sy]
            e = x >= (fs << 22)
            emit[..., st] = e
            words[..., st] = x & 0xFFFF
            x = torch.where(e, x >> 16, x)
            x = ((x // fs) << SRANS_M_BITS) + (x % fs) + cs
        E = emit.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
        Wd = words.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
        cw = E.view(G, iters, 8, 128).sum(-1)
        padded = ((4 + cw.sum(-1) + 7) // 8) * 8  # 16-bit words, whole 16-byte rows
        if int(padded.max()) * 2 > SRANS_CAP:
            raise ValueError("a stage block exceeds the decode slot")
        out.append((torch.tensor(ids, device=dev), iters, x.reshape(G, 256), E, Wd, cw, padded))

    def put(dest, starts, Eb, Wb, cb):
        dest[starts[:, None] + torch.arange(4, device=dev)] = (cb[:, 0::2] | (cb[:, 1::2] << 8)).int()
        pos = starts[:, None] + 4 + torch.cumsum(Eb.long(), 1) - 1
        dest[pos[Eb]] = Wb[Eb].int()

    base, total = [], 0
    for ids, iters, *_rest, padded in out:
        flat = padded[:, 2:].reshape(-1)
        base.append((torch.cumsum(flat, 0) - flat + total).view(len(ids), iters - 2))
        total += int(flat.sum())
    hdr = torch.zeros((336, SRANS_HDR // 2), dtype=torch.int32, device=dev)
    var = torch.zeros(total, dtype=torch.int32, device=dev)
    hflat = hdr.view(-1)
    for (ids, iters, states, E, Wd, cw, padded), st in zip(out, base):
        G = len(ids)
        put(var, st.reshape(-1), E[:, 2:].reshape(-1, 1024), Wd[:, 2:].reshape(-1, 1024), cw[:, 2:].reshape(-1, 8))
        hs = ids[:, None] * (SRANS_HDR // 2) + 640 + 512 * torch.arange(2, device=dev)[None, :]
        put(hflat, hs.reshape(-1), E[:, :2].reshape(-1, 1024), Wd[:, :2].reshape(-1, 1024), cw[:, :2].reshape(-1, 8))
        hdr[ids, 0:512:2] = (states & 0xFFFF).int()
        hdr[ids, 1:512:2] = (states >> 16).int()
        offs = torch.zeros((G, 64), dtype=torch.long, device=dev)
        offs[:, 2:iters] = st // 8
        offs[:, iters] = (st[:, -1] + padded[:, -1]) // 8
        hdr[ids, 512:640:2] = (offs & 0xFFFF).int()
        hdr[ids, 513:640:2] = (offs >> 16).int()

    def u16(v):
        v = v & 0xFFFF
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16).contiguous().view(torch.uint8)

    lut8 = lut.to(dev).to(torch.int32).contiguous().view(torch.uint8)
    coded = torch.cat((lut8, u16(hdr.reshape(-1)), u16(var)))
    return coded.view(-1, 16)


def pack_scales6(tiled_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """[n/tile_n, k/16, tile_n] one-byte scales (any 1-byte dtype) -> (codes uint8 [3/4 bytes], palette uint8 [64]),
    or None when the layer has more than 64 distinct scale bytes. The code of byte b of the flat tensor sits at bits
    [6b, 6b + 6) of the code stream, least significant bit first: four codes per three bytes."""
    flat = tiled_scales.contiguous().view(torch.uint8).reshape(-1)
    if flat.numel() % 4:
        return None
    values = torch.unique(flat)
    if values.numel() > 64:
        return None
    palette = torch.zeros(64, dtype=torch.uint8, device=flat.device)
    palette[: values.numel()] = values
    lut = torch.zeros(256, dtype=torch.uint8, device=flat.device)
    lut[values.long()] = torch.arange(values.numel(), dtype=torch.uint8, device=flat.device)
    codes = lut[flat.long()].view(-1, 4).to(torch.int32)
    c0, c1, c2, c3 = codes.unbind(1)
    packed = torch.stack(
        (c0 | ((c1 & 3) << 6), (c1 >> 2) | ((c2 & 15) << 4), (c2 >> 4) | (c3 << 2)), dim=1
    ).to(torch.uint8).reshape(-1)
    # round trip on the device before anyone relies on it
    q = packed.view(-1, 3).to(torch.int32)
    d = torch.stack((q[:, 0] & 63, (q[:, 0] >> 6) | ((q[:, 1] & 15) << 2), (q[:, 1] >> 4) | ((q[:, 2] & 3) << 4), q[:, 2] >> 2), 1)
    if not torch.equal(palette[d.reshape(-1).long()], flat):
        return None
    return packed, palette


def decode_linear6(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
    fixed_grid: bool = False,
) -> torch.Tensor:
    if not STATS["logged"]:
        STATS["logged"] = True
        logger.info("NVFP4 decode GEMV with 6-bit block scales: %d layers, %d kept uncompressed",
                    STATS["admitted"], STATS["kept"])
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        use_grid = fixed_grid and _gate_up is not None and x.numel() == k and (n, k) == (39936, 6656)
        kernel = _gate_up if use_grid else _gemv6
        if is_gate_up_srans(scales6):
            # only the coded-scale kernel can read this scale form
            if not use_grid or _gate_up_srans is None:
                raise RuntimeError("rANS-coded gate/up scales outside the one-row fixed grid")
            kernel = _gate_up_srans
        status = kernel(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (scale6) launch failed with CUDA error {status}")
    return output


def _decode6_fake(x, tiled_weight, scales6, palette, global_scale, workspace, n, k, fixed_grid=False):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode6", decode_linear6, fake_impl=_decode6_fake)


def self_check6(layer: torch.nn.Module, reference, fixed_grid: bool = False) -> bool:
    """decode_linear6 == reference (the uncompressed tiled GEMV) bitwise on the layer's own weight, for activations with
    large and cancelling entries at several magnitudes."""
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    dev = layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 7 + k)
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        x = x.to(torch.bfloat16)
        got = decode_linear6(x, layer._marlin_decode_weight, layer._marlin_decode_scales6, layer._marlin_decode_palette,
                             layer.weight_global_scale, layer.workspace, n, k, fixed_grid)
        ref = reference(x, layer._marlin_decode_weight, layer._marlin_decode_scales, layer.weight_global_scale,
                        layer.workspace, n, k)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    return True


def _gate_up_probes(layer: torch.nn.Module) -> list[torch.Tensor]:
    """One-row activations for the coded-scale admission: self_check6's large and cancelling entries at three
    magnitudes and two plain Gaussian rows, all BF16."""
    k, dev = layer.input_size_per_partition, layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(GATE_UP_N * 7 + k + 1)
    out = []
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        out.append(x.to(torch.bfloat16))
    for mag in (0.5, 4.0):
        out.append((torch.randn(1, k, generator=gen, device=dev) * mag).to(torch.bfloat16))
    return out


def _gate_up_grid_outputs(layer, weight, scales, probes) -> list[torch.Tensor]:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    return [decode_linear6(x, weight, scales, layer._marlin_decode_palette, layer.weight_global_scale,
                           layer.workspace, n, k, True) for x in probes]


def _six_bit_codes(layer: torch.nn.Module) -> torch.Tensor:
    """The layer's six-bit scale codes, rebuilt from the tiled one-byte scales."""
    packed = pack_scales6(layer._marlin_decode_scales)
    if packed is None or not torch.equal(packed[1], layer._marlin_decode_palette):
        raise ValueError("six-bit codes cannot be rebuilt")
    return packed[0]


def _same_bits(expected, got) -> bool:
    return all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, got))


def check_gate_up_srans(layer: torch.nn.Module) -> bool:
    """For a layer with rANS-coded gate/up scales: True iff the coded kernel on the current buffers equals, bit for
    bit, the admitted six-bit fixed grid with codes rebuilt from the one-byte scales. Depends only on the buffers'
    contents, so it can be re-run after a placement step copies the buffers elsewhere."""
    c = layer._marlin_decode_scales6
    if not is_gate_up_srans(c) or _gate_up_srans is None:
        return False
    probes = _gate_up_probes(layer)
    expected = _gate_up_grid_outputs(layer, layer._marlin_decode_weight, _six_bit_codes(layer), probes)
    return _same_bits(expected, _gate_up_grid_outputs(layer, layer._marlin_decode_weight, c, probes))


def revert_gate_up_srans(layer: torch.nn.Module) -> None:
    """Put rANS-coded gate/up scales back to six-bit codes (into the same storage when the coded buffer still lives
    at the start of the six-bit code buffer it was written into)."""
    c = layer._marlin_decode_scales6
    if is_gate_up_srans(c):
        codes = _six_bit_codes(layer)
        base = c._base
        if (base is not None and base.dim() == 1 and base.dtype == torch.uint8 and base.numel() == codes.numel()
                and base.data_ptr() == c.data_ptr()):
            base.copy_(codes)
            codes = base
        layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)
    layer.marlin_decode6_gate_up_srans = False


def admit_gate_up_srans(layer: torch.nn.Module) -> bool:
    """For an admitted fixed-grid gate/up layer with six-bit codes: encode its scales (a pure function of the tiled
    one-byte scales) and switch _marlin_decode_scales6 to the coded buffer only if the coded kernel reproduces
    every output bit of the six-bit fixed grid on the probes. Otherwise nothing changes."""
    c = layer._marlin_decode_scales6
    if is_gate_up_srans(c):  # already coded (e.g. after a placement move): only re-check
        layer.marlin_decode6_gate_up_srans = check_gate_up_srans(layer)
        return layer.marlin_decode6_gate_up_srans
    layer.marlin_decode6_gate_up_srans = False
    if (_gate_up_srans is None or not getattr(layer, "marlin_decode6_fixed_grid", False)
            or (layer.output_size_per_partition, layer.input_size_per_partition) != (GATE_UP_N, GATE_UP_K)
            or getattr(layer, "_marlin_decode_scales", None) is None):
        return False
    w = layer._marlin_decode_weight
    probes = _gate_up_probes(layer)
    expected = _gate_up_grid_outputs(layer, w, c, probes)
    coded = encode_gate_up_scales(layer._marlin_decode_scales, layer._marlin_decode_palette)
    if not _same_bits(expected, _gate_up_grid_outputs(layer, w, coded, probes)):
        return False
    if int(torch.count_nonzero(layer.workspace)) != 0:
        return False
    if coded.numel() <= c.numel() and c.dim() == 1 and c.is_contiguous():
        # write the coded bytes over the start of the six-bit code buffer, so the scale stream keeps the memory the
        # loader gave it (no late allocation); the temporary is freed
        c[:coded.numel()].copy_(coded.view(-1))
        coded = c[:coded.numel()].view(-1, 16)
        if not _same_bits(expected, _gate_up_grid_outputs(layer, w, coded, probes)):
            c.copy_(_six_bit_codes(layer))
            return False
    layer.register_buffer("_marlin_decode_scales6", coded, persistent=False)
    layer.marlin_decode6_gate_up_srans = True
    STATS["srans_scales"] += layer._marlin_decode_scales.numel()
    STATS["srans_bytes"] += coded.numel()
    return True


def prepare_decode_layout6(layer: torch.nn.Module, reference) -> bool:
    """After the board's prepare_decode_layout: add the 6-bit scale codes and palette, check, register. True if admitted."""
    layer.marlin_decode6_fixed_grid = False
    tiled = getattr(layer, "_marlin_decode_scales", None)
    if tiled is None or getattr(layer, "_marlin_decode_weight", None) is None:
        return False
    k = layer.input_size_per_partition
    packed = pack_scales6(tiled) if k % 128 == 0 else None
    if packed is None:
        STATS["kept"] += 1
        return False
    codes, palette = packed
    layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)
    layer.register_buffer("_marlin_decode_palette", palette, persistent=False)
    if not self_check6(layer, reference):
        del layer._marlin_decode_scales6, layer._marlin_decode_palette
        STATS["kept"] += 1
        return False
    if _gate_up is not None and (layer.output_size_per_partition, k) == (39936, 6656):
        try:
            layer.marlin_decode6_fixed_grid = self_check6(layer, reference, fixed_grid=True)
        except (RuntimeError, ValueError) as exc:
            logger.warning_once("NVFP4 gate/up grid admission failed on this layer: %s", exc)
        if layer.marlin_decode6_fixed_grid:
            STATS["fixed_grid"] += 1
            if _gate_up_srans is not None and os.environ.get("MUSE_GATE_UP_SRANS", "1") != "0":
                try:
                    coded_ok = admit_gate_up_srans(layer)
                except (RuntimeError, ValueError) as exc:
                    logger.warning_once("NVFP4 rANS gate/up scale admission failed on this layer: %s", exc)
                    coded_ok = False
                STATS["gate_up_srans" if coded_ok else "gate_up_srans_rejected"] += 1
    STATS["admitted"] += 1
    return True
