# SPDX-License-Identifier: Apache-2.0
"""M = 1 tiled-layout NVFP4 decode GEMV with 6-bit palette and rANS-coded block scales.

Supports 6-bit palette scales on all layers and in-smem-header rANS-coded scales on MLP
gate/up (336-CTA fixed grid, 704 B slots) and MLP down_proj (48-CTA stream-K, 704 B slots).
Every layer is admitted only after a load-time bitwise check against the uncompressed tiled reference.
"""

import ctypes
import os
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
STATS = {
    "admitted": 0,
    "kept": 0,
    "fixed_grid": 0,
    "gate_up_srans": 0,
    "gate_up_srans_rejected": 0,
    "srans_scales": 0,
    "srans_bytes": 0,
    "logged": False,
}

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin6.so")))
_gemv6 = _lib.muse_marlin6_gemv
_gemv6.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv6.restype = ctypes.c_int
_gate_up = getattr(_lib, "muse_marlin6_gate_up", None)
if _gate_up is not None:
    _gate_up.argtypes = _gemv6.argtypes
    _gate_up.restype = ctypes.c_int
_gate_up_srans = getattr(_lib, "muse_marlin6_gate_up_srans", None)
if _gate_up_srans is not None:
    _gate_up_srans.argtypes = _gemv6.argtypes
    _gate_up_srans.restype = ctypes.c_int
_down_srans = getattr(_lib, "muse_marlin6_down_srans", None)
if _down_srans is not None:
    _down_srans.argtypes = _gemv6.argtypes
    _down_srans.restype = ctypes.c_int
_enc_p1 = getattr(_lib, "muse_srans_hdr_pass1", None)
if _enc_p1 is not None:
    _enc_p1.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p,
    ]
    _enc_p1.restype = ctypes.c_int
_enc_p2 = getattr(_lib, "muse_srans_hdr_pass2", None)
if _enc_p2 is not None:
    _enc_p2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    _enc_p2.restype = ctypes.c_int

GATE_UP_N, GATE_UP_K = 39936, 6656
DOWN_N, DOWN_K = 6656, 19968
SRANS_M_BITS, SRANS_M, SRANS_L = 10, 1024, 1 << 16
SRANS_CAP = 704
GATE_UP_HDR, GATE_UP_B0_OFF = 2688, 1280
DOWN_HDR, DOWN_B0_OFF = 3136, 1728

_SCHED_CACHE: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_sched(kind: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (kind, device)
    cached = _SCHED_CACHE.get(key)
    if cached is not None:
        return cached
    stages: list[tuple[int, int]] = []
    blk_off = [0]
    if kind == "gate_up":
        for col in range(288):
            for kt in range(52):
                stages.append((col, kt))
            blk_off.append(len(stages))
        for j in range(24):
            for kt in range(26, 52):
                stages.append((288 + j, kt))
            blk_off.append(len(stages))
        for j in range(24):
            for kt in range(0, 26):
                stages.append((288 + j, kt))
            blk_off.append(len(stages))
    else:
        # down_proj (6656x19968): 52 cols x 156 k-tiles, 48 blocks x 169 stages
        for b in range(48):
            for i in range(169):
                pos = 169 * b + i
                stages.append((pos // 156, pos % 156))
            blk_off.append(len(stages))
    res = (
        torch.tensor(stages, dtype=torch.int32, device=device).reshape(-1),
        torch.tensor(blk_off, dtype=torch.int32, device=device),
    )
    _SCHED_CACHE[key] = res
    return res


def _srans_table(counts: torch.Tensor, palette: torch.Tensor):
    counts = counts.cpu().long()
    pal = palette.cpu().long()
    total = int(counts.sum())
    f = torch.zeros(64, dtype=torch.long)
    for s in range(64):
        c = int(counts[s])
        if c:
            if int(pal[s]) >= 0x80:
                raise ValueError("rANS embedded BF16 scale LUT requires palette bytes < 0x80")
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
            bf16_scale = (int(pal[s]) & 0x7F) << 4
            lut[int(cum[s]) + slots] = bf16_scale | (slots << 11) | (int(f[s]) << 21)
    return f, cum, lut


def is_gate_up_srans(scales: torch.Tensor) -> bool:
    return scales.dim() == 2 and scales.shape[1] == 16 and scales.dtype == torch.uint8


def _encode_hdr_srans(tiled_scales: torch.Tensor, palette: torch.Tensor, kind: str) -> torch.Tensor:
    if _enc_p1 is None or _enc_p2 is None:
        raise RuntimeError("GPU rANS header encoder kernels unavailable")
    dev = tiled_scales.device
    if kind == "gate_up":
        n_tiles, k_tiles, num_blks = 312, 52, 336
        hdr_bytes, b0_off, off_slots = GATE_UP_HDR, GATE_UP_B0_OFF, 64
    else:
        n_tiles, k_tiles, num_blks = 52, 156, 48
        hdr_bytes, b0_off, off_slots = DOWN_HDR, DOWN_B0_OFF, 176

    raw = tiled_scales.contiguous().view(torch.uint8).reshape(n_tiles, k_tiles, 1024)
    vals = torch.unique(raw)
    inv = torch.full((256,), -1, dtype=torch.long, device=dev)
    pal = palette.long()
    if not torch.equal(pal[: vals.numel()], vals.long()):
        raise ValueError("palette does not list the scale bytes in order")
    inv[pal[: vals.numel()]] = torch.arange(vals.numel(), device=dev)
    sym = inv[raw.long()].to(torch.uint8).contiguous()
    f, cum, lut = _srans_table(torch.bincount(sym.reshape(-1).long(), minlength=64), palette)
    enc_f = f.to(dtype=torch.int32, device=dev)
    enc_c = cum.to(dtype=torch.int32, device=dev)

    stage_col_k, blk_stage_off = _get_sched(kind, dev)
    total_stages = stage_col_k.numel() // 2
    stage_rows16 = torch.empty(total_stages, dtype=torch.int32, device=dev)
    stage_cw = torch.empty(total_stages, dtype=torch.int64, device=dev)
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream().cuda_stream
        st = _enc_p1(
            sym.data_ptr(), k_tiles, stage_col_k.data_ptr(), blk_stage_off.data_ptr(),
            enc_f.data_ptr(), enc_c.data_ptr(), stage_rows16.data_ptr(), stage_cw.data_ptr(),
            num_blks, stream,
        )
        if st:
            raise RuntimeError(f"muse_srans_hdr_pass1 failed with CUDA error {st}")

    max_bytes = int(stage_rows16.max().item()) * 16
    if max_bytes > SRANS_CAP:
        raise ValueError(f"stage block ({max_bytes} B) exceeds decode slot ({SRANS_CAP} B)")

    rows16_h = stage_rows16.cpu().tolist()
    blk_off_h = blk_stage_off.cpu().tolist()
    offs_mat = [[0] * off_slots for _ in range(num_blks)]
    running = 0
    for b in range(num_blks):
        s0, s1 = blk_off_h[b], blk_off_h[b + 1]
        iters = s1 - s0
        for local_s in range(2, iters):
            offs_mat[b][local_s] = running
            running += rows16_h[s0 + local_s]
        offs_mat[b][iters] = running
    var_bytes = running * 16

    hdr_off = 4096
    var_off = hdr_off + num_blks * hdr_bytes
    coded = torch.zeros(var_off + var_bytes, dtype=torch.uint8, device=dev)
    coded[:4096].copy_(lut.to(dtype=torch.int32, device=dev).view(torch.uint8))
    offs_t = torch.tensor(offs_mat, dtype=torch.int32, device=dev)
    hdr_view = coded[hdr_off:var_off].view(num_blks, hdr_bytes)
    hdr_view[:, 1024 : 1024 + off_slots * 4].copy_(offs_t.view(torch.uint8))

    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream().cuda_stream
        bp = coded.data_ptr()
        st = _enc_p2(
            sym.data_ptr(), k_tiles, stage_col_k.data_ptr(), blk_stage_off.data_ptr(),
            enc_f.data_ptr(), enc_c.data_ptr(), stage_cw.data_ptr(),
            hdr_bytes, b0_off, SRANS_CAP, bp + hdr_off, bp + var_off,
            num_blks, stream,
        )
        if st:
            raise RuntimeError(f"muse_srans_hdr_pass2 failed with CUDA error {st}")
    return coded.view(-1, 16)


def encode_gate_up_scales(tiled_scales: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    return _encode_hdr_srans(tiled_scales, palette, "gate_up")


def encode_down_scales(tiled_scales: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    return _encode_hdr_srans(tiled_scales, palette, "down")


def pack_scales6(tiled_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
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
        logger.info("NVFP4 decode GEMV compressed scales: %d layers (%d rANS), %d kept uncompressed",
                    STATS["admitted"], STATS["gate_up_srans"], STATS["kept"])
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        use_grid = fixed_grid and _gate_up is not None and x.numel() == k and (n, k) == (GATE_UP_N, GATE_UP_K)
        kernel = _gate_up if use_grid else _gemv6
        if is_gate_up_srans(scales6):
            if use_grid and _gate_up_srans is not None:
                kernel = _gate_up_srans
            elif (n, k) == (DOWN_N, DOWN_K) and _down_srans is not None:
                kernel = _down_srans
            else:
                raise RuntimeError(f"rANS-coded scales unsupported for shape ({n}, {k})")
        status = kernel(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (scale6/srans) launch failed with CUDA error {status}")
    return output


def _decode6_fake(x, tiled_weight, scales6, palette, global_scale, workspace, n: int, k: int, fixed_grid: bool = False):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode6", decode_linear6, fake_impl=_decode6_fake)


def self_check6(layer: torch.nn.Module, reference, fixed_grid: bool = False) -> bool:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(n * 7 + k)
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=layer.weight.device) * mag
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


def _layer_probes(layer: torch.nn.Module) -> list[torch.Tensor]:
    n, k, dev = layer.output_size_per_partition, layer.input_size_per_partition, layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 7 + k + 1)
    out = []
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        out.append(x.to(torch.bfloat16))
    for mag in (0.5, 4.0):
        out.append((torch.randn(1, k, generator=gen, device=dev) * mag).to(torch.bfloat16))
    return out


def _layer_outputs(layer, weight, scales, probes, fixed_grid: bool) -> list[torch.Tensor]:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    return [decode_linear6(x, weight, scales, layer._marlin_decode_palette, layer.weight_global_scale,
                           layer.workspace, n, k, fixed_grid) for x in probes]


def _six_bit_codes(layer: torch.nn.Module) -> torch.Tensor:
    packed = pack_scales6(layer._marlin_decode_scales)
    if packed is None or not torch.equal(packed[1], layer._marlin_decode_palette):
        raise ValueError("six-bit codes cannot be rebuilt")
    return packed[0]


def _same_bits(expected, got) -> bool:
    return all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, got))


def check_gate_up_srans(layer: torch.nn.Module) -> bool:
    c = layer._marlin_decode_scales6
    fixed = getattr(layer, "marlin_decode6_fixed_grid", False)
    if not is_gate_up_srans(c):
        return False
    probes = _layer_probes(layer)
    expected = _layer_outputs(layer, layer._marlin_decode_weight, _six_bit_codes(layer), probes, fixed)
    return _same_bits(expected, _layer_outputs(layer, layer._marlin_decode_weight, c, probes, fixed))


def revert_gate_up_srans(layer: torch.nn.Module) -> None:
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


def _admit_mlp_srans(layer: torch.nn.Module, kind: str) -> bool:
    c = layer._marlin_decode_scales6
    fixed = (kind == "gate_up")
    if is_gate_up_srans(c):
        layer.marlin_decode6_gate_up_srans = check_gate_up_srans(layer)
        return layer.marlin_decode6_gate_up_srans
    layer.marlin_decode6_gate_up_srans = False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if kind == "gate_up":
        if _gate_up_srans is None or not getattr(layer, "marlin_decode6_fixed_grid", False) or (n, k) != (GATE_UP_N, GATE_UP_K):
            return False
    else:
        if _down_srans is None or (n, k) != (DOWN_N, DOWN_K):
            return False
    if getattr(layer, "_marlin_decode_scales", None) is None:
        return False
    w = layer._marlin_decode_weight
    probes = _layer_probes(layer)
    expected = _layer_outputs(layer, w, c, probes, fixed)
    coded = _encode_hdr_srans(layer._marlin_decode_scales, layer._marlin_decode_palette, kind)
    if not _same_bits(expected, _layer_outputs(layer, w, coded, probes, fixed)):
        return False
    if int(torch.count_nonzero(layer.workspace)) != 0:
        return False
    if coded.numel() <= c.numel() and c.dim() == 1 and c.is_contiguous():
        c[: coded.numel()].copy_(coded.view(-1))
        coded = c[: coded.numel()].view(-1, 16)
        if not _same_bits(expected, _layer_outputs(layer, w, coded, probes, fixed)):
            c.copy_(_six_bit_codes(layer))
            return False
    layer.register_buffer("_marlin_decode_scales6", coded, persistent=False)
    layer.marlin_decode6_gate_up_srans = True
    STATS["srans_scales"] += layer._marlin_decode_scales.numel()
    STATS["srans_bytes"] += coded.numel()
    return True


def admit_gate_up_srans(layer: torch.nn.Module) -> bool:
    return _admit_mlp_srans(layer, "gate_up")


def prepare_decode_layout6(layer: torch.nn.Module, reference) -> bool:
    layer.marlin_decode6_fixed_grid = False
    tiled = getattr(layer, "_marlin_decode_scales", None)
    if tiled is None or getattr(layer, "_marlin_decode_weight", None) is None:
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
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
    if _gate_up is not None and (n, k) == (GATE_UP_N, GATE_UP_K):
        try:
            layer.marlin_decode6_fixed_grid = self_check6(layer, reference, fixed_grid=True)
        except (RuntimeError, ValueError) as exc:
            logger.warning_once("NVFP4 gate/up grid admission failed on this layer: %s", exc)
        if layer.marlin_decode6_fixed_grid:
            STATS["fixed_grid"] += 1
            if _gate_up_srans is not None and os.environ.get("MUSE_GATE_UP_SRANS", "1") != "0":
                try:
                    coded_ok = _admit_mlp_srans(layer, "gate_up")
                except (RuntimeError, ValueError) as exc:
                    logger.warning_once("NVFP4 rANS gate/up scale admission failed on this layer: %s", exc)
                    coded_ok = False
                STATS["gate_up_srans" if coded_ok else "gate_up_srans_rejected"] += 1
    elif _down_srans is not None and (n, k) == (DOWN_N, DOWN_K) and os.environ.get("MUSE_DOWN_SRANS", "1") != "0":
        try:
            coded_ok = _admit_mlp_srans(layer, "down")
        except (RuntimeError, ValueError) as exc:
            logger.warning_once("NVFP4 rANS down_proj scale admission failed on this layer: %s", exc)
            coded_ok = False
        STATS["gate_up_srans" if coded_ok else "gate_up_srans_rejected"] += 1
    STATS["admitted"] += 1
    return True
