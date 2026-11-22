# SPDX-License-Identifier: Apache-2.0
"""decsc: the one-row tile-major Marlin NVFP4 GEMV (harness/kernels/decode) reading 6-bit block-scale codes.

``prepare(layer)`` takes a layer already prepared by muse_marlin_gemv.prepare_decode_layout (tile-major
``_marlin_decode_weight`` / ``_marlin_decode_scales``), builds a table of the layer's distinct scale bytes (at most
64) and one 6-bit index per scale in the same tile-major order, checks that the codes decode to every scale byte and
that the kernel's output equals the decode package's GEMV bit for bit on three inputs, and registers the state
(keyed by the tile-major scale tensor). ``torch.ops.vllm.marlin_tiled_decode_sc6`` is the launcher. Input-independent
and lossless; 25% fewer scale bytes, 2.8% fewer bytes per one-row NVFP4 call. DECSC=0 disables it.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

ENABLED = os.environ.get("DECSC", "1") != "0"
# 6 (default): 6-bit codes only; 5: prefer the 5-bit code (31-entry table + per-stage escapes) where a layer admits
# it. On GB10 the 5-bit kernel was slower than the 6-bit one on every shape (run
# glim-exec-dec2-decsct-20261122-030214: 0.87 vs 1.13 ms per token saved), so it is opt-in.
BITS = 5 if os.environ.get("DECSC_BITS", "6") == "5" else 6
MAX_ESC = 8  # escape entries per (tile, k-tile) stage of the 5-bit code
# (N, K) -> bits for shapes where a code did not measure faster than the record GEMV on GB10 (0: keep 8-bit scales,
# 6: no 5-bit code); filled from tests/decsc_time.py.
SHAPE_LIMIT: dict[tuple[int, int], int] = {}
_lib = ctypes.CDLL(str(Path(__file__).with_name("libdecsc.so")))
_gemv = _lib.muse_marlin_gemv_sc6
_gemv.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv.restype = ctypes.c_int
_gemv5 = _lib.muse_marlin_gemv_sc5
_gemv5.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
_gemv5.restype = ctypes.c_int

# tile-major scale tensor data_ptr -> (bits, codes, table, escapes per stage)
STATES: dict[int, tuple[int, torch.Tensor, torch.Tensor, int]] = {}
stats = {"layers": 0, "layers5": 0, "scale_bytes": 0, "code_bytes": 0, "rejected": 0, "last_reason": ""}


@torch.no_grad()
def encode(tiled_scales: torch.Tensor):
    """[N/tile, K/16, tile] scale bytes -> (codes [N/tile, K/16, tile*6/8] uint8, table [64] uint8), or None for more
    than 64 distinct bytes."""
    u = tiled_scales.view(torch.uint8)
    t, g, w = u.shape
    vals = torch.unique(u)
    if vals.numel() > 64:
        return None
    tab = torch.zeros(64, dtype=torch.uint8, device=u.device)
    tab[: vals.numel()] = vals
    idx = torch.searchsorted(vals, u.reshape(-1)).reshape(t, g, w // 4, 4).to(torch.int32)
    bits = idx[..., 0] | (idx[..., 1] << 6) | (idx[..., 2] << 12) | (idx[..., 3] << 18)
    codes = torch.stack([bits & 255, (bits >> 8) & 255, (bits >> 16) & 255], -1).to(torch.uint8)
    return codes.reshape(t, g, w * 3 // 4).contiguous(), tab


def decode(codes: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    t, g, cw = codes.shape
    c = codes.view(t, g, cw // 3, 3).to(torch.int32)
    bits = c[..., 0] | (c[..., 1] << 8) | (c[..., 2] << 16)
    idx = torch.stack([(bits >> s) & 63 for s in (0, 6, 12, 18)], -1).reshape(t, g, cw // 3 * 4).long()
    return tab[idx]


@torch.no_grad()
def encode5(tiled_scales: torch.Tensor):
    """[N/128, K/16, 128] scale bytes -> (records [stages, 640 + 4 * ceil4(E)] uint8, table [32] uint8, E), or None
    when some stage needs more than MAX_ESC escapes. Stage s = (tile, k-tile) in tile-major order: 8 rows x 80 B of
    5-bit codes (code 31 = escape), then E u32 entries pos | byte << 16 (0xffff = unused)."""
    u = tiled_scales.view(torch.uint8)
    t, g, w = u.shape
    if w != 128 or g % 8:
        return None
    hist = torch.bincount(u.reshape(-1).long(), minlength=256)
    top = torch.argsort(hist, descending=True, stable=True)[:31]
    tab = torch.zeros(32, dtype=torch.uint8, device=u.device)
    tab[:31] = top.to(torch.uint8)
    lut = torch.full((256,), 31, dtype=torch.int64, device=u.device)
    lut[top] = torch.arange(31, device=u.device)
    stages = t * g // 8
    idx = lut[u.long()].view(stages, 1024)
    esc = idx == 31
    cnt = esc.sum(1)
    e = int(cnt.max().item()) if stages else 0
    if e > MAX_ESC:
        return None
    epad = (e + 3) // 4 * 4
    ent = torch.full((stages, max(epad, 1)), 0xFFFF, dtype=torch.int64, device=u.device)
    nz = esc.nonzero()
    if nz.numel():
        st, pos = nz[:, 0], nz[:, 1]
        start = cnt.cumsum(0) - cnt
        rank = torch.arange(st.numel(), device=u.device) - start[st]
        byte = u.view(stages, 1024)[st, pos].long()
        ent[st, rank] = pos | (byte << 16)
    ent = ent[:, :epad]
    pairs = idx.view(stages, 8, 16, 8)
    v = torch.zeros(stages, 8, 16, dtype=torch.int64, device=u.device)
    for j in range(8):
        v |= pairs[..., j] << (5 * j)
    codes = torch.stack([(v >> (8 * b)) & 255 for b in range(5)], -1).to(torch.uint8).view(stages, 640)
    entb = ent.to(torch.int32).view(torch.uint8).view(stages, 4 * epad)
    return torch.cat([codes, entb], 1).contiguous(), tab, e


def decode5(rec: torch.Tensor, tab: torch.Tensor, e: int, t: int, g: int) -> torch.Tensor:
    stages = rec.shape[0]
    c = rec[:, :640].view(stages, 8, 16, 5).long()
    v = sum(c[..., b] << (8 * b) for b in range(5))
    idx = torch.stack([(v >> (5 * j)) & 31 for j in range(8)], -1).view(stages, 1024)
    out = tab.long()[idx]
    if e:
        ent = rec[:, 640:640 + 4 * e].contiguous().view(torch.int32).long().view(stages, e)
        valid = (ent & 0xFFFF) != 0xFFFF
        st = torch.arange(stages, device=rec.device).view(-1, 1).expand(-1, e)[valid]
        pos = (ent & 0xFFFF)[valid]
        out[st, pos] = (ent >> 16)[valid] & 255
    return out.to(torch.uint8).view(t, g, 128)


def decode_linear_sc5(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    records: torch.Tensor,
    table: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
    esc: int,
) -> torch.Tensor:
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    blocks = 48
    partials = x.new_empty((blocks, 16, 128), dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _gemv5(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            records.data_ptr(), table.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, esc,
            blocks, x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"decsc 5-bit GEMV launch failed with CUDA error {status}")
    return output


def _fake5(x, tiled_weight, records, table, global_scale, workspace, n, k, esc):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode_sc5", decode_linear_sc5, fake_impl=_fake5)


def decode_linear_sc6(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    codes: torch.Tensor,
    table: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    # Mirrors muse_marlin_gemv.decode_linear (same blocks, scratch and locks).
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    blocks = 48
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        status = _gemv(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            codes.data_ptr(), table.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k,
            blocks, x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"decsc GEMV launch failed with CUDA error {status}")
    return output


def _fake(x, tiled_weight, codes, table, global_scale, workspace, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode_sc6", decode_linear_sc6, fake_impl=_fake)


def _reject(reason: str):
    stats["rejected"] += 1
    stats["last_reason"] = reason
    return None


def _check_inputs(k: int, device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(k * 7 + 1)
    a = torch.randn(1, k, generator=gen, device=device)
    b = a.clone()
    b[:, ::3] *= 64
    b[:, 1::2] = -b[:, ::2][:, : b[:, 1::2].shape[1]]
    return [a.to(torch.bfloat16), b.to(torch.bfloat16), (a * 1e-3).to(torch.bfloat16)]


def _same_as_record(layer, tw, ts, fn) -> bool:
    from muse_marlin_gemv import decode_linear
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    for x in _check_inputs(k, tw.device):
        ref = decode_linear(x, tw, ts, layer.weight_global_scale, layer.workspace, n, k)
        got = fn(x)
        if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
            return False
    return True


@torch.no_grad()
def prepare5(layer: torch.nn.Module):
    """The verified 5-bit state (records, table, escapes per stage) of a tile-major layer, or None."""
    tw = getattr(layer, "_marlin_decode_weight", None)
    ts = getattr(layer, "_marlin_decode_scales", None)
    if tw is None or ts is None or ts.dim() != 3 or ts.shape[2] != 128:
        return None
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    enc = encode5(ts)
    if enc is None:
        return None
    rec, tab, e = enc
    if not torch.equal(decode5(rec, tab, e, ts.shape[0], ts.shape[1]), ts.view(torch.uint8)):
        return None
    if not _same_as_record(layer, tw, ts, lambda x: decode_linear_sc5(
            x, tw, rec, tab, layer.weight_global_scale, layer.workspace, n, k, e)):
        return None
    return rec, tab, e


@torch.no_grad()
def prepare(layer: torch.nn.Module):
    """Build, verify and register the 5-bit (preferred, where admitted) or 6-bit scale codes of a layer with the
    tile-major decode layout. Returns (bits, codes, table, escapes per stage) or None."""
    if not ENABLED:
        return None
    tw = getattr(layer, "_marlin_decode_weight", None)
    ts = getattr(layer, "_marlin_decode_scales", None)
    if tw is None or ts is None:
        return None
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    limit = SHAPE_LIMIT.get((n, k), 5)
    if limit == 0:
        return None
    tile = 128 if n % 128 == 0 else 64
    if ts.element_size() != 1 or tuple(ts.shape) != (n // tile, k // 16, tile) or not ts.is_contiguous():
        return _reject(f"layer {n}x{k}: unexpected tile-major scale layout {tuple(ts.shape)}")
    enc = encode(ts)
    if enc is None:
        return _reject(f"layer {n}x{k}: more than 64 distinct scale bytes")
    codes, tab = enc
    if not torch.equal(decode(codes, tab), ts.view(torch.uint8)):
        return _reject(f"layer {n}x{k}: codes do not decode to the scales")
    if not _same_as_record(layer, tw, ts, lambda x: decode_linear_sc6(
            x, tw, codes, tab, layer.weight_global_scale, layer.workspace, n, k)):
        return _reject(f"layer {n}x{k}: output differs from the decode GEMV")
    state = (6, codes, tab, 0)
    if BITS == 5 and limit == 5:
        s5 = prepare5(layer)
        if s5 is not None:
            state = (5, s5[0], s5[1], s5[2])
            stats["layers5"] += 1
    STATES[ts.data_ptr()] = state
    stats["layers"] += 1
    stats["scale_bytes"] += ts.numel()
    stats["code_bytes"] += state[1].numel()
    return state
