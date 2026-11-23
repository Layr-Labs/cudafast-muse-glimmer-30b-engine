"""Capture only the live final layer tiles (self_attn gate/o_proj + FFN) while retaining operator shapes."""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes as C
from dataclasses import dataclass, field
from pathlib import Path

import torch
import marlinw_down
from marlinw import swiglu

try:
    import marlinw_attn
except (ImportError, OSError):
    marlinw_attn = None

ROWS = 512
TILE_ROWS = 64
_native = {}
_active = ContextVar("muse_final_slab", default=None)


@dataclass(frozen=True)
class SampleTile:
    gate_address: int
    gate_weight: torch.Tensor
    gate_scales: torch.Tensor
    gate_global: torch.Tensor
    down_address: int
    work: torch.Tensor
    offsets: torch.Tensor
    slots: int
    attn_layers: dict = field(default_factory=dict)

    @contextmanager
    def capture(self):
        calls = [0, 0, 0]
        token = _active.set((self, calls))
        try:
            yield
            expected = [1, 1, len(self.attn_layers)]
            if calls != expected:
                raise RuntimeError(f"Expected live final tiles {expected}, captured {calls}")
        finally:
            _active.reset(token)


def _prepare_attn_proj(proj, device):
    if marlinw_attn is None or not marlinw_attn.ENABLED:
        return None
    path = getattr(proj, "marlin_prompt_path", None)
    if path is None or path[0] != "narrow_attn" or not proj.weight.is_cuda:
        return None
    n = getattr(proj, "output_size_per_partition", 0)
    k = getattr(proj, "input_size_per_partition", 0)
    if (n, k) not in marlinw_attn.SHAPES:
        return None
    source_work = path[1].tolist()
    last = ROWS // TILE_ROWS - 1
    slot_map = {}
    filtered = []
    for it in source_work:
        if it[0] != last:
            continue
        cnt = it[5]
        slot = slot_map.setdefault(it[6], len(slot_map)) if cnt > 1 else 0
        filtered.append((0, it[1], it[2], it[3], it[4], cnt, slot, 0))
    if not filtered:
        return None
    if marlinw_attn.forward_waits(filtered) >= 2 * marlinw_attn.SMS:
        return None
    starts = list(range(len(filtered) + 1))
    return (
        torch.tensor(filtered, device=device, dtype=torch.int32),
        torch.tensor(starts, device=device, dtype=torch.int32),
        len(slot_map),
        n,
        k,
    )


def prepare(target):
    mlp = getattr(target, "mlp", target)
    self_attn = getattr(target, "self_attn", None)
    gate, down = mlp.gate_up_proj, mlp.down_proj
    path = getattr(down, "marlin_prompt_path", None)
    if not (getattr(gate, "marlin_swiglu_ready", False) and
            path is not None and path[0] == "wide" and path[2][0] is not None and
            gate.tp_size == down.tp_size == 1 and gate.weight.is_cuda):
        return None
    device = gate.weight.device
    if (torch.cuda.get_device_capability(device) != (12, 1) or
            torch.cuda.get_device_properties(device).multi_processor_count != marlinw_down.SMS):
        return None
    if device not in _native:
        if not swiglu.prepare(gate):
            return None
        root = Path(__file__).parent
        gate_lib = C.CDLL(str(root / "libmuse_sample_gate.so"))
        down_lib = C.CDLL(str(root / "libmuse_sample_down.so"))
        project_gate = gate_lib.swiglu_gemm
        project_gate.argtypes = swiglu._lib.swiglu_gemm.argtypes
        project_gate.restype = C.c_int
        project_down = down_lib.marlinw_down_gemm
        project_down.argtypes = marlinw_down._launch.argtypes
        project_down.restype = C.c_int
        occupancy = C.c_int()
        with torch.cuda.device(device):
            if gate_lib.swiglu_prepare() or down_lib.marlinw_down_init(C.byref(occupancy)):
                return None
        _native[device] = project_gate, project_down
    source_work, source_offsets, slots = path[2]
    items = source_work.tolist()
    starts = source_offsets.tolist()
    work, pointers = [], [0]
    last = ROWS // TILE_ROWS - 1
    for a, b in zip(starts, starts[1:]):
        group = items[a:b]
        if group[0][0] == last:
            if any(task[0] != last for task in group):
                return None
            work.extend((0, *item[1:]) for item in group)
            pointers.append(len(work))
    if len(pointers) != 1 + marlinw_down.COLUMNS // 128:
        return None
    admitted_weight = getattr(gate, "swiglu_paired_weight", gate.weight)
    attn_layers = {}
    if self_attn is not None:
        for attr in ("output_gate_proj", "o_proj"):
            proj = getattr(self_attn, attr, None)
            if proj is not None:
                entry = _prepare_attn_proj(proj, device)
                if entry is not None:
                    attn_layers[proj.weight.data_ptr()] = entry
    return SampleTile(
        admitted_weight.data_ptr(), gate.weight, gate.weight_scale, gate.weight_global_scale,
        down.weight.data_ptr(), torch.tensor(work, device=device, dtype=torch.int32),
        torch.tensor(pointers, device=device, dtype=torch.int32), slots, attn_layers,
    )


def attn_if_active(x, weight, scales, global_scale, n, k):
    active = _active.get()
    if active is None or marlinw_attn is None:
        return None
    plan, calls = active
    entry = plan.attn_layers.get(weight.data_ptr())
    if entry is None:
        return None
    work, work_ptr, slots, en, ek = entry
    if (n != en or k != ek or x.dim() != 2 or x.shape != (ROWS, k)
            or x.dtype != torch.bfloat16 or x.stride(1) != 1
            or x.stride(0) < k or x.stride(0) % 8 or x.data_ptr() % 16):
        return None
    out = x.new_empty((ROWS, n))
    tile = x[-TILE_ROWS:]
    dest = out[-TILE_ROWS:]
    partial, locks = marlinw_attn._get_scratch(x.device, slots)
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = marlinw_attn._launch(
            tile.data_ptr(), weight.data_ptr(), dest.data_ptr(), partial.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), locks.data_ptr(),
            work.data_ptr(), work_ptr.data_ptr(), TILE_ROWS, n, k, tile.stride(0),
            work_ptr.numel() - 1, stream,
        )
    if status:
        raise RuntimeError(f"Final attn slab launch failed with status {status}")
    calls[2] += 1
    return out


def gate_if_active(x, weight):
    active = _active.get()
    if active is None or weight.data_ptr() != active[0].gate_address:
        return None
    plan, calls = active
    if x.shape != (ROWS, marlinw_down.COLUMNS) or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("Final gate capture requires contiguous BF16 prefill input")
    out = x.new_empty((ROWS, marlinw_down.INNER))
    # Skip zeroing rows 0..447 on the intermediate gate_up buffer: down_if_active only reads out[-64:].
    tile = x[-TILE_ROWS:]
    dest = out[-TILE_ROWS:]
    g, _ = _native[x.device]
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = g(tile.data_ptr(), plan.gate_weight.data_ptr(), plan.gate_scales.data_ptr(),
                   plan.gate_global.data_ptr(), dest.data_ptr(), tile.stride(0), stream)
    if status:
        raise RuntimeError(f"Final gate launch failed with status {status}")
    calls[0] += 1
    return out


def down_if_active(x, weight, scales, global_scale):
    active = _active.get()
    if active is None or weight.data_ptr() != active[0].down_address:
        return None
    plan, calls = active
    if x.shape != (ROWS, marlinw_down.INNER) or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("Final down capture requires contiguous BF16 prefill input")
    out = x.new_empty((ROWS, marlinw_down.COLUMNS))
    out[:-TILE_ROWS].zero_()
    tile = x[-TILE_ROWS:]
    dest = out[-TILE_ROWS:]
    partial, locks = marlinw_down._get_scratch(x.device, plan.slots)
    _, d = _native[x.device]
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = d(tile.data_ptr(), weight.data_ptr(), dest.data_ptr(), partial.data_ptr(),
                   scales.data_ptr(), global_scale.data_ptr(), locks.data_ptr(), plan.work.data_ptr(),
                   plan.offsets.data_ptr(), tile.stride(0), stream)
    if status:
        raise RuntimeError(f"Final down launch failed with status {status}")
    calls[1] += 1
    return out
