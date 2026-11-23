"""Capture only the live final layer tiles (attention gate/o_proj + FFN) while retaining operator shapes."""

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
class AttnSlabPlan:
    n: int
    k: int
    work: torch.Tensor
    work_ptr: torch.Tensor
    slots: int


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
    attn_plans: dict[int, AttnSlabPlan] = field(default_factory=dict)

    @contextmanager
    def capture(self):
        calls = [0, 0]
        token = _active.set((self, calls))
        try:
            yield
            if calls != [1, 1]:
                raise RuntimeError(f"Expected one live gate and down tile, captured {calls}")
        finally:
            _active.reset(token)


def _prepare_attn_proj(proj) -> AttnSlabPlan | None:
    if marlinw_attn is None or proj is None:
        return None
    path = getattr(proj, "marlin_prompt_path", None)
    if path is None or path[0] != "narrow_attn":
        return None
    w, ws, gs = proj.weight, proj.weight_scale, proj.weight_global_scale
    n, k = proj.output_size_per_partition, proj.input_size_per_partition
    if (n, k) not in ((4096, 6656), (6656, 4096)) or not w.is_cuda:
        return None
    dev = w.device
    src_work, _src_ptr, slots = path[1], path[2], path[3]
    last = ROWS // TILE_ROWS - 1
    items = src_work.cpu().tolist()
    c7 = [(0, *it[1:]) for it in items if it[0] == last]
    if not c7:
        return None
    w7 = torch.tensor(c7, device=dev, dtype=torch.int32)
    p7 = torch.arange(len(c7) + 1, device=dev, dtype=torch.int32)
    plan = AttnSlabPlan(n, k, w7, p7, slots)
    # Bitwise check on Layer 51's actual weight against full 512-row marlinw_attn.gemm
    gen = torch.Generator(device=dev).manual_seed(n * 19 + k + 51)
    for mag in (0.05, 2.0, 32.0):
        x = (torch.randn((ROWS, k), generator=gen, device=dev) * mag).to(torch.bfloat16).contiguous()
        ref = marlinw_attn.gemm(x, w, ws, gs, src_work, _src_ptr, slots)
        got = _run_attn_slab(x, w, ws, gs, plan)
        if ref is None or got is None or not torch.equal(ref[-TILE_ROWS:].view(torch.int16), got[-TILE_ROWS:].view(torch.int16)):
            return None
    return plan


def _run_attn_slab(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor, plan: AttnSlabPlan):
    if (
        x.ndim != 2
        or x.shape != (ROWS, plan.k)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, plan.n))
    out[:-TILE_ROWS].zero_()
    tile = x[-TILE_ROWS:]
    dest = out[-TILE_ROWS:]
    partial, locks = marlinw_attn._get_scratch(x.device, plan.slots)
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = marlinw_attn._launch(
            tile.data_ptr(), weight.data_ptr(), dest.data_ptr(), partial.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), locks.data_ptr(),
            plan.work.data_ptr(), plan.work_ptr.data_ptr(), TILE_ROWS, plan.n, plan.k,
            tile.stride(0), plan.work_ptr.numel() - 1, stream,
        )
    if status:
        return None
    return out


def prepare(layer_or_mlp):
    mlp = getattr(layer_or_mlp, "mlp", layer_or_mlp)
    self_attn = getattr(layer_or_mlp, "self_attn", None)
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
        project_gate.restype = C.C_INT if hasattr(C, "C_INT") else C.c_int
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
    attn_plans: dict[int, AttnSlabPlan] = {}
    if self_attn is not None:
        for attr in ("output_gate_proj", "o_proj"):
            proj = getattr(self_attn, attr, None)
            try:
                aplan = _prepare_attn_proj(proj)
            except Exception:
                aplan = None
            if aplan is not None:
                attn_plans[proj.weight.data_ptr()] = aplan
    admitted_weight = getattr(gate, "swiglu_paired_weight", gate.weight)
    return SampleTile(
        admitted_weight.data_ptr(), gate.weight, gate.weight_scale, gate.weight_global_scale,
        down.weight.data_ptr(), torch.tensor(work, device=device, dtype=torch.int32),
        torch.tensor(pointers, device=device, dtype=torch.int32), slots, attn_plans,
    )


def attn_if_active(x, weight, scales, global_scale):
    active = _active.get()
    if active is None:
        return None
    aplan = active[0].attn_plans.get(weight.data_ptr())
    if aplan is None:
        return None
    return _run_attn_slab(x, weight, scales, global_scale, aplan)


def gate_if_active(x, weight):
    active = _active.get()
    if active is None or weight.data_ptr() != active[0].gate_address:
        return None
    plan, calls = active
    if x.shape != (ROWS, marlinw_down.COLUMNS) or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("Final gate capture requires contiguous BF16 prefill input")
    out = x.new_empty((ROWS, marlinw_down.INNER))
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
