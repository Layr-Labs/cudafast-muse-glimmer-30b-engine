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
attn_stats = {"calls": 0}  # final-layer attention slab launches (for the prefill graph's capture log line)


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
    # prefill3: data_ptr of the final layer's post-attention norm weight and of the model's final norm weight; the
    # compiled launches that read them run only rows 448..511 under the capture context (see _restrict_rows).
    post_attn_norm_address: int = 0
    final_norm_address: int = 0
    qkv_plan: object = None  # (weight data_ptr, work list, grid) of the final layer's restricted qkv tiles

    @contextmanager
    def capture(self):
        _restricted_inputs.clear()
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
    if not RESTRICT:
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


def prepare(layer_or_mlp, final_norm=None):
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
        *_restriction_addresses(layer_or_mlp, final_norm),
        _prepare_qkv(getattr(self_attn, "qkv_proj", None)),
    )


def attn_if_active(x, weight, scales, global_scale):
    active = _active.get()
    if active is None:
        return None
    qplan = active[0].qkv_plan
    if qplan is not None and RESTRICT and weight.data_ptr() == qplan[0]:
        out = _run_qkv(x, weight, scales, global_scale, qplan)
        if out is not None:
            restrict_stats["qkv"] += 1
            return out
    aplan = active[0].attn_plans.get(weight.data_ptr())
    if aplan is None:
        return None
    out = _run_attn_slab(x, weight, scales, global_scale, aplan)
    if out is not None:
        attn_stats["calls"] += 1
        if aplan.n == 4096:  # the output gate: the sigmoid(gate) * attention launch that reads it runs 64 rows
            _restricted_inputs.add(out.data_ptr())
    return out


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
    if not RESTRICT:
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


# --------------------------------------------------------------------------- final rows only (prefill3)
# Under the capture context only rows 448..511 of the last layer's output reach the sampled row. The compiled
# per-row launches after the final layer's attention (post-attention norm, sigmoid(gate) * attention of the slabbed
# gate, post-FFN + final norm) are launched on those 64 rows only: the same compiled kernel (or the norm-glue kernel
# that replaces it) with every row tensor offset by 448 rows and the row count 64. Each of these kernels computes a
# row from that row alone (one program per row / one element per index), so rows 448..511 get the same bits; rows
# 0..447 are never read by anything that reaches the sampled row, so the slabs' zero fills are skipped as well.
# MUSE_FINAL_ROWS_ONLY=0 restores 512-row launches and the fills.
import os as _os

RESTRICT = _os.environ.get("MUSE_FINAL_ROWS_ONLY", "1") == "1"
_restricted_inputs = set()
restrict_stats = {"qkv": 0, "post_attn": 0, "sigmoid": 0, "final_norm": 0}
_POST_ATTN = ("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1", "triton_red_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_1")
_FINAL = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3"
_SIGMOID = "triton_poi_fused_mul_sigmoid_0"
_installed_restrict = False


def _restriction_addresses(layer, final_norm):
    ln = getattr(layer, "post_attention_layernorm", None)
    w = getattr(ln, "weight", None)
    fw = getattr(final_norm, "weight", None)
    if not RESTRICT or w is None or fw is None:
        return 0, 0
    _install_restrict()
    return w.data_ptr(), fw.data_ptr()


def _narrow(args, rows_index, per_row):
    """Offset every 512-row tensor argument by 448 rows; the row-count argument becomes 64 (x per_row)."""
    first = ROWS - TILE_ROWS
    out = list(args)
    for i, t in enumerate(args):
        if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == ROWS and t.numel() >= ROWS:
            out[i] = t[first:]
    out[rows_index] = TILE_ROWS * per_row
    return out


def _install_restrict():
    global _installed_restrict
    if _installed_restrict:
        return
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    inner = CachingAutotuner.run

    def run(autotuner, *args, **kwargs):
        active = _active.get()
        if RESTRICT and active is not None and not kwargs.get("benchmark_run", False):
            tile = active[0]
            name = getattr(getattr(autotuner, "fn", None), "__name__", None)
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            if name in _POST_ATTN and len(args) >= 8 and args[-2] == ROWS and tile.post_attn_norm_address \
                    and args[2].data_ptr() == tile.post_attn_norm_address:
                restrict_stats["post_attn"] += 1
                return inner(autotuner, *_narrow(args, len(args) - 2, 1), **kwargs)
            if name == _FINAL and len(args) == 11 and args[9] == ROWS and tile.final_norm_address \
                    and args[6].data_ptr() == tile.final_norm_address:
                restrict_stats["final_norm"] += 1
                return inner(autotuner, *_narrow(args, 9, 1), **kwargs)
            if name == _SIGMOID and len(args) == 4 and args[3] == ROWS * 4096 and any(
                    t.data_ptr() in _restricted_inputs for t in tensors[:2]):
                if all(t.shape == (ROWS, 4096) and t.is_contiguous() for t in tensors[:3]):
                    restrict_stats["sigmoid"] += 1
                    return inner(autotuner, *_narrow(args, 3, 4096), **kwargs)
        return inner(autotuner, *args, **kwargs)

    CachingAutotuner.run = run
    _installed_restrict = True


# The final layer's qkv projection (NVFP4 4608x6656 on the 128x128 TMA tile, no K cuts): the sampled row needs K and V
# of all 512 rows (they enter the KV cache and every row's attention) but Q only of rows 448..511. The restricted plan
# launches the same kernel on the 16 K/V tiles of all row chunks plus the 32 Q tiles of the last 128-row chunk (48
# tiles instead of 144); each tile is computed exactly as in the full launch. Q of rows 0..383 is left unwritten: it
# only feeds those rows' own attention outputs, which reach nothing that is sampled.
_QKV = (4608, 6656)


def _prepare_qkv(proj):
    if not RESTRICT or proj is None or marlinw_attn is None:
        return None
    try:
        import marlinw_down128 as D
    except (ImportError, OSError):
        return None
    path = getattr(proj, "marlin_prompt_path", None)
    if path is None or path[0] != "narrow_attn" or getattr(D, "_tma_shape_main", None) is None:
        return None
    if (proj.output_size_per_partition, proj.input_size_per_partition) != _QKV or not proj.weight.is_cuda:
        return None
    dev = proj.weight.device
    plan = D._plans.get((dev, *_QKV))
    if plan is None:
        plan = D._plans[(dev, *_QKV)] = D.prepare_shape(dev, *_QKV)
    if plan.tma_work is None or plan.tma_grid != 144:
        return None
    rows = plan.tma_work.cpu().tolist()
    keep = [b for b in range(plan.tma_grid) if rows[2 * b][1] >= 32 or rows[2 * b][0] == 384]
    if len(keep) != 48:
        return None
    work = [rows[r] for b in keep for r in (2 * b, 2 * b + 1)] + rows[2 * plan.tma_grid:]
    qplan = (proj.weight.data_ptr(), torch.tensor(work, dtype=torch.int32, device=dev), len(keep), plan.tma_slots)
    w, ws, gs = proj.weight, proj.weight_scale, proj.weight_global_scale
    src_work, src_ptr, slots = path[1], path[2], path[3]
    gen = torch.Generator(device=dev).manual_seed(4608 * 19 + 51)
    for mag in (0.05, 2.0, 32.0):
        x = (torch.randn((ROWS, _QKV[1]), generator=gen, device=dev) * mag).to(torch.bfloat16).contiguous()
        narrow = marlinw_attn.gemm(x, w, ws, gs, src_work, src_ptr, slots)
        tile = D.gemm(x, w, ws, gs, plan, True)
        got = _run_qkv(x, w, ws, gs, qplan)
        if narrow is None or tile is None or got is None:
            return None
        for ref in (narrow, tile):
            if not (torch.equal(ref[384:].view(torch.int16), got[384:].view(torch.int16))
                    and torch.equal(ref[:, 4096:].view(torch.int16), got[:, 4096:].view(torch.int16))):
                return None
    return qplan


def _run_qkv(x, weight, scales, global_scale, qplan):
    import marlinw_down128 as D

    if x.shape != (ROWS, _QKV[1]) or x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.stride(0) % 8 \
            or x.data_ptr() % 16:
        return None
    out = x.new_empty((ROWS, _QKV[0]))
    scratch = D._get_scratch(x.device, qplan[3])
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = D._tma_shape_main(x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                                   scales.data_ptr(), global_scale.data_ptr(), qplan[1].data_ptr(), qplan[2],
                                   _QKV[0], _QKV[1], x.stride(0), stream)
    if status:
        return None
    return out
