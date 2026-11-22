"""Capture only the live final FFN tile while retaining both operator shapes."""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes as C
from dataclasses import dataclass
from pathlib import Path

import torch
import marlinw
import marlinw_down
from marlinw import swiglu

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

logger = init_logger("vllm.muse_final_slab")
ROWS = 512
TILE_ROWS = 64
_CHECK_SCALES = (0.05, 1.0, 16.0)
_native = {}
_down_scratch = {}
_active = ContextVar("muse_final_slab", default=None)


@dataclass(frozen=True)
class SampleTile:
    gate_address: int
    gate_weight: torch.Tensor
    gate_scales: torch.Tensor
    gate_global: torch.Tensor
    down_address: int
    work: torch.Tensor | None = None
    offsets: torch.Tensor | None = None
    slots: int = 0
    gate_plan: marlinw.Plan | None = None
    down_plan: marlinw.Plan | None = None

    # Micro-attribution: Restricting the final prefill MLP layer inside CUDA graph capture to the
    # last 64-row token slab while preserving full operator shapes is inspired by @kaankolcu
    # (Yukon submission e7600084); the marlinw.gemm_last_chunk fallback and final_slab_shape Dynamo
    # routing for un-sliced layers are custom-implemented here.
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


def _check_last_chunk(layer, size_n: int, size_k: int, chunk_plan: marlinw.Plan) -> bool:
    device = layer.weight.device
    gen = torch.Generator(device=device)
    gen.manual_seed(size_n * 53 + size_k)
    x = torch.randn(ROWS, size_k, generator=gen, device=device)
    x[:, ::3] *= 64
    x[1::2, 1::2] = -x[1::2, ::2]
    x = x.to(torch.bfloat16)
    if getattr(layer, "marlin_prompt_path", None) is None:
        ref = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=size_n,
            size_k=size_k,
        )
    else:
        ref = marlinw.gemm(
            x, layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, size_n, size_k, lambda a, c: None,
        )
    if ref is None:
        return False
    got = torch.empty((TILE_ROWS, size_n), dtype=torch.bfloat16, device=device)
    marlinw.gemm_last_chunk(
        x[-TILE_ROWS:], got, layer.weight, layer.weight_scale, layer.weight_global_scale, chunk_plan
    )
    return torch.equal(got.view(torch.int16), ref[-TILE_ROWS:].view(torch.int16))


def prepare(mlp):
    gate, down = mlp.gate_up_proj, mlp.down_proj
    path = getattr(down, "marlin_prompt_path", None)
    if not ((path is None or path[0] == "wide") and gate.tp_size == down.tp_size == 1 and gate.weight.is_cuda):
        return None
    device = gate.weight.device
    cap = torch.cuda.get_device_capability(device)
    if (path is not None
            and path[0] == "wide"
            and getattr(gate, "marlin_swiglu_ready", False)
            and path[2][0] is not None
            and cap == (12, 1)
            and torch.cuda.get_device_properties(device).multi_processor_count == marlinw_down.SMS):
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
        sample_plan = SampleTile(
            admitted_weight.data_ptr(), gate.weight, gate.weight_scale, gate.weight_global_scale,
            down.weight.data_ptr(), torch.tensor(work, device=device, dtype=torch.int32),
            torch.tensor(pointers, device=device, dtype=torch.int32), slots,
        )
        if not _matches_full_launch(sample_plan, gate, down, path[2]):
            logger.warning("Final-row prefill tile differs from the full 512-row launches; "
                           "the prefill graph keeps the full final FFN")
            return None
        logger.info("Final-row prefill tile matched the full 512-row paired SwiGLU and split-K down "
                    "launches bitwise on the last %d rows (%d inputs)", TILE_ROWS, len(_CHECK_SCALES))
        return sample_plan

    gate_path = getattr(gate, "marlin_prompt_path", None)
    if ((gate_path is not None and gate_path[0] != "wide")
            or getattr(gate, "marlin_swiglu_ready", False)
            or getattr(gate, "marlin_swiglu_fp8_ready", False)
            or cap not in ((12, 0), (12, 1))
            or not (marlinw.prepare(gate) and marlinw.prepare(down))
            or gate.output_size_per_partition != 2 * marlinw_down.INNER
            or gate.input_size_per_partition != marlinw_down.COLUMNS
            or down.output_size_per_partition != marlinw_down.COLUMNS
            or down.input_size_per_partition != marlinw_down.INNER):
        return None
    gate_plan = marlinw.get_last_chunk_plan(ROWS, 2 * marlinw_down.INNER, marlinw_down.COLUMNS, device)
    down_plan = marlinw.get_last_chunk_plan(ROWS, marlinw_down.COLUMNS, marlinw_down.INNER, device)
    if gate_plan is None or down_plan is None:
        return None
    if not (_check_last_chunk(gate, 2 * marlinw_down.INNER, marlinw_down.COLUMNS, gate_plan)
            and _check_last_chunk(down, marlinw_down.COLUMNS, marlinw_down.INNER, down_plan)):
        return None
    logger.info("Final-row prefill tile matched the full 512-row Marlin launches bitwise on the last %d rows",
                TILE_ROWS)
    return SampleTile(
        gate.weight.data_ptr(), gate.weight, gate.weight_scale, gate.weight_global_scale,
        down.weight.data_ptr(), gate_plan=gate_plan, down_plan=down_plan,
    )


def _matches_full_launch(plan, gate, down, down_plan):
    """The single-slab gate/up and down launches against the last TILE_ROWS
    rows of the full launches the prefill runs otherwise (the paired SwiGLU and
    the narrow split-K down, each admitted against ops.marlin_gemm at load),
    bitwise, on this layer's weights and generated inputs."""
    device = gate.weight.device
    gen = torch.Generator(device=device)
    gen.manual_seed(6113)
    token = _active.set((plan, [0, 0]))
    try:
        for scale in _CHECK_SCALES:
            x = torch.randn((ROWS, marlinw_down.COLUMNS), generator=gen, device=device)
            x = (x * scale).to(torch.bfloat16)
            got = gate_if_active(x, gate.weight)
            full = swiglu.gemm(x, gate.weight, gate.weight_scale, gate.weight_global_scale)
            if got is None or not torch.equal(got[-TILE_ROWS:].view(torch.int16),
                                              full[-TILE_ROWS:].view(torch.int16)):
                return False
            got = down_if_active(full, down.weight, down.weight_scale, down.weight_global_scale)
            ref = marlinw_down.gemm(full, down.weight, down.weight_scale, down.weight_global_scale,
                                    *down_plan)
            if got is None or ref is None or not torch.equal(got[-TILE_ROWS:].view(torch.int16),
                                                             ref[-TILE_ROWS:].view(torch.int16)):
                return False
        return True
    except (RuntimeError, ValueError) as error:
        logger.warning("Final-row prefill tile check failed: %s", error)
        return False
    finally:
        _active.reset(token)


def gate_if_active(x, weight):
    active = _active.get()
    if active is None or weight.data_ptr() != active[0].gate_address or active[0].gate_plan is not None:
        return None
    plan, calls = active
    if x.shape != (ROWS, marlinw_down.COLUMNS) or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("Final gate capture requires contiguous BF16 prefill input")
    out = x.new_empty((ROWS, marlinw_down.INNER))
    out[:-TILE_ROWS].zero_()
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
    if active is None or weight.data_ptr() != active[0].down_address or active[0].down_plan is not None:
        return None
    plan, calls = active
    if x.shape != (ROWS, marlinw_down.INNER) or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("Final down capture requires contiguous BF16 prefill input")
    out = x.new_empty((ROWS, marlinw_down.COLUMNS))
    out[:-TILE_ROWS].zero_()
    tile = x[-TILE_ROWS:]
    dest = out[-TILE_ROWS:]
    key = (x.device, int(plan.slots))
    scratch = _down_scratch.get(key)
    if scratch is None:
        scratch = (
            x.new_empty((plan.slots, TILE_ROWS, 128), dtype=torch.float32),
            x.new_zeros((plan.slots,), dtype=torch.int32),
        )
        _down_scratch[key] = scratch
    partial, locks = scratch
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


def wide_if_active(x, weight, scales, global_scale):
    active = _active.get()
    if active is None:
        return None
    plan, calls = active
    addr = weight.data_ptr()
    if addr == plan.gate_address and plan.gate_plan is not None:
        chunk_plan = plan.gate_plan
        call_idx = 0
    elif addr == plan.down_address and plan.down_plan is not None:
        chunk_plan = plan.down_plan
        call_idx = 1
    else:
        return None
    if (x.shape != (ROWS, chunk_plan.k) or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or x.data_ptr() % 16):
        raise ValueError("Final wide tile capture requires 16-byte aligned contiguous BF16 input")
    out = x.new_empty((ROWS, chunk_plan.n))
    out[:-TILE_ROWS].zero_()
    marlinw.gemm_last_chunk(
        x[-TILE_ROWS:], out[-TILE_ROWS:], weight, scales, global_scale, chunk_plan
    )
    calls[call_idx] += 1
    return out
