"""pfx: a Marlin-exact NVFP4 W4A16 GEMM for the 512-row prompt pass (see csrc/pfx.cu).

libpfx.so computes, from the same Marlin-format tensors (no repack), exactly what vLLM v0.27.1's
``ops.marlin_gemm`` computes for an NVFP4 linear at M = 512 rows with bf16 activations, fp32
reduce and no atomics, but streams each weight from DRAM once instead of once per 64-row
sub-problem. ``marlin_units`` transcribes Marlin's stripe partition into the unit table that
fixes every output element's summation tree.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "available", "load_error", "marlin_units", "units_tensor", "attrs", "Plan", "gemm"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpfx.so")

THREAD_K, THREAD_N, THREAD_M_BLOCKS = 64, 256, 4
MAX_SEG = 4

_lib = None
_load_error: str | None = None
_ready = False


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.pfx_version.restype = ci
        lib.pfx_smem_bytes.restype = ci
        lib.pfx_scratch_floats.argtypes = [ci]
        lib.pfx_scratch_floats.restype = ci
        lib.pfx_unit_ints.restype = ci
        lib.pfx_init.restype = ci
        lib.pfx_attrs.argtypes = [vp]
        lib.pfx_attrs.restype = ci
        lib.pfx_gemm.argtypes = [vp, cll, vp, vp, vp, vp, cll, ci, ci, vp, ci, vp, ci, vp]
        lib.pfx_gemm.restype = ci
        lib.pfx_version()
        _lib = lib
    except OSError as e:  # library missing or unloadable: callers fall back to stock Marlin
        _load_error = repr(e)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


def _init() -> bool:
    """Sets the kernel's dynamic shared memory limit (once, outside graph capture)."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.pfx_init()
    if rc != 0:
        raise RuntimeError(f"pfx_init failed: {rc}")
    _ready = True
    return True


def attrs() -> dict:
    lib = _load()
    assert lib is not None, _load_error
    out = (ctypes.c_int * 4)()
    rc = lib.pfx_attrs(out)
    if rc != 0:
        raise RuntimeError(f"pfx_attrs failed: {rc}")
    return {"regs": out[0], "static_smem": out[1], "max_threads": out[2], "local_bytes": out[3],
            "dyn_smem": lib.pfx_smem_bytes()}


# ------------------------------------------------------------------------------- Marlin partition
# vLLM v0.27.1 marlin_mm (csrc/libtorch_stable/quantization/marlin/marlin.cu) and the stripe
# partition of marlin_template.h, for bf16 activations, NVFP4 weights (group_blocks 1, no act
# order) and M a multiple of 64 small enough for one launch.

def _launches(m: int, n: int):
    """marlin_mm's row split: (row0, prob_m_split, thread_m_blocks) per kernel launch."""
    max_par = 16 * 8 if n <= 4096 else 16
    out, row, rest = [], 0, m
    while rest:
        par = min(rest // 64, max_par)
        split = par * 64 if par > 0 else rest
        out.append((row, split, min(-(-split // 16), 4)))
        row += split
        rest -= split
    return out


def marlin_units(m: int, n: int, k: int, sms: int = 48):
    """Work units: (row0, col, [(lo, hi) k-tile segments in fold order]) per 64-row sub-problem x
    256-feature column, column-major. A segment is the k-tile range one Marlin CTA reduces; fold
    order is Marlin's lock order (slice_idx 0 = highest K first)."""
    assert m % 64 == 0 and n % THREAD_N == 0 and k % THREAD_K == 0, (m, n, k)
    k_tiles, n_tiles = k // THREAD_K, n // THREAD_N
    tiles = []
    for row0, prob_m, tmb in _launches(m, n):
        assert tmb == THREAD_M_BLOCKS and prob_m % 64 == 0, "M must be a multiple of 64"
        # marlin_mm switches to {128, 64, 128} when the problem has few tiles
        assert not (n // THREAD_N * (-(-prob_m // 64)) * 4 <= sms), "Marlin picks {128, 64, 128} here"
        parallel = prob_m // 64 if prob_m > 64 else 1
        gmn = parallel * n_tiles
        part2 = gmn
        if gmn > sms:
            part2 = gmn % sms
            if part2 * 3 <= sms:
                part2 += sms
        iters = -(-k_tiles * part2 // sms)  # group_blocks 1 < thread_k_blocks 4: no rounding
        for gi in range(gmn):
            par_id, col = divmod(gi, n_tiles)
            if gi < gmn - part2:
                segs = [(0, k_tiles)]
            else:
                cp = gi - (gmn - part2)
                lo, hi = cp * k_tiles, (cp + 1) * k_tiles
                segs = []
                while lo < hi:
                    end = min(hi, (lo // iters + 1) * iters)
                    segs.append((lo - cp * k_tiles, end - cp * k_tiles))
                    lo = end
                segs.sort(key=lambda s: -s[0])
            assert len(segs) <= MAX_SEG, segs
            tiles.append((row0 + par_id * 64, col, segs))
    tiles.sort(key=lambda t: (t[1], t[0]))
    return tiles


def units_tensor(units, device) -> torch.Tensor:
    rows = []
    for row0, col, segs in units:
        lo = [s[0] for s in segs] + [0] * (MAX_SEG - len(segs))
        hi = [s[1] for s in segs] + [0] * (MAX_SEG - len(segs))
        rows.append([row0, col, len(segs), 0] + lo + hi)
    return torch.tensor(rows, dtype=torch.int32, device=device).contiguous()


class Plan:
    """Unit table and scratch for one (M, N, K) on one device."""

    def __init__(self, m: int, n: int, k: int, device, sms: int | None = None, grid: int | None = None):
        lib = _load()
        assert lib is not None, _load_error
        assert _init()
        idx = device.index if device.index is not None else torch.cuda.current_device()
        self.sms = sms or torch.cuda.get_device_properties(idx).multi_processor_count
        self.grid = grid or self.sms
        self.m, self.n, self.k = m, n, k
        self.units_py = marlin_units(m, n, k, self.sms)
        self.units = units_tensor(self.units_py, device)
        self.scratch = torch.zeros(lib.pfx_scratch_floats(self.grid), dtype=torch.float32, device=device)


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, global_scale: torch.Tensor,
         plan: Plan, out: torch.Tensor | None = None) -> torch.Tensor:
    lib = _load()
    assert lib is not None, _load_error
    m, k = x.shape
    n = plan.n
    assert m == plan.m and k == plan.k, (m, k, plan.m, plan.k)
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    assert weight.dtype == torch.int32 and tuple(weight.shape) == (k // 16, 2 * n) and weight.is_contiguous()
    assert weight_scale.dtype == torch.float8_e4m3fn and tuple(weight_scale.shape) == (k // 16, n)
    assert weight_scale.is_contiguous()
    assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    assert out.dtype == torch.bfloat16 and out.stride(1) == 1 and out.stride(0) % 8 == 0
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.pfx_gemm(x.data_ptr(), x.stride(0), weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
                      out.data_ptr(), out.stride(0), n, k, plan.units.data_ptr(), plan.units.shape[0],
                      plan.scratch.data_ptr(), plan.grid, stream)
    if rc != 0:
        raise RuntimeError(f"pfx_gemm failed: {rc} (M={m} N={n} K={k})")
    return out


# ---------------------------------------------------------------------------- engine entry points
# VLLM_PFX=0 disables everything below (the engine then runs stock Marlin for every call).
ENABLED = os.environ.get("VLLM_PFX", "1") != "0"
# The row counts and (N, K) shapes pfx runs. tests/pfx_exact.py verified it bit-exact against
# stock Marlin on every NVFP4 linear shape of Muse Glimmer 30B at M = 512; only the shapes where it
# is also faster are routed here (MLP gate_up 6656 -> 39936, down 19968 -> 6656). Every other call
# keeps stock Marlin.
ROWS = frozenset({512})
VERIFIED_SHAPES = frozenset({(39936, 6656), (6656, 19968), (4608, 6656), (4096, 6656), (6656, 4096)})
# Routed: gate_up only. Its tiles are single-segment under Marlin's partition, so the kernel never
# takes the multi-segment fold through the per-CTA scratch; down (two-segment tiles) stays on stock.
SHAPES = frozenset({(39936, 6656)})
_plans: dict[tuple[int, int, int, int], Plan] = {}
_logged = False


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 linear (after prepare_fp4_layer_for_marlin) of a verified shape and builds
    its per-device plans (load time, outside graph capture). Returns whether pfx may run it."""
    layer._pfx_ok = False
    if not ENABLED or _load() is None or torch.cuda.is_current_stream_capturing():
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and getattr(layer, "bias", None) is None
        and gs is not None and gs.dtype == torch.float32 and gs.numel() == 1
        and w.dtype == torch.int32 and tuple(w.shape) == (k // 16, 2 * n) and w.is_contiguous() and w.is_cuda
        and s.dtype == torch.float8_e4m3fn and tuple(s.shape) == (k // 16, n) and s.is_contiguous()
    )
    if not ok or not _init():
        return False
    idx = w.device.index if w.device.index is not None else torch.cuda.current_device()
    for m in ROWS:
        if (idx, m, n, k) not in _plans:
            _plans[(idx, m, n, k)] = Plan(m, n, k, w.device)
    layer._pfx_ok = True
    return True


def maybe_gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, global_scale: torch.Tensor,
               n: int, k: int) -> torch.Tensor | None:
    """Marlin-exact pfx output for a 2-D bf16 x of a verified row count, else None (caller runs stock)."""
    global _logged
    if not ENABLED or x.dim() != 2 or x.shape[0] not in ROWS or x.shape[1] != k or x.dtype != torch.bfloat16:
        return None
    if not x.is_cuda or x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        return None
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    plan = _plans.get((idx, x.shape[0], n, k))
    if plan is None:
        return None
    out = gemm(x, weight, weight_scale, global_scale, plan)
    if not _logged:
        _logged = True
        import logging

        logging.getLogger("vllm").info("pfx: Marlin-exact NVFP4 GEMM at M=%d (N=%d K=%d, grid %d)",
                                       x.shape[0], n, k, plan.grid)
    return out
