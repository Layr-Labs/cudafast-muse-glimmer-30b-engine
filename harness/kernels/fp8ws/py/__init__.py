# SPDX-License-Identifier: Apache-2.0
"""Zero-VRAM dual-TMA M=512 prefill FP8 GEMM for MuseGlimmer-30B-NVFP4.

From SP5D's Yukon submission 98c1361e-289f-45a3-8471-5df0259bd3b2 (harness/kernels/fp8ws), unchanged apart from SHAPES."""

import ctypes
import os
from pathlib import Path

import torch

M = 512
SHAPES = frozenset(
    {(4608, 6656), (4096, 6656)}
    | ({(6656, 4096), (6656, 19968)} if os.environ.get("FP8WS_6656", "1") != "0" else set())
)
_lib = None
_load_failed = False
_ready_devices: set[int] = set()


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8ws.so")))
        lib.fp8ws_init.argtypes = []
        lib.fp8ws_init.restype = ctypes.c_int
        lib.fp8ws_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.fp8ws_run.restype = ctypes.c_int
        lib.fp8ws_run_dual.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.fp8ws_run_dual.restype = ctypes.c_int
        if hasattr(lib, "fp8ws_run_dual_a"):
            lib.fp8ws_run_dual_a.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
            lib.fp8ws_run_dual_a.restype = ctypes.c_int
        if hasattr(lib, "bf16ws_run_dual_stacked"):
            lib.bf16ws_run_dual_stacked.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
            lib.bf16ws_run_dual_stacked.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def ready(device: torch.device) -> bool:
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)) or _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            if _lib.fp8ws_init() != 0:
                return False
            _ready_devices.add(index)
    return True


def gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """Compute a[512, K] @ b[K, N] (b col-major stride (1, K)) -> BF16[512, N]."""
    k, n = b.shape
    out = torch.empty((M, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8ws_run(
        a.data_ptr(),
        b.data_ptr(),
        sa.data_ptr(),
        sb.data_ptr(),
        out.data_ptr(),
        M,
        n,
        k,
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8ws launch failed: {rc} (n={n}, k={k})")
    return out


def gemm_dual(
    a: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    sa: torch.Tensor,
    sb0: torch.Tensor,
    sb1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (a[512, K] @ b0[K, N0], a[512, K] @ b1[K, N1]) via dual-TMA."""
    k, n0 = b0.shape
    k1, n1 = b1.shape
    assert k == k1
    out0 = torch.empty((M, n0), device=a.device, dtype=torch.bfloat16)
    out1 = torch.empty((M, n1), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8ws_run_dual(
        a.data_ptr(),
        b0.data_ptr(),
        b1.data_ptr(),
        sa.data_ptr(),
        sb0.data_ptr(),
        sb1.data_ptr(),
        out0.data_ptr(),
        out1.data_ptr(),
        M,
        n0,
        n1,
        k,
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8ws dual launch failed: {rc} (n0={n0}, n1={n1}, k={k})")
    return out0, out1


DUAL_A_STAGES = 3


def gemm_dual_a(a0, a1, b0, b1, sa0, sa1, sb0, sb1):
    """(a0[512, K] @ b0[K, 4608], a1[512, K] @ b1[K, 4096]) in one persistent launch, two activations with their own
    scales (prefill3: one layer's qkv and output-gate operands); each output as gemm() computes it."""
    if _lib is None or not hasattr(_lib, "fp8ws_run_dual_a"):
        return None
    k, n0 = b0.shape
    _, n1 = b1.shape
    out0 = torch.empty((M, n0), device=a0.device, dtype=torch.bfloat16)
    out1 = torch.empty((M, n1), device=a0.device, dtype=torch.bfloat16)
    rc = _lib.fp8ws_run_dual_a(a0.data_ptr(), a1.data_ptr(), b0.data_ptr(), b1.data_ptr(), sa0.data_ptr(),
                               sa1.data_ptr(), sb0.data_ptr(), sb1.data_ptr(), out0.data_ptr(), out1.data_ptr(), M, n0,
                               n1, k, DUAL_A_STAGES, torch.cuda.current_stream(a0.device).cuda_stream)
    if rc:
        return None
    return out0, out1


def bf16_gemm_dual_stacked(a: torch.Tensor, w_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute (a[512, 6656] @ w_cat[:4608].T, a[512, 6656] @ w_cat[4608:].T) directly into contiguous outputs."""
    if _lib is None or not hasattr(_lib, "bf16ws_run_dual_stacked"):
        return None
    out0 = torch.empty((M, 4608), device=a.device, dtype=torch.bfloat16)
    out1 = torch.empty((M, 4096), device=a.device, dtype=torch.bfloat16)
    rc = _lib.bf16ws_run_dual_stacked(
        a.data_ptr(), w_cat.data_ptr(), out0.data_ptr(), out1.data_ptr(),
        M, 4608, 4096, 6656, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        return None
    return out0, out1
