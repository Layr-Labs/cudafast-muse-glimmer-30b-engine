# SPDX-License-Identifier: Apache-2.0
"""GB10 512-row FP8 gate/up SwiGLU and 4.00-QMMA/LDSM TMA prefill GEMM dispatch."""
import ctypes
import os
from pathlib import Path
import threading

import torch

_HIDDEN = 6656
_INNER = 19968
_OPROJ_K = 4096
_ROWS = 512
_USE_WS = os.environ.get("MUSE_FP8_SWIGLU_WS", "1") != "0"
_USE_FP8WS_ACCEL = os.environ.get("MUSE_FP8_SWIGLU_FP8WS", "1") != "0"
_USE_DOWN = _USE_FP8WS_ACCEL and os.environ.get("MUSE_FP8_SWIGLU_DOWN", "1") != "0"
_USE_OPROJ = _USE_FP8WS_ACCEL and os.environ.get("MUSE_FP8_OPROJ_256", "1") != "0"
_USE_DUAL_A = _USE_FP8WS_ACCEL and os.environ.get("MUSE_FP8_SWIGLU_DUAL_A", "1") != "0"
_USE_BF16_DUAL = _USE_FP8WS_ACCEL and os.environ.get("MUSE_FP8_SWIGLU_BF16_DUAL", "1") != "0"

_library = None
_sms: dict[int, int] = {}
_ws_ready: dict[int, bool] = {}
_ws_admitted: dict[int, bool] = {}
_down_admitted: dict[int, bool] = {}
_oproj_admitted: dict[int, bool] = {}
_dual_a_admitted: dict[int, bool] = {}
_bf16_dual_admitted: dict[int, bool] = {}
_scratch_buf: dict[int, torch.Tensor] = {}
_lock = threading.Lock()


def _load():
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_fp8_swiglu.so")))
        ptr = ctypes.c_void_p
        ci = ctypes.c_int
        lib.init_silu.argtypes = [ptr]
        lib.init_silu.restype = ci
        lib.prefill.argtypes = [ci] * 3 + [ptr] * 7 + [ctypes.c_size_t, ci, ci, ptr]
        lib.prefill.restype = ci
        if hasattr(lib, "swiglu_ws_init") and hasattr(lib, "swiglu_ws_run"):
            lib.swiglu_ws_init.argtypes = [ptr]
            lib.swiglu_ws_init.restype = ci
            lib.swiglu_ws_run.argtypes = [ptr] * 7
            lib.swiglu_ws_run.restype = ci
        if hasattr(lib, "swiglu_ws_set_pdl"):
            lib.swiglu_ws_set_pdl.argtypes = [ci]
            lib.swiglu_ws_set_pdl.restype = ci
        if hasattr(lib, "muse_fp8_set_gemm_pdl"):
            lib.muse_fp8_set_gemm_pdl.argtypes = [ci]
            lib.muse_fp8_set_gemm_pdl.restype = ci
        if hasattr(lib, "muse_fp8_gemm_6656"):
            lib.muse_fp8_gemm_6656.argtypes = [ci, ci] + [ptr] * 5 + [ci, ptr]
            lib.muse_fp8_gemm_6656.restype = ci
        if hasattr(lib, "muse_fp8_dual_a"):
            lib.muse_fp8_dual_a.argtypes = [ci, ci] + [ptr] * 11
            lib.muse_fp8_dual_a.restype = ci
        if hasattr(lib, "muse_fp8_down_set_next_dead"):
            lib.muse_fp8_down_set_next_dead.argtypes = [ptr, ctypes.c_size_t]
            lib.muse_fp8_down_set_next_dead.restype = None
        if hasattr(lib, "muse_bf16_dual_stacked"):
            lib.muse_bf16_dual_stacked.argtypes = [ci, ci] + [ptr] * 5
            lib.muse_bf16_dual_stacked.restype = ci
        _library = lib
    return _library


def set_pdl(on: bool) -> bool | None:
    lib = _load()
    fn = getattr(lib, "swiglu_ws_set_pdl", None)
    if fn is None:
        return None
    return bool(fn(1 if on else 0))


def _ensure_ws_ready(device: int) -> bool:
    if device in _ws_ready:
        return _ws_ready[device]
    if torch.cuda.is_current_stream_capturing():
        return False
    with _lock, torch.cuda.device(device):
        if device in _ws_ready:
            return _ws_ready[device]
        try:
            if torch.cuda.get_device_capability(device) != (12, 1):
                _ws_ready[device] = False
                return False
            lib = _load()
            stream = torch.cuda.current_stream(device)
            if device not in _sms:
                if lib.init_silu(stream.cuda_stream):
                    _ws_ready[device] = False
                    return False
                stream.synchronize()
                _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
            if _sms[device] != 48 or not hasattr(lib, "swiglu_ws_init"):
                _ws_ready[device] = False
                return False
            ok = lib.swiglu_ws_init(stream.cuda_stream) == 0
            _ws_ready[device] = ok
            return ok
        except Exception:  # noqa: BLE001
            _ws_ready[device] = False
            return False


def _admit_ws(device: int, packed: torch.Tensor) -> bool:
    if not _USE_WS or not _ensure_ws_ready(device) or not hasattr(_library, "swiglu_ws_run"):
        return False
    try:
        dev = packed.device
        stream = torch.cuda.current_stream(device)
        g = torch.Generator(device=dev).manual_seed(20261131)
        a_rand = (torch.randn((_ROWS, _HIDDEN), device=dev, generator=g, dtype=torch.float32) * 12.0).clamp(
            -448.0, 448.0
        ).to(torch.float8_e4m3fn)
        bits = (
            torch.arange(_ROWS * _HIDDEN, device=dev, dtype=torch.int64) * 7
            + torch.arange(_ROWS, device=dev, dtype=torch.int64).repeat_interleave(_HIDDEN)
        )
        a_bits = (bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(_ROWS, _HIDDEN)
        sa = torch.tensor([0.04], dtype=torch.float32, device=dev)
        sb = torch.tensor([0.03], dtype=torch.float32, device=dev)
        sd = torch.tensor([0.08], dtype=torch.float32, device=dev)
        scratch = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
        for act in (a_rand, a_bits):
            ref = torch.empty((_ROWS, _INNER), dtype=torch.float8_e4m3fn, device=dev)
            got = torch.empty((_ROWS, _INNER), dtype=torch.float8_e4m3fn, device=dev)
            rc0 = _library.prefill(
                _ROWS, 2 * _INNER, _HIDDEN, act.data_ptr(), packed.data_ptr(), sa.data_ptr(),
                sb.data_ptr(), sd.data_ptr(), ref.data_ptr(), scratch.data_ptr(),
                scratch.numel(), device, _sms[device], stream.cuda_stream,
            )
            rc1 = _library.swiglu_ws_run(
                act.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(), sd.data_ptr(),
                got.data_ptr(), stream.cuda_stream,
            )
            stream.synchronize()
            if rc0 != 0 or rc1 != 0 or not torch.equal(ref.view(torch.uint8), got.view(torch.uint8)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def prepare(weight):
    """Return a column-major FP8 weight with alternating gate and up columns."""
    if (not weight.is_cuda or weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (_HIDDEN, 2 * _INNER)
            or weight.stride() != (1, _HIDDEN)
            or torch.cuda.get_device_capability(weight.device) != (12, 1)):
        return None
    device = weight.device.index if weight.device.index is not None else torch.cuda.current_device()
    with _lock, torch.cuda.device(device):
        if device not in _sms:
            try:
                lib = _load()
            except OSError:
                return None
            stream = torch.cuda.current_stream(device)
            if lib.init_silu(stream.cuda_stream):
                return None
            stream.synchronize()
            _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    original = weight.t().view(torch.uint8)
    gate, up = original.split(_INNER)
    packed = torch.stack((gate, up), dim=1).reshape(2 * _INNER, _HIDDEN).view(weight.dtype).t()
    if device not in _ws_admitted and not torch.cuda.is_current_stream_capturing():
        admitted = _admit_ws(device, packed)
        with _lock:
            _ws_admitted[device] = admitted
    return packed


def apply(a, b, scale_a, scale_b, down_scale):
    if (a.shape != (_ROWS, _HIDDEN) or a.dtype != torch.float8_e4m3fn or not a.is_contiguous()
            or b.shape != (_HIDDEN, 2 * _INNER) or b.stride() != (1, _HIDDEN)
            or b.dtype != a.dtype or not a.is_cuda or b.device != a.device
            or any(s.numel() != 1 or s.dtype != torch.float32 or s.device != a.device
                   for s in (scale_a, scale_b, down_scale))):
        raise ValueError("invalid GB10 FP8 gate/up operands")
    device = a.device.index if a.device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        out = a.new_empty((_ROWS, _INNER))
        if _USE_WS and _ws_admitted.get(device) and (((a.data_ptr() | b.data_ptr() | out.data_ptr()) & 15) == 0):
            rc = _library.swiglu_ws_run(
                a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
                down_scale.data_ptr(), out.data_ptr(), stream.cuda_stream,
            )
            if rc == 0:
                return out
        if torch.cuda.is_current_stream_capturing():
            scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
        else:
            scratch = _scratch_buf.get(device)
            if scratch is None:
                scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
                _scratch_buf[device] = scratch
        status = _library.prefill(
            _ROWS, 2 * _INNER, _HIDDEN, a.data_ptr(), b.data_ptr(), scale_a.data_ptr(),
            scale_b.data_ptr(), down_scale.data_ptr(), out.data_ptr(), scratch.data_ptr(),
            scratch.numel(), device, _sms[device], stream.cuda_stream,
        )
        if status:
            raise RuntimeError(f"GB10 FP8 gate/up CUTLASS status {status}")
        return out


_L2_DISCARD_V2 = os.environ.get("MUSE_PREFILL_L2_DISCARD_V2", "1") != "0"
_dead_mlp_in = None


def stash_dead_mlp_in(t) -> None:
    global _dead_mlp_in
    if _L2_DISCARD_V2:
        _dead_mlp_in = t


def _consume_dead_mlp_in(live_tensors):
    global _dead_mlp_in
    t = _dead_mlp_in
    _dead_mlp_in = None
    fn = getattr(_library, "muse_fp8_down_set_next_dead", None) if _library is not None else None
    if fn is None:
        return None
    if not _L2_DISCARD_V2 or t is None or not t.is_cuda or not t.is_contiguous():
        fn(0, 0)
        return None
    dp = t.data_ptr()
    db = t.numel() * t.element_size()
    de = dp + db
    for lt in live_tensors:
        if lt is None:
            continue
        lp = lt.data_ptr()
        le = lp + lt.numel() * lt.element_size()
        if not (de <= lp or le <= dp):
            fn(0, 0)
            return None
    fn(dp, db)
    return t


def _install_fp8ws_accel() -> None:
    if not _USE_FP8WS_ACCEL:
        return
    try:
        import fp8ws as _fp8ws
    except ImportError:
        return
    if getattr(_fp8ws, "_muse_fp8_swiglu_hooked", False):
        return

    orig_set_pdl = getattr(_fp8ws, "set_pdl", None)
    orig_gemm = getattr(_fp8ws, "gemm", None)
    orig_dual_a = getattr(_fp8ws, "gemm_dual_a", None)
    orig_bf16_dual = getattr(_fp8ws, "bf16_gemm_dual_stacked", None)

    if orig_set_pdl is not None:
        def _set_pdl_both(on: bool):
            prev = orig_set_pdl(on)
            try:
                lib = _load()
                fn = getattr(lib, "muse_fp8_set_gemm_pdl", None)
                if fn is not None:
                    fn(1 if on else 0)
            except Exception:  # noqa: BLE001
                pass
            return prev

        _fp8ws.set_pdl = _set_pdl_both

    if (_USE_DOWN or _USE_OPROJ) and orig_gemm is not None:
        def _gemm_accel(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
            k_dim = 0
            admit_map = None
            if b.ndim == 2 and a.ndim == 2 and b.shape[1] == _HIDDEN and a.shape[0] == _ROWS:
                if _USE_DOWN and b.shape[0] == _INNER and a.shape[1] == _INNER:
                    k_dim = _INNER
                    admit_map = _down_admitted
                elif _USE_OPROJ and b.shape[0] == _OPROJ_K and a.shape[1] == _OPROJ_K:
                    k_dim = _OPROJ_K
                    admit_map = _oproj_admitted
            if (
                k_dim != 0
                and a.is_cuda
                and b.device == a.device
                and a.dtype == torch.float8_e4m3fn
                and b.dtype == torch.float8_e4m3fn
                and a.is_contiguous()
                and b.stride() == (1, k_dim)
                and ((a.data_ptr() | b.data_ptr()) & 15) == 0
            ):
                dev_idx = a.device.index if a.device.index is not None else torch.cuda.current_device()
                admitted = admit_map.get(dev_idx)
                if admitted is None and not torch.cuda.is_current_stream_capturing():
                    admitted = False
                    if _ensure_ws_ready(dev_idx) and hasattr(_library, "muse_fp8_gemm_6656"):
                        try:
                            dev = a.device
                            st = torch.cuda.current_stream(dev_idx)
                            g = torch.Generator(device=dev).manual_seed(20261127 + k_dim)
                            a_r = (torch.randn((_ROWS, k_dim), device=dev, generator=g, dtype=torch.float32) * 12.0).clamp(
                                -448.0, 448.0
                            ).to(torch.float8_e4m3fn)
                            bits = (
                                torch.arange(_ROWS * k_dim, device=dev, dtype=torch.int64) * 7
                                + torch.arange(_ROWS, device=dev, dtype=torch.int64).repeat_interleave(k_dim)
                            )
                            a_b = (bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(_ROWS, k_dim)
                            ok = True
                            for act in (a, a_r, a_b):
                                ref = orig_gemm(act, b, sa, sb)
                                got = torch.empty((_ROWS, _HIDDEN), device=dev, dtype=torch.bfloat16)
                                if (got.data_ptr() & 15) != 0:
                                    ok = False
                                    break
                                rc = _library.muse_fp8_gemm_6656(
                                    2, -1, act.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                                    got.data_ptr(), k_dim, st.cuda_stream,
                                )
                                st.synchronize()
                                if rc != 0 or not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                                    ok = False
                                    break
                            admitted = ok
                        except Exception:  # noqa: BLE001
                            admitted = False
                    admit_map[dev_idx] = admitted
                if admitted:
                    out = torch.empty((_ROWS, _HIDDEN), device=a.device, dtype=torch.bfloat16)
                    if (out.data_ptr() & 15) == 0:
                        dead_t = _consume_dead_mlp_in((a, b, sa, sb, out)) if k_dim == _INNER else None
                        rc = _library.muse_fp8_gemm_6656(
                            2, -1, a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                            out.data_ptr(), k_dim, torch.cuda.current_stream(dev_idx).cuda_stream,
                        )
                        del dead_t
                        if rc == 0:
                            if k_dim == _INNER:
                                try:
                                    import muse_sandwich as _ms
                                    _ms.stash_dead_down_in(a)
                                except Exception:  # noqa: BLE001
                                    pass
                            return out
            return orig_gemm(a, b, sa, sb)

        _fp8ws.gemm = _gemm_accel

    if _USE_DUAL_A and orig_dual_a is not None:
        def _dual_a_accel(a0, a1, b0, b1, sa0, sa1, sb0, sb1):
            if (
                a0.ndim == 2
                and a1.ndim == 2
                and b0.ndim == 2
                and b1.ndim == 2
                and a0.shape == (_ROWS, _HIDDEN)
                and a1.shape == (_ROWS, _HIDDEN)
                and b0.shape == (_HIDDEN, 4608)
                and b1.shape == (_HIDDEN, 4096)
                and a0.is_cuda
                and a0.device == a1.device == b0.device == b1.device
                and a0.dtype == torch.float8_e4m3fn
                and a1.dtype == torch.float8_e4m3fn
                and b0.dtype == torch.float8_e4m3fn
                and b1.dtype == torch.float8_e4m3fn
                and a0.is_contiguous()
                and a1.is_contiguous()
                and b0.stride() == (1, _HIDDEN)
                and b1.stride() == (1, _HIDDEN)
                and ((a0.data_ptr() | a1.data_ptr() | b0.data_ptr() | b1.data_ptr()) & 15) == 0
            ):
                dev_idx = a0.device.index if a0.device.index is not None else torch.cuda.current_device()
                admitted = _dual_a_admitted.get(dev_idx)
                if admitted is None and not torch.cuda.is_current_stream_capturing():
                    admitted = False
                    if _ensure_ws_ready(dev_idx) and hasattr(_library, "muse_fp8_dual_a"):
                        try:
                            dev = a0.device
                            st = torch.cuda.current_stream(dev_idx)
                            bits = (
                                torch.arange(_ROWS * _HIDDEN, device=dev, dtype=torch.int64) * 7
                                + torch.arange(_ROWS, device=dev, dtype=torch.int64).repeat_interleave(_HIDDEN)
                            )
                            ab0 = (bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(_ROWS, _HIDDEN)
                            ab1 = ((bits + 37) % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(_ROWS, _HIDDEN)
                            ok = True
                            for x0, x1 in ((a0, a1), (ab0, ab1)):
                                ref = orig_dual_a(x0, x1, b0, b1, sa0, sa1, sb0, sb1)
                                if ref is None:
                                    ok = False
                                    break
                                o0 = torch.empty((_ROWS, 4608), device=dev, dtype=torch.bfloat16)
                                o1 = torch.empty((_ROWS, 4096), device=dev, dtype=torch.bfloat16)
                                if ((o0.data_ptr() | o1.data_ptr()) & 15) != 0:
                                    ok = False
                                    break
                                rc = _library.muse_fp8_dual_a(
                                    2, -1, x0.data_ptr(), x1.data_ptr(), b0.data_ptr(), b1.data_ptr(),
                                    sa0.data_ptr(), sa1.data_ptr(), sb0.data_ptr(), sb1.data_ptr(),
                                    o0.data_ptr(), o1.data_ptr(), st.cuda_stream,
                                )
                                st.synchronize()
                                if (
                                    rc != 0
                                    or not torch.equal(ref[0].view(torch.int16), o0.view(torch.int16))
                                    or not torch.equal(ref[1].view(torch.int16), o1.view(torch.int16))
                                ):
                                    ok = False
                                    break
                            admitted = ok
                        except Exception:  # noqa: BLE001
                            admitted = False
                    _dual_a_admitted[dev_idx] = admitted
                if admitted:
                    o0 = torch.empty((_ROWS, 4608), device=a0.device, dtype=torch.bfloat16)
                    o1 = torch.empty((_ROWS, 4096), device=a0.device, dtype=torch.bfloat16)
                    if ((o0.data_ptr() | o1.data_ptr()) & 15) == 0:
                        rc = _library.muse_fp8_dual_a(
                            2, -1, a0.data_ptr(), a1.data_ptr(), b0.data_ptr(), b1.data_ptr(),
                            sa0.data_ptr(), sa1.data_ptr(), sb0.data_ptr(), sb1.data_ptr(),
                            o0.data_ptr(), o1.data_ptr(), torch.cuda.current_stream(dev_idx).cuda_stream,
                        )
                        if rc == 0:
                            return o0, o1
            return orig_dual_a(a0, a1, b0, b1, sa0, sa1, sb0, sb1)

        _fp8ws.gemm_dual_a = _dual_a_accel

    if _USE_BF16_DUAL and orig_bf16_dual is not None:
        def _bf16_dual_accel(a: torch.Tensor, w_cat: torch.Tensor):
            if (
                a.ndim == 2
                and w_cat.ndim == 2
                and a.shape == (_ROWS, _HIDDEN)
                and w_cat.shape == (8704, _HIDDEN)
                and a.is_cuda
                and w_cat.device == a.device
                and a.dtype == torch.bfloat16
                and w_cat.dtype == torch.bfloat16
                and a.is_contiguous()
                and w_cat.is_contiguous()
                and ((a.data_ptr() | w_cat.data_ptr()) & 15) == 0
            ):
                dev_idx = a.device.index if a.device.index is not None else torch.cuda.current_device()
                admitted = _bf16_dual_admitted.get(dev_idx)
                if admitted is None and not torch.cuda.is_current_stream_capturing():
                    admitted = False
                    if _ensure_ws_ready(dev_idx) and hasattr(_library, "muse_bf16_dual_stacked"):
                        try:
                            dev = a.device
                            st = torch.cuda.current_stream(dev_idx)
                            g = torch.Generator(device=dev).manual_seed(20261128)
                            a_r = torch.randn((_ROWS, _HIDDEN), device=dev, generator=g, dtype=torch.bfloat16)
                            ok = True
                            for act in (a, a_r):
                                ref = orig_bf16_dual(act, w_cat)
                                if ref is None:
                                    ok = False
                                    break
                                o0 = torch.empty((_ROWS, 4608), device=dev, dtype=torch.bfloat16)
                                o1 = torch.empty((_ROWS, 4096), device=dev, dtype=torch.bfloat16)
                                if ((o0.data_ptr() | o1.data_ptr()) & 15) != 0:
                                    ok = False
                                    break
                                rc = _library.muse_bf16_dual_stacked(
                                    2, -1, act.data_ptr(), w_cat.data_ptr(),
                                    o0.data_ptr(), o1.data_ptr(), st.cuda_stream,
                                )
                                st.synchronize()
                                if (
                                    rc != 0
                                    or not torch.equal(ref[0].view(torch.int16), o0.view(torch.int16))
                                    or not torch.equal(ref[1].view(torch.int16), o1.view(torch.int16))
                                ):
                                    ok = False
                                    break
                            admitted = ok
                        except Exception:  # noqa: BLE001
                            admitted = False
                    _bf16_dual_admitted[dev_idx] = admitted
                if admitted:
                    o0 = torch.empty((_ROWS, 4608), device=a.device, dtype=torch.bfloat16)
                    o1 = torch.empty((_ROWS, 4096), device=a.device, dtype=torch.bfloat16)
                    if ((o0.data_ptr() | o1.data_ptr()) & 15) == 0:
                        rc = _library.muse_bf16_dual_stacked(
                            2, -1, a.data_ptr(), w_cat.data_ptr(),
                            o0.data_ptr(), o1.data_ptr(), torch.cuda.current_stream(dev_idx).cuda_stream,
                        )
                        if rc == 0:
                            return o0, o1
            return orig_bf16_dual(a, w_cat)

        _fp8ws.bf16_gemm_dual_stacked = _bf16_dual_accel

    _fp8ws._muse_fp8_swiglu_hooked = True


_install_fp8ws_accel()
