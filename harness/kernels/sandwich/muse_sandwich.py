"""Use the native row staging for the two verified Inductor sandwich kernels."""

import ast
from collections import Counter
import ctypes
from functools import cache
from hashlib import sha256
from pathlib import Path


_library = ctypes.CDLL(str(Path(__file__).with_name("libmuse_sandwich.so")))
_launch = _library.muse_sandwich
_pointer = ctypes.c_void_p
_launch.argtypes = [ctypes.POINTER(_pointer), ctypes.c_int, ctypes.c_bool, ctypes.c_bool, _pointer]
_launch.restype = ctypes.c_int
_launch_counts = Counter()
_WIDTH = 6656

_symbols = {
    "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3",
    "triton_red_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_3",
}
_verified_bodies = {
    "096b4dbdbe218cba4368a8a1de38f0bab983e69643a25662d9d9832d8b4bf1a2": False,
    "6611b90dd2aa0ecd3ed3f75b03e5c73895b89554458babc3ab7088c92130e290": True,
}
_inputs = ("*bf16", "*bf16", "*bf16", "*fp32", "*bf16", "*bf16", "*bf16")
_dimensions = ("i32", "i32", "constexpr", "constexpr")


@cache
def _mode(source):
    definitions = ast.parse(source).body
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return None
    kernel = definitions[0]
    kernel.name = "sandwich"
    kernel.decorator_list = []
    body = ast.dump(kernel, include_attributes=False)
    return _verified_bodies.get(sha256(body.encode()).hexdigest())


def _match(autotuner, args):
    meta = autotuner.__dict__.get("_sandwich_meta", 0)
    if meta == 0:
        meta = None
        function = autotuner.fn
        props = autotuner.device_props
        if (getattr(function, "__name__", None) in _symbols
                and props.type == "cuda" and props.cc in (120, 121)
                and len(autotuner.launchers) == 1 and autotuner._cached_launcher):
            config = autotuner.launchers[0].config
            if (config.num_warps == 8 and config.num_stages == 1 and config.num_ctas == 1
                    and config.kwargs == {"XBLOCK": 1, "R0_BLOCK": 1024}):
                quantize = _mode(function.src)
                if quantize is not None:
                    signature = tuple(autotuner.triton_meta["signature"].values())
                    quantization_and_outputs = ("*fp32", "*bf16", "*fp8e4nv") if quantize else ("*bf16",)
                    expected_sig = _inputs + quantization_and_outputs + _dimensions
                    if signature == expected_sig:
                        meta = (quantize, len(signature) - 2)
        if autotuner._cached_launcher:
            autotuner.__dict__["_sandwich_meta"] = meta
    if meta is None:
        return None
    quantize, expected_len = meta
    if len(args) != expected_len:
        return None
    rows, width = args[-2:]
    if type(rows) is not int or type(width) is not int or rows not in (1, 512) or width != 6656:
        return None
    addresses = [tensor.data_ptr() for tensor in args[:-2]]
    if any(address % 8 for address in addresses):
        return None
    return quantize, rows, addresses


# Each matched compiled kernel object is admitted per row count only after its
# first launch outside graph capture: the compiled kernel and the native one
# run on copies of the real arguments and on two generated inputs, and every
# buffer must come out bitwise equal. Launches before that check (inside a
# graph capture) and after a failed check keep the compiled kernel.
_decisions = {}
_check_counts = Counter()
_PROBES = (None, 0.03, 24.0)


def _layout_ok(tensors, quantize, rows):
    rows_numel = rows * _WIDTH
    expected = [rows_numel] * 3 + [rows] + [_WIDTH] * 3
    expected += [1, rows_numel, rows_numel] if quantize else [rows_numel]
    return len(tensors) == len(expected) and all(
        t.is_contiguous() and t.numel() == n for t, n in zip(tensors, expected)
    )


def _self_check(autotuner, original, args, quantize, rows, stream):
    import torch

    tensors, dimensions = list(args[:-2]), args[-2:]
    if not _layout_ok(tensors, quantize, rows):
        return False
    device = tensors[0].device
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)):
        return _compare(autotuner, original, tensors, dimensions, quantize, rows, stream)


def _compare(autotuner, original, tensors, dimensions, quantize, rows, stream):
    import torch

    generator = torch.Generator(device=tensors[0].device)
    generator.manual_seed(rows * 2 + quantize)
    for magnitude in _PROBES:
        native = [tensor.clone() for tensor in tensors]
        if magnitude is not None:
            for i in range(3):
                native[i].copy_(torch.randn(native[i].shape, generator=generator,
                                            device=generator.device) * magnitude)
            native[3].copy_(torch.rand(native[3].shape, generator=generator, device=generator.device)
                            * (2 * _WIDTH * magnitude * magnitude))
        compiled = [tensor.clone() for tensor in native]
        original(autotuner, *compiled, *dimensions, stream=stream)
        pointers = (_pointer * len(native))(*(tensor.data_ptr() for tensor in native))
        if _launch(pointers, rows, quantize, rows == 512, stream):
            return False
        if not all(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
                   for a, b in zip(compiled, native)):
            return False
    return True


def _admitted(autotuner, original, args, quantize, rows, stream):
    import torch

    key = (id(autotuner), rows)
    decision = _decisions.get(key)
    if decision is not None:
        return decision[0]
    if torch.cuda.is_current_stream_capturing():
        _check_counts["capture", rows] += 1
        return False
    try:
        ok = _self_check(autotuner, original, args, quantize, rows, stream)
    except Exception:  # noqa: BLE001 - any failure keeps the compiled kernel
        ok = False
    _decisions[key] = (ok, autotuner)
    _check_counts["admitted" if ok else "rejected", rows] += 1
    return ok


@cache
def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, get_active_debug_mode

    original = CachingAutotuner.run

    def run(autotuner, *args, stream, benchmark_run=False, **kwargs):
        if not benchmark_run and not kwargs and not autotuner.triton_interpret:
            match = _match(autotuner, args)
            if match is not None and not get_active_debug_mode():
                quantize, rows, addresses = match
                if _admitted(autotuner, original, args, quantize, rows, stream):
                    pointers = (_pointer * len(addresses))(*addresses)
                    status = _launch(pointers, rows, quantize, rows == 512, stream)
                    if status:
                        raise RuntimeError(f"CUDA sandwich launch failed with error {status}")
                    _launch_counts[rows, quantize] += 1
                    return None
        return original(autotuner, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)

    CachingAutotuner.run = run


def launch_summary():
    launches = ", ".join(
        f"{count} {'FP8-output' if quantize else 'BF16-output'} at {rows} rows"
        for (rows, quantize), count in sorted(_launch_counts.items())
    )
    if not _check_counts:
        return launches
    checks = (
        f"first-use check against the compiled kernel: "
        f"{sum(c for (k, _), c in _check_counts.items() if k == 'admitted')} admitted "
        f"({', '.join(f'{c} at {r} rows' for (k, r), c in sorted(_check_counts.items()) if k == 'admitted')}), "
        f"{sum(c for (k, _), c in _check_counts.items() if k == 'rejected')} rejected, "
        f"{sum(c for (k, _), c in _check_counts.items() if k == 'capture')} launches in graph capture before a check"
    )
    return f"{launches}; {checks}" if launches else checks
