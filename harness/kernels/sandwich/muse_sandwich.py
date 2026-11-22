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
    function = autotuner.fn
    if getattr(function, "__name__", None) not in _symbols:
        return None
    props = autotuner.device_props
    if props.type != "cuda" or props.cc != 121 or props.multi_processor_count != 48:
        return None
    if len(autotuner.launchers) != 1 or not autotuner._cached_launcher:
        return None
    config = autotuner.launchers[0].config
    if (config.num_warps != 8 or config.num_stages != 1 or config.num_ctas != 1
            or config.kwargs != {"XBLOCK": 1, "R0_BLOCK": 1024}):
        return None
    quantize = _mode(function.src)
    if quantize is None:
        return None
    signature = tuple(autotuner.triton_meta["signature"].values())
    quantization_and_outputs = ("*fp32", "*bf16", "*fp8e4nv") if quantize else ("*bf16",)
    if signature != _inputs + quantization_and_outputs + _dimensions or len(args) != len(signature) - 2:
        return None
    rows, width = args[-2:]
    if type(rows) is not int or type(width) is not int or rows not in (1, 512) or width != 6656:
        return None
    addresses = [tensor.data_ptr() for tensor in args[:-2]]
    if any(address % 8 for address in addresses):
        return None
    return quantize, rows, addresses


@cache
def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, get_active_debug_mode

    original = CachingAutotuner.run

    def run(autotuner, *args, stream, benchmark_run=False, **kwargs):
        if not benchmark_run and not kwargs and not autotuner.triton_interpret:
            match = _match(autotuner, args)
            if match is not None and not get_active_debug_mode():
                quantize, rows, addresses = match
                pointers = (_pointer * len(addresses))(*addresses)
                status = _launch(pointers, rows, quantize, rows == 512, stream)
                if status:
                    raise RuntimeError(f"CUDA sandwich launch failed with error {status}")
                _launch_counts[rows, quantize] += 1
                return None
        return original(autotuner, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)

    CachingAutotuner.run = run


def launch_summary():
    return ", ".join(
        f"{count} {'FP8-output' if quantize else 'BF16-output'} at {rows} rows"
        for (rows, quantize), count in sorted(_launch_counts.items())
    )
