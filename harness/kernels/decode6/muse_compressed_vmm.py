# SPDX-License-Identifier: Apache-2.0
"""Driver-owned, transparently compressed CUDA storage for immutable weights."""
import ctypes
from pathlib import Path

import torch

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_compressed_vmm.so")))
_alloc = _lib.muse_alloc_compressed
_alloc.argtypes = [ctypes.c_size_t]
_alloc.restype = ctypes.c_void_p
_release = _lib.muse_release_compressed
_release.argtypes = [ctypes.c_void_p]
_release.restype = None
_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_capsule.restype = ctypes.py_object
_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_is_valid.restype = ctypes.c_int
_DLPACK_NAME = b"dltensor"


def empty_like(weight: torch.Tensor) -> torch.Tensor | None:
    """Allocate storage with the same layout; return None if not available."""
    if not weight.is_cuda or not weight.is_contiguous():
        return None
    with torch.cuda.device(weight.device):
        # Ensure the primary context exists, including on a noncurrent device.
        torch.cuda.current_stream(weight.device)
        ptr = _alloc(weight.numel() * weight.element_size())
        if ptr is None:
            return None
        capsule = None
        try:
            capsule = _capsule(ptr, _DLPACK_NAME, None)
            data = torch.from_dlpack(capsule)
        except Exception:
            # The DLPack deleter owns the allocation only after consumption.
            if capsule is None or _is_valid(capsule, _DLPACK_NAME):
                _release(ptr)
            raise
        return data.view(weight.dtype).view(weight.shape)
