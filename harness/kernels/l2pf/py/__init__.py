"""l2pf: L2 prefetch of a weight ahead of the bandwidth-bound GEMM that reads it.

The kernels in ``libl2pf.so`` only issue ``cp.async.bulk.prefetch.L2``: they
move bytes that are already in memory into L2, write nothing, and so cannot
change any value a later kernel computes.

``L2PF=0`` in the environment disables every prefetch.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = [
    "LIB_PATH",
    "ENABLED",
    "CHUNK",
    "lib",
    "prefetch",
    "prefetch_desc",
    "stream_read",
    "self_check",
]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libl2pf.so")
ENABLED = os.environ.get("L2PF", "1") != "0"
# A 64 KB bulk prefetch is honoured; larger requests are cut into 64 KB pieces.
CHUNK = 1 << 16

_lib = None


def lib():
    """The loaded library, or None when it is missing or disabled."""
    global _lib
    if _lib is not None or not ENABLED:
        return _lib
    try:
        h = ctypes.CDLL(LIB_PATH)
    except OSError:
        return None
    h.l2pf_prefetch.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    h.l2pf_prefetch.restype = ctypes.c_int
    h.l2pf_prefetch_desc.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    h.l2pf_prefetch_desc.restype = ctypes.c_int
    h.l2pf_stream_read.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    h.l2pf_stream_read.restype = ctypes.c_int
    _lib = h
    return h


def prefetch(t: torch.Tensor, nbytes: int, stream: torch.cuda.Stream,
             blocks: int = 1) -> None:
    """Queue an L2 prefetch of the first nbytes of t's storage on stream."""
    h = lib()
    if h is None:
        raise RuntimeError("l2pf library not available")
    st = t.untyped_storage()
    nbytes = min(int(nbytes), st.nbytes())
    rc = h.l2pf_prefetch(st.data_ptr(), nbytes, CHUNK, blocks, stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch failed: {rc}")


def prefetch_desc(desc_ptr: int, ndesc: int, stream: torch.cuda.Stream,
                  blocks: int = 1, threads: int = 128) -> None:
    """Queue the prefetches of ndesc descriptors at desc_ptr (device memory)."""
    h = lib()
    if h is None:
        raise RuntimeError("l2pf library not available")
    rc = h.l2pf_prefetch_desc(desc_ptr, ndesc, blocks, threads, stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch_desc failed: {rc}")


def stream_read(t: torch.Tensor, sink: torch.Tensor, stream: torch.cuda.Stream | None = None,
                blocks: int = 192) -> None:
    """Pure DRAM streaming read of t (16-byte aligned); never modifies t."""
    h = lib()
    if h is None:
        raise RuntimeError("l2pf library not available")
    nbytes = t.numel() * t.element_size()
    s = torch.cuda.current_stream(t.device) if stream is None else stream
    rc = h.l2pf_stream_read(t.data_ptr(), nbytes, sink.data_ptr(), blocks, s.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_stream_read failed: {rc}")


def self_check(device: torch.device | int | None = None) -> bool:
    """Run all kernels on a scratch buffer and check that no byte of it or of
    the descriptor table changed. Also loads the module before any capture."""
    h = lib()
    if h is None:
        return False
    dev = torch.device("cuda", torch.cuda.current_device() if device is None else
                       (device if isinstance(device, int) else device.index or 0))
    g = torch.Generator(device=dev)
    g.manual_seed(20261121)
    buf = torch.randint(0, 256, (3 << 20,), generator=g, device=dev, dtype=torch.uint8)
    ref = buf.clone()
    base = buf.data_ptr() + (1 << 20)
    desc = torch.tensor(
        [base, 4096, (1024 << 32) | 64, base + 512 * 1024, 0, (CHUNK << 32) | 4],
        dtype=torch.int64).to(dev)
    dref = desc.clone()
    sink = torch.zeros((1,), dtype=torch.int32, device=dev)
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        rc1 = h.l2pf_prefetch(base, 1 << 20, CHUNK, 2, s.cuda_stream)
        rc2 = h.l2pf_prefetch_desc(desc.data_ptr(), 2, 2, 128, s.cuda_stream)
        rc3 = h.l2pf_stream_read(buf.data_ptr(), buf.numel(), sink.data_ptr(), 4, s.cuda_stream)
    s.synchronize()
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    return rc1 == 0 and rc2 == 0 and rc3 == 0 and torch.equal(buf, ref) and torch.equal(desc, dref)
