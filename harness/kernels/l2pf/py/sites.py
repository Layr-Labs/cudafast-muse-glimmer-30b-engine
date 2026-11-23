"""Prefetch the first bytes of every GEMM's weights in a captured CUDA graph.

While a graph is captured, the GEMM entry points the compiled model calls
(``torch.ops._C.marlin_gemm``, ``torch.ops._C.cutlass_scaled_mm``,
``glimmer_fp8_ext.gemm`` and Inductor's
``extern_kernels.mm``) are wrapped. Right after GEMM i is enqueued, a side
stream forks from the capture stream, waits for it, and queues an L2 prefetch
of the first tiles GEMM i+1 will read; the side stream joins back right after
GEMM i+1 is enqueued. In the graph this is one extra kernel node per GEMM
with two extra edges; the GEMMs and every other kernel keep their arguments,
their launch configuration and their order on the main stream.

The order of the GEMMs and their weight tensors are recorded on the eager
warm-up run that precedes the capture. A site is used only if every byte it
prefetches lies inside a model parameter, and only while the captured calls
match the recorded ones; on the first mismatch prefetching stops for the rest
of that capture.

Ranges: the first bytes of the weight (and the same share of Marlin's block
scales). More bytes go ahead of the GEMM that follows the attention op, the
longest window. GEMMs whose weights exceed MAX_WEIGHT (the MLP) stream at full
bandwidth once running, so they get a smaller head sized to the short window
ahead of them: L2PF_MB_MLP_NORM ahead of a GEMM that follows a norm (gate_up),
L2PF_MB_MLP_ACT ahead of one that follows another large GEMM's activation
(down). ``L2PF_MLP=0`` skips them. (MLP heads: ArthurOstapenko's submission
e8fa7600, commit af81404; the compressed FP8 skip below is ours, kept.)
``marlin_descs`` (first tiles of each block's stripe) is kept for experiments;
it is not used by the session.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field

import torch

from . import CHUNK, ENABLED, lib, prefetch_desc, self_check

__all__ = ["Session", "budget", "weight_descs", "head_descs", "marlin_descs", "available"]


def _env_bytes(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(float(v) * (1 << 20)) if v else default


# Bytes per site: after an attention op (the long latency-bound window) and
# after any other GEMM (one fused norm/activation kernel in between). GEMMs
# whose weights exceed MAX_WEIGHT get the MLP budgets below instead: a head
# that fits the window ahead of them (the norm before gate_up, the activation
# before down).
BYTES_AFTER_ATTN = _env_bytes("L2PF_MB_ATTN", 8 << 20)
BYTES_OTHER = _env_bytes("L2PF_MB_OTHER", 2 << 20)
MAX_WEIGHT = _env_bytes("L2PF_MAX_WEIGHT_MB", 64 << 20)
BLOCKS = int(os.environ.get("L2PF_BLOCKS", "1"))
MLP = os.environ.get("L2PF_MLP", "1") != "0"
BYTES_MLP_NORM = _env_bytes("L2PF_MB_MLP_NORM", (9 << 20) // 4)
BYTES_MLP_ACT = _env_bytes("L2PF_MB_MLP_ACT", 1 << 19)
BYTES_RANS_AFTER_ATTN = _env_bytes("L2PF_MB_RANS_ATTN", 0)
BYTES_RANS_QKV = _env_bytes("L2PF_MB_RANS_QKV", 0)
TLB_PREFETCH = os.environ.get("L2PF_TLB", "1") != "0"
TLB_RANS_HEAD = int(os.environ.get("L2PF_TLB_RANS_HEAD_KB", "32")) << 10
TLB_STRIDE = int(os.environ.get("L2PF_TLB_STRIDE_KB", "2048")) << 10
TLB_ALL = os.environ.get("L2PF_TLB_ALL", "1") != "0"
TLB_UNASSISTED = os.environ.get("L2PF_TLB_UNASSISTED", "0") != "0"

# Compressed one-row projections measured with the same captured stream edges.
_RANS_SKIP_PREFETCH = {
    (6656, 4096, True),
    (4608, 6656, False),
}


def available() -> bool:
    return ENABLED and lib() is not None


# ---------------------------------------------------------------- descriptors


def _dense_span(t: torch.Tensor) -> tuple[int, int]:
    """(address, bytes) of the memory t covers; t must be non-overlapping and dense."""
    if not t.is_cuda or t.numel() == 0:
        raise ValueError("not a CUDA tensor")
    expect = 1
    for size, stride in sorted(zip(t.shape, t.stride()), key=lambda d: d[1]):
        if size == 1:
            continue
        if stride != expect:
            raise ValueError("not dense")
        expect *= size
    return t.data_ptr(), t.numel() * t.element_size()


def head_descs(t: torch.Tensor, nbytes: int) -> list[tuple[int, int, int, int]]:
    """The first nbytes of t's memory in 64 KB pieces (row-major [N, K] weights,
    including a transposed [K, N] view of one)."""
    base, span = _dense_span(t)
    n = min(int(nbytes), span) & ~15
    if base % 16 or n <= 0:
        return []
    full, rem = divmod(n, CHUNK)
    out = []
    if full:
        out.append((base, CHUNK, CHUNK, full))
    if rem:
        out.append((base + full * CHUNK, 0, rem, 1))
    return out


def tlb_descs(t: torch.Tensor, start_bytes: int = 0, stride: int = TLB_STRIDE,
              seg: int = 128) -> list[tuple[int, int, int, int]]:
    """One small bulk prefetch per stride across t to warm GMMU page-table entries on the side stream."""
    if stride <= 0 or seg <= 0 or seg % 16:
        return []
    base, span = _dense_span(t)
    if base % 16 or span <= stride:
        return []
    first = ((max(int(start_bytes), stride) + stride - 1) // stride) * stride
    if first + seg > span:
        return []
    cnt = (span - seg - first) // stride + 1
    return [(base + first, stride, seg, cnt)]


def marlin_stripe_starts(n: int, k: int, sms: int, thread_k: int = 128,
                         thread_n: int = 128) -> tuple[list[tuple[int, int, int]], int]:
    """(column tile, k tile, tiles before the stripe changes column) of the first
    tile each block of the one-row Marlin kernel reads, and k_tiles.

    Mirrors the partition at the top of Marlin's kernel for prob_m <= 8
    (parallel = 1): part-1 blocks walk whole columns b, b + sms, ...; part-2
    blocks take ``iters`` consecutive k-tiles in column-major order.
    """
    k_tiles, n_tiles = k // thread_k, n // thread_n
    part2, part1 = n_tiles, 0
    if n_tiles > sms:
        part2 = n_tiles % sms
        if part2 * 3 <= sms:
            part2 += sms
        part1 = (n_tiles - part2) // sms
    iters = math.ceil(k_tiles * part2 / sms)
    starts = []
    for b in range(sms):
        if part1:
            starts.append((b, 0, k_tiles))
            continue
        col, row = divmod(iters * b, k_tiles)
        if col >= part2:
            continue
        starts.append((col + n_tiles - part2, row, min(iters, k_tiles - row)))
    return starts, k_tiles


def marlin_descs(qweight: torch.Tensor, scales: torch.Tensor, n: int, k: int,
                 nbytes: int, sms: int) -> list[tuple[int, int, int, int]]:
    """First tiles of every block's stripe of a Marlin NVFP4 weight.

    qweight is the repacked int32 [k/16, 2n] code tensor: one row holds 16 k
    for all n (8n bytes), a 128-column tile is 1 KB of a row and a 128-deep
    k-tile is 8 rows. scales is the [k/16, n] one-byte block-scale tensor: a
    tile reads 128 bytes from each of its 8 rows.
    """
    if (n % 128 or k % 128 or qweight.dtype != torch.int32 or qweight.shape != (k // 16, 2 * n)
            or scales.element_size() != 1 or scales.shape != (k // 16, n)
            or not qweight.is_contiguous() or not scales.is_contiguous()):
        return head_descs(qweight, nbytes)
    starts, _ = marlin_stripe_starts(n, k, sms)
    if not starts:
        return []
    tile = 8 * 1024 + 8 * 128
    m = max(1, int(nbytes) // (len(starts) * tile))
    qb, sb = qweight.data_ptr(), scales.data_ptr()
    rows = []  # (col, first k-tile, k-tiles)
    for col, row, run in starts:
        rows.append((col, row, min(m, run)))
    out = []
    # Codes: merge blocks whose stripes start on the same k-tile in adjacent
    # columns (part-1 blocks) into one segment per row, up to 64 KB.
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    i = 0
    while i < len(rows):
        col, row, cnt = rows[i]
        j = i + 1
        while (j < len(rows) and rows[j][1] == row and rows[j][2] == cnt
               and rows[j][0] == col + (j - i) and (j - i + 1) * 1024 <= CHUNK):
            j += 1
        w = j - i
        out.append((qb + 8 * row * 8 * n + col * 1024, 8 * n, w * 1024, 8 * cnt))
        out.append((sb + 8 * row * n + col * 128, n, w * 128, 8 * cnt))
        i = j
    return out


def budget(weight_bytes: int, after_attn: bool) -> int:
    """Bytes to prefetch ahead of a GEMM that reads weight_bytes of weights."""
    if weight_bytes > MAX_WEIGHT:
        return 0
    return BYTES_AFTER_ATTN if after_attn else BYTES_OTHER


def mlp_budget(n_tensors: int, after_act: bool) -> int:
    """Bytes of the weight head ahead of a GEMM above MAX_WEIGHT; with block
    scales (an eighth of the code bytes) the two together take the budget."""
    if not MLP:
        return 0
    nbytes = BYTES_MLP_ACT if after_act else BYTES_MLP_NORM
    return nbytes * 8 // 9 if n_tensors > 1 else nbytes


def weight_descs(ws: tuple[torch.Tensor, ...], nbytes: int) -> list[tuple[int, int, int, int]]:
    """The first nbytes of the weight (ws[0]) and the same share of its block
    scales (ws[1], one byte per 16 four-bit codes: an eighth of the bytes)."""
    if nbytes <= 0:
        return []
    out = head_descs(ws[0], nbytes)
    for w in ws[1:]:
        out += head_descs(w, nbytes // 8)
    return out


def _inside(descs, ranges) -> bool:
    for base, pitch, seg, cnt in descs:
        lo, hi = base, base + (cnt - 1) * pitch + seg
        if not any(a <= lo and hi <= b for a, b in ranges):
            return False
    return True


def weight_ranges(module: torch.nn.Module) -> list[tuple[int, int]]:
    """Memory ranges of every parameter and buffer of module."""
    out = []
    for t in list(module.parameters()) + list(module.buffers()):
        if t.is_cuda and t.numel():
            st = t.untyped_storage()
            out.append((st.data_ptr(), st.data_ptr() + st.nbytes()))
    return out


# ---------------------------------------------------------------- capture hooks


class _PacketProxy:
    """Stands in for an op packet: calls of the packet and of its ``.default``
    overload go through wrappers; every other attribute comes from the packet."""

    def __init__(self, packet, call, default):
        self._packet = packet
        self._call = call
        self.default = default

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._packet, name)


def _targets():
    """(owner, attribute, kind, proxy) for every entry point to wrap.

    Inductor's wrappers call ``torch.ops._C.<gemm>.default(...)`` and
    ``extern_kernels.mm(...)``, looked up at call time; the graph module that
    joins the compiled pieces calls the attention op through its packet.
    """
    out = []
    for name, kind in (("marlin_gemm", "marlin"), ("cutlass_scaled_mm", "cutlass")):
        try:
            pkt = getattr(torch.ops._C, name)
            pkt.default  # noqa: B018 - materialize the overload
        except (AttributeError, RuntimeError):
            continue
        out.append((pkt, "default", kind, False))
    try:
        import glimmer_fp8_ext
    except (ImportError, OSError):
        pass
    else:
        # Wrap the actual launch, so falling back from the Python FP8 op
        # records only the stock CUTLASS site through its default overload.
        out.append((glimmer_fp8_ext, "gemm", "cutlass_fp8", False))
    try:
        # Muse Glimmer decode FP8 GEMV (vllm .../scaled_mm/muse_fp8.py): its weight is the packed buffer (args[2]).
        # MUSEFP8_L2PF=0 leaves these GEMMs out of the prefetch chain.
        if os.environ.get("MUSEFP8_L2PF", "1") != "0":
            pkt = torch.ops.vllm.muse_fp8_scaled_mm
            pkt.default  # noqa: B018
            out.append((pkt, "default", "musefp8", False))
    except (AttributeError, RuntimeError):
        pass
    try:
        lane_bf16 = torch.ops.vllm.muse_bf16_lane_linear
        lane_bf16.default  # noqa: B018
        out.append((lane_bf16, "default", "muse_bf16_lane", False))
        try:
            bgpkt = torch.ops.vllm.muse_bf16_gate_fill
            bgpkt.default  # noqa: B018
            out.append((bgpkt, "default", "muse_bf16_gate", False))
        except (AttributeError, RuntimeError):
            pass
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch._inductor.select_algorithm import extern_kernels

        if hasattr(extern_kernels, "mm"):
            out.append((extern_kernels, "mm", "mm", False))
    except Exception:  # noqa: BLE001
        pass
    try:
        ns = torch.ops.vllm
        ns.unified_attention_with_output  # noqa: B018
        out.append((ns, "unified_attention_with_output", "attn", True))
    except (AttributeError, RuntimeError):
        pass
    try:
        tiled = torch.ops.vllm.marlin_tiled_decode
        out.append((tiled, "default", "marlin_tiled", False))
    except (AttributeError, RuntimeError):
        pass
    try:
        # the same GEMV with 6-bit block-scale codes: args[1] tiled codes, args[2] scale codes
        tiled6 = torch.ops.vllm.marlin_tiled_decode6
        out.append((tiled6, "default", "marlin_tiled", False))
    except (AttributeError, RuntimeError):
        pass
    return out


def _weights_of(kind: str, args) -> tuple[torch.Tensor, ...] | None:
    try:
        if kind == "marlin":
            return (args[2], args[4])
        if kind in ("cutlass", "musefp8"):
            return (args[2],)
        if kind in ("cutlass_fp8", "mm"):
            return (args[1],)
        if kind == "muse_bf16_lane" and args[0].shape[0] == 1 and args[0].is_contiguous():
            return (args[2],)
        if kind == "muse_bf16_gate" and args[1].shape[0] == 1 and args[1].is_contiguous():
            return (args[3],)
        if kind == "marlin_tiled":
            return (args[1], args[2])
    except IndexError:
        return None
    return None


@dataclass
class _Site:
    kind: str
    ptr: int
    after_attn: bool
    descs: list = field(default_factory=list)
    off: int = 0
    unassisted: bool = False
    mlp: str = ""  # "norm" / "act": a GEMM above MAX_WEIGHT after a norm / an activation


class Session:
    """Record the GEMM sites of one forward, then prefetch during its capture."""

    def __init__(self, ranges: list[tuple[int, int]] | None):
        self.ranges = ranges
        self.storages = None if ranges is None else {a: b for a, b in ranges}
        self.sites: list[_Site] = []
        self.table: torch.Tensor | None = None
        self.side: torch.cuda.Stream | None = None
        self.sms = 0
        self.used = 0
        self.prefetched = 0
        self.stopped = False
        self._i = 0
        self._pending = False
        self._attn = False
        self._prev_big = False

    # -- patching
    @contextlib.contextmanager
    def _patched(self, on_call, with_attn):
        # A wrapped entry point can be re-entered through itself (a Python
        # dispatch path that looks the overload up by name finds the wrapper);
        # only the outermost call of a nest is a site.
        depth = [0]

        def wrap(fn, kind):
            def wrapper(*args, **kwargs):
                depth[0] += 1
                try:
                    out = fn(*args, **kwargs)
                finally:
                    depth[0] -= 1
                if depth[0] == 0:
                    try:
                        on_call(kind, args)
                    except Exception:  # noqa: BLE001 - never fail the caller
                        self.stopped = True
                return out
            return wrapper

        saved = []
        try:
            for obj, attr, kind, proxy in _targets():
                if kind == "attn" and not with_attn:
                    continue
                orig = getattr(obj, attr)
                if proxy:
                    new = _PacketProxy(orig, wrap(orig, kind), wrap(orig.default, kind))
                else:
                    new = wrap(orig, kind)
                setattr(obj, attr, new)
                saved.append((obj, attr, orig))
            yield
        finally:
            for obj, attr, orig in reversed(saved):
                setattr(obj, attr, orig)

    # -- record (eager warm-up)
    @contextlib.contextmanager
    def record(self):
        if not available() or not self_check():
            yield
            return
        dev = torch.cuda.current_device()
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.side = torch.cuda.Stream(device=dev)
        self.sites = []
        self.stopped = False
        self._attn = False
        self._prev_big = False
        with self._patched(self._on_record, True):
            yield
        if not self.stopped:
            self._build(dev)

    def _weight_key(self, kind, args) -> int | None:
        """Address of the call's weight if it is a model parameter, else None."""
        ws = _weights_of(kind, args)
        if not ws or not all(isinstance(w, torch.Tensor) and w.is_cuda for w in ws):
            return None
        if self.storages is not None and not all(
                w.untyped_storage().data_ptr() in self.storages for w in ws):
            return None
        return ws[0].data_ptr()

    def _on_record(self, kind, args):
        if kind == "attn":
            self._attn = True
            return
        key = self._weight_key(kind, args)
        if key is None:
            return
        ws = _weights_of(kind, args)
        site = _Site(kind, key, self._attn)
        self._attn = False
        prev_big, self._prev_big = self._prev_big, False
        try:
            wbytes = sum(_dense_span(w)[1] for w in ws)
            if wbytes > MAX_WEIGHT:
                self._prev_big = True
                site.mlp = "act" if prev_big else "norm"
                mb_b = mlp_budget(len(ws), prev_big)
                site.descs = weight_descs(ws, mb_b)
                if TLB_PREFETCH and (kind == "musefp8" or TLB_ALL):
                    site.descs += tlb_descs(ws[0], mb_b, TLB_STRIDE)
            else:
                nbytes = budget(wbytes, site.after_attn)
                if kind == "musefp8":
                    a, weight, packed, sa, sb = args
                    k, n = weight.shape
                    if ((n, k, site.after_attn) in _RANS_SKIP_PREFETCH and a.shape[0] == 1
                            and a.is_contiguous() and 0 < packed.numel() != n * k):
                        from vllm.model_executor.kernels.linear.scaled_mm.muse_fp8 import _scales_ok
                        if _scales_ok(sa, sb, n):
                            nbytes = BYTES_RANS_AFTER_ATTN if site.after_attn else BYTES_RANS_QKV
                            if nbytes == 0:
                                site.unassisted = True
                elif kind == "muse_bf16_lane":
                    a, weight, packed, index = args
                    n, k = weight.shape
                    if ((n, k, site.after_attn) in _RANS_SKIP_PREFETCH and a.shape[0] == 1
                            and a.is_contiguous() and index.numel() == 1):
                        nbytes = BYTES_RANS_AFTER_ATTN if site.after_attn else BYTES_RANS_QKV
                        if nbytes == 0:
                            site.unassisted = True
                if site.unassisted and TLB_PREFETCH and TLB_UNASSISTED:
                    site.descs = head_descs(ws[0], TLB_RANS_HEAD) + tlb_descs(ws[0], TLB_RANS_HEAD, TLB_STRIDE)
                else:
                    site.descs = weight_descs(ws, nbytes)
                    if not site.unassisted and TLB_PREFETCH and (
                            kind in ("musefp8", "musefp8_gate", "muse_bf16_lane", "muse_bf16_gate") or TLB_ALL):
                        site.descs += tlb_descs(ws[0], nbytes, TLB_STRIDE)
            if self.ranges is not None and not _inside(site.descs, self.ranges):
                site.descs = []
        except Exception:  # noqa: BLE001 - a site without ranges is still a site
            site.descs = []
        self.sites.append(site)

    def _build(self, dev):
        flat = []
        for s in self.sites:
            s.off = len(flat) // 3
            for base, pitch, seg, cnt in s.descs:
                flat += [base, pitch, (seg << 32) | cnt]
        self.used = sum(1 for s in self.sites if s.descs)
        if flat:
            self.table = torch.tensor(flat, dtype=torch.int64).to(dev)

    # -- apply (capture)
    @contextlib.contextmanager
    def apply(self):
        if self.table is None or self.side is None:
            yield
            return
        self._i = 0
        self._pending = False
        self.stopped = False
        self.prefetched = 0
        try:
            self._fork(torch.cuda.current_stream(), 0)
        except Exception:  # noqa: BLE001
            self.stopped = True
        try:
            with self._patched(self._on_apply, False):
                yield
        finally:
            self._join(torch.cuda.current_stream())

    def _fork(self, main, i):
        if i >= len(self.sites) or not self.sites[i].descs:
            return
        s = self.sites[i]
        self.side.wait_stream(main)
        prefetch_desc(self.table.data_ptr() + 24 * s.off, len(s.descs), self.side,
                      blocks=BLOCKS)
        self._pending = True
        self.prefetched += 1

    def _join(self, main):
        if self._pending:
            main.wait_stream(self.side)
            self._pending = False

    def _on_apply(self, kind, args):
        if kind == "attn" or self.stopped:
            return
        key = self._weight_key(kind, args)
        if key is None:
            return
        main = torch.cuda.current_stream()
        i = self._i
        self._i += 1
        if i >= len(self.sites) or key != self.sites[i].ptr or kind != self.sites[i].kind:
            self.stopped = True
            self._join(main)
            return
        self._join(main)
        self._fork(main, i + 1)

    def summary(self) -> str:
        na = sum(1 for s in self.sites if s.descs and not s.mlp and s.after_attn)
        nb = sum(1 for s in self.sites if s.descs and not s.mlp and not s.after_attn)
        ng = sum(1 for s in self.sites if s.descs and s.mlp == "norm")
        nd = sum(1 for s in self.sites if s.descs and s.mlp == "act")
        mb = sum(sg * c for s in self.sites for _, _, sg, c in s.descs) / (1 << 20)
        ua = sum(1 for s in self.sites if s.unassisted and s.after_attn)
        ub = sum(1 for s in self.sites if s.unassisted and not s.after_attn)
        if ng or nd:
            big = (f"{ng + nd} MLP sites ({ng} after a norm at "
                   f"{BYTES_MLP_NORM / (1 << 20):.2f} MiB, {nd} after an activation at "
                   f"{BYTES_MLP_ACT / (1 << 20):.2f} MiB)")
        else:
            big = "skipped"
        return (f"{self.prefetched} of {len(self.sites)} GEMM sites"
                f"{' (stopped at a mismatch)' if self.stopped else ''}; "
                f"{na} after attention at {BYTES_AFTER_ATTN / (1 << 20):.1f} MiB, "
                f"{nb} others at {BYTES_OTHER / (1 << 20):.1f} MiB, "
                f"weights above {MAX_WEIGHT / (1 << 20):.0f} MiB: {big}; "
                f"{mb:.0f} MiB per step; compressed FP8 sites left without prefetch: "
                f"{ua} output after attention, {ub} QKV")
