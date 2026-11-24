# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
import math
import os
from typing import Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    get_pp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

try:
    # Optional kernel package built from harness/kernels (L2 prefetch of the
    # next GEMM's weights inside the captured FULL graph).
    from l2pf import sites as l2_prefetch
except ImportError:
    l2_prefetch = None

_PERSIST_L2_LIMIT_BYTES = 4 << 20
_cudart_upload_fn: Any = False


def upload_cudagraph(
    graph: torch.cuda.CUDAGraph, stream: Any = None
) -> None:
    """Eagerly upload an instantiated CUDA graph to the device; fail-soft."""
    global _cudart_upload_fn
    try:
        if hasattr(graph, "upload"):
            if stream is not None:
                try:
                    graph.upload(stream)
                    return
                except TypeError:
                    pass
            graph.upload()
            return
        if not hasattr(graph, "raw_cuda_graph_exec"):
            return
        if _cudart_upload_fn is False:
            import ctypes

            for libname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
                try:
                    lib = ctypes.CDLL(libname)
                    fn = lib.cudaGraphUpload
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    fn.restype = ctypes.c_int
                    _cudart_upload_fn = fn
                    break
                except OSError:
                    continue
            else:
                _cudart_upload_fn = None
        if _cudart_upload_fn is not None:
            import ctypes

            exec_ptr = int(graph.raw_cuda_graph_exec())
            if exec_ptr:
                if stream is None:
                    st_ptr = int(torch.cuda.current_stream().cuda_stream)
                elif isinstance(stream, int):
                    st_ptr = stream
                else:
                    st_ptr = int(getattr(stream, "cuda_stream", stream))
                _cudart_upload_fn(ctypes.c_void_p(exec_ptr), ctypes.c_void_p(st_ptr))
    except Exception:  # noqa: BLE001
        pass


def _apply_persisting_window(
    arena: torch.Tensor | None,
    cap_stream: Any,
    miss_streaming: bool = False,
    persist_limit: int = _PERSIST_L2_LIMIT_BYTES,
) -> bool:
    if arena is None:
        return False
    try:
        import l2pf as _l2pf_mod

        if hasattr(_l2pf_mod, "set_persisting_window_ex"):
            return bool(
                _l2pf_mod.set_persisting_window_ex(
                    arena.data_ptr(),
                    arena.numel(),
                    persist_limit,
                    bool(miss_streaming),
                    cap_stream,
                )
            )
        if hasattr(_l2pf_mod, "set_persisting_window"):
            return bool(
                _l2pf_mod.set_persisting_window(
                    arena.data_ptr(), arena.numel(), persist_limit, cap_stream
                )
            )
    except Exception:  # noqa: BLE001
        pass
    return False


def _clear_persisting_window(
    cap_stream: Any, persist_limit: int = _PERSIST_L2_LIMIT_BYTES
) -> None:
    """Clear stream accessPolicyWindow WITHOUT zeroing cudaLimitPersistingL2CacheSize."""
    try:
        import l2pf as _l2pf_mod

        if hasattr(_l2pf_mod, "clear_stream_persisting_window"):
            _l2pf_mod.clear_stream_persisting_window(cap_stream)
        elif hasattr(_l2pf_mod, "set_persisting_window_ex"):
            _l2pf_mod.set_persisting_window_ex(
                0, 0, persist_limit, False, cap_stream
            )
    except Exception:  # noqa: BLE001
        pass


def _apply_l2_fetch_granularity() -> None:
    """R3F-D: Set CUDA max L2 fetch granularity on boot before graph capture; fail-soft."""
    gran = os.environ.get("MUSE_L2_FETCH_GRAN", "128").strip()
    if gran in ("0", "", "false", "no"):
        return
    try:
        import l2pf as _l2pf_mod

        if hasattr(_l2pf_mod, "set_max_l2_fetch_granularity"):
            if _l2pf_mod.set_max_l2_fetch_granularity(int(gran)):
                logger.info("Set CUDA max L2 fetch granularity to %d B", int(gran))
    except Exception:  # noqa: BLE001
        pass


def _call_bool_setter(
    mod: Any, fn_name: str, val: bool | None, default_bool: bool = True
) -> None:
    if mod is None:
        return
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return
    try:
        if val is None:
            try:
                fn(None)
            except TypeError:
                fn(default_bool)
        else:
            fn(bool(val))
    except Exception:  # noqa: BLE001
        pass


def _set_decode_autotune_runtime(
    stages: int | None, one_pass: bool | None, pdl: bool | None
) -> None:
    try:
        import muse_marlin6_gemv as _m6

        fn_st = getattr(_m6, "set_runtime_tma4_stages", None)
        if callable(fn_st):
            try:
                fn_st(stages)
            except TypeError:
                if stages is not None:
                    fn_st(int(stages))
        _call_bool_setter(_m6, "set_runtime_pdl", pdl, True)
    except Exception:  # noqa: BLE001
        pass
    for mod_name in (
        "muse_rans_ext",
        "vllm.model_executor.kernels.linear.scaled_mm.muse_fp8",
        "muse_bf16_lane",
        "vllm.model_executor.models.muse_glimmer_norm_glue",
    ):
        try:
            mod = __import__(mod_name, fromlist=["set_runtime_pdl"])
            _call_bool_setter(mod, "set_runtime_pdl", pdl, True)
        except Exception:  # noqa: BLE001
            pass
    try:
        import muse_gqa16_attn as _gqa

        _call_bool_setter(_gqa, "set_runtime_one_pass", one_pass, True)
        _call_bool_setter(_gqa, "set_runtime_pdl", pdl, True)
    except Exception:  # noqa: BLE001
        pass


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    num_active_loras: int = 0


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        # Memory ranges of the model's weights (set by the model capture) and
        # the L2 prefetch sessions of the captured FULL graphs.
        self.weight_ranges: list[tuple[int, int]] | None = None
        self._live_weight_ranges_fn: Callable[[], list[tuple[int, int]]] | None = None
        self._l2_prefetch_sessions: list[Any] = []

        self._init_candidates()

    def _verify_l2_prefetch(self, l2pf: Any) -> None:
        if l2pf is None or not hasattr(l2pf, "verify_live_ranges"):
            return
        live = (
            self._live_weight_ranges_fn()
            if self._live_weight_ranges_fn is not None
            else self.weight_ranges
        )
        if live is not None:
            l2pf.verify_live_ranges(live)

    def _l2_prefetch_session(
        self, desc: "BatchExecutionDescriptor", *, retain: bool = True
    ) -> Any:
        if (
            l2_prefetch is None
            or desc.cg_mode != CUDAGraphMode.FULL
            or self.weight_ranges is None
            or not l2_prefetch.available()
        ):
            return None
        session = l2_prefetch.Session(self.weight_ranges)
        if retain:
            self._l2_prefetch_sessions.append(session)
        return session

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_token_lora: dict[tuple[int, int], list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )
        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )

        # When using Dynamic SD, num_speculative_tokens is the max number of
        # draft tokens. The scheduler might use a smaller number so we need
        # to capture graphs for all possible values during decode.
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            # uses_dynamic_speculative_decoding() guarantees this is set.
            assert num_spec_per_batch_size is not None
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            # Each entry is (range_start, range_end, num_speculative_tokens).
            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
        else:
            decode_query_lens = [self.decode_query_len]

        for num_tokens, num_active_loras in product(
            capture_sizes, self.lora_capture_cases
        ):
            # Capture uniform decode specfifc graphs if required
            #  (i.e. separate decode routine)
            if separate_decode_routine and decode_mode:
                for decode_query_len in decode_query_lens:
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or rounded_num_tokens > max_cg_capture_size
                        or rounded_num_reqs > self.max_num_reqs
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
                        descs_by_token_lora[
                            (rounded_num_tokens, num_active_loras)
                        ].append(desc)

            if mixed_mode:
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        if not descs_by_token_lora:
            return

        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                for num_active_loras in self.lora_capture_cases:
                    staging_key = (token_cg_size, num_active_loras)
                    if staging_key in descs_by_token_lora:
                        self._candidates[(i, num_active_loras)] = descs_by_token_lora[
                            staging_key
                        ]
            current_range_start = token_cg_size + 1

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device):
            _apply_l2_fetch_granularity()
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup (also records the GEMM sites for the L2 prefetch)
                    l2pf = self._l2_prefetch_session(desc)
                    with l2pf.record() if l2pf is not None else nullcontext():
                        forward_fn(CUDAGraphMode.NONE)
                    self._verify_l2_prefetch(l2pf)

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        self._verify_l2_prefetch(l2pf)
                        persist_set = False
                        cap_stream = current_platform.current_stream()
                        if desc.num_tokens == 1:
                            arena = getattr(
                                getattr(self, "_capture_model", None),
                                "_persisting_norm_arena",
                                None,
                            )
                            persist_set = _apply_persisting_window(
                                arena, cap_stream, False, _PERSIST_L2_LIMIT_BYTES
                            )
                        try:
                            with (
                                torch.cuda.graph(graph, self.pool),
                                l2pf.apply() if l2pf is not None else nullcontext(),
                            ):
                                forward_fn(CUDAGraphMode.NONE)
                                # Join offloader's copy stream after forward to avoid
                                # unjoined stream error. The last layer's start_prefetch
                                # forks copy_stream, but wait_prefetch only happens in
                                # the next forward pass.
                                get_offloader().join_after_forward()
                        finally:
                            if persist_set:
                                _clear_persisting_window(
                                    cap_stream, _PERSIST_L2_LIMIT_BYTES
                                )
                        upload_cudagraph(graph, cap_stream)
                        if l2pf is not None:
                            logger.info(
                                "L2 prefetch for num_tokens=%d: %s",
                                desc.num_tokens,
                                l2pf.summary(),
                            )
                        self.graphs[desc] = graph
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                ):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self._capture_model = model
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        if l2_prefetch is not None:
            self.weight_ranges = l2_prefetch.weight_ranges(model)
            self._live_weight_ranges_fn = lambda: l2_prefetch.weight_ranges(model)
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                if self.is_last_pp_rank:
                    # Last PP rank (common case).
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux
                else:
                    # Non-last PP rank.
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if self.intermediate_tensors is None:
                        self.intermediate_tensors = IntermediateTensors.empty_like(
                            intermediate_tensors
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        self.intermediate_tensors[k][:num_tokens] = v

            return forward_fn

        self._last_create_forward_fn = create_forward_fn
        try:
            super().capture(create_forward_fn, progress_bar_desc)
        finally:
            self._live_weight_ranges_fn = None

    @torch.inference_mode()
    def maybe_autotune_decode_graph(self, runner: Any) -> None:
        """Proposal 09: Whole-graph boot autotuner for M=1 FULL decode graph.

        Captures 4 candidate decode graphs sharing weights & activation pool,
        verifies 0-ULP (`torch.equal` on output logits across 6 synthetic steps
        at pos=512..517), times 16 interleaved rounds (dropping round 0), picks
        winner only if paired gain > 2*sigma and > 0.3%, resets losers, and
        uploads winner. Fail-soft to default captured graph.
        """
        env_val = os.environ.get("MUSE_BOOT_AUTOTUNE")
        if env_val is not None:
            enabled = env_val.strip().lower() not in ("0", "", "false", "no")
        else:
            enabled = False
        if not enabled:
            return

        create_fn = getattr(self, "_last_create_forward_fn", None)
        if (
            create_fn is None
            or not self._graphs_captured
            or not self.is_first_pp_rank
            or not self.is_last_pp_rank
            or self.dp_size != 1
            or self.tp_size != 1
            or self.use_aux_hidden_state_outputs
            or self.hidden_states is None
            or getattr(runner, "lora_config", None) is not None
            or getattr(runner, "speculator", None) is not None
        ):
            return

        target_desc = self.dispatch(1, 1, 1, 0)
        if (
            target_desc.cg_mode != CUDAGraphMode.FULL
            or target_desc.num_tokens != 1
            or target_desc.num_reqs not in (1, None)
            or target_desc not in self.graphs
        ):
            return

        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
        from vllm.v1.kv_cache_interface import AttentionSpec

        bt = runner.block_tables
        ib = runner.input_buffers
        if not bt.block_sizes or bt.cp_size != 1:
            return
        min_bs = min(bt.block_sizes)
        need_blks = (518 + min_bs - 1) // min_bs
        need_total_blks = need_blks * bt.num_kv_cache_groups
        if runner.kv_cache_config.num_blocks <= need_total_blks + 2:
            return

        # Collect unique KV cache tensors and their block dimension for clean fill & zeroing.
        kv_views: list[tuple[torch.Tensor, int]] = []
        seen_kv_ptrs: set[int] = set()
        fwd_ctx = runner.compilation_config.static_forward_context
        for groups in runner.attn_groups:
            for group in groups:
                spec = group.kv_cache_spec
                if not isinstance(spec, AttentionSpec):
                    return
                if group.kv_cache_group_id >= len(runner.kernel_block_sizes):
                    continue
                kbs = runner.kernel_block_sizes[group.kv_cache_group_id]
                if kbs != spec.block_size:
                    return
                bdim = group.backend.get_kv_cache_block_dim(
                    kbs,
                    spec.num_kv_heads,
                    spec.head_size,
                    cache_dtype_str=runner.cache_config.cache_dtype,
                )
                for lname in group.layer_names:
                    attn_mod = fwd_ctx.get(lname)
                    kv = getattr(attn_mod, "kv_cache", None)
                    if isinstance(kv, (list, tuple)) and len(kv) == 1:
                        kv = kv[0]
                    if not isinstance(kv, torch.Tensor):
                        return
                    if kv.data_ptr() not in seen_kv_ptrs:
                        seen_kv_ptrs.add(kv.data_ptr())
                        kv_views.append((kv, bdim))
        if not kv_views:
            return

        orig_graph = self.graphs[target_desc]
        cand_specs = [
            ("C0(st8,1pass,pdl1,L2Norm)", 8, True, True, False),
            ("C2(st8,2pass,pdl1,L2Norm)", 8, False, True, False),
            ("C3(st8,2pass,pdl0,L2Norm)", 8, False, False, False),
        ]
        captured: list[tuple[str, tuple[int, bool, bool, bool], torch.cuda.CUDAGraph, Any]] = []
        installed_winner = False
        arena = getattr(runner.model, "_persisting_norm_arena", None)

        try:
            if l2_prefetch is not None:
                self._live_weight_ranges_fn = lambda: l2_prefetch.weight_ranges(runner.model)
            with graph_capture(device=self.device):
                cap_stream = current_platform.current_stream()
                for cname, st_val, op_val, pdl_val, miss_stream in cand_specs:
                    try:
                        _set_decode_autotune_runtime(st_val, op_val, pdl_val)
                        fn_w = create_fn(target_desc, warmup=True)
                        cand_l2pf = self._l2_prefetch_session(target_desc, retain=False)
                        with cand_l2pf.record() if cand_l2pf is not None else nullcontext():
                            fn_w(CUDAGraphMode.NONE)
                        self._verify_l2_prefetch(cand_l2pf)

                        fn_c = create_fn(target_desc, warmup=False)
                        cg = torch.cuda.CUDAGraph()
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        self._verify_l2_prefetch(cand_l2pf)
                        pset = _apply_persisting_window(
                            arena, cap_stream, miss_stream, _PERSIST_L2_LIMIT_BYTES
                        )
                        try:
                            with (
                                torch.cuda.graph(cg, self.pool),
                                cand_l2pf.apply() if cand_l2pf is not None else nullcontext(),
                            ):
                                fn_c(CUDAGraphMode.NONE)
                                get_offloader().join_after_forward()
                        finally:
                            if pset:
                                _clear_persisting_window(
                                    cap_stream, _PERSIST_L2_LIMIT_BYTES
                                )
                        captured.append(
                            (cname, (st_val, op_val, pdl_val, miss_stream), cg, cand_l2pf)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Boot autotune candidate %s capture skipped: %s", cname, exc
                        )
                        if cname.startswith("C0"):
                            return

            if len(captured) < 2:
                return

            # Build eager reference forward with default runtime config BEFORE staging synthetic pos=512 state.
            _set_decode_autotune_runtime(8, True, True)
            eager_ref_fn = create_fn(target_desc, warmup=False)

            # Populate blocks 1..need_total_blks cleanly (disjoint slice per group g).
            for kv, bdim in kv_views:
                span = kv.narrow(bdim, 1, need_total_blks)
                if span.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    n_el = span.numel()
                    pat = (
                        ((torch.arange(n_el, device=self.device) % 13).float() - 6.0)
                        .mul(0.03125)
                        .to(span.dtype)
                        .reshape(span.shape)
                    )
                    span.copy_(pat)
                else:
                    span.zero_()

            for g in range(bt.num_kv_cache_groups):
                base_g = 1 + g * need_blks
                blk_seq_g = torch.arange(
                    base_g, base_g + need_blks, dtype=torch.int32, device=self.device
                )
                bt.input_block_tables[g][0, :need_blks].copy_(blk_seq_g)

            enc_runner = getattr(runner.model_state, "encoder_runner", None)
            emb_dst = (
                enc_runner.inputs_embeds[:1]
                if (runner.supports_mm_inputs and enc_runner is not None)
                else None
            )

            alive = list(captured)
            for step_s in range(6):
                pos_s = 512 + step_s
                ib.input_ids[0] = 1000 + step_s * 37
                ib.positions[0] = pos_s
                ib.seq_lens[0] = pos_s + 1
                if ib.seq_lens.numel() > 1:
                    ib.seq_lens[1:].zero_()
                ib.query_start_loc[0] = 0
                ib.query_start_loc[1:] = 1
                ib.is_padding[0] = False
                for g in range(bt.num_kv_cache_groups):
                    bs_g = bt.block_sizes[g]
                    b_id = (1 + g * need_blks) + (pos_s // bs_g)
                    bt.slot_mappings[g, 0] = b_id * bs_g + (pos_s % bs_g)
                if emb_dst is not None:
                    h_dim = emb_dst.shape[-1]
                    emb_pat = (
                        (
                            (torch.arange(h_dim, device=self.device) + step_s * 97) % 19
                            - 9.0
                        )
                        .mul(0.03125)
                        .to(emb_dst.dtype)
                        .reshape(emb_dst.shape)
                    )
                    emb_dst.copy_(emb_pat)

                target_blks = {
                    (1 + g * need_blks) + (pos_s // bt.block_sizes[g])
                    for g in range(bt.num_kv_cache_groups)
                }
                for kv, bdim in kv_views:
                    for tb in target_blks:
                        kv.narrow(bdim, tb, 1).zero_()

                self.hidden_states[:1].zero_()
                _set_decode_autotune_runtime(8, True, True)
                eager_ref_fn(CUDAGraphMode.NONE)
                ref_logits = runner.model.compute_logits(self.hidden_states[:1]).clone()

                next_alive = []
                for item in alive:
                    cname, cfg, cg, cl2pf = item
                    for kv, bdim in kv_views:
                        for tb in target_blks:
                            kv.narrow(bdim, tb, 1).zero_()
                    self.hidden_states[:1].zero_()
                    cg.replay()
                    cand_logits = runner.model.compute_logits(self.hidden_states[:1])
                    if torch.equal(cand_logits, ref_logits):
                        next_alive.append(item)
                    else:
                        logger.warning(
                            "Boot autotune candidate %s failed 0-ULP check at pos=%d; disqualified",
                            cname,
                            pos_s,
                        )
                        cg.reset()
                alive = next_alive
                if not alive or not alive[0][0].startswith("C0"):
                    logger.warning(
                        "Boot autotune baseline C0 disqualified; keeping original graph"
                    )
                    return

            if len(alive) < 2:
                logger.info(
                    "Boot autotune: only %s passed 0-ULP; keeping baseline",
                    alive[0][0],
                )
                return

            # Set clean pos=512 state for 16 interleaved timing rounds (drop round 0).
            ib.positions[0] = 512
            ib.seq_lens[0] = 513
            for g in range(bt.num_kv_cache_groups):
                bs_g = bt.block_sizes[g]
                b_id = (1 + g * need_blks) + (512 // bs_g)
                bt.slot_mappings[g, 0] = b_id * bs_g + (512 % bs_g)

            n_cand = len(alive)
            n_rounds = 16
            samples: list[list[float]] = [[] for _ in range(n_cand)]
            stream = torch.cuda.current_stream(self.device)
            ev_pairs = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(n_cand)
            ]
            torch.cuda.synchronize(self.device)
            for r in range(n_rounds):
                for step_k in range(n_cand):
                    c_idx = (r + step_k) % n_cand
                    s_ev, e_ev = ev_pairs[c_idx]
                    s_ev.record(stream)
                    alive[c_idx][2].replay()
                    e_ev.record(stream)
                torch.cuda.synchronize(self.device)
                if r >= 1:
                    for c_idx in range(n_cand):
                        s_ev, e_ev = ev_pairs[c_idx]
                        samples[c_idx].append(s_ev.elapsed_time(e_ev))

            mean_c0 = sum(samples[0]) / len(samples[0])
            win_idx = 0
            best_gain = 0.0
            summary_parts = [f"{alive[0][0]}={mean_c0:.3f}ms"]
            for idx in range(1, n_cand):
                mean_ci = sum(samples[idx]) / len(samples[idx])
                diffs = [t0 - ti for t0, ti in zip(samples[0], samples[idx])]
                m_gain = sum(diffs) / len(diffs)
                var_d = sum((d - m_gain) ** 2 for d in diffs) / max(1, len(diffs) - 1)
                se_d = math.sqrt(var_d / len(diffs))
                rel_gain = m_gain / max(mean_c0, 1e-9)
                summary_parts.append(
                    f"{alive[idx][0]}={mean_ci:.3f}ms(gain={m_gain:+.3f}ms,2sig={2.0 * se_d:.3f}ms,{rel_gain * 100.0:+.2f}%)"
                )
                if m_gain > 2.0 * se_d and rel_gain > 0.003 and m_gain > best_gain:
                    best_gain = m_gain
                    win_idx = idx

            win_name, win_cfg, win_graph, win_l2pf = alive[win_idx]
            for idx, (_cn, _cfg, cg, _cl2) in enumerate(alive):
                if idx != win_idx:
                    cg.reset()
            captured.clear()

            self.graphs[target_desc] = win_graph
            if win_l2pf is not None:
                self._l2_prefetch_sessions.append(win_l2pf)
            _set_decode_autotune_runtime(win_cfg[0], win_cfg[1], win_cfg[2])
            upload_cudagraph(win_graph, stream)
            installed_winner = True
            orig_graph.reset()
            logger.info(
                "Proposal 09 boot autotuner selected %s: %s",
                win_name,
                ", ".join(summary_parts),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Proposal 09 boot autotuner skipped (%s); keeping default graph", exc)
        finally:
            self._live_weight_ranges_fn = None
            self._last_create_forward_fn = None
            if not installed_winner:
                for _cn, _cfg, cg, _cl2 in captured:
                    try:
                        cg.reset()
                    except Exception:  # noqa: BLE001
                        pass
                _set_decode_autotune_runtime(None, None, None)
                self.graphs[target_desc] = orig_graph
                upload_cudagraph(orig_graph, torch.cuda.current_stream(self.device))
            try:
                for kv, bdim in kv_views:
                    kv.narrow(bdim, 1, need_total_blks).zero_()
                for g in range(bt.num_kv_cache_groups):
                    bt.input_block_tables[g].zero_()
                bt.slot_mappings.fill_(PAD_SLOT_ID)
                ib.input_ids[:1].zero_()
                ib.positions[:1].zero_()
                ib.seq_lens.zero_()
                ib.query_start_loc.zero_()
                ib.is_padding.fill_(True)
                if self.hidden_states is not None:
                    self.hidden_states[:1].zero_()
                torch.cuda.synchronize(self.device)
            except Exception:  # noqa: BLE001
                pass

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
