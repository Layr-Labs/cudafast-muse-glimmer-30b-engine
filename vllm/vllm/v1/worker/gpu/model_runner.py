# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

import functools
import gc
import time
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import (
    MultiModalBudget,
    get_dummy_encoder_profile_inputs,
)
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import (
    async_copy_to_gpu,
    set_default_max_concurrency,
)
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import (
    LoraState,
    create_lora_capture_hook,
    get_lora_capture_cases,
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.prefill_graph import PrefillGraph, capture_prefill_graph
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import KVBlockZeroer, copy_kv_cache_blocks_inplace

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        self.output_copy_stream = torch.cuda.Stream(self.device)

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed
        set_default_max_concurrency(vllm_config.max_concurrent_batches)

        # PP broadcast/recv helper. Runs the collective on a side stream.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Data parallelism.
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank

        # Detect EP all2all peer faults to prevent emitting corrupted output.
        # Only meaningful for MoE + DP with an FT-capable all2all backend.
        self.check_ep_fault = False
        if self.dp_size > 1 and self.model_config.is_moe:
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        # Decode context parallelism.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()

        # Speculative decoding.
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in ("eagle3", "dflash", "dspark"):
                # Drafting may require auxiliary hidden states from target model outputs
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.pcp_manager: pcp.PCPManager | None = None

        # Pooling models.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # Multi-module MTP feeds its modules the next num_speculative_steps prefill
        # tokens during chunked prefill. Other speculators only read the immediate
        # next one.
        num_prefill_lookahead = (
            self.num_speculative_steps
            if self.speculative_config is not None
            and self.speculative_config.use_multi_module_mtp()
            else 1
        )
        # General request states.
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )

        # Samplers and decode_query_len created in load_model() after
        # model_state exists (num_new_sampled_tokens_per_step from ModelState).
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None
        # LOCAL PATCH (prefill graph): whole-forward graph for one fresh
        # prefill of a fixed length (vllm/v1/worker/gpu/prefill_graph.py).
        self.prefill_graph: PrefillGraph | None = None

        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.lora_capture_cases = [0]
        if self.lora_config:
            self.lora_capture_cases = get_lora_capture_cases(
                self.lora_config, self.compilation_config
            )

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        # Expert parallelism load balancer.
        self.eplb = EPLBController(self.parallel_config, self.device)

        # LOCAL PATCH (SP5D, Yukon submission 98c1361e): constant one-request sampling metadata, made once instead
        # of an arange and a zeros launch every step. Only read downstream (sampler, logits-index kernels).
        self._cu_num_logits_1_np = np.arange(2, dtype=np.int32)
        self._cu_num_logits_1_gpu = torch.arange(
            2, device=self.device, dtype=torch.int32
        )
        self._expanded_local_pos_1_gpu = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self._ones_1_i32 = torch.ones(1, dtype=torch.int32, device=self.device)
        self._zeros_1_i32 = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._logits_indices_0_gpu = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._ones_1_np = np.ones(1, dtype=np.int32)
        self._false_1_np = np.zeros(1, dtype=np.bool_)
        self._single_decode_prep_enabled = (
            envs.os.environ.get("VLLM_FUSED_DECODE_PREP", "1") != "0"
        )
        self._single_decode_prep_armed = False
        self._single_decode_prep_verified = False
        self._single_decode_prep_failed = False
        self._qsl_const_gpu: dict[int, torch.Tensor] = {}
        for _nt in (1, 2, 512):
            _qsl = [0] + [_nt] * self.max_num_reqs
            self._qsl_const_gpu[_nt] = torch.tensor(
                _qsl, dtype=torch.int32, device=self.device
            )

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # Do not rely on pooling_runner here, since this information is needed
            # on the first PP rank, while pooling_runner is only initialized
            # on the last PP rank.
            tasks.extend(
                PoolingRunner.get_supported_tasks(self.model, self.model_config)
            )
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        self.eplb.prepare_load()
        eplb_models_added = False
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info_once("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if isinstance(self.speculator, DraftModelSpeculator):
                self.speculator.load_model(self.model)
                eplb_models_added = self.eplb.maybe_register_speculator(
                    self.speculator, self.speculative_config, load_dummy_weights
                )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        if not load_dummy_weights:
            # LOCAL PATCH (decode weight placement): move one-row NVFP4 decode
            # weights off slow device memory; same bytes, checked bitwise.
            try:
                from vllm.model_executor.kernels.linear.nvfp4.decode_placement import (
                    place_decode_weights,
                    place_prefill_weights,
                )

                place_decode_weights(self.model)
                place_prefill_weights(self.model)
            except Exception as err:  # noqa: BLE001 - never part of the load's success
                logger.warning("Decode weight placement unavailable: %s", err)
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        if not load_dummy_weights:
            prepare_communication_buffer_for_model(self.model)
            if self.speculator is not None:
                prepare_communication_buffer_for_model(self.speculator.model)

        # Initialize the components that require the model.
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )

        # Initialize samplers. Model states may override via custom_sampler().
        if self.is_last_pp_rank and not self.is_pooling_model:
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                self.sampler, self.rejection_sampler = custom
            elif self.speculative_config is not None:
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

    def get_model(self) -> nn.Module:
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]

    def update_config(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 reads config via self.vllm_config (e.g. in load_model), so keep it
        # in sync with the attributes the v1 helper just replaced.
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        return torch.cuda.current_stream(self.device)

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
            # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens
            if isinstance(spec, MambaSpec):
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
                max_num_blocks = get_block_table_width(
                    max_num_blocks, spec.block_size, token_alignment=None
                )
            else:
                max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        attn_cg_support = attn_cg_support.narrow(
            *self.model_state.get_additional_cg_support()
        )
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
        )
        check_attention_cp_compatibility(self.vllm_config)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            self.kernel_block_sizes,
            self.vllm_config,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): for now since the worker is shared between MRV1 and MRV2,
            # and for spec-decode with MTP we want to make sure the dummy runs use
            # 1+num_speculative_tokens we use max here, this will likely be eventually
            # changed in the worker: https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        # Distribute the remainder evenly so no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.
        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # Execute the model.
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
            )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        self.execute_model_state = None

        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
            assert self.sampler is not None
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device="cpu",
                    ),
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            self.speculator.propose(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings_by_layer,
                last_hidden_states=spec_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=torch.ones(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                num_rejected=torch.zeros(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                last_sampled=self.req_states.last_sampled_tokens,
                next_prefill_tokens=self.req_states.next_prefill_tokens,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                mm_budget = MultiModalBudget(
                    self.vllm_config,
                    self.mm_registry,
                    enable_cache=False,
                )
                dummy_mm_inputs = get_dummy_encoder_profile_inputs(
                    self.mm_registry,
                    mm_budget,
                )
                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )

        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        self.reset_encoder_cache()
        gc.collect()

    def post_kv_cache_wake_up(self) -> None:
        self.block_tables.init_block_table_layout_tensors()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    def profile_cudagraph_memory(self) -> int:
        # NOTE(woosuk): It is TBD whether we keep this API or not.
        return 0

    @torch.inference_mode()
    def capture_model(self) -> int:
        assert self.cudagraph_manager is not None
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            self.cudagraph_manager.capture(
                self.model,
                self.model_state,
                self.input_buffers,
                self.intermediate_tensors,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                has_lora=self.lora_config is not None,
                use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                lora_capture_hook=create_lora_capture_hook(self.lora_config, self),
            )
            if self.speculator is not None:
                self.speculator.capture()

        # LOCAL PATCH (prefill graph): captured apart from the dispatcher's
        # graphs, so decode dispatch and compile ranges stay unchanged.
        self.prefill_graph = capture_prefill_graph(self)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # Call model_state.remove_request *before* req_states.remove_request
        # so the model_state can still look up the slot index.
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pooling_runner is not None:
            self.pooling_runner.remove_request(req_idx)
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        if self.pooling_runner is not None:
            # Preempted docs keep their query-use reservation until rescheduled.
            self.pooling_runner.on_requests_finished(finished_req_ids)
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.pooling_runner is not None:
                assert new_req_data.pooling_params is not None
                self.pooling_runner.add_request(
                    req_id,
                    req_index,
                    new_req_data.pooling_params,
                    new_req_data.prompt_token_ids,
                )

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        if envs.VLLM_MOE_SKIP_PADDING:
            # Mark trailing cudagraph-padding rows so kernels can skip work for
            # them when supported.
            self.input_buffers.is_padding[:num_tokens].fill_(False)
            self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                True
            )
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # batch_idx -> req_id
        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        if num_reqs == 1 and int(idx_mapping_np[0]) == 0:
            idx_mapping = self._zeros_1_i32
        elif num_reqs == 1 and int(idx_mapping_np[0]) == 1:
            idx_mapping = self._ones_1_i32
        else:
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            if num_reqs == 1:
                cu_num_logits_np = self._cu_num_logits_1_np
                cu_num_logits = self._cu_num_logits_1_gpu
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = self._expanded_local_pos_1_gpu
            else:
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(
                    num_reqs + 1, device=self.device, dtype=torch.int32
                )
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(
                    num_reqs, dtype=torch.int32, device=self.device
                )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.decode_query_len
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        _qsl_cached = self._qsl_const_gpu.get(num_tokens) if num_reqs == 1 else None
        if _qsl_cached is not None:
            self.input_buffers.query_start_loc.copy_(_qsl_cached, non_blocking=True)
        else:
            async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens
        num_computed_prefill_tokens_np = computed_prefill_tokens_np[idx_mapping_np]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        # Get prefill tokens if any.
        if np.any(is_prefilling_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        max_seq_len_np = None
        if self.use_pp:
            # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            max_seq_len_np=max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
        )
        return pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is not None:
            return self.pcp_manager.prepare_attn(input_batch)

        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def _try_lmhead_argmax_graph(
        self,
        sample_hidden_states: torch.Tensor,
        lmpost_argmax: Any,
        lmpost_into: Any,
    ) -> torch.Tensor | None:
        if (
            lmpost_into is None
            or envs.os.environ.get("VLLM_LMHEAD_GRAPH", "1") == "0"
            or self.cudagraph_manager is None
            or not self.cudagraph_manager._graphs_captured
            or sample_hidden_states.ndim != 2
            or sample_hidden_states.shape[0] != 1
            or not sample_hidden_states.is_contiguous()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        entries = getattr(self, "_lmhead_cg_entries", None)
        if entries is None:
            entries = {}
            self._lmhead_cg_entries = entries
            self._lmhead_cg_rejected: set[tuple] = set()
            self._lmhead_cg_pool = torch.cuda.graph_pool_handle()
            self._lmhead_cg_cap_stream = torch.cuda.Stream(device=self.device)
            self._lmhead_cg_step = 0
        sig = (
            sample_hidden_states.data_ptr(),
            tuple(sample_hidden_states.shape),
            tuple(sample_hidden_states.stride()),
        )
        entry = entries.get(sig)
        if entry is not None:
            graphs, outs = entry
            slot = self._lmhead_cg_step & 3
            self._lmhead_cg_step += 1
            graphs[slot].replay()
            return outs[slot]
        if sig in self._lmhead_cg_rejected or len(entries) >= 4:
            return None

        mult = self.model.output_multiplier
        cap = self.model.final_logit_softcapping
        raw_ref = self.model.logits_processor(self.model.lm_head, sample_hidden_states)
        if raw_ref is None:
            self._lmhead_cg_rejected.add(sig)
            return None
        ref_tok = lmpost_argmax(raw_ref, mult, cap)
        if ref_tok is None:
            self._lmhead_cg_rejected.add(sig)
            return None
        del raw_ref
        fused_launch = None
        try:
            import lmpost as _lmp
            import muse_marlin6_gemv as _m6
            lm_head = self.model.lm_head
            lp = self.model.logits_processor
            org_vocab = int(getattr(lp, "org_vocab_size", getattr(lm_head, "org_vocab_size", 0)))
            if (
                getattr(lm_head, "tp_size", 1) == 1
                and getattr(lm_head, "bias", None) is None
                and not getattr(lp, "logits_as_input", False)
                and getattr(lp, "soft_cap", None) is None
                and float(getattr(lp, "scale", 1.0)) == 1.0
                and sample_hidden_states.dtype == torch.bfloat16
                and (sample_hidden_states.data_ptr() & 15) == 0
            ):
                lut = _lmp._get_lut(self.device, mult, cap)
                if getattr(lm_head, "marlin_decode6_lmpost_admitted", None) is None:
                    lm_head.marlin_decode6_lmpost_admitted = _m6.admit_tma4_lmpost(
                        lm_head, lut, org_vocab, lmpost_argmax, mult, cap
                    )
                if lm_head.marlin_decode6_lmpost_admitted:
                    fused_launch = (lm_head, lut, org_vocab, _m6.launch_lmhead_lmpost6)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fused lm_head+lmpost setup skipped (%s); using 2-node graph", exc)
            fused_launch = None
        try:
            outs = [torch.empty((1, 1), dtype=torch.int64, device=self.device) for _ in range(4)]
            graphs: list[torch.cuda.CUDAGraph] = []
            main_st = torch.cuda.current_stream(self.device)
            cap_st = self._lmhead_cg_cap_stream
            cap_st.wait_stream(main_st)
            with torch.cuda.stream(cap_st):
                for slot in range(4):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self._lmhead_cg_pool, stream=cap_st):
                        if fused_launch is not None:
                            lm_h, lut_t, ov, fn_f = fused_launch
                            if not fn_f(sample_hidden_states, lm_h, lut_t, outs[slot], ov):
                                raise RuntimeError("fused lm_head+lmpost refused during capture")
                        else:
                            raw_g = self.model.logits_processor(self.model.lm_head, sample_hidden_states)
                            if raw_g is None or not lmpost_into(raw_g, mult, cap, outs[slot]):
                                raise RuntimeError("lmpost_into refused during capture")
                    graphs.append(g)
            main_st.wait_stream(cap_st)
            outs[0].fill_(-1)
            graphs[0].replay()
            if not torch.equal(outs[0], ref_tok):
                self._lmhead_cg_rejected.add(sig)
                logger.warning("lm_head+lmpost graph mismatch vs eager; keeping eager path")
                return ref_tok
            entries[sig] = (graphs, outs)
            self._lmhead_cg_step = 1
            logger.info(
                "Captured %s lm_head+lmpost_argmax CUDA graph (4-slot ring, shape=%s)",
                "1-node fused" if fused_launch is not None else "2-node",
                tuple(sample_hidden_states.shape),
            )
            return outs[0]
        except Exception as exc:  # noqa: BLE001
            self._lmhead_cg_rejected.add(sig)
            logger.warning("lm_head+lmpost graph capture skipped (%s); keeping eager path", exc)
            return ref_tok

    def _try_decode_embed_graph(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> torch.Tensor | None:
        enc_runner = getattr(self.model_state, "encoder_runner", None)
        if (
            enc_runner is None
            or enc_runner.is_realtime
            or getattr(self.model_state, "mm_pruner", None) is not None
            or scheduled_encoder_inputs
            or self.lora_config is not None
            or envs.os.environ.get("VLLM_DECODE_EMBED_GRAPH", "1") == "0"
            or self.cudagraph_manager is None
            or not self.cudagraph_manager._graphs_captured
            or torch.cuda.is_current_stream_capturing()
            or input_batch.num_reqs != 1
            or input_batch.num_tokens != 1
            or input_batch.num_tokens_after_padding != 1
            or int(input_batch.num_computed_tokens_np[0]) < int(input_batch.prefill_len_np[0])
        ):
            return None
        ids_1 = input_batch.input_ids[:1]
        dst_1 = enc_runner.inputs_embeds[:1]
        if not ids_1.is_contiguous() or not dst_1.is_contiguous():
            return None
        entries = getattr(self, "_decode_embed_cg_entries", None)
        if entries is None:
            entries = {}
            self._decode_embed_cg_entries = entries
            self._decode_embed_cg_rejected: set[tuple] = set()
            self._decode_embed_cg_pool = torch.cuda.graph_pool_handle()
            self._decode_embed_cg_cap_stream = torch.cuda.Stream(device=self.device)
        sig = (ids_1.data_ptr(), ids_1.dtype, dst_1.data_ptr(), tuple(dst_1.shape))
        entry = entries.get(sig)
        if entry is not None:
            g, out_view = entry
            g.replay()
            return out_view
        if sig in self._decode_embed_cg_rejected or len(entries) >= 4:
            return None

        ref_out = self.model_state.get_mm_embeddings(
            scheduled_encoder_inputs, input_batch, self.req_states
        )
        if ref_out is None or ref_out.data_ptr() != dst_1.data_ptr() or ref_out.shape != dst_1.shape:
            self._decode_embed_cg_rejected.add(sig)
            return ref_out
        ref_copy = ref_out.clone()
        try:
            import muse_embed as _me

            lm = (
                self.model.get_language_model()
                if hasattr(self.model, "get_language_model")
                else getattr(self.model, "model", self.model)
            )
            if not (hasattr(lm, "embed_tokens") and hasattr(lm, "embed_norm")):
                lm = getattr(self.model, "model", lm)
            if not _me.embed_into(lm, ids_1, dst_1):
                dst_1.copy_(ref_copy)
                self._decode_embed_cg_rejected.add(sig)
                return dst_1
            if not torch.equal(dst_1.view(torch.int16), ref_copy.view(torch.int16)):
                dst_1.copy_(ref_copy)
                self._decode_embed_cg_rejected.add(sig)
                logger.warning("decode muse_embed eager mismatch; keeping eager get_mm_embeddings")
                return dst_1
            main_st = torch.cuda.current_stream(self.device)
            cap_st = self._decode_embed_cg_cap_stream
            cap_st.wait_stream(main_st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cap_st):
                with torch.cuda.graph(g, pool=self._decode_embed_cg_pool, stream=cap_st):
                    if not _me.embed_into(lm, ids_1, dst_1):
                        raise RuntimeError("muse_embed.embed_into refused during decode graph capture")
            main_st.wait_stream(cap_st)
            dst_1.zero_()
            g.replay()
            if not torch.equal(dst_1.view(torch.int16), ref_copy.view(torch.int16)):
                dst_1.copy_(ref_copy)
                self._decode_embed_cg_rejected.add(sig)
                logger.warning("decode muse_embed graph mismatch vs eager; keeping eager path")
                return dst_1
            entries[sig] = (g, dst_1)
            logger.info(
                "Captured M=1 decode muse_embed CUDA graph (shape=%s)",
                tuple(dst_1.shape),
            )
            return dst_1
        except Exception as exc:  # noqa: BLE001
            dst_1.copy_(ref_copy)
            self._decode_embed_cg_rejected.add(sig)
            logger.warning("decode muse_embed graph capture skipped (%s); keeping eager path", exc)
            return dst_1

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
        compact_sample: bool = False,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        # LOCAL PATCH (SP5D, Yukon submission 98c1361e): a one-request step without draft tokens that completes its
        # prompt takes its sampling row as a slice (the row logits_indices would gather), and a greedy step without
        # logprobs, penalties, bias or grammar takes the token from lmpost's fused softcap + argmax.
        can_single_row = (
            input_batch.num_reqs == 1
            and input_batch.num_draft_tokens == 0
            and int(input_batch.num_computed_tokens_np[0])
            + int(input_batch.num_scheduled_tokens[0])
            >= int(input_batch.prefill_len_np[0])
        )
        if compact_sample:
            sample_hidden_states = hidden_states[-1:]
        elif can_single_row:
            logits_row = int(input_batch.query_start_loc_np[1]) - 1
            sample_hidden_states = hidden_states[logits_row : logits_row + 1]
        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
        lmpost_argmax = getattr(self.model, "_lmpost_softcap_argmax", None)
        if (
            can_single_row
            and lmpost_argmax is not None
            and grammar_output is None
            and self.sampler is not None
            and not self.sampler.compute_nans
        ):
            idx_mapping_np = input_batch.idx_mapping_np
            states = self.sampler.sampling_states
            if (
                states.temperature.np[idx_mapping_np[0]] == 0.0
                and states.max_num_logprobs(idx_mapping_np) == NO_LOGPROBS
                and self.sampler.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) == 0
                and not self.sampler._requires_logits_processing(idx_mapping_np)
            ):
                lmpost_into = getattr(self.model, "_lmpost_softcap_argmax_into", None)
                sampled_token_ids = self._try_lmhead_argmax_graph(
                    sample_hidden_states, lmpost_argmax, lmpost_into
                )
                if sampled_token_ids is None:
                    raw_logits = self.model.logits_processor(self.model.lm_head, sample_hidden_states)
                    if raw_logits is not None:
                        sampled_token_ids = lmpost_argmax(
                            raw_logits, self.model.output_multiplier, self.model.final_logit_softcapping
                        )
                if sampled_token_ids is not None:
                    if not getattr(self, "_lmpost_logged", False):
                        self._lmpost_logged = True
                        logger.info("lmpost fused softcap + argmax in use (one-row greedy decode)")
                    return (
                        SamplerOutput(
                            sampled_token_ids=sampled_token_ids,
                            logprobs_tensors=None,
                            num_nans=None,
                            num_sampled=self._ones_1_i32,
                            num_rejected=self._zeros_1_i32,
                        ),
                        self._ones_1_i32,
                        self._zeros_1_i32,
                    )
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    def _can_zero_prep_prefill(self, scheduler_output: SchedulerOutput) -> bool:
        pg = self.prefill_graph
        if (
            pg is None
            or not pg._fast_verified
            or not pg._zp_enabled
            or pg._zp_failed
            or pg.embed_input_ids is None
            or len(scheduler_output.scheduled_new_reqs) != 1
            or len(scheduler_output.scheduled_cached_reqs.req_ids) != 0
            or scheduler_output.total_num_scheduled_tokens != pg.num_tokens
            or len(scheduler_output.num_scheduled_tokens) != 1
            or scheduler_output.scheduled_encoder_inputs
            or scheduler_output.scheduled_spec_decode_tokens
            or scheduler_output.new_block_ids_to_zero
            or scheduler_output.kv_cache_block_copies
            or scheduler_output.has_structured_output_requests
            or self.lora_config is not None
            or self.pcp_manager is not None
            or self.dp_size != 1
            or not self.is_first_pp_rank
            or not self.is_last_pp_rank
            or self.use_dcp
            or self.use_pp
            or self.model_config.rswa_window is not None
            or self.use_aux_hidden_state_outputs
            or self.speculator is not None
            or not self.supports_mm_inputs
            or self.model.requires_raw_input_tokens
            or self.block_tables.cp_size != 1
        ):
            return False
        for bs, kbs in zip(self.block_tables.block_sizes, self.block_tables.kernel_block_sizes):
            if bs != kbs or pg.num_tokens % bs != 0:
                return False
        nr = scheduler_output.scheduled_new_reqs[0]
        if (
            nr.num_computed_tokens != 0
            or nr.prefill_token_ids is None
            or nr.prompt_token_ids is None
            or len(nr.prefill_token_ids) != pg.num_tokens
            or len(nr.prompt_token_ids) != pg.num_tokens
            or nr.mm_features
            or nr.sampling_params is None
            or nr.sampling_params.prompt_logprobs is not None
        ):
            return False
        if self.cudagraph_manager is not None:
            bd, _ = dispatch_cg_and_sync_dp(
                self.cudagraph_manager, 1, pg.num_tokens, pg.num_tokens, 1, 0,
                need_eager=False, num_active_loras=0,
            )
            if bd.cg_mode != CUDAGraphMode.NONE or bd.num_tokens != pg.num_tokens:
                return False
        return True

    def _build_zero_prep_batch(self, req_id: str, n: int) -> InputBatch:
        pg = self.prefill_graph
        assert pg is not None
        req_idx = self.req_states.req_id_to_index[req_id]
        idx_mapping_np = np.array([req_idx], dtype=np.int32)
        if req_idx == 0:
            idx_mapping = self._zeros_1_i32
        elif req_idx == 1:
            idx_mapping = self._ones_1_i32
        else:
            idx_mapping = torch.tensor([req_idx], dtype=torch.int32, device=self.device)
        num_sched = np.array([n], dtype=np.int32)
        qsl_np = np.array([0, n], dtype=np.int32)
        prefill_len_np = np.array([n], dtype=np.int32)
        zero_1_np = np.zeros(1, dtype=np.int32)
        is_pref_np = np.array([True], dtype=np.bool_)
        seq_ub = torch.from_numpy(num_sched.copy())
        return InputBatch(
            req_ids=[req_id],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=self._expanded_local_pos_1_gpu,
            num_scheduled_tokens=num_sched,
            num_tokens=n,
            num_tokens_after_padding=n,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=self.input_buffers.query_start_loc[:2],
            query_start_loc_np=qsl_np,
            seq_lens=self.input_buffers.seq_lens[:1],
            seq_lens_cpu_upper_bound=seq_ub,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=zero_1_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=zero_1_np,
            is_prefilling_np=is_pref_np,
            max_seq_len_np=None,
            input_ids=self.input_buffers.input_ids[:n],
            positions=self.input_buffers.positions[:n],
            is_padding=self.input_buffers.is_padding[:n],
            logits_indices=pg._zp_logits_indices_gpu,
            cu_num_logits=self._cu_num_logits_1_gpu,
            cu_num_logits_np=self._cu_num_logits_1_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
        )

    def _is_single_full_decode(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> tuple[str, int, int, int, int] | None:
        if (
            not self._single_decode_prep_enabled
            or self._single_decode_prep_failed
            or batch_desc.cg_mode != CUDAGraphMode.FULL
            or batch_desc.num_tokens != 1
            or batch_desc.num_reqs != 1
            or self.cudagraph_manager is None
            or batch_desc not in self.cudagraph_manager.graphs
            or scheduler_output.total_num_scheduled_tokens != 1
            or len(scheduler_output.num_scheduled_tokens) != 1
            or scheduler_output.scheduled_spec_decode_tokens
            or self.use_dcp
            or self.pcp_manager is not None
            or self.block_tables.cp_size != 1
            or self.model_state.num_new_sampled_tokens_per_step != 1
        ):
            return None
        req_id = next(iter(scheduler_output.num_scheduled_tokens))
        req_idx = self.req_states.req_id_to_index.get(req_id)
        if req_idx is None:
            return None
        num_comp = int(self.req_states.num_computed_tokens_np[req_idx])
        pref_len = int(self.req_states.prefill_len.np[req_idx])
        num_comp_pref = int(self.req_states.num_computed_prefill_tokens[req_idx])
        if num_comp_pref < pref_len or num_comp < pref_len:
            return None
        return req_id, req_idx, num_comp, pref_len, num_comp_pref

    def _try_prepare_single_decode(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> tuple[InputBatch, tuple[torch.Tensor, ...], torch.Tensor] | None:
        if not self._single_decode_prep_armed:
            return None
        if not self._single_decode_prep_verified and torch.cuda.is_current_stream_capturing():
            return None
        info = self._is_single_full_decode(scheduler_output, batch_desc)
        if info is None:
            return None
        req_id, req_idx, num_comp, pref_len, num_comp_pref = info
        idx_mapping_np = np.array([req_idx], dtype=np.int32)
        if req_idx == 0:
            idx_mapping = self._zeros_1_i32
        elif req_idx == 1:
            idx_mapping = self._ones_1_i32
        else:
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        block_tables, slot_mappings = self.block_tables.prepare_single_decode_step(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        if not self._single_decode_prep_verified:
            got_ids = self.input_buffers.input_ids.clone()
            got_pos = self.input_buffers.positions.clone()
            got_sl = self.input_buffers.seq_lens.clone()
            got_qsl = self.input_buffers.query_start_loc.clone()
            got_pad = self.input_buffers.is_padding.clone()
            got_slots = self.block_tables.slot_mappings.clone()
            got_bts = [bt.clone() for bt in self.block_tables.input_block_tables]
            ref_batch = self.prepare_inputs(scheduler_output, batch_desc)
            ref_bts, ref_slots = self.prepare_attn(ref_batch)
            ok = (
                torch.equal(got_ids, self.input_buffers.input_ids)
                and torch.equal(got_pos, self.input_buffers.positions)
                and torch.equal(got_sl, self.input_buffers.seq_lens)
                and torch.equal(got_qsl, self.input_buffers.query_start_loc)
                and torch.equal(got_pad, self.input_buffers.is_padding)
                and torch.equal(got_slots, self.block_tables.slot_mappings)
                and torch.equal(self._logits_indices_0_gpu, ref_batch.logits_indices)
                and all(
                    torch.equal(g_bt, r_bt)
                    for g_bt, r_bt in zip(got_bts, self.block_tables.input_block_tables)
                )
            )
            if ok:
                self._single_decode_prep_verified = True
                logger.info("Single-token FULL decode fused prep verified bitwise and active")
            else:
                self._single_decode_prep_failed = True
                self._single_decode_prep_armed = False
                logger.warning("Single-token FULL decode fused prep mismatch; keeping stock prep")
            return ref_batch, ref_bts, ref_slots
        seq_ub = torch.from_numpy(np.array([num_comp + 1], dtype=np.int32))
        max_seq_len_np = (
            self.req_states.max_seq_len[idx_mapping_np] if self.use_pp else None
        )
        prompt_lens = (
            self.req_states.prompt_len.gpu[idx_mapping]
            if self.model_config.rswa_window is not None
            else None
        )
        input_batch = InputBatch(
            req_ids=[req_id],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=self._expanded_local_pos_1_gpu,
            num_scheduled_tokens=self._ones_1_np,
            num_tokens=1,
            num_tokens_after_padding=1,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=self.input_buffers.query_start_loc[:2],
            query_start_loc_np=self._cu_num_logits_1_np,
            seq_lens=self.input_buffers.seq_lens[:1],
            seq_lens_cpu_upper_bound=seq_ub,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.array([num_comp], dtype=np.int32),
            prefill_len_np=np.array([pref_len], dtype=np.int32),
            num_computed_prefill_tokens_np=np.array([num_comp_pref], dtype=np.int32),
            is_prefilling_np=self._false_1_np,
            max_seq_len_np=max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:1],
            positions=self.input_buffers.positions[:1],
            is_padding=self.input_buffers.is_padding[:1],
            logits_indices=self._logits_indices_0_gpu,
            cu_num_logits=self._cu_num_logits_1_gpu,
            cu_num_logits_np=self._cu_num_logits_1_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
        )
        return input_batch, block_tables, slot_mappings

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if (
            not dummy_run
            and not is_profile
            and self.prefill_graph is not None
            and self.prefill_graph._zp_verified
            and self._can_zero_prep_prefill(scheduler_output)
        ):
            pg = self.prefill_graph
            nr = scheduler_output.scheduled_new_reqs[0]
            self._single_decode_prep_armed = False
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            if pg.stage_zero_prep(
                self.input_buffers,
                self.block_tables,
                nr.prefill_token_ids,  # type: ignore[arg-type]
                nr.block_ids,
                PAD_SLOT_ID,
                bool(envs.VLLM_MOE_SKIP_PADDING),
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = pg.fast_replay()
                pg._zp_replays += 1
                if pg._zp_replays == 1:
                    logger.info("Prefill graph: zero-prep launch active (host prep overlapped behind graph)")
                # Overlap CPU request/sampler/block-table bookkeeping behind the running 246 ms GPU graph!
                self.add_requests(scheduler_output)
                self.update_requests(scheduler_output)
                self.block_tables.apply_staged_writes()
                input_batch = self._build_zero_prep_batch(nr.req_id, pg.num_tokens)
                self.model_state.preprocess_state(
                    input_batch,
                    tuple(bt[:1] for bt in self.block_tables.input_block_tables),
                    self.kv_cache_config,
                    self.req_states.num_computed_tokens.gpu,
                )
                self.execute_model_state = ExecuteModelState(
                    input_batch=input_batch,
                    attn_metadata=None,
                    slot_mappings_by_layer=None,
                    hidden_states=model_output,
                    aux_hidden_states=None,
                    finished_req_ids=scheduler_output.finished_req_ids,
                    compact_sample=pg.final_sample_tile,
                )
                return None

        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=is_profile or skip_compiled,
            num_active_loras=num_active_loras,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            fast_dec = self._try_prepare_single_decode(scheduler_output, batch_desc)
            if fast_dec is not None:
                input_batch, block_tables, slot_mappings = fast_dec
            else:
                self._single_decode_prep_armed = False
                input_batch = self.prepare_inputs(scheduler_output, batch_desc)
                block_tables, slot_mappings = self.prepare_attn(input_batch)
                if self._is_single_full_decode(scheduler_output, batch_desc) is not None:
                    self._single_decode_prep_armed = True
            # Mamba "align" pre-copy: migrate recurrent state across block
            # boundaries before the forward. Runs only on real batches, and
            # before model_state.prepare_attn gathers num_accepted_tokens so the
            # boundary reset is visible to the attention metadata.
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            self._single_decode_prep_armed = False
            # No actual tokens to run. A dummy run for DP or memory profiling.
            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        # Fast-path check for verified 512-token prefill graph replay: skips 52-layer Python
        # attention metadata construction and recursive _plan() tree walk once the first replay
        # verified zero-copy persistent buffer identity.
        prefill_graph = self.prefill_graph
        fast_prefill_replay = (
            not dummy_run
            and prefill_graph is not None
            and prefill_graph.embed_input_ids is not None
            and batch_desc.cg_mode == CUDAGraphMode.NONE
            and self.supports_mm_inputs
            and self.is_first_pp_rank
            and self.lora_config is None
            and not self.model.requires_raw_input_tokens
            and not scheduler_output.scheduled_encoder_inputs
            and (
                self.encoder_cache is None
                or not any(
                    self.encoder_cache.mm_features.get(rid)
                    for rid in input_batch.req_ids
                )
            )
            and not skip_compiled
            and num_tokens_across_dp is None
            and input_batch.num_reqs == 1
            and input_batch.num_draft_tokens == 0
            and input_batch.num_tokens == prefill_graph.num_tokens
            and input_batch.num_tokens_after_padding == prefill_graph.num_tokens
            and int(input_batch.num_computed_tokens_np[0]) == 0
            and (not prefill_graph.final_sample_tile or (
                self.prompt_logprobs_worker is not None
                and not self.prompt_logprobs_worker.uses_prompt_logprobs[input_batch.idx_mapping_np[0]]
                and input_batch.cu_num_logits_np[-1] == 1
                and self.speculator is None
                and self.pcp_manager is None
                and not self.use_aux_hidden_state_outputs
            ))
            and block_tables is not None
            and slot_mappings is not None
            and prefill_graph.can_fast_replay(
                input_batch, block_tables, slot_mappings, input_batch.input_ids
            )
        )

        # LOCAL PATCH (SP5D, Yukon submission 98c1361e): a FULL-graph replay reads the persistent input, block-table
        # and slot-mapping buffers filled above, not the Python metadata objects, so they are not built for it (no
        # speculator reads them in sample_tokens). FlashAttention-2 metadata building writes no persistent buffer.
        skip_eager_attn_meta = fast_prefill_replay or (
            not dummy_run
            and batch_desc.cg_mode == CUDAGraphMode.FULL
            and self.speculator is None
            and self.cudagraph_manager is not None
            and batch_desc in self.cudagraph_manager.graphs
        )
        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run) and not skip_eager_attn_meta:
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
                # FULL replay reads capture-time metadata buffers. Re-stage them
                # from the zeroed dummy block tables instead of retaining state
                # indices from the previous real batch.
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        # LOCAL PATCH (SP5D, Yukon submission 98c1361e; prefill graph only): a text-only, one-request prefill that
        # the boot prefill graph can replay embeds its token ids inside that graph (prefill_graph.py), so no eager
        # embedding runs before the replay; the eager embedding is computed only if the replay is refused.
        graph_embed = False
        prefill_graph = self.prefill_graph
        if (
            self.supports_mm_inputs
            and self.is_first_pp_rank
            and not dummy_run
            and prefill_graph is not None
            and prefill_graph.embed_input_ids is not None
            and batch_desc.cg_mode == CUDAGraphMode.NONE
            and self.lora_config is None
            and not self.model.requires_raw_input_tokens
            and not scheduler_output.scheduled_encoder_inputs
            and (
                self.encoder_cache is None
                or not any(
                    self.encoder_cache.mm_features.get(rid)
                    for rid in input_batch.req_ids
                )
            )
            and not skip_compiled
            and num_tokens_across_dp is None
            and input_batch.num_reqs == 1
            and input_batch.num_draft_tokens == 0
            and input_batch.num_tokens == prefill_graph.num_tokens
            and input_batch.num_tokens_after_padding == prefill_graph.num_tokens
            and int(input_batch.num_computed_tokens_np[0]) == 0
        ):
            graph_embed = True
        if not fast_prefill_replay:
            if self.supports_mm_inputs and self.is_first_pp_rank:
                # Run MM encoder (if needed) and get multimodal embeddings.
                # Only first PP rank prepares multimodal embeddings.
                if dummy_run or graph_embed:
                    # Obtain mm embeddings of correct shape for compiled model.
                    inputs_embeds = self.model_state.dummy_inputs_embeds(
                        input_batch.num_tokens_after_padding
                    )
                else:
                    scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                    if self.lora_config is not None:
                        set_active_mm_loras(
                            model=self.model,
                            lora_manager=self.lora_manager,
                            encoder_cache=self.encoder_cache,
                            req_id_to_index=self.req_states.req_id_to_index,
                            lora_state=self.lora_state,
                            scheduled_encoder_inputs=scheduled_encoder_inputs,
                        )
                    inputs_embeds = None
                    if self.lora_config is None:
                        inputs_embeds = self._try_decode_embed_graph(
                            scheduled_encoder_inputs, input_batch
                        )
                    if inputs_embeds is None:
                        inputs_embeds = self.model_state.get_mm_embeddings(
                            scheduled_encoder_inputs, input_batch, self.req_states
                        )
                if inputs_embeds is not None and not self.model.requires_raw_input_tokens:
                    input_ids = None

            model_inputs = {
                "input_ids": input_ids,
                "positions": input_batch.positions,
                "inputs_embeds": inputs_embeds,
                "intermediate_tensors": None,
                # NOTE: Values returned by `prepare_inputs` will override the default
                # values above.
                **self.model_state.prepare_inputs(input_batch, self.req_states),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None

                # Prepare the intermediate tensors.
                assert intermediate_tensors is not None
                assert self.intermediate_tensors is not None
                n = input_batch.num_tokens_after_padding
                new_tensors = {
                    k: v[:n]
                    if dummy_run
                    else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                    for k, v in self.intermediate_tensors.tensors.items()
                }
                model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
                del intermediate_tensors

            # Update the EPLB meta.
            self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        # Run model.
        compact_sample = False
        if fast_prefill_replay:
            assert prefill_graph is not None
            if (
                prefill_graph._zp_enabled
                and not prefill_graph._zp_verified
                and not prefill_graph._zp_failed
                and not is_profile
                and self._can_zero_prep_prefill(scheduler_output)
            ):
                try:
                    n_zp = prefill_graph.num_tokens
                    nr_zp = scheduler_output.scheduled_new_reqs[0]
                    ref_ids = self.input_buffers.input_ids[:n_zp].clone()
                    ref_pos = self.input_buffers.positions[:n_zp].clone()
                    ref_sl = self.input_buffers.seq_lens[:1].clone()
                    ref_qsl = self.input_buffers.query_start_loc[:2].clone()
                    ref_slots = self.block_tables.slot_mappings[:, :n_zp].clone()
                    ref_bts = [
                        bt[0, : len(nr_zp.block_ids[g])].clone()
                        for g, bt in enumerate(self.block_tables.input_block_tables)
                    ]
                    ok_stage = prefill_graph.stage_zero_prep(
                        self.input_buffers,
                        self.block_tables,
                        nr_zp.prefill_token_ids,  # type: ignore[arg-type]
                        nr_zp.block_ids,
                        PAD_SLOT_ID,
                        bool(envs.VLLM_MOE_SKIP_PADDING),
                    )
                    zp_batch = self._build_zero_prep_batch(nr_zp.req_id, n_zp)
                    same_zp = (
                        ok_stage
                        and torch.equal(ref_ids, self.input_buffers.input_ids[:n_zp])
                        and torch.equal(ref_pos, self.input_buffers.positions[:n_zp])
                        and torch.equal(ref_sl, self.input_buffers.seq_lens[:1])
                        and torch.equal(ref_qsl, self.input_buffers.query_start_loc[:2])
                        and torch.equal(ref_slots, self.block_tables.slot_mappings[:, :n_zp])
                        and torch.equal(input_batch.logits_indices, zp_batch.logits_indices)
                        and all(
                            torch.equal(
                                r,
                                self.block_tables.input_block_tables[g][
                                    0, : len(nr_zp.block_ids[g])
                                ],
                            )
                            for g, r in enumerate(ref_bts)
                        )
                    )
                    if same_zp:
                        prefill_graph._zp_verified = True
                        logger.info("Prefill graph: zero-prep launch verified bitwise and armed")
                    else:
                        self.input_buffers.input_ids[:n_zp].copy_(ref_ids)
                        self.input_buffers.positions[:n_zp].copy_(ref_pos)
                        self.input_buffers.seq_lens[:1].copy_(ref_sl)
                        self.input_buffers.query_start_loc[:2].copy_(ref_qsl)
                        self.block_tables.slot_mappings[:, :n_zp].copy_(ref_slots)
                        for g, r in enumerate(ref_bts):
                            self.block_tables.input_block_tables[g][
                                0, : len(nr_zp.block_ids[g])
                            ].copy_(r)
                        prefill_graph._zp_failed = True
                        logger.warning("Prefill graph: zero-prep verification mismatch; keeping standard fast_replay")
                except Exception as exc:  # noqa: BLE001
                    prefill_graph._zp_failed = True
                    logger.warning("Prefill graph: zero-prep verification raised (%s); keeping standard fast_replay", exc)
            self.kv_connector.pre_forward(scheduler_output)
            model_output = prefill_graph.fast_replay()
            compact_sample = prefill_graph.final_sample_tile
        elif batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager (NONE): call the raw model directly.
                    model_output = None
                    # LOCAL PATCH (prefill graph): a fresh, unpadded,
                    # one-request prefill of the captured length replays
                    # the boot graph; try_replay refuses any other step.
                    prefill_graph = self.prefill_graph
                    if (
                        prefill_graph is not None
                        and not dummy_run
                        and not skip_compiled
                        and num_tokens_across_dp is None
                        and input_batch.num_reqs == 1
                        and input_batch.num_draft_tokens == 0
                        and input_batch.num_tokens == prefill_graph.num_tokens
                        and input_batch.num_tokens_after_padding
                        == prefill_graph.num_tokens
                        and int(input_batch.num_computed_tokens_np[0]) == 0
                        and (not prefill_graph.final_sample_tile or (
                            self.prompt_logprobs_worker is not None
                            and not self.prompt_logprobs_worker.uses_prompt_logprobs[input_batch.idx_mapping_np[0]]
                            and input_batch.cu_num_logits_np[-1] == 1
                            and self.speculator is None
                            and self.pcp_manager is None
                            and not self.use_aux_hidden_state_outputs
                        ))
                    ):
                        model_output = prefill_graph.try_replay(
                            attn_metadata,
                            slot_mappings_by_layer,
                            model_inputs,
                            input_batch.is_padding,
                            batch_descriptor,
                            raw_input_ids=input_batch.input_ids if graph_embed else None,
                        )
                        compact_sample = model_output is not None and prefill_graph.final_sample_tile
                    if model_output is None:
                        if graph_embed:
                            # Replay refused: the eager embedding this step skipped.
                            model_inputs["inputs_embeds"] = self.model_state.get_mm_embeddings(
                                scheduler_output.scheduled_encoder_inputs,
                                input_batch,
                                self.req_states,
                            )
                        model_output = self.model(**model_inputs)

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            compact_sample=compact_sample,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        compact_sample = self.execute_model_state.compact_sample
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Last rank: sample tokens
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )

        # LOCAL PATCH (two-stage output wait, async_utils.wait_for_output): for
        # a one-request prefill step, mark in stream order where the forward
        # ends and the sampling tail begins. A decode step needs no mark: the
        # host waits for it with the next step already queued.
        forward_done = None
        if input_batch.num_reqs == 1 and input_batch.num_tokens > 1:
            forward_done = torch.cuda.Event(blocking=True)
            forward_done.record(self.main_stream)

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output, compact_sample=compact_sample
        )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        known_single_token = (
            input_batch.num_reqs == 1
            and self.num_speculative_steps == 0
            and not bool(input_batch.is_prefilling_np[0])
            and sampler_output.sampled_token_ids.shape == (1, 1)
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            check_ep_fault=self.check_ep_fault,
            known_single_token=known_single_token,
            forward_done=forward_done,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            # The EAGLE/MTP drafter reads one position ahead of the target.
            # TODO(TheEpicDolphin): Gather MM embeddings for all speculative
            # steps during multi-module MTP.
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            self.postprocess_num_computed_tokens(input_batch)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        assert self.pooling_runner is not None
        pooler_output, finished_mask = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # Build the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            finished_mask=finished_mask,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        self.postprocess_num_computed_tokens(input_batch)
        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            del self.model_state
        if getattr(self, "speculator", None) is not None:
            self.speculator = None
        if hasattr(self, "model"):
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    ########### EPLB methods start ###########
    @property
    def eplb_state(self):
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
            old_num_physical_experts,
        )

    ########### EPLB methods end ###########


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
    compact_sample: bool = False


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int], decode_query_len: int
) -> list[str]:
    # Order decode -> short_extend -> prefill; split_decodes_and_prefills
    # relies on uniform decodes (query_len == decode_query_len) leading.
    key = lambda r: ((num := num_tokens_per_req[r]) != decode_query_len, num)
    return sorted(num_tokens_per_req, key=key)
