# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DFlashSpeculator(DraftModelSpeculator):
    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        from vllm.model_executor.models.qwen3_dflash import (
            dflash_has_any_non_causal,
            dflash_query_len,
        )

        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        # Each request emits (bonus + N sampled masks + optional trailing context masks) query tokens.
        self.num_query_per_req = dflash_query_len(
            self.draft_model_config.hf_config, self.num_speculative_steps
        )
        self.extra_query_tokens = max(
            0, self.num_query_per_req - (1 + self.num_speculative_steps)
        )
        if self.extra_query_tokens > 0:
            logger.info(
                "DFlash bidirectional full-tile query window active: Q=%d (1 anchor + %d sampled + %d trailing mask)",
                self.num_query_per_req,
                self.num_speculative_steps,
                self.extra_query_tokens,
            )

        # Whether the anchor query position is itself a prediction. DFlash default uses
        # the anchor as the bonus token (only mask tokens predict); DSpark samples from
        # the anchor and the N-1 mask token positions. See _prepare_dflash_inputs_kernel
        dflash_config = (
            getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        if dflash_config.get("sample_from_anchor", False):
            raise ValueError(
                "sample_from_anchor=True is not supported for DFlash. "
                "DFlash uses a fixed 1+N query layout where the anchor "
                "is the bonus token."
            )
        self.sample_from_anchor = False

        # Context positions for the K/V precompute. Populated by
        # prepare_dflash_inputs, and processed by the model's
        # precompute_and_store_context_kv method. NOT captured by CUDA graphs.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1
        self._ctx_kv_cg_entries: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._ctx_kv_cg_rejected: set[tuple] = set()
        self._ctx_kv_cg_pool = torch.cuda.graph_pool_handle()
        self._ctx_kv_cg_stream = torch.cuda.Stream(device=device)

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The draft's attention differs from the target's in causality.
        return replace(
            self.vllm_config,
            attention_config=replace(
                self.vllm_config.attention_config,
                use_non_causal=self.requires_non_causal,
            ),
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "%s draft attention (%s) does not support full CUDA graphs; "
                "running the draft eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        # PIECEWISE cudagraphs are not supported for dflash.
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        cg_vllm_config = self.vllm_config
        comp_cfg = self.vllm_config.compilation_config
        if (
            comp_cfg is not None
            and comp_cfg.cudagraph_capture_sizes
            and self.num_query_per_req > comp_cfg.max_cudagraph_capture_size
        ):
            new_sizes = sorted(
                set(comp_cfg.cudagraph_capture_sizes) | {self.num_query_per_req}
            )
            cg_vllm_config = replace(
                self.vllm_config,
                compilation_config=replace(
                    comp_cfg,
                    cudagraph_capture_sizes=new_sizes,
                    max_cudagraph_capture_size=max(
                        comp_cfg.max_cudagraph_capture_size,
                        self.num_query_per_req,
                    ),
                ),
            )

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            cg_vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        # Reset sampling indices to zero to prevent stale values from prior
        # dummy runs from being baked into the captured graph.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]

        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to. Models that share a single group
        # leave this as None and share one context slot mapping.
        self._layer_group_idx: list[int] | None = None
        # Per-KV-group causal, falling back to whether the drafter is all-causal.
        self._group_causal: dict[int, bool] | bool = not self.requires_non_causal
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]] for name in layer_names
            ]
            if hasattr(self.model, "get_draft_attn_causal"):
                self._group_causal = {
                    name_to_gid[name]: layer_causal
                    for name, layer_causal in zip(
                        layer_names, self.model.get_draft_attn_causal()
                    )
                }
        self._single_ctx_group = (
            self._layer_group_idx is None or len(set(self._layer_group_idx)) <= 1
        )
        self._single_ctx_gidx = (
            0 if not self._layer_group_idx else self._layer_group_idx[0]
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        tail_slice_m6: bool = False,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            if (
                tail_slice_m6
                and hasattr(self.model, "model")
                and hasattr(self.model.model, "forward_tail_m6")
            ):
                last_hidden_states = self.model(
                    input_ids=self.input_buffers.input_ids[:num_tokens],
                    positions=self.input_buffers.positions[:num_tokens],
                    inputs_embeds=None,
                    tail_slice_m6=True,
                )
            else:
                last_hidden_states = self.model(
                    input_ids=self.input_buffers.input_ids[:num_tokens],
                    positions=self.input_buffers.positions[:num_tokens],
                    inputs_embeds=None,
                )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        num_sample = num_reqs * self.num_speculative_steps
        use_tail_slice = (
            num_reqs == 1
            and not self.sample_from_anchor
            and num_tokens_padded == 16
            and num_sample == 6
            and attn_metadata is not None
            and os.environ.get("MUSE_DFLASH_FAST_SAMPLE", "1") != "0"
            and os.environ.get("MUSE_DFLASH_LAST_LAYER_SLICE", "1") != "0"
        )
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            tail_slice_m6=use_tail_slice,
        )

        if use_tail_slice and last_hidden_states.shape[0] == num_sample:
            sample_hidden_states = last_hidden_states
        elif (
            num_reqs == 1
            and not self.sample_from_anchor
            and num_tokens_padded >= 1 + num_sample
            and last_hidden_states.is_contiguous()
            and os.environ.get("MUSE_DFLASH_FAST_SAMPLE", "1") != "0"
        ):
            sample_hidden_states = last_hidden_states[1 : 1 + num_sample]
        else:
            sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        if (
            num_reqs == 1
            and self.draft_logits is None
            and not self.use_local_argmax_reduction
            and hasattr(self.model, "draft_argmax")
            and os.environ.get("MUSE_DFLASH_FAST_SAMPLE", "1") != "0"
        ):
            fast_toks = self.model.draft_argmax(sample_hidden_states)
            if fast_toks is not None:
                self.draft_tokens[:1] = fast_toks.view(1, self.num_speculative_steps)
                return
        # sample_pos is the predicted token's position Q; verification keys
        # Gumbel by the predecessor (Q-1). sample_draft adds +1, so pass Q-2.
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 2,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int | None = None,
        causal: bool | Mapping[int, bool] = False,
        query_start_loc_np: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None  # Omitted for DFlash, read from self instead
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
            query_start_loc_np=query_start_loc_np,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        if num_reqs == 1:
            max_seq_len = int(input_batch.seq_lens_cpu_upper_bound.numpy()[0])
        else:
            max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        def _build_ctx_hidden() -> torch.Tensor:
            if aux_hidden_states:
                if (
                    len(aux_hidden_states) == 5
                    and hasattr(self.model, "combine_hidden_states_aux5")
                ):
                    out = self.model.combine_hidden_states_aux5(
                        aux_hidden_states, num_target_tokens
                    )
                    if out is not None:
                        return out if out.is_contiguous() else out.contiguous()
                if aux_hidden_states[0].shape[0] > num_target_tokens:
                    aux_cat = torch.cat(
                        [h[:num_target_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    aux_cat = torch.cat(aux_hidden_states, dim=-1)
                out = self.model.combine_hidden_states(aux_cat)[:num_target_tokens]
            else:
                out = last_hidden_states[:num_target_tokens]
            if not out.is_contiguous():
                out = out.contiguous()
            return out

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            context_hidden_states = _build_ctx_hidden()
            self.model.precompute_and_store_context_kv(
                context_hidden_states,
                self.context_positions[:num_target_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        ran_cpp_prep = False
        ran_fused_tail = False
        if (
            num_reqs == 1
            and not self.sample_from_anchor
            and len(self.draft_kv_cache_group_ids) == 1
            and os.environ.get("MUSE_DFLASH_CPP_PREP", "1") != "0"
        ):
            try:
                import muse_bfx_ext as _bfx

                if hasattr(_bfx, "dflash_prepare_inputs"):
                    from vllm.v1.worker.gpu.spec_decode.dflash import native_leaf as _nl

                    gid0 = self.draft_kv_cache_group_ids[0]
                    bt0 = self.block_tables.input_block_tables[gid0]
                    lm_head = getattr(self.model, "lm_head", None)
                    want_ids_base = (
                        not dummy_run
                        and not is_profile
                        and num_target_tokens > 0
                        and input_batch.input_ids.is_cuda
                        and input_batch.input_ids.is_contiguous()
                    )
                    want_tail = (
                        want_ids_base
                        and lm_head is not None
                        and getattr(lm_head, "prompt_tail_admitted", False)
                    )
                    hist_req_id = input_batch.req_ids[0] if input_batch.req_ids else None
                    hist_buf, hist_len_buf = (
                        _nl.get_hist_bufs(self.device, hist_req_id)
                        if (want_ids_base and _nl.ENABLED)
                        else (None, None)
                    )
                    want_ids = want_tail or (hist_buf is not None)
                    try:
                        _bfx.dflash_prepare_inputs(
                            self.input_buffers.input_ids,
                            self.input_buffers.positions,
                            self.input_buffers.query_start_loc,
                            self.input_buffers.seq_lens,
                            self.block_tables.slot_mappings[gid0],
                            self.context_positions,
                            self._context_slot_mappings[0],
                            self.sample_indices,
                            self.sample_pos,
                            self.sample_idx_mapping,
                            self.temperature,
                            self.seeds,
                            input_batch.positions,
                            input_batch.query_start_loc,
                            input_batch.idx_mapping,
                            last_sampled,
                            next_prefill_tokens,
                            num_sampled,
                            num_rejected,
                            temperature,
                            seeds,
                            bt0,
                            bt0.stride(0),
                            self.parallel_drafting_token_id,
                            self.block_tables.kernel_block_sizes[gid0],
                            self.num_query_per_req,
                            self.num_speculative_steps,
                            self.max_num_reqs,
                            self.max_num_tokens,
                            self.max_model_len,
                            PAD_SLOT_ID,
                            input_batch.input_ids if want_ids else None,
                            num_target_tokens if want_ids else 0,
                            lm_head.in_static_vocab if want_tail else None,
                            lm_head.tail_ids if want_tail else None,
                            lm_head.tail_count if want_tail else None,
                            lm_head.tail_seen if want_tail else None,
                            hist_buf,
                            hist_len_buf,
                        )
                    except TypeError:
                        _bfx.dflash_prepare_inputs(
                            self.input_buffers.input_ids,
                            self.input_buffers.positions,
                            self.input_buffers.query_start_loc,
                            self.input_buffers.seq_lens,
                            self.block_tables.slot_mappings[gid0],
                            self.context_positions,
                            self._context_slot_mappings[0],
                            self.sample_indices,
                            self.sample_pos,
                            self.sample_idx_mapping,
                            self.temperature,
                            self.seeds,
                            input_batch.positions,
                            input_batch.query_start_loc,
                            input_batch.idx_mapping,
                            last_sampled,
                            next_prefill_tokens,
                            num_sampled,
                            num_rejected,
                            temperature,
                            seeds,
                            bt0,
                            bt0.stride(0),
                            self.parallel_drafting_token_id,
                            self.block_tables.kernel_block_sizes[gid0],
                            self.num_query_per_req,
                            self.num_speculative_steps,
                            self.max_num_reqs,
                            self.max_num_tokens,
                            self.max_model_len,
                            PAD_SLOT_ID,
                            input_batch.input_ids if want_tail else None,
                            num_target_tokens if want_tail else 0,
                            lm_head.in_static_vocab if want_tail else None,
                            lm_head.tail_ids if want_tail else None,
                            lm_head.tail_count if want_tail else None,
                            lm_head.tail_seen if want_tail else None,
                        )
                    ran_cpp_prep = True
                    ran_fused_tail = want_tail
            except Exception:  # noqa: BLE001
                ran_cpp_prep = False
                ran_fused_tail = False

        if not ran_cpp_prep:
            # Support multiple draft KV cache groups by preparing inputs once for each
            for i, gid in enumerate(self.draft_kv_cache_group_ids):
                prepare_dflash_inputs(
                    self.input_buffers,
                    self.block_tables.slot_mappings[gid],
                    self.context_positions,
                    self._context_slot_mappings[i],
                    self.sample_indices,
                    self.sample_pos,
                    self.sample_idx_mapping,
                    self.temperature,
                    self.seeds,
                    input_batch,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    self.block_tables.input_block_tables[gid],
                    self.block_tables.kernel_block_sizes[gid],
                    self.parallel_drafting_token_id,
                    self.num_query_per_req,
                    self.num_speculative_steps,
                    self.max_num_reqs,
                    self.max_num_tokens,
                    self.max_model_len,
                    self.sample_from_anchor,
                )

        # Pre-insert context K/V into the cache. Runs eagerly outside the captured graph
        # because the context shape varies per step. During dummy runs the block tables
        # are placeholders, so we skip the cache write to avoid clobbering real entries.
        # Each layer uses the context slots of its own kv-cache group.
        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif getattr(self, "_single_ctx_group", False):
            context_slots = self._context_slot_mappings[self._single_ctx_gidx][
                :num_target_tokens
            ]
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][:num_target_tokens]
        ctx_pos_slice = self.context_positions[:num_target_tokens]
        if (
            not ran_fused_tail
            and not dummy_run
            and not is_profile
            and num_reqs == 1
            and hasattr(self.model, "update_draft_prompt_tail")
        ):
            self.model.update_draft_prompt_tail(
                input_batch.input_ids[:num_target_tokens],
                ctx_pos_slice,
                num_rejected=num_rejected,
                extra_ids=self.input_buffers.input_ids[:1],
            )

        ran_ctx_graph = False
        if (
            os.environ.get("VLLM_DFLASH_CTX_KV_GRAPH", "1") != "0"
            and not dummy_run
            and not is_profile
            and num_reqs == 1
            and 1 <= num_target_tokens <= self.num_query_per_req
            and not torch.cuda.is_current_stream_capturing()
            and self.query_cudagraph_manager is not None
            and self.query_cudagraph_manager._graphs_captured
        ):
            if (
                aux_hidden_states
                and len(aux_hidden_states) == 5
                and isinstance(context_slots, torch.Tensor)
            ):
                cg_key: tuple = (
                    num_target_tokens,
                    aux_hidden_states[0].data_ptr(),
                    aux_hidden_states[0].stride(0),
                    aux_hidden_states[1].data_ptr(),
                    aux_hidden_states[2].data_ptr(),
                    aux_hidden_states[3].data_ptr(),
                    aux_hidden_states[4].data_ptr(),
                    ctx_pos_slice.data_ptr(),
                    context_slots.data_ptr(),
                )
            else:
                if aux_hidden_states:
                    src_sig = tuple(
                        (h.data_ptr(), h.dtype, tuple(h.shape), tuple(h.stride()))
                        for h in aux_hidden_states
                    )
                else:
                    src_sig = (
                        (
                            last_hidden_states.data_ptr(),
                            last_hidden_states.dtype,
                            tuple(last_hidden_states.shape),
                            tuple(last_hidden_states.stride()),
                        ),
                    )
                pos_sig = (
                    ctx_pos_slice.data_ptr(),
                    ctx_pos_slice.dtype,
                    tuple(ctx_pos_slice.shape),
                    tuple(ctx_pos_slice.stride()),
                )
                slot_sig = (
                    tuple(
                        (s.data_ptr(), s.dtype, tuple(s.shape), tuple(s.stride()))
                        if s is not None
                        else None
                        for s in context_slots
                    )
                    if isinstance(context_slots, list)
                    else (
                        (
                            context_slots.data_ptr(),
                            context_slots.dtype,
                            tuple(context_slots.shape),
                            tuple(context_slots.stride()),
                        )
                        if context_slots is not None
                        else None,
                    )
                )
                cg_key = (num_target_tokens, src_sig, pos_sig, slot_sig)
            cg = self._ctx_kv_cg_entries.get(cg_key)
            if cg is not None:
                cg.replay()
                ran_ctx_graph = True
            elif (
                cg_key not in self._ctx_kv_cg_rejected
                and len(self._ctx_kv_cg_entries) < 64
            ):
                context_hidden_states = _build_ctx_hidden()
                self.model.precompute_and_store_context_kv(
                    context_hidden_states,
                    ctx_pos_slice,
                    context_slots,
                )
                ran_ctx_graph = True
                try:
                    main_st = torch.cuda.current_stream(self.device)
                    cap_st = self._ctx_kv_cg_stream
                    cap_st.wait_stream(main_st)
                    try:
                        g = torch.cuda.CUDAGraph(keep_graph=True)
                    except TypeError:
                        g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(cap_st):
                        with torch.cuda.graph(
                            g, pool=self._ctx_kv_cg_pool, stream=cap_st
                        ):
                            ctx_h = _build_ctx_hidden()
                            self.model.precompute_and_store_context_kv(
                                ctx_h,
                                ctx_pos_slice,
                                context_slots,
                            )
                    main_st.wait_stream(cap_st)
                    if os.environ.get("VLLM_CUDAGRAPH_UPLOAD", "1") != "0" and hasattr(
                        g, "raw_cuda_graph_exec"
                    ):
                        try:
                            import muse_bfx_ext as _bfx_up

                            _bfx_up.cuda_graph_upload(int(g.raw_cuda_graph_exec()))
                        except Exception:
                            pass
                    self._ctx_kv_cg_entries[cg_key] = g
                except Exception as exc:
                    self._ctx_kv_cg_rejected.add(cg_key)
                    logger.warning(
                        "DFlash context-KV CUDA graph capture skipped (%s); keeping eager path",
                        exc,
                    )

        if not ran_ctx_graph:
            context_hidden_states = _build_ctx_hidden()
            self.model.precompute_and_store_context_kv(
                context_hidden_states,
                ctx_pos_slice,
                context_slots,
            )

        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            draft_attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
            draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, :num_tokens_padded],
                self.kv_cache_config,
            )
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]

    def emit_cpp_prep_for_capture(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> bool:
        """Emit dflash_prepare_inputs onto current capture stream for bs=1 verify."""
        if (
            input_batch.num_reqs != 1
            or self.sample_from_anchor
            or len(self.draft_kv_cache_group_ids) != 1
            or os.environ.get("MUSE_DFLASH_CPP_PREP", "1") == "0"
        ):
            return False
        import muse_bfx_ext as _bfx
        from vllm.v1.worker.gpu.spec_decode.dflash import native_leaf as _nl

        gid0 = self.draft_kv_cache_group_ids[0]
        bt0 = self.block_tables.input_block_tables[gid0]
        lm_head = getattr(self.model, "lm_head", None)
        num_target_tokens = int(input_batch.num_tokens)
        want_ids_base = (
            num_target_tokens > 0
            and input_batch.input_ids.is_cuda
            and input_batch.input_ids.is_contiguous()
        )
        want_tail = (
            want_ids_base
            and lm_head is not None
            and getattr(lm_head, "prompt_tail_admitted", False)
        )
        hist_buf = _nl._S.hist if (want_ids_base and _nl.ENABLED and _nl._S.ready) else None
        hist_len_buf = _nl._S.hist_len if hist_buf is not None else None
        want_ids = want_tail or (hist_buf is not None)
        _bfx.dflash_prepare_inputs(
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            self.block_tables.slot_mappings[gid0],
            self.context_positions,
            self._context_slot_mappings[0],
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            input_batch.positions,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            temperature,
            seeds,
            bt0,
            bt0.stride(0),
            self.parallel_drafting_token_id,
            self.block_tables.kernel_block_sizes[gid0],
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            PAD_SLOT_ID,
            input_batch.input_ids if want_ids else None,
            num_target_tokens if want_ids else 0,
            lm_head.in_static_vocab if want_tail else None,
            lm_head.tail_ids if want_tail else None,
            lm_head.tail_count if want_tail else None,
            lm_head.tail_seen if want_tail else None,
            hist_buf,
            hist_len_buf,
        )
        return True

    def get_decode_compound_child_graphs(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        num_target_tokens: int = 7,
    ) -> tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph] | None:
        """Return (ctx_kv_cg, query_full_cg) for bs=1 steady decode if already captured."""
        if (
            self.query_cudagraph_manager is None
            or not self.query_cudagraph_manager._graphs_captured
            or not aux_hidden_states
            or len(aux_hidden_states) != 5
            or not getattr(self, "_single_ctx_group", False)
        ):
            return None
        context_slots = self._context_slot_mappings[self._single_ctx_gidx][:num_target_tokens]
        ctx_pos_slice = self.context_positions[:num_target_tokens]
        cg_key: tuple = (
            num_target_tokens,
            aux_hidden_states[0].data_ptr(),
            aux_hidden_states[0].stride(0),
            aux_hidden_states[1].data_ptr(),
            aux_hidden_states[2].data_ptr(),
            aux_hidden_states[3].data_ptr(),
            aux_hidden_states[4].data_ptr(),
            ctx_pos_slice.data_ptr(),
            context_slots.data_ptr(),
        )
        ctx_cg = self._ctx_kv_cg_entries.get(cg_key)
        if ctx_cg is None:
            return None
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            1,
            self.num_query_per_req,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=False,
        )
        if batch_desc.cg_mode != CUDAGraphMode.FULL or batch_desc not in self.query_cudagraph_manager.graphs:
            return None
        return ctx_cg, self.query_cudagraph_manager.graphs[batch_desc]

    def get_decode_compound_diamond_graphs(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        num_target_tokens: int = 7,
    ) -> tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph, torch.cuda.CUDAGraph] | None:
        """Return (ctx_proj_cg, ctx_kv_post_cg, query_full_cg) so fc+fkv run in parallel with post_prep."""
        if (
            os.environ.get("MUSE_COMPOUND_DIAMOND_FC", "0") in ("0", "", "false", "no")
            or self.query_cudagraph_manager is None
            or not self.query_cudagraph_manager._graphs_captured
            or not aux_hidden_states
            or len(aux_hidden_states) != 5
            or not getattr(self, "_single_ctx_group", False)
            or not hasattr(self.model, "combine_hidden_states_aux5")
        ):
            return None
        dmodel = getattr(self.model, "model", None)
        cache_entry = getattr(dmodel, "_dflash_kv_post_cache", None) if dmodel is not None else None
        if cache_entry is None or cache_entry[1] is None:
            return None
        _, kv_post_fn, k_caches, v_caches, block_size = cache_entry
        context_slots = self._context_slot_mappings[self._single_ctx_gidx][:num_target_tokens]
        ctx_pos_slice = self.context_positions[:num_target_tokens]
        if not context_slots.is_contiguous() or not ctx_pos_slice.is_contiguous():
            return None
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            1,
            self.num_query_per_req,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=False,
        )
        if batch_desc.cg_mode != CUDAGraphMode.FULL or batch_desc not in self.query_cudagraph_manager.graphs:
            return None
        query_cg = self.query_cudagraph_manager.graphs[batch_desc]

        d_key = (
            num_target_tokens,
            aux_hidden_states[0].data_ptr(),
            aux_hidden_states[0].stride(0),
            aux_hidden_states[1].data_ptr(),
            aux_hidden_states[2].data_ptr(),
            aux_hidden_states[3].data_ptr(),
            aux_hidden_states[4].data_ptr(),
            ctx_pos_slice.data_ptr(),
            context_slots.data_ptr(),
        )
        d_cache = getattr(self, "_compound_diamond_cg_cache", None)
        if d_cache is None:
            d_cache = {}
            self._compound_diamond_cg_cache = d_cache
        hit = d_cache.get(d_key)
        if hit is not None:
            return hit[0], hit[1], query_cg
        try:
            cos_sin_cache = dmodel._rope_cos_sin_cache
            if cos_sin_cache.dtype != torch.bfloat16:
                cast_cache = getattr(dmodel, "_rope_cos_sin_cache_cast", None)
                if cast_cache is None or cast_cache.dtype != torch.bfloat16 or cast_cache.device != self.device:
                    cast_cache = cos_sin_cache.to(device=self.device, dtype=torch.bfloat16)
                    dmodel._rope_cos_sin_cache_cast = cast_cache
                cos_sin_cache = cast_cache
            slots_list = [context_slots] * int(dmodel._num_attn_layers)

            proj_pool = torch.cuda.graph_pool_handle()
            post_pool = torch.cuda.graph_pool_handle()
            main_st = torch.cuda.current_stream(self.device)
            cap_st = self._ctx_kv_cg_stream
            cap_st.wait_stream(main_st)
            ctx_proj_cg = torch.cuda.CUDAGraph(keep_graph=True)
            ctx_kv_post_cg = torch.cuda.CUDAGraph(keep_graph=True)
            all_kv_ref = None
            with torch.cuda.stream(cap_st):
                with torch.cuda.graph(ctx_proj_cg, pool=proj_pool, stream=cap_st):
                    ctx_h = self.model.combine_hidden_states_aux5(aux_hidden_states, num_target_tokens)
                    if ctx_h is None or not ctx_h.is_contiguous():
                        raise RuntimeError("combine_hidden_states_aux5 non-contiguous in diamond capture")
                    all_kv_flat = dmodel._project_context_kv_flat(ctx_h)
                    if not all_kv_flat.is_contiguous():
                        all_kv_flat = all_kv_flat.contiguous()
                    all_kv_ref = all_kv_flat
                with torch.cuda.graph(ctx_kv_post_cg, pool=post_pool, stream=cap_st):
                    kv_post_fn(
                        all_kv_ref,
                        ctx_pos_slice,
                        dmodel._k_norm_weights,
                        cos_sin_cache,
                        k_caches,
                        v_caches,
                        slots_list,
                        int(dmodel._num_kv_heads),
                        int(block_size),
                        float(dmodel._rms_norm_eps),
                        bool(dmodel._rope_is_neox),
                    )
            main_st.wait_stream(cap_st)
            d_cache[d_key] = (ctx_proj_cg, ctx_kv_post_cg, all_kv_ref, proj_pool, post_pool)
            return ctx_proj_cg, ctx_kv_post_cg, query_cg
        except Exception as exc:
            logger.warning("Compound diamond fc+fkv split skipped (%s); using monolithic ctx_kv graph", exc)
            return None

    def warmup_ctx_kv_graphs(self, aux_base: list[torch.Tensor] | None) -> int:
        """Pre-capture and cudaGraphUpload _ctx_kv_cg_entries for num_target_tokens in 1..7 at boot."""
        if (
            self._ctx_kv_cg_stream is None
            or not aux_base
            or len(aux_base) != 5
            or not getattr(self, "_single_ctx_group", False)
        ):
            return 0
        captured = 0
        for n_tok in range(1, 8):
            aux_n = [h[:n_tok] for h in aux_base]
            context_slots = self._context_slot_mappings[self._single_ctx_gidx][:n_tok]
            ctx_pos_slice = self.context_positions[:n_tok]
            cg_key = (
                n_tok,
                aux_n[0].data_ptr(),
                aux_n[0].stride(0),
                aux_n[1].data_ptr(),
                aux_n[2].data_ptr(),
                aux_n[3].data_ptr(),
                aux_n[4].data_ptr(),
                ctx_pos_slice.data_ptr(),
                context_slots.data_ptr(),
            )
            if cg_key in self._ctx_kv_cg_entries or cg_key in self._ctx_kv_cg_rejected:
                continue
            try:
                main_st = torch.cuda.current_stream(self.device)
                cap_st = self._ctx_kv_cg_stream
                cap_st.wait_stream(main_st)
                try:
                    g = torch.cuda.CUDAGraph(keep_graph=True)
                except TypeError:
                    g = torch.cuda.CUDAGraph()
                with torch.cuda.stream(cap_st):
                    with torch.cuda.graph(g, pool=self._ctx_kv_cg_pool, stream=cap_st):
                        ctx_h = self.model.combine_hidden_states_aux5(aux_n, n_tok)
                        if ctx_h is None:
                            ctx_h = self.model.combine_hidden_states(torch.cat(aux_n, dim=-1))[:n_tok]
                        if not ctx_h.is_contiguous():
                            ctx_h = ctx_h.contiguous()
                        self.model.precompute_and_store_context_kv(ctx_h, ctx_pos_slice, context_slots)
                main_st.wait_stream(cap_st)
                if os.environ.get("VLLM_CUDAGRAPH_UPLOAD", "1") != "0" and hasattr(g, "raw_cuda_graph_exec"):
                    try:
                        import muse_bfx_ext as _bfx_up

                        _bfx_up.cuda_graph_upload(int(g.raw_cuda_graph_exec()))
                    except Exception:
                        pass
                self._ctx_kv_cg_entries[cg_key] = g
                captured += 1
            except Exception:
                self._ctx_kv_cg_rejected.add(cg_key)
        return captured


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)
    q_slot = tl.where(query_pos < max_model_len, q_slot, PAD_SLOT_ID)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the first
    # num_speculative_steps mask tokens (offsets 1..num_speculative_steps) are
    # sampled; any trailing masks beyond that act purely as bidirectional context.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = (
        is_query
        & (query_off >= sample_off)
        & (query_off < sample_off + num_speculative_steps)
    )
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        # Copy sampling state.
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
