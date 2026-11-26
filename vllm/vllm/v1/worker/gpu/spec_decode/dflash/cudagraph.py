# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        progress_bar_desc: str = "Capturing CUDA graphs",
        cascade_clone_target_tokens: int = 0,
    ) -> None:
        self._cascade_cloned_id: int = 0

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state
            want_cascade_clone = (
                not warmup
                and cascade_clone_target_tokens > 0
                and desc.cg_mode == CUDAGraphMode.FULL
                and num_reqs == 1
                and num_tokens == cascade_clone_target_tokens
                and self._cascade_cloned_id == 0
            )

            def _run(cg_mode: CUDAGraphMode) -> None:
                forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )
                if want_cascade_clone and torch.cuda.is_current_stream_capturing():
                    try:
                        import muse_bfx_ext as _bfx

                        if hasattr(_bfx, "dflash_cascade_clone_capturing_graph"):
                            self._cascade_cloned_id = int(
                                _bfx.dflash_cascade_clone_capturing_graph()
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "DFlash cascade graph clone skipped (%s); keeping 1-pass draft",
                            exc,
                        )

            return _run

        super().capture(create_forward_fn, progress_bar_desc)
