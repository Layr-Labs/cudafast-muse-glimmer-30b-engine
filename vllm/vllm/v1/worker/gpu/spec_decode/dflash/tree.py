# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification of six DFlash candidates packed into seven target rows."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = init_logger(__name__)

CANDIDATES = 6
ROWS = CANDIDATES + 1
# Reserve at least one of the six candidates for a second continuation.
MAX_PATH = CANDIDATES - 1
TOP_K = 8
PROPOSAL_TEMPERATURE = 0.6


@triton.jit
def _stage_tree(
    top_ids_ptr,
    top_scores_ptr,
    tokens_ptr,
    positions_ptr,
    parents_ptr,
    remap_ptr,
    rows: tl.constexpr,
    max_path: tl.constexpr,
    top_k: tl.constexpr,
):
    edge_count: tl.constexpr = 8 * top_k
    edges = tl.arange(0, edge_count)
    nodes = tl.arange(0, 8)
    edge_parent = edges // top_k
    edge_rank = edges % top_k
    root_position = tl.load(positions_ptr)
    node_score = tl.where(nodes == 0, 0.0, float("-inf"))
    node_depth = tl.full((8,), 0, tl.int32)
    node_path = tl.full((8,), 0, tl.int32)
    used_edge = tl.full((edge_count,), False, tl.int1)

    tl.store(remap_ptr, -1)
    tl.store(remap_ptr + 1, rows - 1)
    tl.store(remap_ptr + 2, -1)
    tl.store(remap_ptr + 3, 0)
    tl.store(parents_ptr, -1)

    # Best-first expansion under the factorized draft distribution. At each
    # iteration the frontier contains only children of already selected nodes.
    for node in range(1, rows):
        parent_depth = tl.gather(node_depth, edge_parent, 0)
        parent_score = tl.gather(node_score, edge_parent, 0)
        slot = parent_depth * top_k + edge_rank
        score = tl.load(top_scores_ptr + slot, parent_depth < max_path, -1e20)
        score = tl.where(score > -1e20, score, -1e20)
        eligible = (edge_parent < node) & (parent_depth < max_path) & ~used_edge
        joint_score = tl.where(eligible, parent_score + score, float("-inf"))
        best_edge = tl.argmax(joint_score, 0, tie_break_left=True)
        parent = best_edge // top_k
        depth = tl.sum(tl.where(nodes == parent, node_depth, 0), 0) + 1
        path = tl.sum(tl.where(nodes == parent, node_path, 0), 0) | (node << (4 * depth))
        token = tl.load(top_ids_ptr + (depth - 1) * top_k + best_edge % top_k)
        tl.store(tokens_ptr + node, token)
        tl.store(positions_ptr + node, root_position + depth)
        tl.store(parents_ptr + node, parent)
        tl.store(remap_ptr + 1 + 3 * node, rows - 1 - depth)
        tl.store(remap_ptr + 2 + 3 * node, rows)
        tl.store(remap_ptr + 3 + 3 * node, path)
        node_score = tl.where(nodes == node, tl.max(joint_score, 0), node_score)
        node_depth = tl.where(nodes == node, depth, node_depth)
        node_path = tl.where(nodes == node, path, node_path)
        used_edge |= edges == best_edge


@triton.jit
def _commit_tree(
    greedy_ptr,
    tokens_ptr,
    positions_ptr,
    parents_ptr,
    remap_ptr,
    output_ptr,
    path_ptr,
    sampled_ptr,
    rejected_ptr,
    rows: tl.constexpr,
    max_path: tl.constexpr,
):
    nodes = tl.arange(0, 8)
    parents = tl.load(parents_ptr + nodes, nodes < rows, -1)
    tokens = tl.load(tokens_ptr + nodes, nodes < rows, -1)
    greedy = tl.load(greedy_ptr + nodes, nodes < rows, -1)
    root_position = tl.load(positions_ptr)
    tl.store(positions_ptr + nodes, root_position + nodes, nodes < rows)
    tl.store(remap_ptr, 0)

    node = tl.full((), 0, tl.int32)
    num_sampled = tl.full((), 0, tl.int32)
    for depth in range(rows):
        if node < rows and depth <= max_path:
            token = tl.sum(tl.where(nodes == node, greedy, 0), 0)
            tl.store(output_ptr + depth, token)
            tl.store(path_ptr + depth, node)
            num_sampled += 1
            match = (parents == node) & (tokens == token) & (nodes < rows)
            node = tl.min(tl.where(match, nodes, rows), 0)
        else:
            tl.store(output_ptr + depth, -1)
            tl.store(path_ptr + depth, depth)
    tl.store(sampled_ptr, num_sampled)
    tl.store(rejected_ptr, rows - num_sampled)


@triton.jit
def _compact_kv(
    layouts_ptr, slots_ptr, path_ptr, sampled_ptr, slot_group_stride: tl.constexpr
):
    layout = layouts_ptr + tl.program_id(0) * 6
    cache = tl.load(layout).to(tl.pointer_type(tl.uint16))
    block_stride = tl.load(layout + 1)
    head_stride = tl.load(layout + 2)
    slot_stride = tl.load(layout + 3)
    block_size = tl.load(layout + 4)
    slots = slots_ptr + tl.load(layout + 5) * slot_group_stride
    values = tl.arange(0, 512)
    head_offset = (values // 256) * head_stride + values % 256
    num_sampled = tl.load(sampled_ptr)
    # A child's physical row follows its parent's. Moving the path from
    # shallow to deep ensures an in-place write cannot destroy a later source.
    for dst in range(1, num_sampled):
        src = tl.load(path_ptr + dst)
        if src != dst:
            src_slot = tl.load(slots + src)
            dst_slot = tl.load(slots + dst)
            src_offset = src_slot // block_size * block_stride + src_slot % block_size * slot_stride
            dst_offset = dst_slot // block_size * block_stride + dst_slot % block_size * slot_stride
            data = tl.load(cache + src_offset + head_offset)
            tl.store(cache + dst_offset + head_offset, data)


class DraftTreeVerifier:
    def __init__(
        self,
        layouts: list[tuple[int, ...]],
        remap: torch.Tensor,
    ):
        self.device = remap.device
        self.remap = remap
        self.layouts = torch.tensor(layouts, dtype=torch.int64, device=self.device)
        self.parents = torch.empty(ROWS, dtype=torch.int32, device=self.device)
        self.path = torch.empty(ROWS, dtype=torch.int64, device=self.device)
        self.request_id: str | None = None

    @classmethod
    def create(cls, caches, cache_groups, draft_layer_names, layers, slots):
        try:
            import muse_gqa16_attn
        except (ImportError, OSError):
            return None

        layouts = []
        for group_id, group in enumerate(cache_groups):
            for name in group.layer_names:
                if name in draft_layer_names:
                    continue
                cache = caches.get(name)
                layer = layers.get(name)
                if (
                    cache is None
                    or not getattr(getattr(layer, "impl", None), "_gqa16_decode_eligible", False)
                    or cache.dtype != torch.bfloat16
                    or cache.ndim != 4
                    or cache.shape[1] != 2
                    or cache.shape[3] != 256
                    or cache.stride(3) != 1
                    or cache.shape[2] % 16
                ):
                    return None
                layouts.append((
                    cache.data_ptr(), cache.stride(0), cache.stride(1),
                    cache.stride(2), cache.shape[2], group_id,
                ))
        if not layouts:
            return None
        remap = torch.zeros(1 + 3 * ROWS, dtype=torch.int32, device=slots.device)
        verifier = cls(layouts, remap)
        try:
            verifier._warmup(slots)
        except Exception:  # noqa: BLE001
            logger.exception("DFlash tree verifier failed to initialize")
            return None
        if not muse_gqa16_attn.bind_tree_remap(remap):
            return None
        logger.info("DFlash tree verifier armed for %d target KV layers", len(layouts))
        return verifier

    def _warmup(self, slots: torch.Tensor) -> None:
        ids = torch.ones((MAX_PATH, TOP_K), dtype=torch.int64, device=self.device)
        scores = torch.full(ids.shape, -2.0, dtype=torch.float32, device=self.device)
        tokens = torch.zeros(ROWS, dtype=torch.int32, device=self.device)
        positions = torch.arange(ROWS, dtype=torch.int64, device=self.device)
        self.stage(scores, ids, tokens, positions)
        # The synthetic target rejects every child, so the compaction kernel
        # compiles and runs without touching the resident's real KV cache.
        greedy = torch.zeros(ROWS, dtype=torch.int64, device=self.device)
        output = self.commit(greedy, tokens, positions)
        _compact_kv[(self.layouts.shape[0],)](
            self.layouts, slots, self.path, output.num_sampled, slots.stride(0),
        )
        assert output.num_sampled.item() == 1

    def stage(
        self, scores: torch.Tensor, ids: torch.Tensor,
        input_ids: torch.Tensor, positions: torch.Tensor,
    ) -> None:
        _stage_tree[(1,)](
            ids, scores, input_ids, positions, self.parents, self.remap,
            ROWS, MAX_PATH, TOP_K, num_warps=1,
        )

    def commit(
        self, greedy: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor,
    ) -> SamplerOutput:
        output = torch.empty((1, ROWS), dtype=torch.int64, device=self.device)
        sampled = torch.empty(1, dtype=torch.int32, device=self.device)
        rejected = torch.empty_like(sampled)
        _commit_tree[(1,)](
            greedy, input_ids, positions, self.parents, self.remap, output,
            self.path, sampled, rejected, ROWS, MAX_PATH, num_warps=1,
        )
        return SamplerOutput(
            sampled_token_ids=output,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=sampled,
            num_rejected=rejected,
        )

    def compact(
        self, hidden: torch.Tensor, aux_hidden: list[torch.Tensor],
        slots: torch.Tensor, sampled: torch.Tensor,
    ) -> None:
        _compact_kv[(self.layouts.shape[0],)](
            self.layouts, slots, self.path, sampled, slots.stride(0),
        )
        for states in (hidden, *aux_hidden):
            states[:ROWS].copy_(states.index_select(0, self.path))
