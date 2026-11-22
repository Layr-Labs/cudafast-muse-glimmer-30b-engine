"""Generation backends the resident drives.

A backend owns ONE live greedy request at a time and exposes it as a stream of
engine steps. `start(tokens)` prefills `tokens` and returns the first step's
emitted tokens; `step()` runs one more engine step (one verify cycle when
speculation is on) and returns what it emitted; `abort()` drops the request.
`start(tokens, max_tokens)` bounds the request: the engine emits at most
`max_tokens` tokens and then stops on its own, so nothing decodes on after the
caller has what it asked for.

Every emitted token carries the top-k RAW LOGITS of the position that produced
it (highest first), so the resident can answer `top_logits` for whatever
frontier it holds without another forward pass.

`VllmBackend` is the scored engine. `FakeBackend` is a deterministic stand-in
with the same step semantics (including multi-token speculative steps), so the
resident's wire and bookkeeping are tested with no GPU and no vLLM install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# DETERMINISM (measured on GB10, docs/muse-glimmer-port-notes.md "Findings").
# With vLLM's default kernels the same prompt gives different logits run to
# run: the Marlin NVFP4 MoE is nondeterministic even on a fixed input, and this
# checkpoint's FP8-quantized activations amplify a last-bit difference into a
# changed greedy token within a few steps. The serial control leg must be
# reproducible, so the reference engine runs the humming W4A16 MoE with its
# batch-invariant kernels, and the Mamba FP8 projections on CUTLASS rather than
# FlashInfer's autotuned FP8 GEMM. Set before vLLM is imported; the engine-core
# process inherits them.
os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from typing import Protocol

TOP_K = 8


@dataclass(frozen=True)
class Emit:
    """One emitted token and the top-k (token, raw logit) of its position."""

    token: int
    top: tuple[tuple[int, float], ...]


class Backend(Protocol):
    vocab_size: int
    eos_token: int
    depth: int  # draft tokens per cycle; 0 = serial
    ctx_size: int
    ident: str
    model_path: str

    def start(
        self,
        tokens: list[int],
        max_tokens: int | None = None,
        need_logprobs: bool = True,
    ) -> list[Emit]: ...

    def step(self) -> list[Emit]: ...

    def abort(self) -> bool:
        """Drop the request. True when it was still generating."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# vLLM (the scored engine)
# --------------------------------------------------------------------------- #


@dataclass
class VllmConfig:
    """Everything the fixture pins for the engine. `tools/serve-up.sh` exports
    each value; nothing here has a silent default the fixture does not name."""

    model_dir: str
    depth: int
    ctx_size: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    kv_cache_memory_bytes: int
    ident: str
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "VllmConfig":
        def need(name: str) -> str:
            value = os.environ.get(name, "")
            if not value:
                raise RuntimeError(f"{name} is unset; tools/serve-up.sh exports it")
            return value

        depth = int(need("RESIDENT_SPEC_DEPTH"))
        if not 0 <= depth <= 6:
            raise RuntimeError(f"RESIDENT_SPEC_DEPTH={depth} is outside 0..6")
        return cls(
            model_dir=need("RESIDENT_MODEL_DIR"),
            depth=depth,
            ctx_size=int(need("RESIDENT_CTX_SIZE")),
            gpu_memory_utilization=float(need("RESIDENT_GPU_MEMORY_UTILIZATION")),
            max_num_batched_tokens=int(need("RESIDENT_MAX_NUM_BATCHED_TOKENS")),
            kv_cache_memory_bytes=int(need("RESIDENT_KV_CACHE_MEMORY_BYTES")),
            ident=os.environ.get("RESIDENT_IDENT", "vllm"),
        )


def _checkpoint_is_multimodal(model_dir: str) -> bool:
    """True when the checkpoint's config.json declares a vision tower.

    Read from the snapshot on disk, never from the model registry: the engine
    must decide before vLLM is imported whether the renderer will need a
    tokenizer (see the PORT DRIFT note in VllmBackend).
    """
    import json

    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(config.get("vision_config"), dict)


class VllmBackend:
    """One `LLMEngine`, one greedy request at a time.

    THE PINNED ENGINE SHAPE (fixture `engine_env`): batch 1, no prefix cache
    (a phase must never reuse another phase's prefill), raw-logit top-k,
    mamba state in fp32 and the checkpoint's own quantization as declared by
    its `hf_quant_config.json`. The only thing that differs between legs is
    `speculative_config`, and only through `RESIDENT_SPEC_DEPTH`.
    """

    def __init__(self, cfg: VllmConfig):
        from vllm import EngineArgs, LLMEngine, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        spec = None
        if cfg.depth > 0:
            spec = {"method": "mtp", "num_speculative_tokens": cfg.depth}
        # PORT DRIFT (vLLM 0.27.1, Muse Glimmer 30B): a multimodal checkpoint
        # needs a tokenizer even when the caller only sends token IDs, because
        # the renderer builds the model's multimodal processor at engine start
        # and that processor raises "You cannot pass text prompts when
        # `skip_tokenizer_init=True`". A text-only checkpoint keeps the old
        # pinned shape (tokenizer skipped); a checkpoint whose config carries a
        # vision tower loads the tokenizer and still takes token-ID prompts.
        multimodal = _checkpoint_is_multimodal(cfg.model_dir)
        args = EngineArgs(
            model=cfg.model_dir,
            tokenizer=cfg.model_dir,
            skip_tokenizer_init=not multimodal,
            trust_remote_code=False,
            dtype="auto",
            seed=0,
            max_model_len=cfg.ctx_size,
            max_num_seqs=1,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_prefix_caching=False,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            # PINNED, not profiled: the profiled KV size differs box to box
            # (CUDA-graph memory estimates vary), which changes the cache
            # geometry and, with it, this model's numerics. A fixed size makes
            # every box identical (port notes section 7.2).
            kv_cache_memory_bytes=cfg.kv_cache_memory_bytes,
            mamba_ssm_cache_dtype="float32",
            logprobs_mode="raw_logits",
            max_logprobs=TOP_K,
            speculative_config=spec,
            moe_backend="humming",
            cudagraph_capture_sizes=[1, 512] if cfg.depth == 0 else [1, 2, 512],
            enable_flashinfer_autotune=False,
            async_scheduling=False,
            disable_log_stats=True,
        )
        self._engine = LLMEngine.from_engine_args(args)
        self._SamplingParams = SamplingParams
        self._delta = RequestOutputKind.DELTA
        hf = self._engine.model_config.hf_config
        # PORT DRIFT (Muse Glimmer 30B): a composite config keeps vocab_size and
        # eos_token_id in its text sub-config, not at the top level. Read the
        # text config when the top level does not carry them, so the same code
        # serves a flat text-only config and a nested multimodal one.
        text_hf = hf
        if not hasattr(hf, "vocab_size") and hasattr(hf, "get_text_config"):
            text_hf = hf.get_text_config()
        self.vocab_size = int(text_hf.vocab_size)
        eos = text_hf.eos_token_id
        self.eos_token = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.depth = cfg.depth
        self.ctx_size = cfg.ctx_size
        self.ident = cfg.ident
        self.model_path = cfg.model_dir
        self._rid = 0
        self._live: str | None = None
        self._fast_decode_ready = False
        self._fast_decode_failed = False
        self._fast_p512_ready = False
        self._fast_max_steps = 4096
        self._fast_window_limit = int(getattr(text_hf, "sliding_window", None) or cfg.ctx_size)
        self._warm_up()

    # BOOT WARM-UP. The first prefill and the first decode steps after the
    # engine is built run cold (lazy kernel JIT, autotune, allocator growth):
    # measured on GB10, the first seed-length prefill is ~2.8x slower than every
    # later one. benchd times the FIRST prefill a leg's resident serves, so the
    # resident pays that cost here, before it reports ready, on a synthetic
    # input-independent prompt -- never the benchmark's tokens. It is the same
    # class of work as CUDA graph capture at boot. The prompt is the fixture's
    # `seed_tokens` long, so the warm-up JITs the exact shapes the timed
    # prefill runs.
    WARMUP_PROMPT_TOKENS = 512
    WARMUP_DECODE_STEPS = 8
    WARMUP_ROUNDS = 1

    def _warm_up(self) -> None:
        prompt = [(i * 7919) % 30000 + 1000 for i in range(self.WARMUP_PROMPT_TOKENS)]
        for _ in range(self.WARMUP_ROUNDS):
            self.start(prompt, need_logprobs=True)
            for _ in range(self.WARMUP_DECODE_STEPS):
                self.step()
            self.abort()
        if self.depth == 0:
            emitted = self.start(prompt, need_logprobs=False)
            self._init_fast_decode()
            if self._fast_decode_ready:
                warm_toks = self.serial_run_fast(emitted[0].token, 4)
                import sys
                print(
                    f"resident: fast decode warmup {'ok' if warm_toks is not None else 'skipped'}",
                    file=sys.stderr,
                    flush=True,
                )
            self.abort()
            if self._fast_p512_ready:
                p512_em = self._prefill_fast_512(prompt)
                if p512_em is None or p512_em[0].token != emitted[0].token:
                    import sys
                    print(
                        f"resident: fast prefill512 self-check failed ({p512_em} vs {emitted[0].token}); disabling",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._fast_p512_ready = False
                else:
                    import sys
                    print("resident: fast prefill512 self-check ok", file=sys.stderr, flush=True)

    def _init_fast_decode(self) -> None:
        """Capture lightweight pre-embed and post-logits/argmax CUDA graphs for
        zero-sync multi-step serial decode (in-process V2 GPUModelRunner)."""
        if self._fast_decode_ready or self._fast_decode_failed or self.depth != 0:
            return
        try:
            import torch
            from muse_kernels import load_muse_kernels
            from vllm.config.compilation import CUDAGraphMode

            if not load_muse_kernels():
                self._fast_decode_failed = True
                return
            ec_client = self._engine.engine_core
            core = getattr(ec_client, "engine_core", None)
            if core is None:
                self._fast_decode_failed = True
                return
            runner = core.model_executor.driver_worker.model_runner
            cg_mgr = getattr(runner, "cudagraph_manager", None)
            if cg_mgr is None or not cg_mgr.graphs:
                self._fast_decode_failed = True
                return
            full_graph = None
            for desc, graph in cg_mgr.graphs.items():
                if desc.cg_mode == CUDAGraphMode.FULL and desc.num_tokens == 1 and desc.num_reqs == 1:
                    full_graph = graph
                    break
            if full_graph is None or cg_mgr.hidden_states is None:
                self._fast_decode_failed = True
                return

            device = runner.device
            self._fast_core = core
            self._fast_runner = runner
            self._fast_full_graph = full_graph
            self._fast_out_tokens = torch.empty(
                self._fast_max_steps, dtype=torch.int32, device=device
            )
            self._fast_step_idx = torch.zeros(1, dtype=torch.int32, device=device)
            self._fast_step_limit = torch.ones(1, dtype=torch.int32, device=device)
            self._fast_idx_map = torch.zeros(1, dtype=torch.int32, device=device)
            self._fast_lm_blk_val = torch.empty(192, dtype=torch.float32, device=device)
            self._fast_lm_blk_idx = torch.empty(192, dtype=torch.int32, device=device)
            self._fast_supports_mm = bool(
                runner.supports_mm_inputs
                and hasattr(runner.model_state, "encoder_runner")
                and runner.model_state.encoder_runner is not None
            )

            lm_head = getattr(runner.model, "lm_head", None)
            use_fused_lmhead = bool(
                lm_head is not None
                and getattr(lm_head, "_muse_fast_lmhead_id", -1) >= 0
                and getattr(lm_head, "bias", None) is None
                and hasattr(runner.model, "output_multiplier")
            )
            lm_mul = float(getattr(runner.model, "output_multiplier", 1.0)) if use_fused_lmhead else 1.0
            lm_cap = float(getattr(runner.model, "final_logit_softcapping", None) or 0.0) if use_fused_lmhead else 0.0

            pool = cg_mgr.pool
            input_ids_1 = runner.input_buffers.input_ids[:1]
            hidden_1 = cg_mgr.hidden_states[:1]

            def _pre_op() -> None:
                if self._fast_supports_mm:
                    emb = runner.model.model.embed_input_ids(input_ids_1)
                    runner.model_state.encoder_runner.inputs_embeds[:1].copy_(emb)

            def _post_op() -> None:
                if use_fused_lmhead:
                    rp = getattr(lm_head, "_muse_rans", None)
                    tm = getattr(lm_head, "_muse_tm", None)
                    if rp is not None:
                        bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp
                        torch.ops.muse_fast.nvfp4_rans_lmhead_m1_argmax_step_update(
                            hidden_1,
                            bq_tm,
                            sc_stream,
                            stream_chunk_off,
                            init_states,
                            lut,
                            lm_head.weight_global_scale,
                            lm_mul,
                            lm_cap,
                            self._fast_lm_blk_val,
                            self._fast_lm_blk_idx,
                            self._fast_out_tokens,
                            self._fast_step_idx,
                            self._fast_step_limit,
                            runner.input_buffers.input_ids,
                            runner.input_buffers.positions,
                            runner.input_buffers.seq_lens,
                            runner.block_tables.input_block_table_ptrs,
                            runner.block_tables.block_sizes_tensor,
                            runner.block_tables.slot_mappings,
                        )
                    elif tm is not None:
                        bq_tm, bs_tm, b_base, tile_s_off, tile_s_mode = tm
                        torch.ops.muse_fast.nvfp4_tm_lmhead_m1_argmax_step_update(
                            hidden_1,
                            bq_tm,
                            bs_tm,
                            b_base,
                            tile_s_off,
                            tile_s_mode,
                            lm_head.weight_global_scale,
                            lm_mul,
                            lm_cap,
                            self._fast_lm_blk_val,
                            self._fast_lm_blk_idx,
                            self._fast_out_tokens,
                            self._fast_step_idx,
                            self._fast_step_limit,
                            runner.input_buffers.input_ids,
                            runner.input_buffers.positions,
                            runner.input_buffers.seq_lens,
                            runner.block_tables.input_block_table_ptrs,
                            runner.block_tables.block_sizes_tensor,
                            runner.block_tables.slot_mappings,
                            getattr(lm_head, "_muse_tm_mask", 2),
                        )
                    else:
                        torch.ops.muse_fast.nvfp4_lmhead_m1_argmax_step_update(
                            hidden_1,
                            lm_head.weight,
                            lm_head.weight_scale,
                            lm_head.weight_global_scale,
                            lm_mul,
                            lm_cap,
                            self._fast_lm_blk_val,
                            self._fast_lm_blk_idx,
                            self._fast_out_tokens,
                            self._fast_step_idx,
                            self._fast_step_limit,
                            runner.input_buffers.input_ids,
                            runner.input_buffers.positions,
                            runner.input_buffers.seq_lens,
                            runner.block_tables.input_block_table_ptrs,
                            runner.block_tables.block_sizes_tensor,
                            runner.block_tables.slot_mappings,
                        )
                else:
                    logits = runner.model.compute_logits(hidden_1)
                    torch.ops.muse_fast.greedy_argmax_step_update(
                        logits,
                        self._fast_out_tokens,
                        self._fast_step_idx,
                        self._fast_step_limit,
                        runner.input_buffers.input_ids,
                        runner.input_buffers.positions,
                        runner.input_buffers.seq_lens,
                        runner.block_tables.input_block_table_ptrs,
                        runner.block_tables.block_sizes_tensor,
                        runner.block_tables.slot_mappings,
                    )

            # Snapshot 512-token prefill metadata BEFORE overwriting positions/seq_lens for M=1 capture.
            p512_snaps = (
                runner.input_buffers.positions[:512].clone(),
                runner.input_buffers.seq_lens[:1].clone(),
                runner.input_buffers.query_start_loc[:2].clone(),
                runner.block_tables.slot_mappings[:, :512].clone(),
                [bt[:1].clone() for bt in runner.block_tables.input_block_tables],
            )

            with torch.inference_mode():
                # Ensure valid dummy indices during warmup/capture
                self._fast_step_idx.zero_()
                self._fast_step_limit.fill_(1)
                input_ids_1.zero_()
                runner.input_buffers.positions[:1].zero_()
                runner.input_buffers.seq_lens[:1].fill_(1)
                _pre_op()
                _post_op()
                torch.cuda.synchronize()

                self._fast_pre_graph = None
                if self._fast_supports_mm:
                    pre_g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pre_g, pool=pool):
                        _pre_op()
                    self._fast_pre_graph = pre_g

                post_g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(post_g, pool=pool):
                    _post_op()
                    if self._fast_supports_mm:
                        _pre_op()
                self._fast_post_graph = post_g
                torch.cuda.synchronize()

            self._fast_decode_ready = True
            self._init_fast_prefill_512(
                runner, cg_mgr, lm_head, use_fused_lmhead, lm_mul, lm_cap, p512_snaps
            )
            import sys
            print(
                f"resident: fast decode ready (mm={self._fast_supports_mm}, fused_lmhead={use_fused_lmhead}, p512={self._fast_p512_ready}, window={self._fast_window_limit})",
                file=sys.stderr,
                flush=True,
            )
        except Exception as err:
            import sys
            print(f"resident: fast decode init skipped: {err}", file=sys.stderr, flush=True)
            self._fast_decode_failed = True

    def _init_fast_prefill_512(
        self,
        runner,
        cg_mgr,
        lm_head,
        use_fused_lmhead: bool,
        lm_mul: float,
        lm_cap: float,
        p512_snaps: tuple,
    ) -> None:
        """Capture fast whole-forward M=512 prefill graph + fused LM-head argmax
        for max_tokens=1, need_logprobs=False timed prefill calls."""
        try:
            import torch

            p512_desc = cg_mgr.get_prefill_full_desc(512)
            if p512_desc is None or not use_fused_lmhead:
                return
            rp = getattr(lm_head, "_muse_rans", None)
            if rp is None:
                return
            p512_graph = cg_mgr.graphs.get(p512_desc)
            if p512_graph is None or cg_mgr.hidden_states is None or cg_mgr.hidden_states.shape[0] < 512:
                return

            device = runner.device
            self._p512_graph = p512_graph
            self._p512_host_ids = torch.empty(512, dtype=torch.int32, pin_memory=True)
            self._p512_out_tok = torch.empty(1, dtype=torch.int32, device=device)
            (
                self._p512_pos_snap,
                self._p512_seq_snap,
                self._p512_qsl_snap,
                self._p512_slot_snap,
                self._p512_bt_snaps,
            ) = p512_snaps

            input_ids_512 = runner.input_buffers.input_ids[:512]
            hidden_last = cg_mgr.hidden_states[511:512]
            bq_tm, sc_stream, stream_chunk_off, init_states, lut = rp

            def _pre512() -> None:
                runner.input_buffers.positions[:512].copy_(self._p512_pos_snap)
                runner.input_buffers.seq_lens[:1].copy_(self._p512_seq_snap)
                runner.input_buffers.query_start_loc[:2].copy_(self._p512_qsl_snap)
                runner.block_tables.slot_mappings[:, :512].copy_(self._p512_slot_snap)
                for dst_bt, snap_bt in zip(
                    runner.block_tables.input_block_tables, self._p512_bt_snaps
                ):
                    dst_bt[:1].copy_(snap_bt)
                if self._fast_supports_mm:
                    emb = runner.model.model.embed_input_ids(input_ids_512)
                    runner.model_state.encoder_runner.inputs_embeds[:512].copy_(emb)

            def _post512() -> None:
                torch.ops.muse_fast.nvfp4_rans_lmhead_m1_argmax_only(
                    hidden_last,
                    bq_tm,
                    sc_stream,
                    stream_chunk_off,
                    init_states,
                    lut,
                    lm_head.weight_global_scale,
                    lm_mul,
                    lm_cap,
                    self._fast_lm_blk_val,
                    self._fast_lm_blk_idx,
                    self._p512_out_tok,
                )

            with torch.inference_mode():
                _pre512()
                _post512()
                torch.cuda.synchronize()
                pool = cg_mgr.pool
                pre_g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(pre_g, pool=pool):
                    _pre512()
                post_g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(post_g, pool=pool):
                    _post512()
                torch.cuda.synchronize()
                self._p512_pre_graph = pre_g
                self._p512_post_graph = post_g

            self._fast_p512_ready = True
        except Exception as err:
            import sys
            print(f"resident: fast prefill512 init skipped: {err}", file=sys.stderr, flush=True)
            self._fast_p512_ready = False

    def _prefill_fast_512(self, tokens: list[int]) -> list[Emit] | None:
        if not self._fast_p512_ready or len(tokens) != 512:
            return None
        if len(self._fast_core.scheduler.running) != 0:
            return None
        import torch

        with torch.inference_mode():
            self._p512_host_ids.copy_(torch.tensor(tokens, dtype=torch.int32))
            self._fast_runner.input_buffers.input_ids[:512].copy_(
                self._p512_host_ids, non_blocking=True
            )
            self._p512_pre_graph.replay()
            self._p512_graph.replay()
            self._p512_post_graph.replay()
            tok = int(self._p512_out_tok.item())
        return [Emit(tok, ())]

    def serial_run_fast(self, first_token: int, count: int) -> list[int] | None:
        """Zero-sync on-GPU multi-step serial decode loop.

        Returns the `count` generated tokens `[T_1, ..., T_count]` or None if
        the request is not eligible (falling back to standard engine steps).
        """
        if not self._fast_decode_ready or self._live is None or count < 2:
            return None
        if count > self._fast_max_steps:
            return None
        core = self._fast_core
        scheduler = core.scheduler
        if len(scheduler.running) != 1:
            return None
        req = scheduler.running[0]
        if req.is_finished():
            return None
        sp = req.sampling_params
        if sp is None or sp.temperature != 0.0 or sp.logprobs is not None:
            return None
        start_pos = req.num_computed_tokens
        if start_pos + count + 1 >= min(self.ctx_size, self._fast_window_limit):
            return None
        if req.num_output_tokens + count > req.max_tokens:
            return None

        import torch

        runner = self._fast_runner
        req_idx = runner.req_states.req_id_to_index.get(req.request_id)
        if req_idx is None:
            return None

        # Pre-allocate KV cache blocks for all `count` decode positions at once.
        new_blocks = scheduler.kv_cache_manager.allocate_slots(req, count)
        if new_blocks is None:
            return None
        new_block_ids = new_blocks.get_block_ids()
        if any(new_block_ids):
            runner.block_tables.append_block_ids(req_idx, new_block_ids, overwrite=False)
            runner.block_tables.apply_staged_writes()

        with torch.inference_mode():
            self._fast_idx_map.fill_(req_idx)
            runner.block_tables.gather_block_tables(self._fast_idx_map, num_reqs_padded=1)
            torch.ops.muse_fast.init_decode_loop(
                first_token,
                start_pos,
                count,
                self._fast_step_idx,
                self._fast_step_limit,
                runner.input_buffers.input_ids,
                runner.input_buffers.positions,
                runner.input_buffers.seq_lens,
                runner.input_buffers.query_start_loc,
                runner.block_tables.input_block_table_ptrs,
                runner.block_tables.block_sizes_tensor,
                runner.block_tables.slot_mappings,
            )

            pre_g = self._fast_pre_graph
            full_g = self._fast_full_graph
            post_g = self._fast_post_graph
            if pre_g is not None:
                pre_g.replay()
            for _ in range(count):
                full_g.replay()
                post_g.replay()

            tokens = [int(t) for t in self._fast_out_tokens[:count].tolist()]

            # Keep scheduler and runner request states consistent for subsequent
            # steps or clean abort().
            new_computed = start_pos + count
            req.num_computed_tokens = new_computed
            req.append_output_token_ids(tokens)
            runner.req_states.num_computed_tokens_np[req_idx] = new_computed
            runner.req_states.num_computed_tokens.gpu[req_idx] = new_computed
            runner.req_states.total_len.gpu[req_idx] = new_computed + 1
            runner.req_states.last_sampled_tokens[req_idx, 0] = tokens[-1]

        return tokens

    def _params(
        self,
        prompt_len: int,
        max_tokens: int | None = None,
        need_logprobs: bool = True,
    ):
        room = max(1, self.ctx_size - prompt_len - self.depth - 1)
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=room if max_tokens is None else min(room, max_tokens),
            ignore_eos=True,
            logprobs=TOP_K if need_logprobs else None,
            detokenize=False,
            output_kind=self._delta,
        )

    def _collect(self) -> list[Emit]:
        emitted: list[Emit] = []
        for out in self._engine.step():
            if out.request_id != self._live:
                continue
            comp = out.outputs[0]
            lps = comp.logprobs
            if lps:
                for token, lp in zip(comp.token_ids, lps):
                    top: tuple[tuple[int, float], ...] = ()
                    if lp:
                        ranked = sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)
                        top = tuple((int(t), float(v.logprob)) for t, v in ranked[:TOP_K])
                    emitted.append(Emit(int(token), top))
            else:
                for token in comp.token_ids:
                    emitted.append(Emit(int(token), ()))
        return emitted

    def start(
        self,
        tokens: list[int],
        max_tokens: int | None = None,
        need_logprobs: bool = True,
    ) -> list[Emit]:
        self.abort()
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError(f"{len(tokens)} tokens do not fit the {self.ctx_size}-token context")
        if (
            self._fast_p512_ready
            and max_tokens == 1
            and not need_logprobs
            and len(tokens) == 512
        ):
            fast_em = self._prefill_fast_512(tokens)
            if fast_em is not None:
                return fast_em
        self._rid += 1
        self._live = f"r{self._rid}"
        self._engine.add_request(
            self._live,
            {"prompt_token_ids": list(tokens)},
            self._params(len(tokens), max_tokens, need_logprobs=need_logprobs),
        )
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the engine finished the request without emitting a token")
            emitted = self._collect()
        return emitted

    def step(self) -> list[Emit]:
        if self._live is None:
            raise RuntimeError("step with no live request")
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the request reached the context limit")
            emitted = self._collect()
        return emitted

    def abort(self) -> bool:
        running = False
        if self._live is not None:
            running = self._engine.has_unfinished_requests()
            self._engine.abort_request([self._live])
            # Drain the abort so the next request starts on an idle engine.
            while self._engine.has_unfinished_requests():
                self._engine.step()
            self._live = None
        return running

    def close(self) -> None:
        self.abort()


# --------------------------------------------------------------------------- #
# Fake (tests only)
# --------------------------------------------------------------------------- #


class FakeBackend:
    """Deterministic greedy "model": the next token is a fixed hash of the last
    two context tokens. With depth > 0 a step emits a variable number of tokens
    (1..depth+1), every one of them the same greedy continuation, which is the
    contract a correct speculative engine keeps."""

    def __init__(self, depth: int = 0, vocab_size: int = 131072, ctx_size: int = 8192):
        self.vocab_size = vocab_size
        self.eos_token = 2
        self.depth = depth
        self.ctx_size = ctx_size
        self.ident = f"fake depth={depth}"
        self.model_path = "/fake/model"
        self._ctx: list[int] | None = None
        self._cycle = 0
        self._left: int | None = None
        self.starts = 0

    def _next(self, ctx: list[int]) -> int:
        a = ctx[-1] if ctx else 0
        b = ctx[-2] if len(ctx) > 1 else 0
        return (a * 31 + b * 17 + 7) % 1000 + 20

    def _emit(self, ctx: list[int]) -> Emit:
        token = self._next(ctx)
        top = tuple(
            (token if i == 0 else (token + i) % self.vocab_size, 10.0 - i) for i in range(TOP_K)
        )
        return Emit(token, top)

    def start(
        self,
        tokens: list[int],
        max_tokens: int | None = None,
        need_logprobs: bool = True,
    ) -> list[Emit]:
        del need_logprobs
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError("context full")
        self.starts += 1
        self._ctx = list(tokens)
        self._cycle = 0
        self._left = None if max_tokens is None else max_tokens - 1
        first = self._emit(self._ctx)
        self._ctx.append(first.token)
        return [first]

    def step(self) -> list[Emit]:
        if self._ctx is None:
            raise RuntimeError("step with no live request")
        # vLLM finishes a request at its max_tokens; so does the fake.
        if self._left == 0:
            raise RuntimeError("the request reached its max_tokens")
        self._cycle += 1
        n = 1 + (self._cycle % (self.depth + 1)) if self.depth else 1
        if self._left is not None:
            n = min(n, self._left)
            self._left -= n
        out = []
        for _ in range(n):
            e = self._emit(self._ctx)
            self._ctx.append(e.token)
            out.append(e)
        return out

    def abort(self) -> bool:
        running = self._ctx is not None and self._left != 0
        self._ctx = None
        return running

    def close(self) -> None:
        self.abort()
