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
frontier it holds without another forward pass. `start(..., top_logits=False)`
drops them for a request whose caller reads only the greedy tokens (the timed
prefill and the free run): vLLM then skips the top-k selection over the
vocabulary, its device-to-host copy and the per-token Logprob objects on every
step. The tokens are the same either way: with greedy sampling the logprobs are
a side read of the raw logits, never an input to the argmax or to the
rejection sampler (vllm/v1/worker/gpu/sample/sampler.py).

`VllmBackend` is the scored engine. `FakeBackend` is a deterministic stand-in
with the same step semantics (including multi-token speculative steps), so the
resident's wire and bookkeeping are tested with no GPU and no vLLM install.
"""

from __future__ import annotations

import os
import sys
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
from typing import Protocol

TOP_K = 8


def _log(msg: str) -> None:
    print(f"resident: {msg}", file=sys.stderr, flush=True)


def _shutdown_inductor_workers() -> None:
    """Shut down idle torch._inductor compile_worker subprocesses once warmup/capture is done."""
    if os.environ.get("VLLM_SHUTDOWN_COMPILE_WORKERS", "1") == "0":
        return
    try:
        from torch._inductor import async_compile
        pool_fn = getattr(async_compile, "_pool_set", None)
        if pool_fn is not None:
            for pool in list(pool_fn):
                shutdown = getattr(pool, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            pool_fn.clear()
    except Exception:
        pass


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
    draft_model_path: str  # the separate drafter's directory; "" when none is loaded

    def start(
        self, tokens: list[int], max_tokens: int | None = None, top_logits: bool = True
    ) -> list[Emit]: ...

    def step(self) -> list[Emit]: ...

    def begin_spec_run(self, count: int) -> None:
        """Bound and pipeline the one timed speculative run."""
        ...

    def end_spec_run(self) -> None:
        """Return to non-lookahead stepping before the timed reply."""
        ...

    def abort(self) -> bool:
        """Drop the request. True when it was still generating."""
        ...

    def settle(self) -> None:
        """Finish the bookkeeping of a request that has completed on its own.
        The resident calls it after it has written a reply."""
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
    # The speculative decoder at depth > 0 (SPEC_METHODS). "mtp" reads an MTP
    # layer inside the target checkpoint; "dflash" loads the separate drafter at
    # `draft_model_dir`, the contract's pinned drafter that tools/serve-up.sh
    # mounts read-only. Both come from serve-up.sh, never from the submission.
    spec_method: str = "mtp"
    draft_model_dir: str = ""
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
        method = os.environ.get("RESIDENT_SPEC_METHOD", "mtp")
        draft_dir = os.environ.get("RESIDENT_DRAFT_MODEL_DIR", "")
        check_spec_method(method, depth, draft_dir)
        return cls(
            model_dir=need("RESIDENT_MODEL_DIR"),
            depth=depth,
            ctx_size=int(need("RESIDENT_CTX_SIZE")),
            gpu_memory_utilization=float(need("RESIDENT_GPU_MEMORY_UTILIZATION")),
            max_num_batched_tokens=int(need("RESIDENT_MAX_NUM_BATCHED_TOKENS")),
            kv_cache_memory_bytes=int(need("RESIDENT_KV_CACHE_MEMORY_BYTES")),
            ident=os.environ.get("RESIDENT_IDENT", "vllm"),
            spec_method=method,
            draft_model_dir=draft_dir,
        )


# The decoders the resident can build. "mtp" is the in-checkpoint MTP layer
# (unchanged); "dflash" is a separate DFlash drafter loaded from its own
# directory.
SPEC_METHODS = ("mtp", "dflash")


def check_spec_method(method: str, depth: int, draft_dir: str) -> None:
    """Refuse a decoder the resident cannot build, or a drafter it cannot find.

    A dflash leg needs the drafter directory (with its config.json); an mtp or
    serial leg must not name one, so a stray path can never reach the engine.
    """
    if method not in SPEC_METHODS:
        raise RuntimeError(f"RESIDENT_SPEC_METHOD={method!r} is not one of {', '.join(SPEC_METHODS)}")
    if method == "dflash" and depth > 0:
        if not draft_dir:
            raise RuntimeError("RESIDENT_SPEC_METHOD=dflash needs RESIDENT_DRAFT_MODEL_DIR; tools/serve-up.sh exports it")
        if not os.path.isfile(os.path.join(draft_dir, "config.json")):
            raise RuntimeError(f"RESIDENT_DRAFT_MODEL_DIR={draft_dir} holds no config.json")
    elif draft_dir:
        raise RuntimeError(
            f"RESIDENT_DRAFT_MODEL_DIR is set for a {'serial' if depth == 0 else method} leg; only a dflash leg loads a drafter"
        )


def speculative_config(cfg: VllmConfig) -> dict | None:
    """vLLM's `speculative_config` for this leg; None for a serial leg."""
    if cfg.depth == 0:
        return None
    if cfg.spec_method == "dflash":
        return {"method": "dflash", "model": cfg.draft_model_dir, "num_speculative_tokens": cfg.depth}
    return {"method": "mtp", "num_speculative_tokens": cfg.depth}


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
        from vllm.model_executor.draft_vocab import configure_draft_vocab
        from vllm.sampling_params import RequestOutputKind
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.executor.abstract import Executor

        try:
            import muse_sandwich
        except ModuleNotFoundError:
            muse_sandwich = None
        if muse_sandwich is not None:
            muse_sandwich.install()

        spec = speculative_config(cfg)
        if cfg.depth:
            draft_vocab_size = int(os.environ.get("MUSE_DRAFT_VOCAB_SIZE", "98304"))
            draft_vocab_mode = os.environ.get("MUSE_DRAFT_VOCAB_MODE", "ascii_bpe")
            configure_draft_vocab(cfg.model_dir, draft_vocab_size, draft_vocab_mode)
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
            # Read each checkpoint shard in one pass instead of memory-mapping
            # it: the same bytes reach the same parameters, in a shorter load.
            safetensors_load_strategy=(
                "lazy" if os.environ.get("VLLM_RESIDENT_EAGER_LOAD", "1") == "0" else "eager"
            ),
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
            disable_log_stats=True,
        )
        # Drive the single-request engine in this process. `from_engine_args`
        # would ignore `enable_multiprocessing=False` whenever the default-on
        # VLLM_ENABLE_V1_MULTIPROCESSING flag is set, so build the engine from
        # its parts: same config, same executor class, but `multiprocess_mode=False`
        # selects the synchronous in-process client (no engine-core process, no
        # IPC hop per step). Startup is done afterwards: keep this process on
        # the fastest cores (a no-op where all cores are alike).
        if os.environ.get("VLLM_RESIDENT_INPROC", "1") == "0":
            self._engine = LLMEngine.from_engine_args(args)
        else:
            usage_context = UsageContext.ENGINE_CONTEXT
            engine_config = args.create_engine_config(usage_context)
            self._engine = LLMEngine(
                vllm_config=engine_config,
                executor_class=Executor.get_class(engine_config),
                log_stats=not args.disable_log_stats,
                usage_context=usage_context,
                multiprocess_mode=False,
            )
        from vllm.v1.worker import hostfast

        pinned = hostfast.pin_to_fast_cpus()
        _log(
            f"process pinned to CPUs {hostfast.format_cpus(pinned)}"
            if pinned
            else "CPU affinity unchanged"
        )
        if pinned:
            tickled = hostfast.start_idle_tickle(pinned)
            if tickled:
                _log(f"started {tickled} SCHED_IDLE deep-idle tickler threads")
        # REPLY FIRST, THEN CLEAN UP. When a bounded request (a timed prefill)
        # has emitted its last token, the engine core still owns the request's
        # cleanup and a queued empty batch. The stock engine's core loop does
        # that work while the output is already on its way to the caller; the
        # in-process client did it inline, before `start` could return the
        # token. With `defer_settle` the client leaves it to `settle()`, which
        # the resident calls right after it has written the reply (and which
        # `abort`, so every `start`, runs first in any case). The same work
        # runs, in the stock order. VLLM_RESIDENT_DEFER_SETTLE=0 keeps it inline.
        client = self._engine.engine_core
        self._settle_core = None
        if os.environ.get("VLLM_RESIDENT_DEFER_SETTLE", "1") != "0" and callable(
            getattr(client, "settle", None)
        ):
            client.defer_settle = True
            self._settle_core = client.settle
        # DFlash's V2 runner can apply speculative rejection corrections and
        # draft-token updates while two batches are in flight. Keep that
        # machinery dormant for prefill/correctness, though: a globally async
        # engine would queue decode work before free_decode_begin replies. The
        # timed run enables it only after its exact token count is known.
        self._async_core = None
        self._spec_run_pipelined = False
        core = getattr(client, "engine_core", None)
        if cfg.depth > 0 and core is not None and core.batch_queue is not None:
            core.step_fn = core.step
            self._async_core = core
        # VLLM_RESIDENT_ARGMAX_ONLY=0 keeps the top-k on every request.
        self._argmax_only = os.environ.get("VLLM_RESIDENT_ARGMAX_ONLY", "1") != "0"
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
        self.draft_model_path = cfg.draft_model_dir if cfg.depth > 0 and cfg.spec_method == "dflash" else ""
        self._rid = 0
        self._live: str | None = None
        self._warm_up()
        _shutdown_inductor_workers()
        if muse_sandwich is not None and (summary := muse_sandwich.launch_summary()):
            _log(f"native sandwich dispatch during initialization: {summary}")
        if muse_sandwich is not None and hasattr(muse_sandwich, "fold_summary"):
            _log(f"gate-quant fold: {muse_sandwich.fold_summary()}")

    # BOOT WARM-UP. The first prefill and the first decode steps after the
    # engine is built run cold (lazy kernel JIT, autotune, allocator growth):
    # measured on GB10, the first seed-length prefill is ~2.8x slower than every
    # later one. benchd times the FIRST prefill a leg's resident serves, so the
    # resident pays that cost here, before it reports ready, on a synthetic
    # input-independent prompt -- never the benchmark's tokens. It is the same
    # class of work as CUDA graph capture at boot. The prompt is the fixture's
    # `seed_tokens` long, so the warm-up JITs the exact shapes the timed
    # prefill runs.
    WARMUP_PROMPT_TOKENS = 4096
    WARMUP_DECODE_STEPS = 64
    WARMUP_ROUNDS = 3
    WARMUP_SEED_TOKENS = 512
    WARMUP_SEED_DECODE_STEPS = 4

    def _warm_up(self) -> None:
        prompt = [(i * 7919) % 30000 + 1000 for i in range(self.WARMUP_PROMPT_TOKENS)]
        for i in range(self.WARMUP_ROUNDS):
            # Both request shapes the phases use: the timed phases' greedy-only
            # requests and the correctness gates' top-k ones (the latter JIT
            # the logprobs kernels), ending on the timed shape.
            self.start(prompt, top_logits=(self.WARMUP_ROUNDS - i) % 2 == 0)
            for _ in range(self.WARMUP_DECODE_STEPS):
                self.step()
            self.abort()
        for i in range(3):
            self.start(prompt[: self.WARMUP_SEED_TOKENS], top_logits=False)
            if i == 0:
                for _ in range(self.WARMUP_SEED_DECODE_STEPS):
                    self.step()
            self.abort()

    def _params(self, prompt_len: int, max_tokens: int | None = None, top_logits: bool = True):
        room = max(1, self.ctx_size - prompt_len - self.depth - 1)
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=room if max_tokens is None else min(room, max_tokens),
            ignore_eos=True,
            # None, not 0: 0 still runs the logprobs path for the sampled
            # token. None leaves the sampler on its greedy path alone.
            logprobs=TOP_K if top_logits else None,
            detokenize=False,
            output_kind=self._delta,
        )

    def _collect(self) -> list[Emit]:
        emitted: list[Emit] = []
        for out in self._engine.step():
            if out.request_id != self._live:
                continue
            comp = out.outputs[0]
            if not comp.logprobs:
                emitted.extend(Emit(int(token), ()) for token in comp.token_ids)
                continue
            for token, lp in zip(comp.token_ids, comp.logprobs):
                top: tuple[tuple[int, float], ...] = ()
                if lp:
                    ranked = sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)
                    top = tuple((int(t), float(v.logprob)) for t, v in ranked[:TOP_K])
                emitted.append(Emit(int(token), top))
        return emitted

    def start(
        self, tokens: list[int], max_tokens: int | None = None, top_logits: bool = True
    ) -> list[Emit]:
        self.abort()
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError(f"{len(tokens)} tokens do not fit the {self.ctx_size}-token context")
        self._rid += 1
        self._live = f"r{self._rid}"
        self._engine.add_request(
            self._live,
            {"prompt_token_ids": list(tokens)},
            self._params(len(tokens), max_tokens, top_logits or not self._argmax_only),
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

    def begin_spec_run(self, count: int) -> None:
        core = self._async_core
        if core is None:
            return
        if self._spec_run_pipelined:
            raise RuntimeError("a speculative run is already pipelined")
        if core.batch_queue:
            raise RuntimeError("cannot pipeline with an occupied batch queue")
        if self._live is None:
            raise RuntimeError("cannot pipeline without a live request")
        request = core.scheduler.requests.get(self._live)
        # A bounded sync (used by diagnostics and correctness probes) may have
        # completed the request already.  Its next cycle will start a fresh
        # request, so retain the synchronous path for that run.
        if request is None or not self._engine.has_unfinished_requests():
            return
        # max_tokens counts every output after the prompt, including the seed
        # token already produced by free_decode_begin. This bound lets the
        # async scheduler prove that no N+1 batch is needed, so all GPU work
        # belongs to free_decode_run(N)'s timed window.
        request.max_tokens = request.num_output_tokens + count
        if request.sampling_params is not None:
            request.sampling_params.max_tokens = request.max_tokens
        core.step_fn = core.step_with_batch_queue
        self._spec_run_pipelined = True

    def end_spec_run(self) -> None:
        core = self._async_core
        if core is None or not self._spec_run_pipelined:
            return
        if core.batch_queue:
            raise RuntimeError("pipelined speculative run left a batch in flight")
        core.step_fn = core.step
        self._spec_run_pipelined = False

    def abort(self) -> bool:
        running = False
        if self._live is not None:
            running = self._engine.has_unfinished_requests()
            self._engine.abort_request([self._live])
            # Drain the abort so the next request starts on an idle engine.
            while self._engine.has_unfinished_requests():
                self._engine.step()
            self._live = None
        if self._settle_core is not None:
            # Nothing is live now: the core may still own a deferred cleanup.
            self._settle_core()
        if getattr(self, "_spec_run_pipelined", False):
            async_core = self._async_core
            assert async_core is not None
            async_core.step_fn = async_core.step
            self._spec_run_pipelined = False
        return running

    def settle(self) -> None:
        if self._settle_core is None:
            return
        self._settle_core()
        if self._live is not None and not self._engine.has_unfinished_requests():
            # The request completed on its own: do now, after the reply, what
            # the next `start` would otherwise do first inside its window.
            self.abort()

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

    def __init__(
        self, depth: int = 0, vocab_size: int = 131072, ctx_size: int = 8192, draft_model_path: str = ""
    ):
        self.vocab_size = vocab_size
        self.eos_token = 2
        self.depth = depth
        self.ctx_size = ctx_size
        self.ident = f"fake depth={depth}"
        self.model_path = "/fake/model"
        self.draft_model_path = draft_model_path
        self._ctx: list[int] | None = None
        self._cycle = 0
        self._left: int | None = None
        self._top = True
        self.starts = 0
        # settle() calls: the resident makes one after every reply.
        self.settles = 0
        # The top_logits flag of every start, in order (tests read it).
        self.start_modes: list[bool] = []

    def _next(self, ctx: list[int]) -> int:
        a = ctx[-1] if ctx else 0
        b = ctx[-2] if len(ctx) > 1 else 0
        return (a * 31 + b * 17 + 7) % 1000 + 20

    def _emit(self, ctx: list[int]) -> Emit:
        token = self._next(ctx)
        if not self._top:
            return Emit(token, ())
        top = tuple(
            (token if i == 0 else (token + i) % self.vocab_size, 10.0 - i) for i in range(TOP_K)
        )
        return Emit(token, top)

    def start(
        self, tokens: list[int], max_tokens: int | None = None, top_logits: bool = True
    ) -> list[Emit]:
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError("context full")
        self.starts += 1
        self.start_modes.append(top_logits)
        self._top = top_logits
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

    def begin_spec_run(self, count: int) -> None:
        pass

    def end_spec_run(self) -> None:
        pass

    def abort(self) -> bool:
        running = self._ctx is not None and self._left != 0
        self._ctx = None
        return running

    def settle(self) -> None:
        self.settles += 1

    def close(self) -> None:
        self.abort()
