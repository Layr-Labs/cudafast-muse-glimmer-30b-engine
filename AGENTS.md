# Agent guide — Muse Glimmer 30B NVFP4 engine

This file is the working contract for coding agents in this repository.
`CLAUDE.md` is a symbolic link to it.

Read [README.md](README.md) first and [TASK.md](TASK.md) second. This file adds
the operational rules for iterating here. The ranked track is
`muse-glimmer-30b-cuda-v1`.

## Goal

Make the target prefill and decode faster on a DGX Spark, single stream, without
changing what it computes: the output must match the golden token for token
(the token-tolerance budget is `0.0`).

## Authorities

| Question | File |
|---|---|
| What you may edit | `benchmark.json` `editablePaths`: `vllm`, `harness`, `mtp-head.manifest.json` |
| Pins, engine configuration, serve depths, scoring | `fixtures/muse_glimmer_30b_cuda_v1_track.json` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |

## Where to work

| Lever | Where |
|---|---|
| Engine arguments (CUDA graphs, compile level, scheduler, async scheduling) | `harness/vllm_resident/backends.py` `VllmBackend` |
| Per-step overhead between benchd and vLLM | `harness/vllm_resident/resident.py`, `harness/protocol-adapter/src/` |
| Model code (the 52-layer decoder) | `vllm/vllm/model_executor/models/muse_glimmer.py` |
| Attention (sliding window, FP8 projections) | `vllm/vllm/model_executor/layers/attention/`, `.../quantization/` |
| Dense MLP (NVFP4 W4A16) | `vllm/vllm/model_executor/layers/linear.py`, `.../quantization/` |
| New CUDA kernels | `harness/kernels/` + `harness/kernels/build.sh` |

This track is serial only: `allowed_modes` is `[serial]` and there is no
speculative lever. `mtp-head.manifest.json` is still an editable path, but it
must declare `"source": "none"`.

## Rules

- **Never edit outside `editablePaths`.** That covers `tools/`, `fixtures/`,
  `setup.sh`, `benchmark.sh`, `benchmark.json`, `.github/` and
  `correctness_prompts/`, all trusted. A submission that touches them is
  refused.
- **Never change a numerical format.** Section 3.4 of the contract lists them.
  Keep the MLP and `lm_head` at W4A16_NVFP4 (group 16), the attention
  projections at FP8 w8a8, the vision tower and embeddings at BF16, and the
  KV cache at the fixture's `kv_cache_dtype`, exactly as declared.
- **Never reuse work across requests or phases.** That rules out prefix
  caching, memoized prompts and state kept across `sync`. The resident's
  `invalidate` must leave the engine with nothing from the previous phase.
  The timed prefill follows one unmeasured warm-up prefill of the same prompt
  in the same phase; compute both in full and carry nothing between them.
- **Never move prefill work out of `free_decode_begin`** (contract section 5.1.1).
- **The timed window demands an exact match.** If your timed stream leaves
  the golden at any token, even at a logit tie, the run fails (contract
  section 5.4.1). The ranked run and the local modes apply the same rule.
- **Keep the engine reproducible.** Run `tools/determinism-probe.py` against a
  booted resident; it must report one stream at serial; this track has no other
  mode. Today the probe starts the worker at a fixed path
  (`~/muse-glimmer/repo/.build/release/mlxfast-runtime-worker`), not a binary
  of the current checkout. The fixture disables
  `FlashInferFP8ScaledMMLinearKernel` because this checkpoint quantizes its
  attention projections to FP8 (`docs/muse-glimmer-port-notes.md`).
- **Keep the resident wire intact.** The Rust client in
  `harness/protocol-adapter/src/resident.rs` and `resident.py` must agree.
  `test_adapter_e2e.py` proves it on the serial path.
- **Keep `IMPLEMENTED_DEPTH` in `resident.py` and `ENGINE_IMPLEMENTED_DEPTH` in
  `session_backend.rs` equal.** Tests on both sides check it.
- **Report raw counters only.** The adapter never reports a rate, a time or a
  ratio. benchd does all measurement.
- **Keep `harness/protocol-adapter/target/` and `__pycache__/` out of a
  submission.** They count against the byte cap.

## Checks before you submit

```bash
cargo fmt --manifest-path harness/protocol-adapter/Cargo.toml -- --check
cargo clippy --manifest-path harness/protocol-adapter/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident
python3 tools/lint-benchmark-manifest.py
```

These commands check the adapter, the resident and the manifest with no GPU.

```bash
./setup.sh && ./tools/local-baseline.sh
```

This command runs one real leg. It needs a Spark or another sm_121 box with the
pinned image.

## Environment rules on a ranked box

Never export the following in a ranked job; `tools/ranked-box-preflight.sh`
refuses each of them by name:

- **Resident sockets:** `DS4_RESIDENT_SOCKET` and
  `BENCH_WORKER_RESIDENT_SOCKET`. benchd injects them per leg, and an inherited
  one is a third resident.
- **Engine configuration:** anything prefixed `RESIDENT_`, `VLLM_`, `HF_`,
  `TRANSFORMERS_`, `TORCHINDUCTOR_` or `TRITON_`. The fixture's `engine_env`
  is the only engine configuration.
- **Test seams:** `SERVE_UP_LAUNCHER` and `STAGE_VLLM_OVERLAY_BASE`.
- **Measurement skips:** `BENCHD`, `MLXFAST_SKIP_*` and
  `MLXFAST_LOCAL_COOL_GATE`.

## Notes

Keep your own notes, such as what sped the engine up and what the gate refused,
in an untracked local file such as `notes.local.md` (add it to your local git
ignore, for example `.git/info/exclude`). Never commit anything outside
`editablePaths`: that includes this file and `docs/`.
