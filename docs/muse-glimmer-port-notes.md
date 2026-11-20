# Engineering log — Muse Glimmer 30B NVFP4 CUDA port

This is the port's own record: what this tree is, what the port added, the
numbers behind the track fixture, and what is still open. The challenge
folder's `cuda/findings.md` is the numbered finding register (`F-CUDA-nn`);
this log is the narrative. The base engine's own engineering log stays in the
organizer's base repository, at the commit this tree was cloned from.

## 1. What this tree is

- **Engine base.** Cloned from the organizer's CUDA engine repository for the
  previous track, at commit
  `2ba41bff7a27c861a3f1a32fd8ca8780402181bf`.
- **Engine.** vLLM `v0.27.1` (`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`),
  Python and Triton source VENDORED under `vllm/`, layered over the compiled
  extensions of the pinned container image
  (`docker.io/vllm/vllm-openai:v0.27.1`, arm64 digest
  `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c`).
  `vllm/VENDOR.json` records the base and every local patch.
- **Track.** `muse-glimmer-30b-cuda-v1`, SERIAL ONLY: `allowed_modes` is
  `[serial]`, the fixture declares one serve depth (`serial`), and the
  checkpoint ships no MTP or draft tensors.
- **Box class.** NVIDIA DGX Spark (GB10, `sm_121`, aarch64), driver floor
  `580.159.03`.

## 2. What the port added

Commit `59080ac` (branch `muse-glimmer-nvfp4-port`) is the overlay that makes
the pinned image serve this checkpoint. It adds, inside `vllm/` (the editable
surface):

| File | What it is |
|---|---|
| `vllm/vllm/model_executor/models/muse_glimmer.py` | the model: a 52-layer text decoder (39 sliding-attention layers with window 2048, 13 full-attention layers), a dense MLP, `MuseGlimmerForConditionalGeneration` |
| `vllm/vllm/transformers_utils/configs/muse_glimmer.py` | the config class (`text_config` holds the decoder, `vision_config` the tower) |
| `vllm/vllm/transformers_utils/processors/muse_glimmer.py` | the multimodal processor (the scored text track never passes an image) |
| `vllm/vllm/reasoning/muse_glimmer_reasoning_parser.py`, `vllm/vllm/tool_parsers/muse_glimmer_tool_parser.py` | the chat-template side; both are inert with thinking off |
| `vllm/vllm/model_executor/models/registry.py`, `transformers_utils/config.py`, two `__init__.py` files | registrations |

Two resident commits came with it:

- `bdf6539` — the resident loads the tokenizer for a multimodal checkpoint
  under a token-ID protocol (benchd feeds ids, never text, so the processor is
  not on the timed path).
- `87630f9` — the resident reads `vocab_size` and `eos_token_id` from the text
  sub-config rather than the top-level config, which is where this checkpoint
  puts them.

`vllm/VENDOR.json` also keeps four local patches inherited from the base (the
GB10 NVML memory-clock fallback, the humming batch-invariant switch, the pinned
Mamba SSD autotune configs, and the MTP-drafter skip). The last one is inert
here: no leg ever arms a drafter.

## 3. The track fixture

`fixtures/muse_glimmer_30b_cuda_v1_track.json` is the authority. The tool that
authored it is `cuda/actions/author-track-fixture.py` in the challenge folder;
it reads the pins file, the snapshot's `config.json` and
`generation_config.json`, and the engine map's contract block, and refuses if a
snapshot value contradicts a value this track recorded.

Commands behind every number:

```bash
# the 14 weight-file pins
ssh <box> 'cd $HOME/muse-glimmer-30b/weights && sha256sum -c SHA256SUMS'
  -> 14 lines "OK"
# tokenizer.json = c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d
```

Model facts, from the snapshot's own files:

| Field | Value | Source |
|---|---|---|
| architecture | `MuseGlimmerForConditionalGeneration` | `config.json` |
| model types | `muse_glimmer`, text `muse_glimmer_text` | `config.json` |
| text layers | 52 (39 sliding_attention, 13 full_attention) | `config.json` |
| sliding window | 2048 | `config.json` |
| hidden / intermediate | 6656 / 19968 (dense MLP) | `config.json` |
| heads | 32 query, 2 KV, head_dim 128 | `config.json` |
| vocab | 202048 | `config.json` |
| positions | 131072 | `config.json` |
| tokens | bos 200000, eos 200001 + 200008, pad 200018 | `generation_config.json` |
| vision tower | 50 layers, hidden 1536, patch 14, merge 2 | `config.json` |
| tensors | 2222 total, 809 vision, no `experts`, no `mtp` | `model.safetensors.index.json` |
| quantization | 393 layers: 213 `W4A16_NVFP4` (group 16, MLP + `lm_head`), 180 `FP8` (attention projections) | `config.json` `quantization_config` |

**Formats (frozen).** The MLP projections and `lm_head` are W4A16_NVFP4 with
group size 16; the attention projections are FP8 (w8a8 float8); the vision
tower and the embeddings are BF16; `ignore` is empty.

Not carried over from the base fixture, because this model has none of these:
`allowed_modes [serial, mtp]` and the six draft depths, `speculative_oracles`,
`mamba_ssm_cache_dtype`, `moe_backend`, `speculative_method`,
`VLLM_HUMMING_BATCH_INVARIANT`.

KEPT: `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`. This
checkpoint quantizes its attention projections to FP8, and that kernel is not
batch-invariant, so the same prompt would otherwise yield two token streams.
That is what makes the serial tapes stable, and the fixture pins it in
`serve_configuration.engine_args.deterministic_kernels`.

## 4. Determinism

A golden is only useful if two boots of the same tree produce the same token
stream. The port's determinism rests on three declarations in the fixture:

- `serve_configuration.engine_env` (context 8192, batched tokens 8192, memory
  fraction 0.45, KV cache 4 GiB, offline HF/transformers);
- `serve_configuration.engine_args` (batch 1, prefix caching off,
  `kv_cache_dtype auto`, `logprobs_mode raw_logits`, seed 0);
- `serve_configuration.engine_args.deterministic_kernels` (the FlashInfer FP8
  scaled-MM kernel disabled).

`tools/determinism-probe.py` is the check: against a booted resident it must
report one stream. Run it after any kernel change.

## 5. What has been measured

| Measurement | Result | Where |
|---|---|---|
| Engine build in the pinned container | 2222 tensors, two boots of the same tree identical | organizer receipt |
| Candidate vs vendor runtime parity | 0.0 divergence over 384 tokens | organizer receipt |
| Resident boot + serve on a ranked box | boot completes, requests served, load and request exit 0 | organizer receipt |
| Ranked pair (control vs candidate) | not yet run in the local cycle | — |

## 6. Container and host facts that cost time

- **Boot is slow.** A cold resident boot on a Spark takes several minutes
  (about 400 s in the first runs) because the engine loads 24.7 GB of weights.
  Do not wrap a boot in a short foreground timeout; run it in the background
  and read its log.
- **The compile cache is per leg.** The resident runs with a fresh vLLM,
  Triton and inductor cache each leg, so a "warm" boot is not available by
  design.
- **GB10 exposes no NVML memory clock.** A kernel that tunes against memory
  bandwidth must fall back to a fixed figure (the base's patch uses 273 GB/s).
- **Unified memory means "free" is a moving number.** The box reports 121 GiB
  of unified memory with about 107 GiB free when idle; the resident's own cap
  is the fixture's `RESIDENT_GPU_MEMORY_UTILIZATION`.

## 7. Test state on a control host

- `python3 tools/lint-benchmark-manifest.py`: 30 checks pass, 3 FAIL
  (`live_goldens pool-9/10/11 have no pinned serial golden in
  timed_prompt_pool`). That is true until the organizer records the pool
  (CUDA-L09) and pins it; it is not a defect of this tree.
- `python3 -m unittest discover -s harness/vllm_resident`: 22 tests, OK
  (7 skipped).
- `python3 tools/check-doc-consistency.py`: passes; it is the tripwire that the
  participant documents and the manifest say the same thing.
- Red on a control host and not caused by this port (same failure sets on the
  base tree): `test-pool-refresh.sh`, `test-serve-up-boot-stop.sh`,
  `test-serve-up-plan-memory.sh`, `test-live-golden-rotation.sh`,
  `test-local-baseline.sh` — `mapfile` / `stat -c` need GNU userland — and the
  docker cases of `test-ranked-box-preflight-env.sh` section 8.

## 8. What is still open

1. **The pool recording** (CUDA-L09): the three serial goldens, recorded on a
   Spark and pinned in the fixture; until then the preflight refuses a ranked
   job by name.
2. **The reference pins**: `reference_engine_commit` names the tree
   that carries the fixture, and the fixture's `baseline_reference_commit`
   names the commit that first carries it, so the fixture pin is one organizer
   commit behind by construction. `tools/ranked-box-preflight.sh` accepts the
   pin as an ancestor of the staged tree, with the tree clean, and prints a WARN
   naming both.
3. **Box calibration** (dev cycle): every box needs a version-3 calibration at
   or under the fixture's `calibration_max_cv_percent` (1%).
4. **The ranked pair** (dev and prod cycles): no ranked job has run on this
   track, so any score quoted for it would be the base engine's, not this
   one's.
