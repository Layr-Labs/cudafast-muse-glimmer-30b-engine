# Muse Glimmer 30B NVFP4 on DGX Spark: inference speed challenge

A ranked engineering challenge: make NVIDIA's `Muse-Glimmer-30B-NVFP4`
prefill and decode faster on a DGX Spark, single stream, without changing what
the model computes.

**[Overview](OVERVIEW.md)** · **[Task and rules](TASK.md)** ·
**[Participant contract](docs/participant-contract.md)** ·
**[Scoring contract](docs/scoring-contract.md)** ·
**[Sandbox design](docs/sandbox-design.md)** ·
**[Pool refresh and final ranking](docs/pool-refresh-and-final-rescore.md)**

## At a glance

| | |
|---|---|
| **Model** | [`nvidia/Muse-Glimmer-30B-NVFP4`](https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4), pinned by revision and per-file sha256 |
| **Hardware** | NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) |
| **Engine** | vLLM v0.27.1 in a pinned container; teams edit `vllm/` and `harness/` |
| **Score** | `prefill_gain^0.25 × decode_gain^0.75`, each gain measured against a trusted reference run on the same box, same prompt, same job |
| **Correctness** | the timed window must match the reference golden token for token: the fixture declares no timed token tolerance (contract section 5.4) |
| **Fairness** | run-to-run spread is small: the engine base's reference against itself scored 0.996, sd 0.005 (8-box soak of the base engine); this track's own null control is an organizer step and is not published here |
| **Integrity** | sandboxed, network-less execution; hidden prompts rotated every 3–4 days; final ranking re-measured on a held-out pool |
| **Track** | `muse-glimmer-30b-cuda-v1` |

[OVERVIEW.md](OVERVIEW.md) explains each of these in one page: how scoring
works, what prevents cheating, and how the system was tested.

## What runs

```text
benchd (pinned prebuilt, trusted)
  │  spawns one per leg, Engine Protocol v1 (NDJSON over stdio)
  ▼
.build/release/mlxfast-runtime-worker      ← harness/protocol-adapter (Rust, editable)
  │  NDJSON over a Unix socket
  ▼
vLLM resident, one per leg                 ← harness/vllm_resident (Python, editable)
  │  runs inside the pinned container: --network none, snapshot and tree read-only
  ▼
vLLM v0.27.1: compiled extensions from the pinned image,
Python/Triton source from vllm/vllm/       ← vendored (editable)
  ▼
the pinned safetensors snapshot            ← trusted, verified by bytes + sha256
```

- **The resident loads the model once per leg.** benchd spawns one adapter per
  leg for the warm-up, timed prefill, timed decode and correctness phases (on
  the last pair a second one for the pinned correctness golden), and each
  adapter connects instead of loading.
- **benchd boots each leg's resident itself,** through that leg's own
  `tools/serve-up.sh --boot`, and stops it with `--stop`.
- **The adapter reports raw counters only.** benchd does all timing and
  scoring.

## Setup

The toolchain needed:

- cargo
- docker with the NVIDIA container runtime
- jq, python3, curl, sha256sum

A DGX Spark has all of them after the standard DGX OS setup plus rustup.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchd from its release channel and verifies
the bytes against the channel manifest. The channel is the `main` branch of the
public `Layr-Labs/benchd` repository; the fetch needs no token. A verified pair
in `benchd-bin/` or in `BENCHD_DIST_LOCAL` needs no network.

```bash
./setup.sh
```

This command runs `tools/engine/build.sh` and then verifies the target
snapshot. The engine build does four things:

1. builds the adapter and stages it at `.build/release/mlxfast-runtime-worker`;
2. pulls the pinned image `vllm/vllm-openai:v0.27.1` by its arm64 digest;
3. stages `.build/vllm-overlay`: the image's installed vLLM, with this tree's
   `vllm/vllm/` copied over it;
4. builds `harness/kernels/` in the container when
   `harness/kernels/build.sh` exists.

On a ranked box the adapter build (and any `build.rs`) and the kernels build
run as an unprivileged sandbox user with no network. Cargo dependencies must
come from crates.io, and are vendored before an offline build; `build.sh` may
write only to `$KERNELS_OUT`. [The contract, section
3.6](docs/participant-contract.md#36-where-your-code-runs) lists the rules.

The snapshot step downloads the 14 pinned files (about 24.7 GB) into
`reference_weights/Muse-Glimmer-30B-NVFP4/`. Each file
must match its pinned byte count and sha256. On a ranked box the snapshot is
pre-staged and named by `MLXFAST_TARGET_SNAPSHOT_DIR`, and `setup.sh` only
verifies it.

```bash
./tools/local-baseline.sh
```

This command boots a serial resident and runs benchd's local-iterate leg over
the public golden `correctness_prompts/public-longcopy-gate-english-512.golden.json`.
The organizer recorded that golden on a Spark with `tools/record-goldens.sh`,
and this tree ships it. `tools/local-baseline.sh` checks its sha256 and byte
count before any run work. The track is serial only, so one golden covers
every local run. A local run seals
`score: null`: only the paired ranked path produces a score.

```bash
yukon setup
```

This command runs `setupCommand`: `./tools/fetch-benchd.sh`, then
`./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`. Off a ranked box it runs
`./tools/local-baseline.sh` and seals `score.json` with `"score": null`. Only
the ranked runner gives a score.

## Iterating without a GPU

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident -v
```

These commands test the adapter against its mock engine, and the resident's
wire and bookkeeping against a fake backend. `test_adapter_e2e` drives the real
adapter binary against the real resident on the serial path.

```bash
tools/three-flows-dry-run.sh
```

This command runs the real benchd binaries (the pinned `main` channel) over the
same fake resident. It covers golden authoring, a local serial leg and the
paired lifecycle. An MTP leg is a positive case only when the fixture's
`allowed_modes` names mtp; on this serial-only track the script checks that an
MTP declaration is refused. Its `B4` check (the sealed vllm-resident identity)
fails against the pinned benchd, which seals `engine-text-withheld` in that
field.

## Repository structure

| Path | Trust | What it is |
|---|---|---|
| `vllm/vllm/` | editable | vLLM v0.27.1 Python source (`vllm/VENDOR.json` records the base) |
| `harness/protocol-adapter/` | editable | Engine Protocol v1 adapter (Rust) |
| `harness/vllm_resident/` | editable | the resident: `resident.py` (wire, bookkeeping), `backends.py` (vLLM + fake) |
| `harness/kernels/` | editable, optional | custom CUDA/C++ kernels built by `harness/kernels/build.sh` |
| `mtp-head.manifest.json` | editable | the speculation declaration; on this serial-only track it must stay `"source": "none"` |
| `fixtures/muse_glimmer_30b_cuda_v1_track.json` | trusted | the track contract: pins, engine configuration, serve depths, scoring semantics |
| `benchmark.json` | trusted | editable paths, commands, byte budget, scoring constants |
| `setup.sh`, `benchmark.sh`, `tools/` | trusted | build, serve, preflight, measure, calibrate, stage, organizer tools |
| `correctness_prompts/` | trusted | the public prompt text and the public golden |
| `.github/` | trusted | CI (`ci.yml`) and the ranked pipeline (`benchmark.yml`) |
| `docs/` | — | the contract, the resident wire, CI coverage, the sandbox design, the submission restriction spec, pool refresh and final re-score, the engineering log |

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

<!-- YUKON-CLI-INSTALL: owner to supply the public install source -->
Install the `yukon` CLI first; its install source is published with the challenge.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login
```

This command signs you in to the default Yukon API. It prints a URL and an
eight-character code; open the URL in any browser, check the request time and
IP address it shows, and approve with your GitHub account. The command then
finishes by itself, also over SSH. Approve only a code from a login you
started.

If device sign-in is not available, create an API key on the Yukon website's
API keys page and put it in the environment instead of on the command line, so
it stays out of your shell history:

```bash
read -rs YUKON_API_TOKEN && export YUKON_API_TOKEN
```

```bash
yukon clone cudafast-muse-glimmer-30b
```

This command clones the benchmark repository.

Before you submit, remove build artifacts from the editable paths. The archive
packages `vllm/`, `harness/` and `mtp-head.manifest.json` as they sit on disk.
The byte budget (`editableSurfaceByteBudget.maxTotalBytes` in
`benchmark.json`) is 41,943,040 bytes, and the static review enforces it. A
stale adapter `target/` tree or a `__pycache__/` directory uses that room and
counts against the budget. To see how much of it the tracked editable files
use, run:

```bash
git ls-files -z vllm harness mtp-head.manifest.json | xargs -0 cat | wc -c
```

```bash
cargo clean --manifest-path harness/protocol-adapter/Cargo.toml
find vllm harness -name __pycache__ -type d -prune -exec rm -rf {} +
```

These commands remove the adapter's build tree and Python bytecode caches.
`./setup.sh` and the ranked box rebuild what they need from source.

```bash
yukon submit --model "<exact model name>" --harness "<harness name>" --note-file submission-note.md
```

This command uploads your editable-path archive. The note is public markdown.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. The static review enforces
the byte budget and refuses symlinks and special files, and the intake keeps
only the editable paths
([docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)). `yukon submit` does not run a local test first, and no local
run blocks the upload. Run `yukon run` yourself before you submit.

## The pinned artifacts

| Artifact | Pin |
|---|---|
| Target | `nvidia/Muse-Glimmer-30B-NVFP4` @ `47818374517751c48c55cde2621594926b1888b6`, 14 files by bytes + sha256 (24.7 GB) |
| Speculation | none: the checkpoint ships no draft tensors and the fixture's `allowed_modes` is `[serial]` |
| Engine source | vLLM `v0.27.1` @ `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Engine image | `docker.io/vllm/vllm-openai:v0.27.1` @ `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c` (arm64) |
| benchd | `Layr-Labs/benchd` dist `8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729`, built from source `3101418f95eb074cc9c100f8dc61cc85cc924155` (`PINNED_COMMIT` and `PINNED_SOURCE_COMMIT` in `tools/fetch-benchd.sh`); the dist's `dist/linux-aarch64/benchd.manifest.json` pins each binary's sha256 and bytes |
| Goldens | `timed_prompt_pool[]` (the 3 serial goldens that `live_goldens` names: `pool-9`, `pool-10`, `pool-11`, three 512-token raw-continuation coding prompts: the first 512 ids of a text, no chat template and no special ids) and the hidden correctness pin in the fixture. The fixture pins each pool golden's `sha256` and `bytes`; each is staged on every box and never committed (see [OVERVIEW.md](OVERVIEW.md)) |

## The ranked run

`.github/workflows/benchmark.yml` is dispatched once per submission. It runs
two jobs:

1. **A `surface-check` on a Spark runner** (no GPU, no credential). It first
   checks the protected base: the diff from the merge base with the default
   branch to the dispatched commit must stay inside that base's
   `editablePaths` (`.github/scripts/enforce-modifiable-surface.sh`, run as
   the base holds it). This check is a second line. The primary guard is the
   Yukon intake, which builds an entry from the base plus only the
   `editablePaths` files. The check runs in the workflow of the dispatched
   commit, so a commit that changes that workflow can remove it. Then it runs the manifest lint and the static review of
   the editable surface, which enforces the byte budget and refuses symlinks
   and special files.
2. **A `ranked` job on a self-hosted Spark** labelled
   `[self-hosted, Linux, ARM64, muse-glimmer-30b-cuda-v1]`. It
   runs these steps in order:
   - `tools/ranked-box-preflight.sh`
   - `./setup.sh`
   - `tools/draw-live-golden.sh`: the fixture declares `live_goldens`, so
     nothing is drawn. Every job measures every live golden, one pair per
     prompt; `score.json` identifies a golden only by its sha256.
   - under the box-wide GPU lock and a quiescence wait,
     `tools/muse-glimmer-measure-and-score.sh`, which execs
     `benchd iterate --mode official` and seals `score.json`
   - after sandbox cleanup (no participant process remains), **Reveal the
     sealed score** prints a commit-reveal block to the job log: a per-job
     nonce drawn before sandbox start (runner-private, never printed earlier),
     the exact bytes of `score.json` if present, and an end line binding
     nonce to the file's sha256 and byte length (or `none` when absent).
   - **Upload score artifact** then uploads that same `score.json` as
     `benchmark-results-<run_id>/score.json` (the `benchmark.json`
     `scoreArtifact`), only when the reveal step printed it after a clean
     sandbox cleanup. The reveal block stays the tamper-evident record
     ([docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)
     §10).

The ranked job holds no credential. Goldens, the reference workspace, the
calibration file and the benchd pair are staged on the box out of band.

Participant code (the builds, the resident and the adapter) runs as an
unprivileged sandbox user with no network, reading the checkout read-only and
writing only its own per-job and per-leg directories. The adapter's stderr is not
returned, the score record carries no engine-supplied text, and nothing
persists between jobs or legs.

### The ranked box toolchain

| | |
|---|---|
| **Hardware** | NVIDIA DGX Spark (GB10, aarch64) |
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.17.0-1026-nvidia |
| **Driver** | NVIDIA 580.159.03 (the fixture's `driver_min`), CUDA driver API 13.0; Docker 29.2.1, NVIDIA Container Toolkit 1.19.1 |
| **Engine image** | `docker.io/vllm/vllm-openai:v0.27.1` @ `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c` (Ubuntu 22.04.5 LTS) |
| **In the image** | vLLM 0.27.1, PyTorch 2.13.0+cu130, FlashInfer 0.6.16.post3, Python 3.12.13; CUDA 13.0 (nvcc V13.0.88), gcc 11.4.0 |
| **CUDA arch** | `sm_121` (the fixture's `serve_configuration.toolchain.cuda_arch`) |
| **Rust on the host** | 1.98.1 (`1.98.1-aarch64-unknown-linux-gnu`), offline `cargo build --release --locked` on vendored crates.io dependencies |

The host has no nvcc. `harness/kernels/build.sh`, when present, runs inside the
pinned image. The adapter (`harness/protocol-adapter`) builds on the host. The
vLLM Python and Triton source in `vllm/vllm/` runs on the compiled extensions
of the pinned image.

The hosted CI does not run the engine on a GPU. A green hosted CI does not
prove that your kernels compile for `sm_121` with nvcc 13.0, or that the
resident boots. The ranked job builds your candidate in `./setup.sh`; a build
failure there fails the submission before any scoring.

## Organizer runbook

1. **Stage each Spark.**
   - `tools/box-runner/install.sh` sets up the runner supervisor
     (`tools/box-runner/README.md`).
   - `sudo tools/box-runner/setup-sandbox-user.sh` creates the sandbox user
     once per box ([sandbox design, section 6](docs/sandbox-design.md#6-operator-steps)).
   - Pre-pull the engine image by digest.
   - Stage the snapshot and export `MLXFAST_TARGET_SNAPSHOT_DIR`.
   - Stage benchd with `tools/fetch-benchd.sh`.
2. **Stage the goldens** on every box. The fixture pins every golden of the
   live pool (one serial golden per prompt, no per-depth oracles);
   `tools/pool-refresh.sh` records, stages and pins a new generation, and
   a box is restaged from the jump box copy
   ([where the goldens live](docs/pool-refresh-and-final-rescore.md#where-the-goldens-live)).
3. **Stage the reference tree** on every box with
   `tools/stage-baseline-workspace.sh`, at the fixture's
   `baseline_reference_commit`.
4. **Calibrate every box** with `tools/calibrate-box.sh <runner name> <out>`
   (3 passes; the box name must be the box's runner name, which the ranked
   job passes as `--box`). Every per-prompt CV must be at most the fixture's
   `calibration_max_cv_percent` (1%), or the file gets no band
   ([reproduce, section 3](docs/reproduce.md#3-recalibrate-a-box)).
5. **Arm the track:** set `official_scoring_enabled: true` once the null
   control lands at about 1.000. The shipped fixture is armed: the flag is
   `true` ([contract, section 5.5](docs/participant-contract.md#55-arming)).
6. **Run the contest.** The intake service dispatches `benchmark.yml` once per
   submission. Every 3 to 4 days, replace the live pool; at the close, freeze
   and run the final re-score on the held-out pool. Both procedures are in
   [`docs/pool-refresh-and-final-rescore.md`](docs/pool-refresh-and-final-rescore.md).
   `tools/router/` is the organizer's own dispatcher, used for load tests and
   for the final re-score (`tools/router/README.md`).

[`docs/muse-glimmer-port-notes.md`](docs/muse-glimmer-port-notes.md) (the
engineering log) records the cluster verification and the load-test findings.

## Documentation

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, engine configuration, scoring semantics | `fixtures/muse_glimmer_30b_cuda_v1_track.json` |
| The score formula, direction, units, rounding, correctness gate and variance limits, each with its source | `docs/scoring-contract.md` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |
| What CI covers | `docs/ci-coverage.md` |
| How participant code is sandboxed on a box | `docs/sandbox-design.md` |
| What a submission may contain, and who enforces it | `docs/submission-restriction-spec.md` |
| Pool refresh and final re-score | `docs/pool-refresh-and-final-rescore.md` |
| How this track was derived, and what was measured | `docs/muse-glimmer-port-notes.md` (engineering log) |
| How to reproduce benchd, the goldens, a calibration and a ranked score | `docs/reproduce.md` |
| What a measured run executes | benchd, resolved from the pinned channel by `tools/fetch-benchd.sh` |
