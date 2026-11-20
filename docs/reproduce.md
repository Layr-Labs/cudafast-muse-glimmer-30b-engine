# Reproduce the track's results

Organizer document. Each section gives the commands that re-create one input
of a ranked score, and the recorded result to compare against.

| Input | Section | Recorded result |
|---|---|---|
| benchd binaries | [1](#1-rebuild-benchd-from-source) | byte-identical rebuild of the pinned dist from its pinned source |
| Hidden goldens | [2](#2-re-record-the-hidden-goldens) | pinned in the fixture; the recording log is not in this repository |
| Box calibration | [3](#3-recalibrate-a-box) | not yet for this track: calibrating a box is a dev-cycle step |
| Ranked score | [4](#4-reproduce-a-ranked-score) | not yet for this track: no ranked job has run |

## 1. Rebuild benchd from source

benchd is pinned in `tools/fetch-benchd.sh`. Its `PINNED_COMMIT` is the dist
commit and `PINNED_SOURCE_COMMIT` the source commit that dist was built from;
`dist/linux-aarch64/benchd.manifest.json` in that dist pins each binary's
sha256 and bytes.

**Recorded result:** the engine base's organizers rebuilt the pinned dist from
its source on an x86_64 Linux host with Docker and no layer cache, and both
binaries matched the manifest byte for byte. `tools/fetch-benchd.sh` does not
rebuild: it fetches the pinned dist and verifies each binary against that
manifest. The source build command is not published in this repository.

## 2. Re-record the hidden goldens

The live pool is `pool-9`, `pool-10` and `pool-11`: three coding prompts, each
one user message rendered through the snapshot's pinned chat template with
thinking off, exactly 512 tokens. This track is serial only, so each prompt
has ONE serial golden — there are no per-depth oracles. The fixture pins every
file by sha256 and bytes: `timed_prompt_pool[]` pins the three serial goldens
and the hidden correctness pin repeats the default live golden's serial tape.

**Where the inputs live.** The prompt texts and token files are organizer
material and are never committed here, because sandboxed participant code can
read the ranked checkout. The organizer archive (a private repository; access
by the organizer) holds the texts, the token files, the goldens and
`SHA256SUMS` (see [pool refresh, "Where the goldens live"](pool-refresh-and-final-rescore.md#where-the-goldens-live)).

**Steps**, on two Sparks, each with this repository checked out, the snapshot
staged, the pinned benchd pair in `benchd-bin/`, and the engine built
(`./setup.sh`, which also stages the vLLM overlay). Use absolute paths:

```bash
export MLXFAST_TARGET_SNAPSHOT_DIR=<snapshot> BENCHD_BIN_DIR="$PWD/benchd-bin"
export SERVE_UP_SOCKET_DIR="$HOME/rsock"; mkdir -p "$SERVE_UP_SOCKET_DIR"   # serve-up.sh refuses /tmp
TOKENS="$HOME/rec/tokens"; OUT="$HOME/rec/goldens"; mkdir -p "$TOKENS" "$OUT"
```

1. Tokenize. The pinned image carries `tokenizers` and `jinja2`; the snapshot
   must be mounted at the same path:

   ```bash
   S=$MLXFAST_TARGET_SNAPSHOT_DIR
   docker run --rm --network none -v "$HOME:$HOME" -v "$S:$S:ro" -w "$PWD" \
     --entrypoint python3 docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c \
     tools/author-prompts.py --snapshot "$S" --out "$TOKENS" --chat \
       pool-9=pool-9.txt pool-10=pool-10.txt pool-11=pool-11.txt
   sha256sum "$TOKENS"/*.tokens.json   # record these with the generation
   ```

   `author-prompts.py --chat` refuses a rendering that is not exactly
   `seed_tokens` (512) ids; it never trims a chat prompt.

2. Record, under the box's GPU lock:

   ```bash
   flock /tmp/mtplx-gpu-exclusive.lock \
     tools/record-goldens.sh --prompts "$TOKENS" --out "$OUT" \
       --only pool-9 --only pool-10 --only pool-11
   ```

   One serial resident records the three serial goldens; the serial step is
   the only step this track runs, and `--steps serial` is implied. Every file
   is re-validated with `benchd validate-golden --contract`.

3. Compare. `(cd "$OUT" && sha256sum *.json)` on both boxes must give the same
   3 lines, and each sha256 and byte count must equal the fixture's pin once
   the pool is pinned.

**Recorded result:** the fixture pins the three pool goldens (`goldens.status`
is `RECORDED`, and each `timed_prompt_pool[]` entry carries `bytes` and
`sha256`). The recording log is held by the organizer, not in this repository.
The preflight refuses by name a fixture that still carries a
`MUSE-GLIMMER-CUDA-PENDING-ORGANIZER` sentinel.

## 3. Recalibrate a box

A calibration file is the box's health band for the serial-control leg.

1. Stage the reference tree at the fixture's `baseline_reference_commit`,
   from a local bundle or mirror (the ranked job holds no credential):

   ```bash
   REF="$HOME/muse-glimmer/reference"     # absolute path
   tools/stage-baseline-workspace.sh "$REF" --source <bundle-or-mirror>
   ```

2. Calibrate, with the box's runner name (the ranked job passes it as
   `--box`; a band under another name is refused):

   ```bash
   export MLXFAST_TARGET_SNAPSHOT_DIR=<snapshot> \
          MLXFAST_MUSE_GLIMMER_GOLDEN_DIR=<staged goldens> \
          MLXFAST_BASELINE_WORKSPACE="$REF" BENCHD_BIN_DIR=<benchd pair> \
          SERVE_UP_SOCKET_DIR="$HOME/rsock"
   NEW="$HOME/muse-glimmer/calibration.new.json"   # absolute path
   tools/calibrate-box.sh <runner name> "$NEW" --passes 3
   # for a pool that is not yet live (refresh, final re-score), add
   # --fixture <proposed fixture> so the bands are for the new prompts
   ```

   `calibrate-box.sh` takes the GPU lock itself; do not wrap it in `flock` of
   the same lock.

3. Check the file: `version` is 3, `reference_commit` and
   `benchd_source_commit` are the pins, there is one `prompts[]` entry per live
   golden, and every `prefill_cv` and `decode_cv` is at most
   `calibration_max_cv_percent` / 100 (0.01).
4. Install it at the path the runner's `MLXFAST_BASELINE_CALIBRATION` names,
   keep the old file under a dated name, and run
   `tools/ranked-box-preflight.sh` with the runner environment.

Recalibrate after any change to the live pool, the reference commit or benchd.

**Recorded result:** none yet. Calibrating a box is a dev-cycle step for this
track (`CUDA-D-*`); the local cycle stops at the owner's contract approval.
Until then a box has no band, and `tools/ranked-box-preflight.sh` refuses a
ranked job on it.

## 4. Reproduce a ranked score

Never re-type the benchd command: the workflow adds the GPU lock, the sandbox,
the cleanup and the nonce-bound reveal.

1. Dispatch `.github/workflows/benchmark.yml` on a fixed ref and record the
   run id and its `headSha`:

   ```bash
   gh workflow run benchmark.yml --repo <repo> --ref <ref>
   gh run list --repo <repo> --workflow benchmark.yml --branch <ref> --commit <sha> \
     --event workflow_dispatch --limit 1 --json databaseId,headSha,createdAt
   # check that createdAt is your dispatch time and headSha is <sha>
   ```

2. Wait for the run. Both jobs, the measurement step and **Sandbox cleanup**
   must succeed.
3. Read the score from the reveal block, which binds a per-job nonce to the
   file's sha256 and length (`MUSE-GLIMMER-SCORE-BEGIN` /
   `MUSE-GLIMMER-SCORE-END`):

   ```bash
   gh run view <run_id> --repo <repo> --log | cut -f3- > job.log
   python3 -c 'import sys, json; sys.path.insert(0, "tools/router"); import router; \
     print(json.dumps(router.score_from_log(open("job.log").read()), indent=1))'
   ```

   `cut -f3-` drops the job and step columns, leaving the timestamped lines
   the parser expects. `score_from_log` returns `None` when the block is
   missing or its bytes do not match the end line.
4. Download the artifact and check that its `score.json` has the sha256 and
   byte count of the reveal block's end line:

   ```bash
   gh run download <run_id> --repo <repo> -n benchmark-results-<run_id> -D art
   sha256sum art/score.json; wc -c < art/score.json
   grep -o 'MUSE-GLIMMER-SCORE-END .*' job.log | tail -1
   ```

**Recorded results.** None for this track: no ranked job has run on it. The
local cycle's own receipts (boot, parity, contract) are in the challenge
folder's `cuda/` tree, and the engine base's ranked runs are recorded in the
base repository. When this track's first ranked job runs, record the run id,
the box, the score and the reveal digest here.
