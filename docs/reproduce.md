# Reproduce the track's results

Organizer document. Each section gives the commands that re-create one input
of a ranked score, and the recorded result to compare against.

| Input | Section | Recorded result |
|---|---|---|
| benchd binaries | [1](#1-rebuild-benchd-from-source) | byte-identical rebuild of the pinned dist from its pinned source, from the public repository with no token |
| Hidden goldens | [2](#2-re-record-the-hidden-goldens) | recorded: the fixture pins the three serial goldens (`timed_prompt_pool`); the recording log is not in this repository |
| Box calibration | [3](#3-recalibrate-a-box) | per box: each calibration file stays on its box; none is published |
| Ranked score | [4](#4-reproduce-a-ranked-score) | none published yet for this track |

## 1. Rebuild benchd from source

benchd is pinned in `tools/fetch-benchd.sh`. Its `PINNED_COMMIT` is the dist
commit and `PINNED_SOURCE_COMMIT` the source commit that dist was built from;
`dist/linux-aarch64/benchd.manifest.json` in that dist pins each binary's
sha256 and bytes.

The public channel ships the verified binaries and their manifest.
`tools/fetch-benchd.sh` only downloads those binaries and checks them against
that manifest; it rebuilds nothing. The same public repository also holds the
pinned source commit, so a rebuild needs no token and no private repository.
To reproduce the binaries, build the pinned source in the bench repository's
own reproducible container (`scripts/build-dist-linux.sh`, which uses
`deploy/Dockerfile.dist-linux`) and compare the hashes. Run this from this
repository's root on a host with Docker and network access to GitHub; the pins
are read from `tools/fetch-benchd.sh`, never retyped:

```bash
src="$(sed -n 's/^PINNED_SOURCE_COMMIT="\(.*\)"$/\1/p' tools/fetch-benchd.sh)"
dist="$(sed -n 's/^PINNED_COMMIT="\(.*\)"$/\1/p' tools/fetch-benchd.sh)"
repo="$(sed -n 's|^BASE_URL=.*raw.githubusercontent.com/\([^}]*\)}"$|\1|p' tools/fetch-benchd.sh)"
work="$(mktemp -d)"
git clone "https://github.com/${repo}.git" "${work}/bench"
git -C "${work}/bench" checkout "${src}"
git -C "${work}/bench" show "${dist}:dist/linux-aarch64/benchd.manifest.json" > "${work}/manifest.json"
(cd "${work}/bench" && BENCHD_DIST_REPO="https://github.com/${repo}.git" \
  BENCHD_DIST_BRANCH="$(jq -r .branch "${work}/manifest.json")" \
  BENCHD_DIST_NO_CACHE=1 BENCHD_DIST_OUT="${work}/rebuild" \
  ./scripts/build-dist-linux.sh "${src}")
jq -r '.binaries | to_entries[] | "\(.value.sha256)  \(.key)"' "${work}/manifest.json" > "${work}/SHA256SUMS"
(cd "${work}/rebuild" && shasum -a 256 -c "${work}/SHA256SUMS")
```

The checkout is detached, so `BENCHD_DIST_BRANCH` gives the build the branch
name the dist manifest records. The last command must print `OK` for `benchd`
and for `record-correctness-golden`. Run the build on an aarch64 host, or set
`BENCHD_DIST_TARGET` only to check another platform's directory. Without a
token the build script prints a warning that a private repository would fail;
the public repository needs none.

**Recorded result:** on 2026-11-20 the pinned source was rebuilt with these
commands from the public repository, with no token and no layer cache, on an
aarch64 Linux Docker engine (Docker 29.5.2 on an Apple silicon Mac). Both
binaries matched the pinned dist manifest byte for byte. Repeat the rebuild
whenever the pin moves.

## 2. Re-record the hidden goldens

The live pool is `pool-9`, `pool-10` and `pool-11`: three coding prompts in
the raw-continuation form: the first 512 ids of each text with the pinned
tokenizer, no chat template and no special ids. This track is serial only,
so each prompt has ONE serial golden — there are no per-depth oracles. The
fixture pins every file by sha256 and bytes: `timed_prompt_pool[]` pins the
three serial goldens and the hidden correctness pin repeats the default live
golden's serial tape.

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

1. Tokenize. The pinned image carries `tokenizers`; the snapshot must be
   mounted at the same path:

   ```bash
   S=$MLXFAST_TARGET_SNAPSHOT_DIR
   docker run --rm --network none -v "$HOME:$HOME" -v "$S:$S:ro" -w "$PWD" \
     --entrypoint python3 docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c \
     tools/author-prompts.py --snapshot "$S" --out "$TOKENS" \
       pool-9=pool-9.txt pool-10=pool-10.txt pool-11=pool-11.txt
   sha256sum "$TOKENS"/*.tokens.json   # record these with the generation
   ```

   Without `--chat`, `author-prompts.py` keeps the first `seed_tokens` (512)
   ids of each text and refuses a text that is shorter. Do not pass `--chat`:
   a chat-template rendering adds special ids that the live pool does not
   have.

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

**Recorded result:** the pool is recorded. `timed_prompt_pool[]` pins the
serial golden of `pool-9`, `pool-10` and `pool-11` by bytes and sha256
(`goldens.status` is `RECORDED`), and the fixture carries no
`MUSE-GLIMMER-CUDA-PENDING-ORGANIZER` sentinel (the preflight refuses a ranked
job by name while one is present). The recording log is held by the
organizer, not in this repository. A re-recording must reproduce those pins
byte for byte.

### 2b. Record the warm-up golden

The fixture's `warmup_golden` is one more golden, never scored: every
warm-up pass runs its prompt. It is recorded the same way, from its own
organizer prompt `warmup.txt`, which must differ from every live prompt
(benchd refuses a warm-up prompt or seed equal to a live one with
`PROMPT-REUSE-REFUSED`):

```bash
tools/author-prompts.py --snapshot "$S" --out "$TOKENS" warmup=warmup.txt   # in the pinned image, as in step 1; raw form, no --chat
flock /tmp/mtplx-gpu-exclusive.lock \
  tools/record-goldens.sh --prompts "$TOKENS" --out "$OUT" --only warmup --steps serial
(cd "$OUT" && sha256sum warmup.golden.json && wc -c < warmup.golden.json)
```

Both boxes must give the same sha256. `tools/calibrate-box.sh` refuses the
warm-up (`warmup-form-mismatch`) when its prefill length differs from the live
goldens' or when it holds a special or chat-template id that none of them
holds (`tools/check-warmup-form.sh`, with the snapshot's pinned
`tokenizer.json`). Upload the file to the R2 key the
fixture's `warmup_golden.r2_path` names, then write its sha256 and bytes into
`warmup_golden`; a `MUSE-GLIMMER-CUDA-PENDING-ORGANIZER` sentinel there makes
`tools/ranked-box-preflight.sh` refuse a ranked job. Stage the file on every
box with `tools/fetch-goldens.sh --all`, then recalibrate every box
(section 3): the warm-up prompt and `decode_golden_offset` change what a
calibration measures.

**Recorded result:** the warm-up golden is recorded on a Spark in the
raw-continuation form of the live pool (512 prefill ids, 0 special ids), and
the fixture pins it: sha256
`1f9c9d81c7478afd3fbd10a88f4c742afe6e220399ef6c3bacd90d8a83618756`, 35014
bytes.

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

**Recorded result:** none published. Each calibration file stays on its box.
A box with no valid version-3 file has no band, and
`tools/ranked-box-preflight.sh` refuses a ranked job on it.

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

**Recorded results.** None published for this track yet. The engine base's
ranked runs are recorded in the base repository. When a ranked job of this
track is published, record the run id, the commit, the score and the reveal
digest here.
