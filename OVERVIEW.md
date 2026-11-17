# Muse Glimmer 30B NVFP4 challenge: overview

A one-page summary of the challenge: what it is, how a submission is scored,
what was built and why, how it was tested, and what is left before launch.
Every point links to the document that has the detail.

## The challenge

- **Task.** Make `nvidia/Muse-Glimmer-30B-NVFP4` prefill and
  decode faster on a DGX Spark, single stream, without changing what it
  computes: the token-tolerance budget is `0.0`.
- **Track.** `muse-glimmer-30b-cuda-v1`, on 8 DGX Sparks
  (`spark-1` … `spark-8`).
- **What a team may edit.** Only `vllm/`, `harness/` and
  `mtp-head.manifest.json` (a declaration that must stay `none`: the track is
  serial only). Everything else is trusted and is replaced by the organizer's
  copy.
- **What a team may not change.** The numerical formats
  ([contract §3.4](docs/participant-contract.md)).
- **Where to start.** [TASK.md](TASK.md) is the one-page rules;
  [docs/participant-contract.md](docs/participant-contract.md) says why each
  rule exists.

## How a submission flows

1. **Intake (the intake service).**
   - Takes a team's archive and keeps only the editable paths.
   - Overlays them onto the armed base branch, replacing what is there.
   - Pushes a commit and dispatches `.github/workflows/benchmark.yml` at it.
   - [docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)
     lists the intake service's responsibilities.
2. **Checks.** A self-hosted runner on one Spark takes the job. It runs:
   - the manifest lint;
   - the static review (byte caps, surface);
   - the box preflight (staged assets, pins, no credentials);
   - the engine build, inside the sandbox.
3. **Measurement.** benchd (the pinned, trusted measurement binary) runs 3
   pairs. Each pair is a trusted serial-control leg, then the candidate leg,
   on the same box and the same prompt.
4. **Result.** benchd seals `score.json` during measurement. After sandbox
   cleanup, the job reveals it in a nonce-bound block on the job log; the
   intake service parses that block ([docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)
   §10).

## How the score is computed

- **Formula.** `composite = prefill_gain^0.25 × decode_gain^0.75`. Each gain
  is control time / candidate time, measured side by side in the same job
  ([contract §5.1](docs/participant-contract.md)).
  - **Why:** a stored baseline would drift with each box's temperature and
    age. Measuring the reference next to the candidate cancels that out.
- **Speed floor.** Both prefill and decode must reach at least 0.95× the
  reference, or the run fails the floor.
- **Correctness.**
  - The candidate's tokens must match the reference golden exactly: the
    token-tolerance budget is `0.0`.
  - There is one mode, so one tape: every leg replays the serial golden.
    benchd has no near-tie replay.
  - **Risk:** BF16 logit ties make an exact match brittle for honest engines.
    A kernel change that resolves a tie differently fails the run.
- **Box health band.** Each box is calibrated. A control leg that runs slower
  than its calibration (by more than 5% on prefill or 3% on decode) means the
  box is unhealthy, and no score is sealed.
  - **Why:** a slow or hot box must never decide a ranking. The bands were set
    from measured variation (3% on decode after 2% refused good runs).
- **Determinism.** The reference engine is made byte-reproducible: the
  fixture disables `FlashInferFP8ScaledMMLinearKernel`, because this
  checkpoint quantizes its attention projections to FP8 and that kernel is not
  batch-invariant, and every boot is pinned (context, batched tokens, memory
  fraction, seed), so the goldens are stable
  ([port log](docs/muse-glimmer-port-notes.md)).

## Security: what stops cheating

- **Sandbox**
  ([docs/sandbox-design.md](docs/sandbox-design.md)).
  - All participant code (build, kernels, adapter, resident) runs as a
    separate user, `nemosub`.
  - It has no network (not even loopback or DNS) and no cron/at. The
    firewall rule is re-applied at every boot, and the runner refuses to start
    without it.
  - It gets read-only access to the checkout and none to the runner's files
    or credentials.
  - Its leftovers are wiped at job start, before the timed measurement, after
    every candidate leg, between control-leg retry attempts, and at job end.
  - **Why:** without it, a submission could read the hidden prompts and
    goldens, or rewrite the measuring tools.
- **Sealed score.** `score.json` names each prompt only by its sha256. It
  carries no prompt text. The pinned benchd (source `b65af98`, bench pull
  requests 421 to 424) keeps engine-written text, audit fields, the hello
  identity and engine-derived numbers out of the sealed output: they are
  sealed as digests or omitted, and a full copy goes only to a private audit
  file on the box. The sealed output and the failure messages name no token
  value.
  - **Remaining risk:** whether a run failed, and the kind of failure, stay
    visible, so a failed run can still carry a very small signal.
- **A score that cannot be forged or erased from the log.** Participant code
  can print into the same job log, so the score is revealed only after
  sandbox cleanup, tagged with a per-job secret drawn before any participant
  code runs. Only that tagged block counts, and a run whose code ran but
  sealed no valid score is the team's failure, never infrastructure.
  - **Why:** otherwise a team could print a fake score line to void any run it
    did not like.
- **Hidden, rotating prompts.** Each ranked job measures every live golden:
  three hidden coding prompts, one pair per prompt.
  - The pool is replaced every 3–4 days.
  - Each prompt is a coding task, rendered as a user message in the model's
    own chat template.
  - **Why:** a prompt learned through timing side channels, and hard-coded,
    loses its value within days.
- **Final re-score.**
  - At the close, each of the top 10 teams' best submission is re-measured 5
    times (one run per slot), across boxes, on a held-out pool.
  - That pool was recorded before the contest and never staged on a box.
  - Each slot is its first run that scored or failed because of the submission
    (failure counts as 0.0); infrastructure runs are re-dispatched until they
    resolve or hit the retry cap (`1 + max-infra-retries` attempts per slot).
    One extra re-dispatch per slot is allowed after the first candidate boot
    failure (a second boot failure in that slot counts as 0.0).
    A run that fails because of the submission, or a reference-leg or band
    failure that persists after re-runs, counts as 0.0 for that slot.
  - The final ranking is the mean of those 5 slot values
    ([docs/pool-refresh-and-final-rescore.md](docs/pool-refresh-and-final-rescore.md)).
  - **Why:** anything learned during the contest is worth nothing at the end,
    day-to-day prompt luck does not pick the winner, and a team cannot drop
    slow runs by failing them on purpose.
- **One automatic re-run.** If the trusted reference engine itself diverges
  from its golden (a rare transient, about 1 in 100 legs), the whole
  measurement re-runs once from a clean sandbox, and only the second attempt
  counts ([contract §5.5](docs/participant-contract.md)).
  - The same single re-run covers a candidate that fails to boot before any
    of its timing exists (pair 1), which an honest engine can hit from a GPU
    or driver transient, and a pair-1 reference (serial-control) leg that
    fails to boot (PR #28).
  - A second divergence, or a second pair-1 reference boot failure, ends the
    job as an infrastructure outcome (exit 75). In the contest such a
    post-setup reference-leg or band failure counts as 0.0 toward selection;
    in the final re-score its slot is re-dispatched first. A reference boot
    failure in a later pair is final (`refused`).
  - **Why:** a team must never lose a run to a fault in our own reference.
  - **Why so narrow:** a submission cannot turn a crash or out-of-memory into
    a free second timing draw.

## How it was tested

### This track (local cycle, 2026-11-17)

| Test | Result |
|---|---|
| Engine tree builds in the pinned container (CUDA-L04B) | two boots of the same tree identical, 2222 tensors |
| Candidate vs vendor runtime (CUDA-LE5) | parity 0.0 over 384 tokens |
| Resident boot and serve on spark-3 (CUDA-LE2) | boot completes, requests served, load and request exit 0 |
| Reference vs candidate, ranked path | not yet run: the local cycle stops at the owner's contract approval (CUDA-LE4) |

The local cycle's numbers, commands and receipts are in the challenge folder
(`cuda/HANDOFF.md`, `cuda/artifacts/`). The engineering log records what the
port changed and every finding ([port log](docs/muse-glimmer-port-notes.md)).

### The engine base it derives from (2026-10-19/20, spark-1 to spark-4)

| Test | Result |
|---|---|
| Ranked run on the base repository (unchanged engine) | every step succeeded, `passed: true`, artifact uploaded; the base track's own score |
| Engine that returns wrong tokens | refused, no score |
| Calibration, 4 boxes, 3 prompts each | every CV under 1% (limit 1% at v9.1.1) |
| Goldens | byte-identical on two recording boxes and on every staged box |
| benchd | a no-cache rebuild from source equals the pinned dist byte for byte |
| GPU-free checks | adapter fmt, clippy, tests and build; resident and router tests; manifest lint; doc consistency; every `tools/test-*.sh` |
| 3-hour load test (spark-1, 3, 4; seeded mix; organizer router) | 22 jobs, 0 infrastructure failures; oversize file refused; trusted-path edits dropped; variants fail correctness (exact-match timed window) |

Those runs measured the base engine and its own goldens. This track re-proves
every check it inherits at its own pin (`cuda/HANDOFF.md`).

### Base-engine soak (all 8 Sparks, 4096-token seed, fork benchd)

| Test | Result |
|---|---|
| 8-hour load test (186 jobs, realistic submission mix) | unchanged engines score mean 0.9959, sd 0.0058; every variant ended as expected |
| Fairness (unchanged engine vs itself) | 19/19 pass, mean 0.9955, sd 0.0046 |
| Rotation smoke test and 3-hour soak (81 submissions, all 8 prompts) | unchanged engines 30/31 pass, mean 0.9959; garbage 10/10 rejected by replay; depth-7, oversize-file and trusted-path submissions refused or dropped; no infrastructure failures |
| Goldens | every live-pool and held-out golden recorded on two boxes each, byte-identical |
| Attack tests on the boxes | fake score lines in the build log ignored (real score kept); a 50 GiB `/dev/shm` flood by the sandbox user wiped and memory restored before the next reference leg; garbage rejected by replay |

These are the engine base's soak results, kept as the port's design evidence;
the full data is in the base repository's own engineering log.

## Fleet state

- **Boxes.** This track stages its boxes in the dev cycle. Re-verify each
  ranked box before launch: runner registration, the staged `pool-9/10/11`
  goldens and a version-3 calibration. spark-3 is the lane's box (idle,
  GPU free, 121 GB memory, 3.3 TB disk).
- **Branches.** `main` is the contest base. `official_scoring_enabled` is
  `true` in the fixture, and the reference tree is the fixture's
  `baseline_reference_commit`.
- **Where the goldens live.** Scoring reads only each Spark's local copy
  (`/srv/muse-glimmer/goldens`, unreadable by the sandbox user); a ranked job
  never fetches goldens. The live pool is also kept in an
  owner-only folder on the jump box, used to restage a box. The held-out
  final pool lives only in the organizer's private archive until the close.
  Goldens are never committed to this repository, because participant code can
  read the checkout ([pool refresh doc](docs/pool-refresh-and-final-rescore.md#where-the-goldens-live)).
- **Cost.** No GitHub-hosted runner is used, and CI is manual only. The
  ranked job uploads one artifact: the sealed `score.json` that Yukon reads.

## Before launch

| Item | Owner |
|---|---|
| Confirm which score source intake reads: the job log (as built) or the `benchmark.json` `scoreArtifact` upload | Intake service |
| Parse the job-log score exactly as `docs/submission-restriction-spec.md` §10 states (nonce-tagged block only) | Intake service |
| Confirm intake returns only the sealed score to teams | Intake service |
| Arm the base on opening day (`official_scoring_enabled: true`) | Organizer |
| Decide what participants see (a stripped copy without organizer docs, pool names and the sandbox threat model) | Organizer |

## Document map

| Question | Document |
|---|---|
| The rules | [TASK.md](TASK.md) |
| Why each rule exists | [docs/participant-contract.md](docs/participant-contract.md) |
| Engine wire protocol | [docs/resident.md](docs/resident.md) |
| Sandbox design | [docs/sandbox-design.md](docs/sandbox-design.md) |
| Intake responsibilities | [docs/submission-restriction-spec.md](docs/submission-restriction-spec.md) |
| Pool refresh and final re-score (runbook) | [docs/pool-refresh-and-final-rescore.md](docs/pool-refresh-and-final-rescore.md) |
| What CI checks | [docs/ci-coverage.md](docs/ci-coverage.md) |
| Engineering findings and test results | [docs/muse-glimmer-port-notes.md](docs/muse-glimmer-port-notes.md) |
| Pins, bands, pool, depths | `fixtures/muse_glimmer_30b_cuda_v1_track.json` |
