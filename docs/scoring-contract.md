# Muse Glimmer 30B NVFP4 — scoring contract

> Last checked: 2026-11-20, against this repository's release tree and
> the benchd source that `tools/fetch-benchd.sh` pins (source commit
> `3101418`, published at dist commit `8ce0ca7` of the public `Layr-Labs/benchd` channel).

This page states how one ranked run of `muse-glimmer-30b-cuda-v1` becomes a
score, and when it becomes no score. Every line names its source. A source in
this repository is `file:line`. A source in benchd is `benchd <path>:<line>`,
a path in the pinned benchd source tree. When this page and the code
disagree, the code wins; section 1 of the
[participant contract](participant-contract.md) gives the full order of
authority.

## 1. The contract in nine lines

| # | Item | What the code does | Source |
|---|---|---|---|
| 1 | Formula | Each pair: `composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75`, with `gain = control leg seconds per token / candidate leg seconds per token`. The score is the arithmetic mean of the 3 per-pair composites. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:6-7,30-33`; `benchmark.json:47`; benchd `crates/bench-core/src/score.rs:42-51,88`; benchd `crates/benchd/src/official.rs:2022-2030,2136` |
| 2 | Direction | Higher is better: `direction` is `+`. A faster candidate leg has a gain above 1. | `benchmark.json:7`; benchd `crates/bench-core/src/score.rs:50` |
| 3 | Units | Inputs are seconds per token. Prefill: the prefill time divided by the 512 prompt tokens. Decode: the time of `free_decode_run(128)` divided by 128. A gain is a ratio of two seconds-per-token figures, so the gains and the score have no unit. | benchd `crates/bench-core/src/score.rs:25-37`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:46,52` |
| 4 | Rounding, ties | benchd does not round the score. `score.json` holds the full double, so two runs tie only when the two values are equal. benchd has no tie-break. When the organizer selects each team's best submission and the top N teams, ties go to the most recent submission. In the final re-score, complete teams with an equal mean share a rank (1, 1, 3); there is no further tie-break. | benchd `crates/benchd/src/official.rs:2136`; benchd `crates/benchd/src/score.rs:554-575` (only diagnostic fields are rounded); `docs/participant-contract.md:262-265,527-528,540-543`; `tools/router/router.py` `select_best` and `rescore_report` |
| 5 | Fail-closed correctness | Timed window: the prefill token, the seed token and each of the 128 committed decode tokens must equal the serial tape. One different token fails the run. Untimed correctness gate: each teacher-forced case of 64 steps passes with at most 10% of its steps not accepted (a top-logit tie within 1e-6 is accepted). Any failure seals `score: null` and `passed: false`. | benchd `crates/benchd/src/main.rs:5264-5265` and `crates/benchd/src/official.rs:1093-1096` (the fixture declares no `timed_token_tolerance_per_thousand`); benchd `crates/bench-runner/src/timing.rs:1031,1220,1284`; benchd `crates/bench-core/src/conformance.rs:448-493`; benchd `crates/bench-core/src/constants.rs:642`; benchd `crates/benchd/src/score.rs:24-27`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:45` |
| 6 | Estimator | Prefill: the one timed pass, after 1 unmeasured warm-up pass on the never-scored `warmup_golden` prompt; pair k decodes from the next live golden (`decode_golden_offset` 1). Decode: one 128-token free-run window per leg, one sample. Score: the arithmetic mean over the 3 pairs. Health band: the arithmetic mean of the calibration passes. | benchd `crates/bench-runner/src/timing.rs:1052-1068,1378-1386`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:48-49`; benchd `crates/benchd/src/official.rs:2136`; benchd `crates/benchd/src/baseline.rs:1034-1035` |
| 7 | Trial count | 3 pairs per job, so 6 legs: 3 serial-control legs and 3 candidate legs. Each leg: 1 warm-up and 1 timed prefill pass, then 1 decode window of 128 tokens. Calibration: 3 passes per live golden. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:6,46,48-49`; `docs/participant-contract.md:314-322,475-476`; `tools/calibrate-box.sh:93` |
| 8 | Max variance | Calibration only. The calibration refuses when the sample coefficient of variation (CV) of its passes is above `calibration_max_cv_percent` = 1.0% on either axis. A ranked run checks the CV in the calibration file against the same limit when it loads the file. The scored legs have no CV limit. Each control leg must be at most the calibration mean x 1.05 (prefill) and x 1.03 (decode); the low side is not checked. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:36-42`; benchd `crates/benchd/src/baseline.rs:157-171,436-475,514-530,961-965,1041-1051`; benchd `crates/benchd/src/main.rs:4903-4909` |
| 9 | Over-budget variance writes no score | A calibration over the CV limit writes no file, and with no valid file the preflight and benchd refuse the ranked run: no score. A control leg above its band refuses the run with `SERIAL-CONTROL-LEG-OUTSIDE-BAND` and seals `score: null`. The spread of the candidate legs is not checked: no run is refused for candidate variance. A pair whose decode gain is above the `decode_speedup_ceiling` of 5.0 refuses the run with `SPEEDUP-ABOVE-CEILING` and seals `score: null` (section 2). | benchd `crates/benchd/src/baseline.rs:1047-1051,460-475`; `docs/participant-contract.md:482-491`; benchd `crates/benchd/src/score.rs:26`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:21`; benchd `crates/benchd/src/official.rs:1695-1702,2070-2110` |

**The mode label.** `benchmark.json` `scoring.mode` is
`qwen-native-mtp-paired-decode-only`. It is a legacy compatibility value:
benchd writes that string into every `score.json`, and
`tools/lint-benchmark-manifest.py` requires the manifest to state the same
string, so readers can match the two. The name comes from the track family
that first ruled this paired regime. It does not describe this track's
model or modes: this track runs serial plus the pinned DFlash drafter. The label changes only when
benchd, `benchmark.json` and every reader of `score.json` adopt a new name
together.

## 2. The gates on every pair

The fixture declares `official_pair_combine: "mean"`, so every pair is scored
and every pair is gated against its own control leg. The first pair that fails
stops the run, and the run seals `score: null`
(benchd `crates/benchd/src/official.rs:1689-1700,2037-2068`).

**What a failure leaves behind.** A failed gate in this table happens after
measurement, so benchd seals a failure record: `score.json` with
`passed: false` and `score: null`, revealed and uploaded like a passing one.
A refusal before measurement (the preflight, a staged-asset or pin check,
setup, a missing calibration) writes no `score.json` at all; the job is red
and the log carries the refusal (`.github/workflows/benchmark.yml`, "TWO
FAILURE SHAPES"). Intake must accept both: no file is a refused run, a sealed
`score: null` is a failed run.

| Gate | Value | Source |
|---|---|---|
| Composite is finite | required | benchd `crates/benchd/src/official.rs:2032-2036` |
| Prefill gain floor | 0.95 | `fixtures/muse_glimmer_30b_cuda_v1_track.json:20` |
| Decode gain floor | 0.95 | `fixtures/muse_glimmer_30b_cuda_v1_track.json:19` |
| Control leg health band | at most mean x 1.05 (prefill), mean x 1.03 (decode) | `fixtures/muse_glimmer_30b_cuda_v1_track.json:36-42`; benchd `crates/benchd/src/baseline.rs:436-475` |
| Gain ceilings | decode at most 5.0, prefill at most 2.0 (equal passes) | `fixtures/muse_glimmer_30b_cuda_v1_track.json:21-22`; benchd `crates/bench-core/src/contract.rs:184-188`; benchd `crates/bench-core/src/score.rs:237-258` |
| Early-decode limit | seed prefill window per token at most 1.7 x the timed prefill per token, on each leg | `fixtures/muse_glimmer_30b_cuda_v1_track.json:23`; benchd `crates/benchd/src/official.rs:1438-1460,1598-1615,1957-1983` |

**The gain ceilings.** `benchmark.json:58-59` states `decodeSpeedupCeiling` 5.0
and `prefillSpeedupCeiling` 2.0.
benchd reads its ceilings from the fixture only
(benchd `crates/benchd/src/main.rs:5256-5257`), and the fixture declares
`decode_speedup_ceiling` 5.0 and `prefill_speedup_ceiling` 2.0
(`fixtures/muse_glimmer_30b_cuda_v1_track.json:21-22`).
benchd refuses a fixture whose ceiling is not finite or not above 1.0
(benchd `crates/bench-core/src/contract.rs:1254-1267,1383-1387`). Before the
floors and the band, benchd checks every pair against both ceilings. The first
pair whose decode gain is above 5.0, or whose prefill gain is above 2.0, stops
the run with `SPEEDUP-ABOVE-CEILING: pair <k> of 3: <decode|prefill> speedup
<gain> exceeds ceiling <ceiling>` and seals `score: null` (benchd
`crates/benchd/src/official.rs:1695-1702,1719-1728,2070-2110`). A gain equal to
its ceiling passes; a gain that is not finite fails (benchd
`crates/bench-core/src/score.rs:237-258`). The prefill ceiling caps any
prefill trick: with no ceiling, a prefill gain of 100 alone would multiply the
composite by 100 ^ 0.25, about 3.16.

**The token-tolerance field.** The fixture's `token_tolerance_budget` (0.0,
`fixtures/muse_glimmer_30b_cuda_v1_track.json:55`) is not read by benchd. The
exact rule of the timed window comes from the absent
`timed_token_tolerance_per_thousand` (row 5).

## 3. What the sealed score is

benchd `iterate --mode official` is the only writer of the sealed `score.json`
(`.github/workflows/benchmark.yml:114-117`). The file holds `score`, `passed`
and `metrics`. `score` is a finite number on a passed run and `null` on any
failure (benchd `crates/benchd/src/score.rs:24-27`). A job that is refused
before benchd measures, for example by the preflight or for a missing or
invalid calibration file (row 9), ends with no `score.json` at all
(`.github/workflows/benchmark.yml`, "TWO FAILURE SHAPES"). Intake must handle both
results: no file, and a sealed file whose `score` is `null`. Every pair stays in
`metrics.paired_legs` as measured
(`docs/participant-contract.md:254-256`).
