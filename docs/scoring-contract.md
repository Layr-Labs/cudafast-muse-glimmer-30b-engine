# Muse Glimmer 30B NVFP4 — scoring contract

> Last checked: 2026-11-20, against this repository's `main` at `2c89be0` and
> the benchd source that `tools/fetch-benchd.sh` pins (source commit
> `8ce0ca7`, published at `d0500fc` of the development bench channel).

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
| 1 | Formula | Each pair: `composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75`, with `gain = control leg seconds per token / candidate leg seconds per token`. The score is the arithmetic mean of the 3 per-pair composites. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:6-7,21-24`; `benchmark.json:47`; benchd `crates/bench-core/src/score.rs:42-51,88`; benchd `crates/benchd/src/official.rs:2022-2030,2136` |
| 2 | Direction | Higher is better: `direction` is `+`. A faster candidate leg has a gain above 1. | `benchmark.json:7`; benchd `crates/bench-core/src/score.rs:50` |
| 3 | Units | Inputs are seconds per token. Prefill: the prefill time divided by the 512 prompt tokens. Decode: the time of `free_decode_run(128)` divided by 128. A gain is a ratio of two seconds-per-token figures, so the gains and the score have no unit. | benchd `crates/bench-core/src/score.rs:25-37`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:37,43` |
| 4 | Rounding, ties | benchd does not round the score. `score.json` holds the full double, so two runs tie only when the two values are equal. benchd has no tie-break. When the organizer selects each team's best submission, equal means go to the most recent run. In the final ranking, equal means are ordered by team name, and each team gets its own rank. | benchd `crates/benchd/src/official.rs:2136`; `tools/router/router.py:458-461,637-645`; benchd `crates/benchd/src/score.rs:554-575` (only diagnostic fields are rounded); `docs/participant-contract.md:261-264,486` |
| 5 | Fail-closed correctness | Timed window: the prefill token, the seed token and each of the 128 committed decode tokens must equal the serial tape. One different token fails the run. Untimed correctness gate: each teacher-forced case of 64 steps passes with at most 10% of its steps not accepted (a top-logit tie within 1e-6 is accepted). Any failure seals `score: null` and `passed: false`. | benchd `crates/benchd/src/main.rs:5264-5265` and `crates/benchd/src/official.rs:1093-1096` (the fixture declares no `timed_token_tolerance_per_thousand`); benchd `crates/bench-runner/src/timing.rs:1031,1220,1284`; benchd `crates/bench-core/src/conformance.rs:448-493`; benchd `crates/bench-core/src/constants.rs:642`; benchd `crates/benchd/src/score.rs:24-27`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:36` |
| 6 | Estimator | Prefill: the median of the 5 timed passes, after 1 unmeasured warm-up pass. Decode: one 128-token free-run window per leg, one sample. Score: the arithmetic mean over the 3 pairs. Health band: the arithmetic mean of the calibration passes. | benchd `crates/bench-runner/src/timing.rs:1052-1068,1378-1386`; `fixtures/muse_glimmer_30b_cuda_v1_track.json:39-40`; benchd `crates/benchd/src/official.rs:2136`; benchd `crates/benchd/src/baseline.rs:1034-1035` |
| 7 | Trial count | 3 pairs per job, so 6 legs: 3 serial-control legs and 3 candidate legs. Each leg: 1 warm-up and 5 timed prefill passes, then 1 decode window of 128 tokens. Calibration: 3 passes per live golden. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:6,37,39-40`; `docs/participant-contract.md:304-312,434-435`; `tools/calibrate-box.sh:90` |
| 8 | Max variance | Calibration only. The calibration refuses when the sample coefficient of variation (CV) of its passes is above `calibration_max_cv_percent` = 1.0% on either axis. A ranked run checks the CV in the calibration file against the same limit when it loads the file. The scored legs have no CV limit. Each control leg must be at most the calibration mean x 1.05 (prefill) and x 1.03 (decode); the low side is not checked. | `fixtures/muse_glimmer_30b_cuda_v1_track.json:27-33`; benchd `crates/benchd/src/baseline.rs:157-171,436-475,514-530,961-965,1041-1051`; benchd `crates/benchd/src/main.rs:4903-4909` |
| 9 | Over-budget variance writes no score | A calibration over the CV limit writes no file, and with no valid file the preflight and benchd refuse the ranked run: no score. A control leg above its band refuses the run with `SERIAL-CONTROL-LEG-OUTSIDE-BAND` and seals `score: null`. The spread of the candidate legs is not checked: no run is refused for candidate variance. | benchd `crates/benchd/src/baseline.rs:1047-1051,460-475`; `docs/participant-contract.md:440-449`; benchd `crates/benchd/src/score.rs:26` |

## 2. The gates on every pair

The fixture declares `official_pair_combine: "mean"`, so every pair is scored
and every pair is gated against its own control leg. The first pair that fails
stops the run, and the run seals `score: null`
(benchd `crates/benchd/src/official.rs:1689-1700,2037-2068`).

| Gate | Value | Source |
|---|---|---|
| Composite is finite | required | benchd `crates/benchd/src/official.rs:2032-2036` |
| Prefill gain floor | 0.95 | `fixtures/muse_glimmer_30b_cuda_v1_track.json:20` |
| Decode gain floor | 0.95 | `fixtures/muse_glimmer_30b_cuda_v1_track.json:19` |
| Control leg health band | at most mean x 1.05 (prefill), mean x 1.03 (decode) | `fixtures/muse_glimmer_30b_cuda_v1_track.json:27-33`; benchd `crates/benchd/src/baseline.rs:436-475` |
| Gain ceiling | none applied (see below) | benchd `crates/bench-core/src/contract.rs:184-188`; benchd `crates/bench-core/src/score.rs:239-258` |

**The gain ceiling.** `benchmark.json:58` states `decodeSpeedupCeiling` 5.0.
benchd reads its ceilings from the fixture only
(benchd `crates/benchd/src/main.rs:5256-5257`), and the fixture declares no
`decode_speedup_ceiling` or `prefill_speedup_ceiling`. benchd skips a ceiling
that is not declared, so no gain ceiling is applied today.

**The token-tolerance field.** The fixture's `token_tolerance_budget` (0.0,
`fixtures/muse_glimmer_30b_cuda_v1_track.json:46`) is not read by benchd. The
exact rule of the timed window comes from the absent
`timed_token_tolerance_per_thousand` (row 5).

## 3. What the sealed score is

benchd `iterate --mode official` is the only writer of the sealed `score.json`
(`.github/workflows/benchmark.yml:114-120`). The file holds `score`, `passed`
and `metrics`. `score` is a finite number on a passed run and `null` on any
failure (benchd `crates/benchd/src/score.rs:24-27`). A job that is refused
before benchd measures, for example by the preflight or for a missing or
invalid calibration file (row 9), ends with no `score.json` at all
(`.github/workflows/benchmark.yml:118-120`). Intake must handle both
results: no file, and a sealed file whose `score` is `null`. Every pair stays in
`metrics.paired_legs` as measured
(`docs/participant-contract.md:253-255`).

**The scoring mode label.** `benchmark.json` `scoring.mode` is
`qwen-native-mtp-paired-decode-only`. It is a legacy compatibility value:
benchd writes this string into every `score.json`, and
`tools/lint-benchmark-manifest.py:107` requires it. It does not describe this
track, which is serial only and has no MTP head.
