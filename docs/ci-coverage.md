# What CI covers

`.github/workflows/ci.yml` runs on manual dispatch only (`workflow_dispatch`):
its jobs need GitHub-hosted runners, so run the same checks locally before a
push (`AGENTS.md`, "Checks before you submit"). It holds no credential.

## `lint` (x86 ubuntu)

| Step | What it proves |
|---|---|
| egress scan | no workflow references a secret, inherits one, asks for write permission, or uploads anything but the ranked `score.json` (the ranked workflow uploads that one file, after sandbox cleanup) |
| actionlint | the workflows parse and type-check (pinned binary, sha256-verified) |
| shellcheck | every tracked `*.sh` outside `vllm/` passes at warning severity |
| manifest lint | `benchmark.json` against the fixture: schema, paths, commands, scoring constants (the decode and prefill ceilings included, `tools/test-lint-scoring-limits.sh`), byte-budget fallbacks, per-leg serve verbs; a tree without the optional `mtp-head.manifest.json` passes the lint and the whole-surface static review, and the scoring reader still refuses an invalid one (`tools/test-optional-editable-path.sh`) |
| benchd platform / recorder | `tools/fetch-benchd.sh` refuses a wrong platform or an unverified second binary |
| adapter | `cargo fmt`, `clippy -D warnings` and `cargo test` (mock and scripted sessions) |
| docs agree with the manifest | `tools/check-doc-consistency.py`, `tools/test-check-doc-odd-names.sh` (a tracked name with a space or a newline does not break it) |
| participant engine edit | the editable list, benchd's harness-hash roster, and that a `vllm/vllm/` edit reaches the staged overlay while compiled extensions survive |
| serve-up boot/stop, memory plan | the real `serve-up.sh` and the real resident (fake backend) |
| facade, local baseline, setup onboarding | the entry points, through a real resident boot |
| yukon run | `benchmarkCommand` reaches the ranked script on a ranked box and `tools/local-baseline.sh` off one |
| spec declaration | the declaration envelope |
| modifiable-surface gate | `tools/test-enforce-modifiable-surface.sh`: `.github/scripts/enforce-modifiable-surface.sh` refuses a change outside the base's `editablePaths`, a self-widened `benchmark.json`, and every spelling of `benchd-bin` / `benchd.pin` (name, case, same inode), and a protected file renamed into `editablePaths` (the diff runs with `--no-renames -z`), a committed gitlink, and an optional editable path that is not a regular file; the ranked workflow runs it from the merge base |
| ranked preflight | every environment refusal, and the section 8 paired-baseline refusals on a scaffolded box |
| calibrator, stager, golden fetcher, box runner | their mechanics, with stub collaborators |
| warm-up form | `tools/test-check-warmup-form.sh`: `tools/check-warmup-form.sh` refuses a warm-up golden whose prefill length differs from the live goldens' or that holds a special or chat-template id none of them holds, and passes a raw-form one; `tools/test-calibrate-box.sh` case 13: the calibrator refuses a chat-form warm-up by name |
| sandbox | `tools/test-sandbox-kill.sh`, `tools/test-sandbox-names.sh` (one sandbox user per runner; a shared one is refused), `tools/test-sandbox-grant-checkout.sh`, `tools/test-sandbox-wipe.sh` (with its failure cases), `tools/test-sandbox-wipe-blocked-scan.sh` (a path participant code makes unreadable is wiped or fails the wipe, never passed), `tools/test-run-engine-sandboxed.sh`, `tools/test-sandbox-leg-isolation.sh` (no file one candidate leg writes is readable by the next; a leftover refuses the leg), `tools/test-gpu-lock-sandbox.sh` (as a real second user: the box GPU lock is neither writable nor readable by the sandbox user, and a sandbox-owned lock is refused), `tools/test-kernels-staged-sandbox.sh` (as a real second user: no staged kernel file or directory is writable by the sandbox user), `tools/test-sandbox-home-sealed.sh` (as a real second user: a home the sandbox user owns fails the wipe; a sealed home takes no data in its times, mode or attributes), `tools/test-sandbox-writable-roots.sh` (as a real second user: a listed root only the sandbox user can see is wiped, a missing one fails the wipe, and the preflight inventory refuses an ACL-writable directory, a writable file the sandbox user does not own and a writable directory hidden in an unlistable one), `tools/test-resident-log-sandbox.sh` (as a real second user: a sandboxed boot with no `SERVE_UP_LOG_DIR`, or with a log the sandbox user could read, is refused), `tools/test-engine-build-sandbox.sh`, `tools/test-wait-for-memory.sh` |
| determinism probe | `tools/test-determinism-probe.sh`: the probe runs this checkout's engine |
| test discovery | `tools/test-ci-tests.sh`: `tools/ci-tests.sh` finds and runs a new test file, skips a listed exception with its reason, and fails on a stale or unexplained exception |

**How the tests are found.** The table names the suites as they are today; CI
does not keep this list. One step, `tools/ci-tests.sh run tools`, finds every
`tools/test-*.sh`, `tools/test_*.py` and `tools/box-runner/test/test-*.sh` on
disk and runs each one, so a new test file runs in CI with no step to add
(`tools/ci-tests.sh list tools` prints the files). It prints one `PASS` or
`FAIL` line per file and runs every file even after a failure.

**The exceptions.** `tools/ci-test-exceptions.tsv` lists, one
`<file><TAB><reason>` line each, the discovered test files CI does not run
because they need a GPU (or anything else a hosted runner lacks). Today it
lists none: every test file runs. `tools/ci-tests.sh check` and
`tools/check-doc-consistency.py` compare the list with the files on disk and
fail on an entry that names no test file, gives no reason, or repeats; the
consistency check also fails when `ci.yml` has no `run:` step for a suite.

## `router` (x86 ubuntu)

`tools/ci-tests.sh run router` (each `tools/router/test_*.py` through
`unittest`): the router's submit, select,
freeze, re-score and retry logic (final ties included), the load-test
variants, and the rendered systemd units. No network: a local bare repository
and fake `gh` and `systemctl` stand in.

## `resident` (x86 ubuntu)

`tools/ci-tests.sh run resident` (each `harness/vllm_resident/test_*.py`
through `unittest`, after the release adapter build): the Python resident's
unit and wire tests, and `test_adapter_e2e.py`. That test
runs the real `cuda-engine` against the real resident and requires the serial
stream.

## `engine-overlay` (arm64 ubuntu, no GPU)

This job does three things:

1. pulls the pinned engine image by digest;
2. runs `tools/stage-vllm-overlay.sh`;
3. imports `vllm.model_executor.models.muse_glimmer` through
   the overlay, asserting that vLLM resolved from the overlay.

## What CI cannot cover

Anything that needs a GPU:

- the engine's numerics;
- speed;
- the goldens;
- calibration;
- the ranked run.

The engineering log (`docs/muse-glimmer-port-notes.md`) section 6 lists the
on-box verification, and section 7 the measured findings.
`tools/three-flows-dry-run.sh` runs the real benchd binaries over the fake
resident. It is not in CI, because it downloads benchd.
