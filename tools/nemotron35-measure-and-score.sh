#!/usr/bin/env bash
#
# nemotron35-measure-and-score.sh -- benchmark.json's benchmarkCommand /
# preSubmitCommand entry point for track nemotron3.5-lightning-30b-a3b-cuda-v1.
#
# It drives `benchd iterate --mode official` -- the SOLE scored path -- against
# the track's LIVE golden. `iterate --mode official` is timed-first, spawns the
# sandboxed runtime worker, runs the full correctness set, gates on the official
# floor/bands, and SEALS the artifact itself: it writes score.json in the
# {score, metrics} shape Yukon's ScoreFileSchema reads, its `.sha256` sidecar,
# and the per-mode benchmark-integrity sidecar. There is NO results.json to
# convert -- benchd is the sole writer of the sealed score, so this script no
# longer post-processes one. This script is TRUSTED-side tooling: it is NOT in
# editablePaths, so a submission cannot rewrite the measurement pipeline from
# inside its own archive.
#
# PAIRED, WITH A PER-BOX BASELINE (contract §5.1). The scored model is TWO
# LEGS on the SAME box in the SAME job, over the ONE live golden:
#
#   leg 1  the SERIAL CONTROL, on the organizer-staged REFERENCE tree
#          (--baseline-workspace), speculation off, always;
#   leg 2  the CANDIDATE, on this checkout, at its declared draft depth.
#
# composite = (ref_prefill_spt / cand_prefill_spt)^0.25
#           * (ref_decode_spt / cand_decode_spt)^0.75, floors unchanged.
#
# NO PAIR IS STORED ANYWHERE. Not in a benchd constant, not in the fixture, not
# in a golden: the denominator is MEASURED, here, next to the candidate. The
# per-box calibration file (--baseline-calibration) is a HEALTH BAND for leg 1
# only -- benchd refuses the run when leg 1 lands outside this box's band -- and
# it is never divided by.
#
# The LIVE golden is read FROM the fixture, never hardcoded: the fixture's
# `live_golden` names the default, `live_golden_rotation` (when declared) the
# pool the ranked workflow draws this job's prompt from, and the prompt's
# `timed_prompt_pool[]` entry pins it ({sha256, bytes}). The pins are forwarded to benchd as
# --golden-sha256/--golden-bytes, which re-verifies the raw bytes BEFORE parse
# and refuses on any mismatch (the integrity pin). --contract carries the arm
# gate: benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true.
#
# The design rationale lives in docs/nemotron35-port-notes.md section 9.
#
# Usage:
#   ./tools/nemotron35-measure-and-score.sh                 # full measure + seal
#   ./tools/nemotron35-measure-and-score.sh --preflight-only # pre-GPU dry-run, no engine
#
# Env:
#   MLXFAST_SCORE_PATH               Where benchd seals the {score, metrics} JSON
#                                     (--score-path). Defaults to score.json
#                                     (benchmark.json always sets this explicitly).
#                                     The .sha256 sidecar and benchmark-integrity
#                                     sidecar are sealed beside it by benchd.
#   MLXFAST_BASELINE_WORKSPACE        The built REFERENCE tree the serial-control
#                                     leg runs on, passed as --baseline-workspace.
#                                     Box-only, staged out of band at the fixture's
#                                     baseline_reference_commit and verified by
#                                     tools/ranked-box-preflight.sh section 8.
#                                     REQUIRED on a real run; refused by name when
#                                     unset.
#   MLXFAST_BASELINE_CALIBRATION      This box's calibration file, passed as
#                                     --baseline-calibration. It is the HEALTH
#                                     BAND for the serial-control leg, never the
#                                     denominator. Written by tools/calibrate-box.sh.
#                                     REQUIRED on a real run; refused by name when
#                                     unset.
#   MLXFAST_LIVE_GOLDEN_CHOICE        The prompt the ranked workflow drew for this
#                                     job (tools/draw-live-golden.sh). REQUIRED on
#                                     a real run when the fixture declares
#                                     live_golden_rotation, and refused unless it
#                                     is in live_golden_rotation.pool. Forwarded
#                                     to benchd as --live-prompt, which re-resolves
#                                     its pins from the fixture and seals the name.
#   MLXFAST_NEMOTRON35_GOLDEN_DIR         Directory holding the staged timed-pool
#                                     golden files. The LIVE golden is resolved
#                                     from it as <live_golden>.golden.json (the
#                                     basename of the fixture entry's r2_path).
#                                     Box-only, staged out of band and
#                                     pin-verified by tools/ranked-box-preflight.sh.
#   MLXFAST_ENGINE_BIN               The Engine Protocol v1 adapter benchd spawns
#                                     (`--engine`). Default: the fixed staged path
#                                     tools/stage-cuda-engine.sh writes,
#                                     .build/release/mlxfast-runtime-worker. The
#                                     adapter attaches to the leg's vLLM resident
#                                     (booted per leg by benchd, below).
#   SANDBOX_USER                       When set (ranked box: nemosub), the
#                                     staged engine must be the sandbox shim
#                                     tools/stage-cuda-engine.sh writes: it execs
#                                     tools/run-engine-sandboxed.sh on the real
#                                     adapter at <engine>.sandboxed. benchd still
#                                     gets the normal staged path as `--engine`,
#                                     so the serial-control leg resolves the
#                                     reference workspace's own binary at that
#                                     path. Refused when the shim, the
#                                     .sandboxed adapter, the wrapper or
#                                     passwordless sudo to SANDBOX_USER is
#                                     unusable.
#   MLXFAST_WEIGHTS_PATH             Weights directory, passed as `--weights`.
#                                     Default: the on-box target snapshot
#                                     ./setup.sh verifies in place,
#                                     MLXFAST_TARGET_SNAPSHOT_DIR.
#   BENCHD                           Path to the benchd binary. Default: the
#                                     binary ./tools/fetch-benchd.sh resolves from
#                                     the dist channel and verifies against the
#                                     channel's benchd.manifest.json
#                                     (benchd-bin/benchd). No cargo build: the
#                                     ranked box has no Rust toolchain, so benchd
#                                     ships prebuilt and pinned by sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *)
      echo "nemotron35-measure-and-score.sh: unrecognized argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# jq is required for reading the fixture (live_golden + pins) and the trackId.
if ! command -v jq >/dev/null 2>&1; then
  echo "nemotron35-measure-and-score.sh: jq is required (it reads the fixture pins and trackId)." >&2
  exit 1
fi

# Resolve the PINNED benchd. benchd ships as a channel PREBUILT
# (./tools/fetch-benchd.sh, verified against the channel's
# benchd.manifest.json; the channel manifest is the pin (contract §1), so
# measurement fixes ship bench-side with no engine commit). fetch-benchd.sh
# accepts an already-present benchd-bin/benchd whose sha256 and bytes match the
# manifest beside it -- the offline path on the ranked box, which has no Rust
# toolchain -- and otherwise downloads and verifies it. It never yields an
# unverified binary, so this refuses rather than measuring against unpinned
# scoring code.
#
# BENCHD= from the caller is honoured and NOT hash-checked: that is a
# deliberate "use this other binary" for benchd development.
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
if [[ ! -x "${BENCHD}" ]]; then
  echo "nemotron35-measure-and-score.sh: benchd not found at ${BENCHD}." >&2
  echo "  fetch it: ./tools/fetch-benchd.sh   (resolves the dist channel, verifies the manifest)" >&2
  echo "  or set BENCHD to an existing binary." >&2
  exit 1
fi

CONTRACT="${SCRIPT_DIR}/fixtures/nemotron35_lightning_30b_a3b_track.json"
if [[ ! -f "${CONTRACT}" ]]; then
  echo "nemotron35-measure-and-score.sh: track contract fixture missing: ${CONTRACT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# THE TRACK ID. `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every
# mode -- its `-{platform}-v{N}` suffix keys the acceptance bands, and the
# official path resolves it BEFORE the timed run. The
# value is benchmark.json's `trackId` (the PLATFORM id). benchd treats
# env != contract as a hard error, so a caller that pre-set a DIFFERENT value is
# refused here rather than silently overridden.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${SCRIPT_DIR}/benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "nemotron35-measure-and-score.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this script will not guess one." >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "nemotron35-measure-and-score.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this script will not override either." >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# ---------------------------------------------------------------------------
# THE LIVE GOLDEN + ITS PIN, read FROM the fixture (never hardcoded). The
# fixture's `live_golden` names the DEFAULT prompt. When the fixture declares
# `live_golden_rotation`, the scored prompt is instead the one the ranked
# workflow drew for THIS job (tools/draw-live-golden.sh, /dev/urandom), handed
# in as MLXFAST_LIVE_GOLDEN_CHOICE; it must be a member of
# live_golden_rotation.pool. Both legs of every pair measure that one prompt.
# Its timed_prompt_pool[] entry -- matched by the basename of its r2_path
# (<name>.golden.json) -- carries the {sha256, bytes} pin.
DEFAULT_LIVE_GOLDEN="$(jq -r '.live_golden // empty' "${CONTRACT}")"
CHOICE="${MLXFAST_LIVE_GOLDEN_CHOICE:-}"
if [[ "$(jq -r 'has("live_golden_rotation")' "${CONTRACT}")" == "true" ]]; then
  if [[ -z "${CHOICE}" ]]; then
    if [[ "${PREFLIGHT_ONLY}" != "1" ]]; then
      echo "nemotron35-measure-and-score.sh: the fixture declares live_golden_rotation but MLXFAST_LIVE_GOLDEN_CHOICE is unset; the ranked workflow draws the scored prompt per job (tools/draw-live-golden.sh). Refusing rather than scoring a prompt nobody drew." >&2
      exit 1
    fi
    CHOICE="${DEFAULT_LIVE_GOLDEN}"
  fi
  if ! jq -e --arg c "${CHOICE}" '.live_golden_rotation.pool // [] | index($c) != null' "${CONTRACT}" >/dev/null; then
    echo "nemotron35-measure-and-score.sh: MLXFAST_LIVE_GOLDEN_CHOICE='${CHOICE}' is not in live_golden_rotation.pool ($(jq -c '.live_golden_rotation.pool' "${CONTRACT}")); refusing." >&2
    exit 1
  fi
  LIVE_GOLDEN_NAME="${CHOICE}"
else
  # No rotation: live_golden, exactly as before rotation existed.
  if [[ -n "${CHOICE}" && "${CHOICE}" != "${DEFAULT_LIVE_GOLDEN}" ]]; then
    echo "nemotron35-measure-and-score.sh: MLXFAST_LIVE_GOLDEN_CHOICE='${CHOICE}' but the fixture declares no live_golden_rotation; the only scored prompt is live_golden '${DEFAULT_LIVE_GOLDEN}'. Refusing." >&2
    exit 1
  fi
  LIVE_GOLDEN_NAME="${DEFAULT_LIVE_GOLDEN}"
fi
if [[ -z "${LIVE_GOLDEN_NAME}" ]]; then
  echo "nemotron35-measure-and-score.sh: fixture declares no live_golden; there is no golden to score over." >&2
  exit 1
fi

LIVE_GOLDEN_BASENAME="${LIVE_GOLDEN_NAME}.golden.json"
# The pool entry whose r2_path ends in /<live_golden>.golden.json. Its pin is the
# integrity pin forwarded to benchd.
live_golden_entry="$(jq -c --arg base "/${LIVE_GOLDEN_BASENAME}" \
  'first(.timed_prompt_pool[] | select(.r2_path | endswith($base)))' "${CONTRACT}")"
if [[ -z "${live_golden_entry}" || "${live_golden_entry}" == "null" ]]; then
  echo "nemotron35-measure-and-score.sh: live_golden '${LIVE_GOLDEN_NAME}' has no timed_prompt_pool entry (looked for an r2_path ending in /${LIVE_GOLDEN_BASENAME})." >&2
  exit 1
fi
LIVE_GOLDEN_SHA256="$(printf '%s' "${live_golden_entry}" | jq -r '.sha256 // empty')"
LIVE_GOLDEN_BYTES="$(printf '%s' "${live_golden_entry}" | jq -r '.bytes // empty')"

# THE SERIAL GOLDEN, kept aside before any per-depth override below. The
# serial-control leg (leg 1) is serial by construction, so benchd verifies it
# against THIS tape (--control-golden); the per-depth tape is the candidate
# leg's oracle only. On the MLX engine the depth-1 tape forks from the serial
# tape at step 1, so a control leg checked against it dies at step 1.
SERIAL_GOLDEN_BASENAME="${LIVE_GOLDEN_BASENAME}"
SERIAL_GOLDEN_SHA256="${LIVE_GOLDEN_SHA256}"
SERIAL_GOLDEN_BYTES="${LIVE_GOLDEN_BYTES}"
if ! printf '%s' "${LIVE_GOLDEN_SHA256}" | grep -Eq '^[0-9a-f]{64}$' \
  || ! printf '%s' "${LIVE_GOLDEN_BYTES}" | grep -Eq '^[1-9][0-9]*$'; then
  echo "nemotron35-measure-and-score.sh: live_golden '${LIVE_GOLDEN_NAME}' is unarmed or malformed (sha256='${LIVE_GOLDEN_SHA256}', bytes='${LIVE_GOLDEN_BYTES}'); nothing can be pin-verified against it." >&2
  exit 1
fi

# PER-DEPTH TIMED ORACLE. The timed leg replays the engine's FREE-RUN under the
# declared speculative spec, and a free-run trajectory forks from the serial one
# at near-tie argmaxes (docs/participant-contract.md section 11.4(c)). So the
# timed oracle must be authored under the SAME spec the leg replays: the serial
# golden resolved above is the launch reference (MTP-0), and each permitted MTP
# draft depth has its OWN golden. The spec label is derived through the one
# trusted deriver that ALSO boots the serve (tools/spec-declaration.sh describe
# -> serial|mtp1..mtp6), so the oracle can never disagree with the draft
# depth that actually ran. This overrides ONLY the timed-oracle file and its pin;
# the correctness gate is untouched, because every per-depth golden carries the
# SAME serial cases[] oracle, so token fidelity is still measured against serial.
SPEC_DESC="$("${SCRIPT_DIR}/tools/spec-declaration.sh" describe)"
if [[ "${SPEC_DESC}" != "serial" ]]; then
  # speculative_oracles.<prompt> maps a spec label to that prompt's per-depth
  # golden's {r2_path, sha256, bytes}. A declared depth with no entry is REFUSED -- there
  # is no timed oracle authored for it, and scoring it against the serial oracle
  # would fail at step 0 (the very divergence this map exists to resolve). The
  # depth is already envelope-checked by spec-declaration.sh; this is the
  # oracle-existence half of the same fail-closed posture.
  spec_entry="$(jq -c --arg n "${LIVE_GOLDEN_NAME}" --arg k "${SPEC_DESC}" '.speculative_oracles[$n][$k] // empty' "${CONTRACT}")"
  if [[ -z "${spec_entry}" || "${spec_entry}" == "null" ]]; then
    echo "nemotron35-measure-and-score.sh: declared spec '${SPEC_DESC}' has no speculative_oracles['${LIVE_GOLDEN_NAME}'] entry in ${CONTRACT}; no timed oracle is authored for that prompt at that draft depth. Refusing rather than scoring it against the serial oracle." >&2
    exit 1
  fi
  LIVE_GOLDEN_BASENAME="$(printf '%s' "${spec_entry}" | jq -r '.r2_path // empty')"
  LIVE_GOLDEN_BASENAME="${LIVE_GOLDEN_BASENAME##*/}"
  LIVE_GOLDEN_SHA256="$(printf '%s' "${spec_entry}" | jq -r '.sha256 // empty')"
  LIVE_GOLDEN_BYTES="$(printf '%s' "${spec_entry}" | jq -r '.bytes // empty')"
  if ! printf '%s' "${LIVE_GOLDEN_SHA256}" | grep -Eq '^[0-9a-f]{64}$' \
    || ! printf '%s' "${LIVE_GOLDEN_BYTES}" | grep -Eq '^[1-9][0-9]*$' \
    || [[ -z "${LIVE_GOLDEN_BASENAME}" || "${LIVE_GOLDEN_BASENAME}" != *.golden.json ]]; then
    echo "nemotron35-measure-and-score.sh: speculative_oracles['${LIVE_GOLDEN_NAME}']['${SPEC_DESC}'] is unarmed or malformed (file='${LIVE_GOLDEN_BASENAME}', sha256='${LIVE_GOLDEN_SHA256}', bytes='${LIVE_GOLDEN_BYTES}')." >&2
    exit 1
  fi
  echo "nemotron35-measure-and-score.sh: declared spec ${SPEC_DESC}; timed oracle is ${LIVE_GOLDEN_BASENAME} (sha256 ${LIVE_GOLDEN_SHA256}, ${LIVE_GOLDEN_BYTES} bytes)" >&2
fi

# ---------------------------------------------------------------------------
# The LIVE golden FILE, resolved from the existing box staging convention:
# MLXFAST_NEMOTRON35_GOLDEN_DIR holds the staged pool, and the live golden is
# <live_golden>.golden.json within it. The goldens are box-only, staged out of
# band and pin-verified by tools/ranked-box-preflight.sh.
GOLDEN_DIR="${MLXFAST_NEMOTRON35_GOLDEN_DIR:-}"
LIVE_GOLDEN_PATH=""
if [[ -n "${GOLDEN_DIR}" ]]; then
  LIVE_GOLDEN_PATH="${GOLDEN_DIR}/${LIVE_GOLDEN_BASENAME}"
fi

# ---------------------------------------------------------------------------
# --preflight-only: a pre-GPU DRY RUN that exercises the arm gate and (when the
# golden is staged) the integrity pin, WITHOUT spawning the engine or loading the
# model. No score is written. The full run below enforces both refusals through
# benchd itself; this is the loud early gate.
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # ARM GATE (mirror of benchd's enforce_official_scoring_enabled): an official
  # run refuses unless the fixture declares official_scoring_enabled: true. false
  # and ABSENT both refuse -- an absent arm state is not an armed one. benchd
  # enforces this itself on the real run; mirroring it here makes the pre-GPU
  # dry-run honest rather than a rubber stamp.
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  if [[ "${armed}" != "true" ]]; then
    echo "nemotron35-measure-and-score.sh: preflight REFUSING -- fixtures/nemotron35_lightning_30b_a3b_track.json does not declare official_scoring_enabled: true (got '${armed}')." >&2
    echo "  benchd refuses to seal an official scoring artifact for an unarmed track; this dry-run mirrors that refusal." >&2
    exit 1
  fi
  echo "nemotron35-measure-and-score.sh: preflight -- arm gate OK (official_scoring_enabled: true), live_golden ${LIVE_GOLDEN_NAME} (sha256 ${LIVE_GOLDEN_SHA256}, ${LIVE_GOLDEN_BYTES} bytes)" >&2

  # INTEGRITY PIN: when the live golden is staged, validate-golden re-verifies its
  # raw bytes against the pin AND load-validates it (reference-model pin from
  # --contract), with NO engine spawned. When it is not staged (e.g. an off-box
  # pre-submit), the pin is still enforced by benchd on the real run -- the old
  # measure-job --preflight-only never checked the golden either, so this is not a
  # weakening; it is a stronger check whenever the golden is present.
  if [[ -n "${LIVE_GOLDEN_PATH}" && -f "${LIVE_GOLDEN_PATH}" ]]; then
    exec "${BENCHD}" validate-golden \
      --golden "${LIVE_GOLDEN_PATH}" \
      --golden-sha256 "${LIVE_GOLDEN_SHA256}" \
      --golden-bytes "${LIVE_GOLDEN_BYTES}" \
      --contract "${CONTRACT}"
  fi
  echo "nemotron35-measure-and-score.sh: preflight -- live golden not staged (MLXFAST_NEMOTRON35_GOLDEN_DIR unset or ${LIVE_GOLDEN_BASENAME} absent); benchd re-verifies the {sha256, bytes} pin on the real run." >&2
  exit 0
fi

# ---------------------------------------------------------------------------
# A REAL RUN needs BOTH halves of the paired baseline, by name. Neither has a
# default and neither may be guessed: without the reference workspace there is no
# serial-control leg to divide by, and without this box's calibration there is
# nothing to say whether that leg landed where this box lands.
BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION:-}"
if [[ -z "${BASELINE_WORKSPACE}" ]]; then
  echo "nemotron35-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE is unset." >&2
  echo "  The scored path is PAIRED: the serial-control leg runs on the staged reference tree at the fixture's baseline_reference_commit. Stage it with tools/stage-baseline-workspace.sh and export the path. Refusing rather than scoring against a stored pair." >&2
  exit 1
fi
if [[ ! -d "${BASELINE_WORKSPACE}" ]]; then
  echo "nemotron35-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE=${BASELINE_WORKSPACE} is not a directory." >&2
  exit 1
fi
if [[ -z "${BASELINE_CALIBRATION}" ]]; then
  echo "nemotron35-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION is unset." >&2
  echo "  Every ranked box carries its own calibration file -- the health band the serial-control leg must land inside. Calibrate this box with tools/calibrate-box.sh and export the path." >&2
  exit 1
fi
if [[ ! -f "${BASELINE_CALIBRATION}" ]]; then
  echo "nemotron35-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION=${BASELINE_CALIBRATION} is not a file." >&2
  exit 1
fi

# A BOX-PRESET SERVE SPEC IS REFUSED ON THE PAIRED PATH. Under one leg a preset
# that AGREED with the declaration was harmless. Under two legs it is not: the
# value reaches the environment of BOTH legs' serve scripts, and leg 1 must be
# serial whatever the candidate declares. So it is refused here as well as in
# tools/ranked-box-preflight.sh section 7.
if [[ -n "${SERVE_UP_SPECULATIVE:-}" ]]; then
  echo "nemotron35-measure-and-score.sh: SERVE_UP_SPECULATIVE=${SERVE_UP_SPECULATIVE} is preset in the environment; on the paired path it would reach the serial-control leg's serve too, and that leg is serial regardless of the candidate's declaration." >&2
  exit 1
fi

# AN INHERITED RESIDENT SOCKET IS REFUSED. benchd boots each leg's resident
# itself and injects that leg's socket into that leg's worker spawns. A socket
# already in the environment would be a THIRD resident -- one neither leg booted
# -- and every worker that inherited it would measure whatever is on the other
# end. benchd refuses one in its own environment; this refuses it earlier, by
# name.
for var in DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  if [[ -n "${value}" ]]; then
    echo "nemotron35-measure-and-score.sh: ${var}=${value} is set in the environment; benchd boots each leg's resident and injects its socket, so an inherited one would point both legs' workers at a resident neither leg booted." >&2
    exit 1
  fi
done

# A REAL RUN needs the staged live golden.
if [[ -z "${GOLDEN_DIR}" || ! -d "${GOLDEN_DIR}" ]]; then
  cat >&2 <<EOF
nemotron35-measure-and-score.sh: MLXFAST_NEMOTRON35_GOLDEN_DIR is unset or missing.
  The live golden (${LIVE_GOLDEN_BASENAME}) is staged onto the box out of band
  (docs/nemotron35-port-notes.md section 7) and this job holds no credential
  to fetch it. There is nothing this script can do here except refuse.
EOF
  exit 1
fi
if [[ ! -f "${LIVE_GOLDEN_PATH}" ]]; then
  echo "nemotron35-measure-and-score.sh: live golden not found at ${LIVE_GOLDEN_PATH}" >&2
  echo "  live_golden is '${LIVE_GOLDEN_NAME}' (fixtures/nemotron35_lightning_30b_a3b_track.json); stage ${LIVE_GOLDEN_BASENAME} into MLXFAST_NEMOTRON35_GOLDEN_DIR." >&2
  exit 1
fi

# --engine. `iterate` spawns the Engine Protocol v1 adapter, which connects to
# the leg's vLLM resident. Default to the fixed staged path
# tools/stage-cuda-engine.sh writes, so the documented invocation is
# self-sufficient; MLXFAST_ENGINE_BIN overrides it for engine development.
ENGINE_BIN="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/mlxfast-runtime-worker}"
if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "nemotron35-measure-and-score.sh: engine binary missing or not executable: ${ENGINE_BIN}" >&2
  echo "  build + stage it (./setup.sh, or tools/stage-cuda-engine.sh), or set MLXFAST_ENGINE_BIN." >&2
  exit 1
fi

SANDBOX_WRAPPER="${SCRIPT_DIR}/tools/run-engine-sandboxed.sh"
if [[ -n "${SANDBOX_USER:-}" ]]; then
  if [[ ! -x "${SANDBOX_WRAPPER}" ]]; then
    echo "nemotron35-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but the engine wrapper is missing or not executable: ${SANDBOX_WRAPPER}" >&2
    exit 1
  fi
  # The staged engine must be exactly the shim stage-cuda-engine.sh writes, and
  # the adapter it names a regular executable file; otherwise the candidate leg
  # would run outside the sandbox.
  # The shim bakes in this step's SANDBOX_* and RUNNER_TEMP (same job, same
  # values as at staging); stage-cuda-engine.sh --print-shim is its one source.
  real_engine="$(cd "$(dirname -- "${ENGINE_BIN}")" && pwd -P)/$(basename -- "${ENGINE_BIN}").sandboxed"
  expected_shim="$("${SCRIPT_DIR}/tools/stage-cuda-engine.sh" --print-shim)" || exit 1
  if [[ -L "${ENGINE_BIN}" || "$(cat -- "${ENGINE_BIN}" 2>/dev/null)" != "${expected_shim}" ]]; then
    echo "nemotron35-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but ${ENGINE_BIN} is not the sandbox shim; stage it with SANDBOX_USER set (tools/engine/build.sh)." >&2
    exit 1
  fi
  if [[ -L "${real_engine}" || ! -f "${real_engine}" || ! -x "${real_engine}" ]]; then
    echo "nemotron35-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but the sandboxed adapter is missing or not a regular executable file: ${real_engine}" >&2
    exit 1
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "nemotron35-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but sudo is not on PATH." >&2
    exit 1
  fi
  if ! sudo -n -u "${SANDBOX_USER}" -- true 2>/dev/null; then
    echo "nemotron35-measure-and-score.sh: SANDBOX_USER=${SANDBOX_USER} but passwordless sudo to that user is unusable (docs/sandbox-design.md section 6a)." >&2
    exit 1
  fi
  echo "nemotron35-measure-and-score.sh: candidate leg engine ${ENGINE_BIN} is the sandbox shim (adapter ${real_engine} as ${SANDBOX_USER})" >&2
fi

# --weights. benchd resolves the target weights from `--weights`. Default to the
# on-box target snapshot ./setup.sh verifies in place
# (MLXFAST_TARGET_SNAPSHOT_DIR); MLXFAST_WEIGHTS_PATH overrides.
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_SNAPSHOT_DIR:-}}"
if [[ -z "${WEIGHTS_PATH}" || ! -d "${WEIGHTS_PATH}" ]]; then
  echo "nemotron35-measure-and-score.sh: the target weights directory is missing or unset: '${WEIGHTS_PATH}'" >&2
  echo "  Set MLXFAST_TARGET_SNAPSHOT_DIR to the on-box target snapshot (./setup.sh verifies it), or MLXFAST_WEIGHTS_PATH to override." >&2
  exit 1
fi

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"

# THE SOLE SCORED PATH: benchd iterate --mode official over the live golden.
# benchd measures BOTH legs -- the serial control on --baseline-workspace and the
# candidate on this checkout -- and the score is the ratio of the two. There is no
# OFFICIAL_BASELINE constant and no stored pair in the golden. benchd SEALS
# score.json (the {score, metrics} shape Yukon reads), score.json.sha256, and the
# benchmark-integrity sidecar itself; this script does no post-conversion.
#
# --golden-sha256/--golden-bytes are the INTEGRITY PIN (benchd re-verifies the
# raw bytes before parse and refuses on mismatch). --contract carries the ARM
# GATE (benchd refuses, pre-GPU, unless official_scoring_enabled: true).
# --baseline-calibration carries this box's HEALTH BAND: benchd refuses the run
# when the control leg lands outside it, and never divides by it.

# --control-golden: the serial tape for the serial-control leg. A benchd that
# does not know the flag verifies leg 1 against the candidate's per-depth tape,
# which is only the same tape at depth 0; a speculative declaration on such a
# benchd is refused rather than measured against the wrong oracle.
CONTROL_GOLDEN_PATH="${GOLDEN_DIR}/${SERIAL_GOLDEN_BASENAME}"
CONTROL_ARGS=()
if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--control-golden'; then
  if [[ ! -f "${CONTROL_GOLDEN_PATH}" ]]; then
    echo "nemotron35-measure-and-score.sh: serial golden not found at ${CONTROL_GOLDEN_PATH}; the serial-control leg has no tape to verify against." >&2
    exit 1
  fi
  CONTROL_ARGS=(--control-golden "${CONTROL_GOLDEN_PATH}" --control-golden-sha256 "${SERIAL_GOLDEN_SHA256}" --control-golden-bytes "${SERIAL_GOLDEN_BYTES}")
  echo "nemotron35-measure-and-score.sh: serial-control leg verifies against ${SERIAL_GOLDEN_BASENAME} (sha256 ${SERIAL_GOLDEN_SHA256}, ${SERIAL_GOLDEN_BYTES} bytes)" >&2
elif [[ "${SPEC_DESC}" != "serial" ]]; then
  echo "nemotron35-measure-and-score.sh: REFUSING -- the declaration is ${SPEC_DESC} but this benchd has no --control-golden; the serial-control leg would be verified against the ${SPEC_DESC} tape." >&2
  exit 1
fi
iterate_official=(
  "${BENCHD}" iterate
  --engine "${ENGINE_BIN}"
  --weights "${WEIGHTS_PATH}"
  --golden "${LIVE_GOLDEN_PATH}"
  --mode official
  --score-path "${SCORE_PATH}"
  --golden-sha256 "${LIVE_GOLDEN_SHA256}"
  --golden-bytes "${LIVE_GOLDEN_BYTES}"
  --contract "${CONTRACT}"
  --baseline-workspace "${BASELINE_WORKSPACE}"
  --baseline-calibration "${BASELINE_CALIBRATION}"
)
iterate_official+=( ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"} )

# --live-prompt names the prompt this job scores. benchd resolves that prompt's
# serial pin and its per-depth oracle pin from --contract itself, refuses any
# rotation-pool prompt the fixture does not fully pin, checks the pins above
# against its own. benchd logs the name but never seals it (a prompt name hints
# at the hidden text); the sealed record identifies the golden only by its sha256
# (metrics.baseline_golden_sha256). A benchd that does
# not know the flag can score only the default live_golden.
if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--live-prompt'; then
  if [[ "$(jq -r 'has("live_golden_rotation")' "${CONTRACT}")" == "true" ]]; then
    iterate_official+=( --live-prompt "${LIVE_GOLDEN_NAME}" )
  fi
elif [[ "${LIVE_GOLDEN_NAME}" != "${DEFAULT_LIVE_GOLDEN}" ]]; then
  echo "nemotron35-measure-and-score.sh: REFUSING -- this job drew '${LIVE_GOLDEN_NAME}' but the pinned benchd has no --live-prompt, so it can neither resolve that prompt's pins nor seal its name." >&2
  exit 1
fi

# --box names the runner the band is checked against. benchd reads RUNNER_NAME
# when the flag is absent, so this is the explicit form of what would happen
# anyway -- and it puts the box in the command line the log records.
if [[ -n "${RUNNER_NAME:-}" ]]; then
  iterate_official+=( --box "${RUNNER_NAME}" )
fi

# The candidate legs must REQUEST the declared spec. benchd's official path
# sends free_decode_begin with no spec unless it is told one, and the
# adapter then resolves serial, so a declared MTP leg would be measured as
# serial while its oracle is the MTP one. benchd carries the declared depth
# with --mtp-depth N; a benchd that does not know the flag cannot run an MTP
# leg, and that is a refusal, never a silent serial run.
if [[ "${SPEC_DESC}" != "serial" ]]; then
  DECLARED_DEPTH="$("${SCRIPT_DIR}/tools/spec-declaration.sh" draft-len)"
  if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--mtp-depth'; then
    iterate_official+=( --mtp-depth "${DECLARED_DEPTH}" )
    echo "nemotron35-measure-and-score.sh: benchd will request the declared spec (--mtp-depth ${DECLARED_DEPTH})" >&2
  else
    echo "nemotron35-measure-and-score.sh: the pinned benchd does not accept --mtp-depth, so it cannot request the declared ${SPEC_DESC} leg; refusing rather than measuring a serial leg against the ${SPEC_DESC} oracle" >&2
    exit 1
  fi
fi

# TWO SERVES, ONE AT A TIME -- AND benchd SEQUENCES THEM.
#
# tools/serve-up.sh boots ONE vLLM resident, which reserves a fixed fraction of
# the box's unified memory and the GPU for as long as it runs. Under the RETIRED single-leg model this
# script wrapped the whole benchd invocation in ONE such serve. Under the paired
# model that is no longer possible, for two separate reasons:
#
#   * leg 1 must run the REFERENCE tree's engine. A candidate-tree resident
#     serving both legs would divide the candidate by itself.
#   * the two residents must not overlap. serve-up.sh's own memory plan requires
#     the whole reservation free BEFORE it loads, and a second resident would
#     also share the GPU with the leg being measured. So leg 1's serve boots, leg 1 is measured, that serve
#     is torn down, and only then does leg 2's serve boot.
#
# So this script boots NO serve. It hands benchd both trees, and benchd boots
# each leg's resident in turn through the PER-LEG VERBS, run from that leg's own
# workspace:
#
#   tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE
#   tools/serve-up.sh --stop --socket PATH
#
#   leg 1  ${MLXFAST_BASELINE_WORKSPACE}/tools/serve-up.sh
#            --boot --spec serial --draft-len 0
#          ALWAYS serial: the flag is the authority in --boot mode, and
#          serve-up.sh refuses a disagreeing SERVE_UP_SPECULATIVE rather than
#          honouring it. So the control leg is serial whatever the candidate
#          declares.
#   leg 2  this tree's tools/serve-up.sh --boot at the DECLARED spec, which
#          benchd derives from the --mtp-depth it is given below.
#
# benchd reads the socket back from the first line of the --socket-out file and
# injects it into that leg's worker spawns as DS4_RESIDENT_SOCKET and
# BENCH_WORKER_RESIDENT_SOCKET. It refuses a socket inherited in its OWN
# environment, so nothing else in the job may export one -- the loop above
# refuses that earlier, by name.
#
# --stop runs on success and on failure alike, and it is idempotent, so a leg
# that fails mid-measurement still frees the box for the other leg.
#
# Both serves run inside the workflow step's GPU lock; neither takes a lock of
# its own.
#
# ONE RETRY PER JOB (single budget), for either:
#   (a) a control-leg TOKEN DIVERGENCE (SERIAL-CONTROL-LEG-FAILED: benchmark
#       <prefill|free-run decode seed|free-run decode> token mismatch ...; ~1
#       in 100 legs, port notes 7.11), or
#   (b) a pair-1 CANDIDATE resident boot failure before any candidate timing is
#       sealed (LEG-SERVE-BOOT-FAILED: the candidate leg's ... with
#       metrics.paired_legs empty — pair 2's candidate boot keeps pair 1 in
#       paired_legs and is never retried).
# Either cause re-runs the whole measurement once from scratch after the
# teardown below. The second attempt's seal is the result; the first attempt's
# timings are discarded, never compared. Every other failure is final,
# INCLUDING the other control-leg failures (spawn, session, protocol, cool
# gate, memory): pair 2's control leg runs after the candidate's leg, so a
# candidate could provoke those on purpose to buy a second draw. A second
# control-leg token divergence exits 75: an infrastructure outcome, not a
# submission failure. The attempt count is logged as control_leg_attempts=N; it
# is not sealed, because score.json is pinned by its .sha256 and
# benchmark-integrity.json and only benchd writes it.
#
# The job log carries benchd's score marker exactly once (tools/router reads
# it), so each attempt's stdout (benchd's copy of score.json) and marker are
# held in a private directory and printed only for the attempt that counts.
HELD="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/measure-held.XXXXXX")"
trap 'rm -rf -- "${HELD}"' EXIT
INTEGRITY_PATH="$(dirname -- "${SCORE_PATH}")/benchmark-integrity.json"

run_attempt() {
  rm -f -- "${HELD}/stdout" "${HELD}/marker"
  set +e
  "${iterate_official[@]}" 2>&1 >"${HELD}/stdout" \
    | awk -v held="${HELD}/marker" 'index($0, "benchd iterate: wrote score.json") { print > held; next } { print; fflush() }' >&2
  local rc="${PIPESTATUS[0]}"
  set -e
  return "${rc}"
}

# Only THIS attempt's seal counts: the marker must have been held from this run
# (the seal files are removed before every attempt), and the error must be the
# reference's own token divergence (benchd bench-runner's TokenMismatch labels,
# directly or inside the reference warmup leg).
control_leg_diverged() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false and ((.metrics.error // "") | test(
      "^SERIAL-CONTROL-LEG-FAILED: (reference warmup leg failed: )?benchmark (prefill|free-run decode seed|free-run decode) token mismatch( |:|$)"))' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

# Pair 1 only: benchd seals no paired_legs row until a pair completes; a
# candidate boot failure on pair 2 carries pair 1 in paired_legs (audit trail).
# benchd omits paired_legs when it is empty (score.rs: skip_serializing_if), so
# an ABSENT key is the pair-1 case; the pinned benchd always seals the completed
# pairs on a candidate boot refusal (official.rs, with_measured_pairs).
pair1_candidate_boot_failed() {
  [[ -f "${HELD}/marker" ]] || return 1
  jq -e '.passed == false
    and ((.metrics.error // "") | test("^LEG-SERVE-BOOT-FAILED: the candidate leg"))
    and ((.metrics.paired_legs // []) | length == 0)' \
    "${SCORE_PATH}" >/dev/null 2>&1
}

measurement_retry_warranted() {
  control_leg_diverged || pair1_candidate_boot_failed
}
remove_seal() { rm -f -- "${SCORE_PATH}" "${SCORE_PATH}.sha256" "${INTEGRITY_PATH}"; }

# Nothing from attempt 1 may reach attempt 2 (contract: no reuse across runs):
# its seal is deleted, every sandbox process is killed, everything the sandbox
# user left in the shared temp dirs and its home is removed, and so is all of
# this job's per-job directory the sandbox user can write -- the same five the
# workflow's cleanup step empties. None is read at run time: tools/engine/build.sh
# copies the adapter and the kernels into .build/ before measurement. Memory
# must be free again before benchd boots the next resident. benchd has already
# stopped each leg's resident (--stop runs on failure too).
# Kill stragglers, wipe shared dirs, wait for memory (ranked sandbox only).
sandbox_reset_shared() {
  [[ -n "${SANDBOX_USER:-}" ]] || return 0
  local left=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 >/dev/null 2>&1 || true
    left="$(pgrep -u "${SANDBOX_USER}" 2>/dev/null | tr '\n' ' ' || true)"
    [[ -z "${left// /}" ]] && break
    sleep 0.5
  done
  if [[ -n "${left// /}" ]]; then
    echo "nemotron35-measure-and-score.sh: ${SANDBOX_USER} processes survived kill -KILL -1: ${left}" >&2
    return 1
  fi
  "${SCRIPT_DIR}/tools/sandbox-wipe.sh" || return 1
  bash "${SCRIPT_DIR}/tools/wait-for-memory.sh"
}

clear_sandbox_job_dirs() {
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || return 1
  local sub
  for sub in cargo-home target kernels-out sock adapter-home; do
    sub="${SANDBOX_ROOT}/${SANDBOX_JOB}/${sub}"
    [[ -d "${sub}" ]] || continue
    sudo -n -u "${SANDBOX_USER}" -- chmod -R u+rwX "${sub}" >/dev/null 2>&1 || true
    sudo -n -u "${SANDBOX_USER}" -- find "${sub}" -mindepth 1 -delete || return 1
    [[ -z "$(find "${sub}" -mindepth 1 -print -quit)" ]] || return 1
  done
}

between_attempts() {
  remove_seal
  if [[ -n "${SANDBOX_USER:-}" ]]; then
    sandbox_reset_shared || return 1
    clear_sandbox_job_dirs || return 1
  else
    bash "${SCRIPT_DIR}/tools/wait-for-memory.sh"
  fi
}

attempts=1
rc=0
remove_seal
if [[ -n "${SANDBOX_USER:-}" ]]; then
  if ! sandbox_reset_shared; then
    echo "::error title=Pre-measurement sandbox reset failed::no score sealed: sandbox leftovers from the submission could not be removed" >&2
    exit 75
  fi
fi
run_attempt || rc=$?
if [[ "${rc}" -ne 0 ]] && measurement_retry_warranted; then
  if control_leg_diverged; then
    echo "nemotron35-measure-and-score.sh: attempt 1 DISCARDED, the serial control leg diverged: $(jq -r '.metrics.error' "${SCORE_PATH}")" >&2
  else
    echo "nemotron35-measure-and-score.sh: attempt 1 DISCARDED, the pair-1 candidate resident did not boot (no candidate timing sealed): $(jq -r '.metrics.error' "${SCORE_PATH}")" >&2
  fi
  if ! between_attempts; then
    echo "::error title=Measurement retry teardown failed::the box could not be reset after a retriable infrastructure failure; infrastructure outcome, not a submission failure" >&2
    exit 75
  fi
  echo "nemotron35-measure-and-score.sh: re-running the whole measurement once (attempt 2 of 2)" >&2
  attempts=2
  rc=0
  run_attempt || rc=$?
fi
cat -- "${HELD}/stdout"
if [[ -f "${HELD}/marker" ]]; then cat -- "${HELD}/marker" >&2; fi
echo "nemotron35-measure-and-score.sh: control_leg_attempts=${attempts}" >&2
if [[ "${rc}" -ne 0 && "${attempts}" -eq 2 ]] && control_leg_diverged; then
  echo "::error title=Serial control leg diverged twice::infrastructure outcome, not a submission failure; re-dispatch this job" >&2
  exit 75
fi
exit "${rc}"
