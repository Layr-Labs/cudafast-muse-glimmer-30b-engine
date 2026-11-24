#!/usr/bin/env bash
# Run the public local fixture through the normal benchmark entry point.
# benchd continues to own correctness, cooling, timing, and sealed artifacts.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/local-baseline.sh

Run the public Muse Glimmer CUDA local baseline after ./setup.sh. No organizer goldens,
reference workspace, or calibration file are required. This measures local
correctness and timing on one leg; it does not produce a ranked score.

Optional environment overrides (relative paths resolve from the checkout):
  MLXFAST_ENGINE_BIN                .build/release/mlxfast-runtime-worker
  MLXFAST_CORRECTNESS_GOLDEN_PATH    the public serial golden:
                                    correctness_prompts/public-longcopy-gate-english-512.golden.json
                                    (committed; its sha256 and bytes are pinned below
                                    and forwarded to benchd)
  MLXFAST_WEIGHTS_PATH              $MLXFAST_TARGET_SNAPSHOT_DIR, else
                                    reference_weights/Muse-Glimmer-30B-NVFP4
  MLXFAST_SCORE_PATH                score.local-iterate.json

The existing benchmark entry point verifies benchd, boots one resident engine
through tools/serve-up.sh, and enables the cool gate. It runs one leg: the
paired ranked path (tools/muse-glimmer-measure-and-score.sh) is not involved.
USAGE
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"

export MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-.build/release/mlxfast-runtime-worker}"
serial_manifest=""
# The free run must match its golden exactly. The one public golden (recorded
# on a Spark by tools/record-goldens.sh and committed under
# correctness_prompts/) is a SERIAL tape. When the shipped declaration enables
# the drafter, this local-only path uses a temporary serial declaration. A
# caller's own declaration is never overridden. A speculative leg (the pinned
# drafter at depth N) needs that depth's own golden: name it with
# MLXFAST_CORRECTNESS_GOLDEN_PATH.
if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  spec="$("${REPO_ROOT}/tools/spec-declaration.sh" describe)" || exit 1
  if [[ "${spec}" != serial ]]; then
    if [[ "${SPEC_DECLARATION_MANIFEST+x}" == x ]]; then
      echo "local-baseline.sh: the caller's SPEC_DECLARATION_MANIFEST derives ${spec}, but the public golden is a serial tape; set MLXFAST_CORRECTNESS_GOLDEN_PATH to a ${spec} golden" >&2
      exit 1
    fi
    serial_manifest="$(mktemp "${TMPDIR:-/tmp}/muse-glimmer-serial-declaration.XXXXXX")"
    trap 'rm -f -- "${serial_manifest}"' EXIT
    if ! jq -c -f "${REPO_ROOT}/tools/testlib/serial-declaration.jq" \
      "${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json" > "${serial_manifest}"; then
      echo "local-baseline.sh: failed to build the temporary serial declaration" >&2
      exit 1
    fi
    export SPEC_DECLARATION_MANIFEST="${serial_manifest}"
    echo "local-baseline.sh: the shipped declaration derives ${spec}; this local run measures SERIAL against the public serial golden (no public ${spec} golden). Set MLXFAST_CORRECTNESS_GOLDEN_PATH to a ${spec} golden to run the drafter." >&2
  fi
  MLXFAST_CORRECTNESS_GOLDEN_PATH="correctness_prompts/public-longcopy-gate-english-512.golden.json"
  # The committed public golden is pinned: benchmark.sh refuses a copy that is
  # not these bytes before any run work. A caller's own pin still wins.
  export MLXFAST_CORRECTNESS_GOLDEN_SHA256="${MLXFAST_CORRECTNESS_GOLDEN_SHA256:-8f8b2717f5674a52fdbb93d8f82caf90f0cbe8d8834cd75145dd0f4495099b74}"
  export MLXFAST_CORRECTNESS_GOLDEN_BYTES="${MLXFAST_CORRECTNESS_GOLDEN_BYTES:-30819}"
fi
export MLXFAST_CORRECTNESS_GOLDEN_PATH
export MLXFAST_WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_SNAPSHOT_DIR:-reference_weights/Muse-Glimmer-30B-NVFP4}}"
export MLXFAST_SCORE_PATH="${MLXFAST_SCORE_PATH:-score.local-iterate.json}"

if [[ ! -x "${MLXFAST_ENGINE_BIN}" ]]; then
  echo "local-baseline.sh: engine not executable: ${MLXFAST_ENGINE_BIN}" >&2
  echo "local-baseline.sh: run ./setup.sh to build and stage the engine, or set MLXFAST_ENGINE_BIN" >&2
  exit 1
fi
if [[ ! -d "${MLXFAST_WEIGHTS_PATH}" ]]; then
  echo "local-baseline.sh: target snapshot directory not found: ${MLXFAST_WEIGHTS_PATH}" >&2
  echo "local-baseline.sh: run ./setup.sh to download and verify the snapshot, or set MLXFAST_TARGET_SNAPSHOT_DIR" >&2
  exit 1
fi

{
  echo "local-baseline.sh: public local baseline (unranked)"
  echo "local-baseline.sh: golden=${MLXFAST_CORRECTNESS_GOLDEN_PATH}"
  echo "local-baseline.sh: weights=${MLXFAST_WEIGHTS_PATH}"
  echo "local-baseline.sh: result=${MLXFAST_SCORE_PATH}; score=null is expected without ranked paired scoring"
} >&2

if [[ -z "${serial_manifest}" ]]; then
  exec "${REPO_ROOT}/benchmark.sh" --local-iterate
fi

"${REPO_ROOT}/benchmark.sh" --local-iterate &
pid=$!
trap 'kill -TERM "${pid}" 2>/dev/null || true' INT TERM

rc=0
wait "${pid}" || rc=$?
while kill -0 "${pid}" 2>/dev/null; do
  wait "${pid}" || rc=$?
done

trap - INT TERM
rm -f -- "${serial_manifest}"
trap - EXIT
exit "${rc}"
