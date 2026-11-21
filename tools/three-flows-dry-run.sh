#!/usr/bin/env bash
#
# three-flows-dry-run.sh -- run the track's flows off the box, with the REAL
# benchd binaries, against the resident on its FAKE backend.
#
# WHAT IT IS FOR. Every plumbing break a laptop can find must be found on a
# laptop first. On a host with NO GPU, NO container and NO checkpoint this runs:
#
#   FLOW A  golden authoring: record-correctness-golden drives the staged
#           adapter -> resident for every mode the fixture's allowed_modes
#           names, exactly as the organizer does on a Spark
#           (tools/record-goldens.sh).
#   FLOW B  the self-benchmark: benchd iterate --mode local-iterate on the
#           candidate leg, and the SEALED fields are checked:
#           passed_correctness, effective_spec_mode/_depth and the
#           engine_backend identity (plus the speculative counters when the
#           fixture allows mtp). A mode the fixture does not allow is a
#           REFUSAL case: an MTP declaration on a serial-only track is refused
#           by tools/spec-declaration.sh before any leg boots.
#   FLOW C  the paired path's lifecycle: benchd calibrate-baseline boots the
#           REFERENCE tree's own serve-up.sh per pass. The fake engine's
#           microsecond timings cannot meet the fixture's CV gate, so the expected
#           outcome is a CALIBRATION-CV-EXCEEDED refusal AFTER both passes were
#           measured -- which is what proves the per-leg boot works.
#
# THE ENGINE IS THE FAKE. The REAL tools/serve-up.sh boots the REAL resident
# (SERVE_UP_LAUNCHER=host, RESIDENT_BACKEND=fake) and the REAL cuda-engine
# adapter attaches. The lifecycle and the wire are the real ones; the tokens
# and every timing are meaningless.
#
# THE TRACK. benchd main is contract-first: every verb gets --contract, and the
# fixture declares the model identity (muse_glimmer, 202048, 52 layers, a
# seed_tokens-long prompt) and the measurement window. The fake backend emits ids far
# inside the vocabulary, so its goldens load under that identity.
#
# Env:
#   THREE_FLOWS_WORK  work directory (default: a fresh mktemp -d, kept)
#   BENCHD_BIN_DIR    where tools/fetch-benchd.sh installs the pair
#                     (default: <work>/benchd-bin)
#   DRY_RUN_DEPTH     the MTP depth flows A and B exercise (default 3); a
#                     positive case only when allowed_modes names mtp
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
# shellcheck source=tools/testlib/synthetic-window.sh
. "${ROOT_DIR}/tools/testlib/synthetic-window.sh"
synthetic_window_env

say() { printf '\n=== %s\n' "$*"; }
die() { printf 'three-flows-dry-run: %s\n' "$*" >&2; exit 2; }
FAILS=0
check() { # check LABEL CMD... -- a named assertion
  local label="$1"; shift
  if "$@"; then printf 'PASS  %s\n' "${label}"; else printf 'FAIL  %s\n' "${label}"; FAILS=$((FAILS + 1)); fi
}

WORK="${THREE_FLOWS_WORK:-$(mktemp -d)}"
mkdir -p "${WORK}"
SOCKDIR="$(mktemp -d /tmp/3flows.XXXXXX)"
trap 'rm -rf "${SOCKDIR}"' EXIT
TRACK="$(jq -r .trackId "${ROOT_DIR}/benchmark.json")"
FIXTURE="${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
REPO="$(jq -r .target.upstream_model_id "${FIXTURE}")"
REV="$(jq -r .target.upstream_revision "${FIXTURE}")"
DEPTH="${DRY_RUN_DEPTH:-3}"
# The positive cases come from the fixture, never from this script: an MTP leg
# is expected to pass only when allowed_modes names mtp.
MTP_ALLOWED="$(jq -r 'if ((.allowed_modes // []) | index("mtp")) != null then 1 else 0 end' "${FIXTURE}")"
export MLXFAST_QWEN_MTP_TRACK_ID="${TRACK}"
export SERVE_UP_SOCKET_DIR="${SOCKDIR}" SERVE_UP_LOG_DIR="${WORK}/serve-log"
synthetic_window_meminfo "${WORK}/meminfo"
export SERVE_UP_MEMINFO="${WORK}/meminfo"
WEIGHTS="${WORK}/snapshot"
synthetic_window_weights "${WEIGHTS}"
export MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}"

say "benchd (the pinned channel)"
export BENCHD_BIN_DIR="${BENCHD_BIN_DIR:-${WORK}/benchd-bin}"
"${ROOT_DIR}/tools/fetch-benchd.sh" >/dev/null || die "cannot resolve benchd (set BENCHD_BRANCH to an existing channel)"
BENCHD="${BENCHD_BIN_DIR}/benchd"
RECORDER="${BENCHD_BIN_DIR}/record-correctness-golden"
[[ -x "${BENCHD}" && -x "${RECORDER}" ]] || die "the channel pair is incomplete in ${BENCHD_BIN_DIR}"

say "build + stage the adapter"
cargo build --release --locked -q --manifest-path "${ROOT_DIR}/harness/protocol-adapter/Cargo.toml" \
  || die "the adapter did not build"
"${ROOT_DIR}/tools/stage-cuda-engine.sh" >/dev/null || die "the adapter did not stage"
ENGINE_REL=".build/release/mlxfast-runtime-worker"

python3 -c "import sys; print(list(range(100, 100 + int(sys.argv[1]))))" "$(jq -r .seed_tokens "${FIXTURE}")" > "${WORK}/prompt.json"

# --- FLOW A: golden authoring -----------------------------------------------
say "flow A: record-correctness-golden over the resident (allowed_modes: $(jq -c '.allowed_modes' "${FIXTURE}"))"
record() { # record SPEC_ARGS... -- writes ${WORK}/<label>.golden.json
  local label="$1"; shift
  (cd "${ROOT_DIR}" && env "$@" SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" tools/serve-up.sh \
     "${RECORDER}" --worker-bin "${ENGINE_REL}" --weights "${WEIGHTS}" \
     --prompt-tokens "${WORK}/prompt.json" --case-name botany \
     --benchmark-free-run --track "${TRACK}" --contract "${FIXTURE}" \
     --model-provenance-repo "${REPO}" --model-provenance-rev "${REV}" \
     --out "${WORK}/${label}.golden.json") \
     > "${WORK}/record-${label}.log" 2>&1
}
check "A1 serial golden recorded" record serial SERVE_UP_SPECULATIVE=0
if [[ "${MTP_ALLOWED}" == 1 ]]; then
  check "A2 mtp${DEPTH} golden recorded" record "mtp${DEPTH}" SERVE_UP_SPECULATIVE=1 "SERVE_UP_SPEC_DRAFT_LEN=${DEPTH}"
  same_cases() { # same_cases A B -- two goldens carry identical serial cases[]
    [[ "$(jq -c .cases "$1")" == "$(jq -c .cases "$2")" ]]
  }
  check "A3 serial and mtp${DEPTH} goldens carry the same serial cases[]" \
    same_cases "${WORK}/serial.golden.json" "${WORK}/mtp${DEPTH}.golden.json"
fi

# --- FLOW B: the self-benchmark ---------------------------------------------
say "flow B: benchd iterate --mode local-iterate (serial; --mtp-depth ${DEPTH} as a $([[ "${MTP_ALLOWED}" == 1 ]] && echo positive || echo refusal) case)"
iterate() { # iterate LABEL GOLDEN [benchd args...] -- env via SPEC_ENV array
  local label="$1" golden="$2"; shift 2
  (cd "${ROOT_DIR}" && env "${SPEC_ENV[@]}" SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" tools/serve-up.sh \
     "${BENCHD}" iterate --engine "${ENGINE_REL}" --weights "${WEIGHTS}" --golden "${golden}" \
     --mode local-iterate --contract "${FIXTURE}" --score-path "${WORK}/${label}.score.json" "$@") \
     > "${WORK}/iterate-${label}.log" 2>&1
}
sealed() { # sealed LABEL JQ -- true when the sealed metrics satisfy JQ
  jq -e ".metrics | ${2}" "${WORK}/${1}.score.json" >/dev/null 2>&1
}
SPEC_ENV=(SERVE_UP_SPECULATIVE=0)
check "B1 serial leg runs" iterate serial "${WORK}/serial.golden.json"
check "B2 serial leg passed correctness" sealed serial '.passed_correctness == true'
check "B3 serial leg sealed effective_spec serial" sealed serial '.effective_spec_mode == "serial"'
check "B4 serial leg sealed the vllm-resident identity" sealed serial '(.engine_backend | startswith("vllm-resident load_epoch="))'
SPEC_ENV=(SERVE_UP_SPECULATIVE=1 "SERVE_UP_SPEC_DRAFT_LEN=${DEPTH}")
if [[ "${MTP_ALLOWED}" == 1 ]]; then
  check "B5 mtp${DEPTH} leg runs" iterate "mtp${DEPTH}" "${WORK}/mtp${DEPTH}.golden.json" --mtp-depth "${DEPTH}"
  check "B6 mtp${DEPTH} leg passed correctness" sealed "mtp${DEPTH}" '.passed_correctness == true'
  check "B7 mtp${DEPTH} leg sealed effective_spec mtp/${DEPTH}" sealed "mtp${DEPTH}" ".effective_spec_mode == \"mtp\" and .effective_spec_depth == ${DEPTH}"
  check "B8 mtp${DEPTH} leg sealed speculative counters" sealed "mtp${DEPTH}" '.spec_drafted_total > 0 and .spec_accepted_total >= 0 and .spec_rounds > 0'
else
  # REFUSAL: allowed_modes does not name mtp, so an MTP declaration must be
  # refused before any leg boots. The gate is tools/spec-declaration.sh, which
  # serve-up.sh, the ranked preflight and tools/local-baseline.sh all resolve
  # the serve spec through. benchd's local-iterate mode does not check
  # allowed_modes itself (an --mtp-depth leg over the serial golden seals
  # passed=true), so it is not the gate tested here.
  printf '{"version":1,"source":"none","spec":{"enabled":true,"num_speculative_tokens":%s}}\n' "${DEPTH}" \
    > "${WORK}/mtp${DEPTH}.declaration.json"
  refused() { # refused DECLARATION -- spec-declaration.sh refuses it by name
    ! SPEC_DECLARATION_MANIFEST="$1" "${ROOT_DIR}/tools/spec-declaration.sh" describe \
      > "${WORK}/declaration-refusal.log" 2>&1 \
      && grep -q 'REFUSING' "${WORK}/declaration-refusal.log"
  }
  check "B5 an mtp${DEPTH} declaration is refused before any leg (allowed_modes does not name mtp)" \
    refused "${WORK}/mtp${DEPTH}.declaration.json"
fi

# --- FLOW C: the paired lifecycle -------------------------------------------
say "flow C: benchd calibrate-baseline boots the REFERENCE tree's own serve-up"
REF="${WORK}/reference"
rm -rf "${REF}"
git clone -q --no-local "${ROOT_DIR}" "${REF}" || die "cannot clone the reference tree"
mkdir -p "${REF}/.build/release"
cp "${ROOT_DIR}/${ENGINE_REL}" "${REF}/${ENGINE_REL}"
"${BENCHD}" calibrate-baseline --baseline-workspace "${REF}" --engine "${ENGINE_REL}" \
  --golden "${WORK}/serial.golden.json" --weights "${WEIGHTS}" --out "${WORK}/calibration.json" \
  --track "${TRACK}" --contract "${FIXTURE}" \
  --passes 2 --box dry-run-box --benchd-source-commit "$(jq -r .source_commit "${BENCHD_BIN_DIR}/benchd.manifest.json")" \
  > "${WORK}/calibrate.log" 2>&1
check "C1 both control passes were measured on the reference tree" grep -q 'pass 2/2 measured' "${WORK}/calibrate.log"
check "C2 the fake engine is refused by the CV gate (expected off-box)" grep -q 'CALIBRATION-CV-EXCEEDED' "${WORK}/calibrate.log"

say "result (logs in ${WORK})"
if [[ "${FAILS}" -eq 0 ]]; then
  echo "OK: every flow behaved as designed"
  exit 0
fi
echo "FAILED: ${FAILS} assertion(s)"
exit 1
