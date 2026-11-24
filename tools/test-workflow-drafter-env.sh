#!/usr/bin/env bash
# test-workflow-drafter-env.sh -- the ranked job's "Runner environment carries
# the box paths and the toolchain" step, cut from .github/workflows/benchmark.yml,
# requires MLXFAST_DRAFTER_DIR ONLY when the entry's declaration switches
# speculation on (tools/spec-declaration.sh speculative = 1).
#
# Cases (a scratch root: the fixture, the trusted spec-declaration.sh and a
# declaration per case; box directories in a temp dir; missing tools stubbed):
#   1. serial declaration, no MLXFAST_DRAFTER_DIR: passes;
#   2. absent declaration, no MLXFAST_DRAFTER_DIR: passes;
#   3. mtp2 declared, no MLXFAST_DRAFTER_DIR: refused, naming the variable;
#   4. mtp2 declared, MLXFAST_DRAFTER_DIR a directory: passes;
#   5. an invalid declaration (depth 7): refused, never read as serial.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

# The step's run block, from its first line to its "runner environment ok" echo.
BLOCK="$(awk '/- name: Runner environment carries the box paths and the toolchain/ {s = 1}
  s && /run: \|/ {p = 1; next}
  p {print}
  p && /echo "runner environment ok:/ {exit}' "${ROOT_DIR}/.github/workflows/benchmark.yml" | sed 's/^          //')"
grep -q 'drafter_spec="$(./tools/spec-declaration.sh speculative)"' <<<"${BLOCK}" \
  || { echo "FAIL: the runner-environment step does not derive the drafter need from tools/spec-declaration.sh" >&2; exit 1; }

R="${WORK}/root"
mkdir -p "${R}/tools" "${R}/fixtures"
cp "${ROOT_DIR}/tools/spec-declaration.sh" "${R}/tools/"
cp "${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json" "${R}/fixtures/"
STUB="${WORK}/bin"
mkdir -p "${STUB}"
for tool in cargo docker git jq python3 curl sha256sum nvidia-smi flock; do
  command -v "${tool}" >/dev/null 2>&1 || { printf '#!/bin/sh\nexit 0\n' > "${STUB}/${tool}"; chmod 755 "${STUB}/${tool}"; }
done
BOX="${WORK}/box"
mkdir -p "${BOX}/snapshot" "${BOX}/benchd" "${BOX}/reference" "${BOX}/goldens" "${BOX}/drafter"
: > "${BOX}/calibration.json"

# The serial declaration: the contract's pin, drafter off (not the shipped
# file: the base may switch it on). The other cases edit it.
SERIAL_DECL="${WORK}/serial.manifest.json"
jq -f "${ROOT_DIR}/tools/testlib/serial-declaration.jq" "${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json" > "${SERIAL_DECL}"

# run_step DECL [VAR=VALUE...] -- DECL: serial | absent | mtp<N> | bad
run_step() {
  local decl="$1"; shift
  rm -f "${R}/mtp-head.manifest.json"
  case "${decl}" in
    absent) ;;
    serial) cp "${SERIAL_DECL}" "${R}/mtp-head.manifest.json" ;;
    bad) jq '.spec.enabled = true | .spec.num_speculative_tokens = 7' "${SERIAL_DECL}" > "${R}/mtp-head.manifest.json" ;;
    mtp*) jq --argjson n "${decl#mtp}" '.spec.enabled = true | .spec.num_speculative_tokens = $n' "${SERIAL_DECL}" > "${R}/mtp-head.manifest.json" ;;
  esac
  (cd "${R}" && env -u MLXFAST_DRAFTER_DIR PATH="${STUB}:${PATH}" \
      MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}/snapshot" BENCHD_BIN_DIR="${BOX}/benchd" \
      MLXFAST_BASELINE_WORKSPACE="${BOX}/reference" MLXFAST_MUSE_GLIMMER_GOLDEN_DIR="${BOX}/goldens" \
      MLXFAST_BASELINE_CALIBRATION="${BOX}/calibration.json" "$@" \
      bash -c "set -euo pipefail
${BLOCK}" 2>&1)
}

out="$(run_step serial)"; rc=$?
[[ ${rc} -eq 0 ]] && grep -q 'runner environment ok: .* drafter=none' <<<"${out}" \
  && ok "case 1: a serial declaration needs no MLXFAST_DRAFTER_DIR" || fail "case 1: rc=${rc}: $(tail -2 <<<"${out}")"
out="$(run_step absent)"; rc=$?
[[ ${rc} -eq 0 ]] && ok "case 2: an absent declaration needs no MLXFAST_DRAFTER_DIR" || fail "case 2: rc=${rc}: $(tail -2 <<<"${out}")"
out="$(run_step mtp2)"; rc=$?
[[ ${rc} -ne 0 ]] && grep -q '::error::MLXFAST_DRAFTER_DIR is not exported' <<<"${out}" \
  && ok "case 3: mtp2 declared with no MLXFAST_DRAFTER_DIR is refused by name" || fail "case 3: rc=${rc}: $(tail -2 <<<"${out}")"
out="$(run_step mtp2 MLXFAST_DRAFTER_DIR="${BOX}/drafter")"; rc=$?
[[ ${rc} -eq 0 ]] && grep -q "drafter=${BOX}/drafter" <<<"${out}" \
  && ok "case 4: mtp2 declared with the drafter staged passes" || fail "case 4: rc=${rc}: $(tail -2 <<<"${out}")"
out="$(run_step bad)"; rc=$?
[[ ${rc} -ne 0 ]] && grep -q 'not a contract-permitted draft depth' <<<"${out}" \
  && ok "case 5: an invalid declaration is refused, never read as serial" || fail "case 5: rc=${rc}: $(tail -2 <<<"${out}")"

if [[ ${failures} -eq 0 ]]; then
  echo "test-workflow-drafter-env.sh: all 5 cases passed"
else
  echo "test-workflow-drafter-env.sh: ${failures} case(s) FAILED" >&2
  exit 1
fi
