#!/usr/bin/env bash
# Unit test for tools/spec-declaration.sh. Offline, no GPU, no serve -- it drives
# the trusted declaration->serve-spec deriver against the REAL contract fixture
# with a series of throwaway manifests, so the envelope it enforces is the one
# that actually ships.
#
# WHAT THIS PROVES, and it is the part worth having: the envelope REFUSALS. The
# CUDA track's draft depth is a serve-BOOT parameter, and this script is the one
# trusted bridge that turns the participant's declaration into that parameter --
# so a non-serial declaration on a SERIAL-ONLY track must be REFUSED here,
# before any serve boots. muse-glimmer-30b-cuda-v1 declares
# mtp_head.permitted_draft_depths [] (contract supported_optimization_modes
# [serial], and the checkpoint ships no MTP tensors), so every depth 1..9 is
# refused and only the serial forms are accepted. If a future fixture edit ever
# arms a draft depth, the depth-1 case below turns this test red -- that is the
# tripwire, in the other direction.
#
# It also pins the rest of the DECLARATION, which the same script validates:
# source, the max_bytes cap and the top-level key set (issue #24 work item A).
#
# Usage: tools/test-spec-declaration.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DECL="${SCRIPT_DIR}/tools/spec-declaration.sh"
CONTRACT="${SCRIPT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fails=0
mani=0

# Write a throwaway manifest and echo its path.
manifest() {
  mani=$((mani + 1))
  local path="${WORK}/manifest-${mani}.json"
  printf '%s' "$1" > "${path}"
  printf '%s' "${path}"
}

# expect_ok <label> <manifest-json> <subcommand> <expected-stdout>
expect_ok() {
  local label="$1" json="$2" sub="$3" want="$4" got rc
  got="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" "${sub}" 2>/dev/null)"
  rc=$?
  if [[ ${rc} -eq 0 && "${got}" == "${want}" ]]; then
    echo "PASS ${label} (${sub} => ${got})"
  else
    echo "FAIL ${label}: ${sub} rc=${rc} got='${got}' want rc=0 '${want}'"
    fails=$((fails + 1))
  fi
}

# expect_refuse <label> <manifest-json> <needle-in-stderr>
expect_refuse() {
  local label="$1" json="$2" needle="$3" err rc
  err="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" draft-len 2>&1 >/dev/null)"
  rc=$?
  if [[ ${rc} -ne 0 && "${err}" == *"${needle}"* ]]; then
    echo "PASS ${label} (refused: ${err##*REFUSING -- })"
  else
    echo "FAIL ${label}: expected refusal containing '${needle}', got rc=${rc} err='${err}'"
    fails=$((fails + 1))
  fi
}

# --- the contract this test pins --------------------------------------------
# The whole point is to read the SHIPPING envelope, not a copy. Assert it up
# front so a reader knows exactly what the accept/refuse split below rests on.
permitted="$(jq -c '.mtp_head.permitted_draft_depths' "${CONTRACT}")"
if [[ "${permitted}" != "[]" ]]; then
  echo "FAIL contract envelope: mtp_head.permitted_draft_depths is ${permitted}, expected [] -- this track is serial only (contract supported_optimization_modes [serial]); update this test deliberately or revert the fixture"
  fails=$((fails + 1))
fi

# --- serial (the no-op) -----------------------------------------------------
expect_ok  "absent-spec-is-serial"        '{}' describe serial
expect_ok  "disabled-is-serial"           '{"spec":{"enabled":false,"num_speculative_tokens":0}}' describe serial
expect_ok  "enabled-zero-is-serial"       '{"spec":{"enabled":true,"num_speculative_tokens":0}}'  describe serial

# --- the serial-only restriction: every draft depth is refused --------------
# 1..8 are inside the structural 0..8 ceiling but OUTSIDE permitted_draft_depths
# (empty), so they must be refused by the contract-membership check. This is the
# tripwire for "no speculative arm": arming a depth reddens the depth-1 case.
expect_refuse "mtp1-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":1}}' "not a contract-permitted draft depth"
expect_refuse "mtp2-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":2}}' "not a contract-permitted draft depth"
expect_refuse "mtp3-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":3}}' "not a contract-permitted draft depth"
expect_refuse "mtp4-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":4}}' "not a contract-permitted draft depth"
expect_refuse "mtp5-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":5}}' "not a contract-permitted draft depth"
expect_refuse "mtp6-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":6}}' "not a contract-permitted draft depth"
expect_refuse "mtp7-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":7}}' "not a contract-permitted draft depth"
expect_refuse "mtp8-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":8}}' "not a contract-permitted draft depth"
# 9 is beyond the structural ceiling and refuses there, before the membership check.
expect_refuse "mtp9-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":9}}' "outside the permitted range 0..8"

# --- malformed declarations refuse ------------------------------------------
expect_refuse "negative-refused"    '{"spec":{"enabled":true,"num_speculative_tokens":-1}}' "outside the permitted range 0..8"
expect_refuse "non-integer-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1.5}}' "must be an integer"
expect_refuse "unknown-key-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1,"foo":1}}' "unknown key"
expect_refuse "bad-enabled-refused" '{"spec":{"enabled":"yes","num_speculative_tokens":1}}' "must be a boolean"

# --- the declaration AROUND the spec block (issue #24 work item A) ----------
# docs/participant-contract.md 4.1 promises two things this validator did not
# check, because it read `.spec` and nothing else: `pinned` is the only accepted
# source, and there is no `arm` key. Each promise gets its refusal case and the
# accepting side gets one too, so a fix that simply refused everything would not
# pass. max_bytes is recorded, not read, so any value passes.
expect_refuse "source-remote-refused"    '{"source":"remote"}'    'the only accepted source is "pinned"'
expect_refuse "source-in-branch-refused" '{"source":"in_branch"}' 'source "in_branch" is not accepted'
expect_refuse "unknown-top-key-refused"    '{"version":1,"arm":"dflash","spec":{"enabled":true,"num_speculative_tokens":1}}' \
  "unknown top-level key(s): arm"
# The shipped declaration is legal, a max_bytes above the staged head's size is
# legal too (the key is not read), and a declaration that states only a version
# and a spec block is the ordinary case.
expect_ok "shipped-declaration-accepted" '{"source":"pinned","max_bytes":2147483648}' describe serial
expect_ok "max-bytes-not-read"           '{"source":"pinned","max_bytes":3000000000}' describe serial
# version + spec is the ordinary shape, but a spec that ARMS a depth is refused
# on this serial-only track (see the mtp cases above).
expect_refuse "version-plus-spec-refused" '{"version":1,"spec":{"enabled":true,"num_speculative_tokens":1}}' "not a contract-permitted draft depth"

if [[ ${fails} -eq 0 ]]; then
  echo "OK: all spec-declaration envelope cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
