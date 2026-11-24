#!/usr/bin/env bash
# Unit test for tools/spec-declaration.sh. Offline, no GPU, no serve -- it drives
# the trusted declaration->serve-spec deriver against the REAL contract fixture
# with a series of throwaway manifests, so the envelope it enforces is the one
# that actually ships.
#
# WHAT THIS PROVES: the drafter is the CONTRACT's. muse-glimmer-30b-cuda-v1 on
# the drafter track pins one DFlash drafter (fixture mtp_head: source "pinned",
# repository, full revision, sha256 per file) and permits draft depths 1..6. A
# declaration may switch that drafter on and pick a permitted depth. Everything
# else is REFUSED here, before any serve boots:
#   * a pin that differs from the contract's in the repository, the revision
#     (a short sha included) or any sha256, or that adds or drops a file;
#   * any other source ("remote", "in_branch", "none");
#   * any key that would point the engine at weights of the submission's own;
#   * a depth outside permitted_draft_depths, or a decoder the contract does
#     not list.
# The accepting side is pinned too, so a fix that simply refused everything
# would not pass.
#
# Usage: tools/test-spec-declaration.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DECL="${SCRIPT_DIR}/tools/spec-declaration.sh"
CONTRACT="${SCRIPT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
SHIPPED="${SCRIPT_DIR}/mtp-head.manifest.json"
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

# decl <jq-filter> -- the PINNED declaration with the filter applied, compact.
# The pin is read from the contract, so the accepted form is exactly the
# contract's and every refusal below differs from it in one named way.
PINNED="$(jq -c '{version: 1, source: .mtp_head.source,
  upstream_model_id: .mtp_head.upstream_model_id, upstream_revision: .mtp_head.upstream_revision,
  sha256: ([.mtp_head.files[] | {key: .path, value: .sha256}] | from_entries),
  spec: {decoder: "dflash", enabled: false, num_speculative_tokens: 0}}' "${CONTRACT}")"
decl() { jq -c "$1" <<<"${PINNED}"; }

# --- the contract this test pins --------------------------------------------
envelope="$(jq -c '{source: .mtp_head.source, decoder: .mtp_head.decoder, repo: .mtp_head.upstream_model_id,
  rev: .mtp_head.upstream_revision, depths: .mtp_head.permitted_draft_depths, modes: .allowed_modes,
  files: [.mtp_head.files[].path]}' "${CONTRACT}")"
want_envelope='{"source":"pinned","decoder":"dflash","repo":"meta-models/Muse-Glimmer-30B-assistant","rev":"c676cce91658f9568aaf264a416211e2782735b7","depths":[1,2,3,4,5,6],"modes":["serial","mtp"],"files":["config.json","model.safetensors"]}'
if [[ "${envelope}" == "${want_envelope}" ]]; then
  echo "PASS contract envelope (${envelope})"
else
  echo "FAIL contract envelope: ${envelope}, expected ${want_envelope} -- update this test deliberately or revert the fixture"
  fails=$((fails + 1))
fi
jq -e '.mtp_head.files | length == 2 and all(.[]; (.sha256 | test("^[0-9a-f]{64}$")) and (.bytes | type == "number" and . > 0))' "${CONTRACT}" >/dev/null \
  && echo "PASS contract pins every drafter file by bytes and sha256" \
  || { echo "FAIL contract: an mtp_head.files entry lacks a 64-hex sha256 or a positive byte count"; fails=$((fails + 1)); }

# --- the shipped declaration --------------------------------------------------
# The exact pin, either with the drafter off (serial) or switched on at a
# permitted depth whose timed oracle every live golden pins (the drafter
# track's base ships mtp4). Anything else in the spec fails here.
got="$(SPEC_DECLARATION_MANIFEST="${SHIPPED}" "${DECL}" describe 2>&1)"
shipped_n="$(jq -r '.spec.num_speculative_tokens // 0' "${SHIPPED}")"
if jq -e --argjson p "${PINNED}" '. == $p' "${SHIPPED}" >/dev/null && [[ "${got}" == "serial" ]]; then
  echo "PASS shipped declaration is the exact pin, disabled (describe => ${got})"
elif jq -e --argjson p "${PINNED}" --argjson n "${shipped_n}" '. == ($p | .spec.enabled = true | .spec.num_speculative_tokens = $n)' \
    "${SHIPPED}" >/dev/null 2>&1 && [[ "${got}" == "mtp${shipped_n}" ]] \
  && jq -e --arg k "mtp${shipped_n}" '(.live_goldens | length) > 0 and all(.live_goldens[] as $g | .speculative_oracles[$g][$k];
       . != null and (.sha256 | test("^[0-9a-f]{64}$")) and (.bytes | type == "number" and . > 0))' "${CONTRACT}" >/dev/null; then
  echo "PASS shipped declaration is the exact pin, on at a permitted depth with every live golden's oracle pinned (describe => ${got})"
else
  echo "FAIL shipped declaration: describe='${got}'; the file is not the exact pin with the drafter off, nor on at a permitted depth whose oracles the contract pins"
  fails=$((fails + 1))
fi

# --- serial (the no-op) -----------------------------------------------------
expect_ok  "absent-spec-is-serial"        '{}' describe serial
expect_ok  "disabled-is-serial"           '{"spec":{"enabled":false,"num_speculative_tokens":0}}' describe serial
expect_ok  "enabled-zero-is-serial"       '{"spec":{"enabled":true,"num_speculative_tokens":0}}'  describe serial
expect_ok  "pinned-disabled-is-serial"    "$(decl .)" describe serial

# --- pinned accepted: switch on + pick a permitted depth --------------------
for d in 1 2 3 4 5 6; do
  expect_ok "pinned-depth-${d}-accepted" "$(decl ".spec.enabled = true | .spec.num_speculative_tokens = ${d}")" describe "mtp${d}"
done
expect_ok "pinned-decoder-is-dflash"   "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 3')" decoder dflash
expect_ok "pinned-speculative-is-1"    "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 3')" speculative 1
expect_ok "pinned-draft-len"           "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 3')" draft-len 3
expect_ok "absent-decoder-defaults"    '{"version":1,"spec":{"enabled":true,"num_speculative_tokens":2}}' decoder dflash
expect_ok "absent-source-enabled-ok"   '{"version":1,"spec":{"enabled":true,"num_speculative_tokens":2}}' describe mtp2
expect_ok "max-bytes-not-read"         "$(decl '.max_bytes = 3000000000')" describe serial

# --- wrong revision refused -------------------------------------------------
expect_refuse "wrong-revision-refused" "$(decl '.upstream_revision = "0000000000000000000000000000000000000000"')" \
  'the contract pins revision c676cce91658f9568aaf264a416211e2782735b7'
expect_refuse "short-revision-refused" "$(decl '.upstream_revision = "c676cce9"')" 'names upstream_revision "c676cce9"'
expect_refuse "main-revision-refused"  "$(decl '.upstream_revision = "main"')" 'names upstream_revision "main"'
expect_refuse "missing-revision-refused" "$(decl 'del(.upstream_revision)')" 'names upstream_revision "null"'
expect_refuse "wrong-repo-refused"     "$(decl '.upstream_model_id = "someone/own-drafter"')" 'accepts no other drafter'

# --- wrong sha refused --------------------------------------------------------
expect_refuse "wrong-weights-sha-refused" "$(decl '.sha256["model.safetensors"] = ("0" * 64)')" 'accepts no other file or digest'
expect_refuse "wrong-config-sha-refused"  "$(decl '.sha256["config.json"] = ("f" * 64)')"     'accepts no other file or digest'
expect_refuse "extra-file-sha-refused"    "$(decl '.sha256["own.safetensors"] = ("a" * 64)')" 'accepts no other file or digest'
expect_refuse "dropped-file-sha-refused"  "$(decl 'del(.sha256["config.json"])')"            'accepts no other file or digest'
expect_refuse "sha-as-string-refused"     "$(decl '.sha256 = ("0" * 64)')"                   'needs sha256 as an object'
expect_refuse "pin-without-source-refused" "$(decl 'del(.source)')" 'only accepted with source "pinned" stated beside them'

# --- depth outside the list refused ----------------------------------------
expect_refuse "depth-7-refused" "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 7')" "not a contract-permitted draft depth"
expect_refuse "depth-8-refused" "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 8')" "not a contract-permitted draft depth"
expect_refuse "depth-9-refused" "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 9')" "outside the permitted range 0..8"
expect_refuse "negative-refused" '{"spec":{"enabled":true,"num_speculative_tokens":-1}}' "outside the permitted range 0..8"

# --- entry-supplied weights refused -----------------------------------------
# Every way a declaration could point the engine at weights of its own: a path
# key at the top or in the spec block, a source that loads from the branch or
# the network, and the old "none" source of the serial track.
expect_refuse "weight-path-top-refused"  "$(decl '.path = "harness/my-drafter"')"            "cannot name weights or a path"
expect_refuse "draft-model-top-refused"  "$(decl '.draft_model = "/home/runner/own"')"        "cannot name weights or a path"
expect_refuse "weights-dir-top-refused"  "$(decl '.weights_dir = "harness/w"')"               "cannot name weights or a path"
expect_refuse "spec-model-refused"       "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 2 | .spec.model = "harness/my-drafter"')" "cannot name weights or a path"
expect_refuse "spec-draft-dir-refused"   "$(decl '.spec.draft_model_dir = "/tmp/x"')"         "cannot name weights or a path"
expect_refuse "source-in-branch-refused" "$(decl '.source = "in_branch"')" 'source "in_branch" is not accepted'
expect_refuse "source-remote-refused"    "$(decl '.source = "remote"')"    'source "remote" is not accepted'
expect_refuse "source-none-refused"      '{"source":"none"}'               'the contract'"'"'s mtp_head.source is "pinned"'

# --- decoder ----------------------------------------------------------------
expect_refuse "decoder-mtp-refused"      "$(decl '.spec.decoder = "mtp"')"   "spec.decoder='mtp' is not this track's decoder"
expect_refuse "decoder-eagle-refused"    "$(decl '.spec.decoder = "eagle"')" "spec.decoder='eagle' is not this track's decoder"
expect_refuse "decoder-empty-refused"    "$(decl '.spec.decoder = ""')"      "is not this track's decoder"
expect_refuse "decoder-type-refused"     "$(decl '.spec.decoder = 1')"       "spec.decoder must be a string"

# --- the wire mode ----------------------------------------------------------
# A speculative leg rides the wire as mode mtp. A contract whose allowed_modes
# lacks it runs serial only, whatever the declaration says.
SERIAL_ONLY="${WORK}/serial-only-contract.json"
jq '.allowed_modes = ["serial"]' "${CONTRACT}" > "${SERIAL_ONLY}"
err="$(SPEC_DECLARATION_MANIFEST="$(manifest "$(decl '.spec.enabled = true | .spec.num_speculative_tokens = 2')")" \
  SPEC_DECLARATION_CONTRACT="${SERIAL_ONLY}" "${DECL}" describe 2>&1 >/dev/null)" && rc=0 || rc=$?
[[ ${rc} -ne 0 && "${err}" == *"does not name the wire mode mtp"* ]] \
  && echo "PASS no-mtp-mode-refused (refused: ${err##*REFUSING -- })" \
  || { echo "FAIL no-mtp-mode-refused: rc=${rc} err='${err}'"; fails=$((fails + 1)); }

# --- malformed declarations refuse ------------------------------------------
expect_refuse "non-integer-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1.5}}' "must be an integer"
expect_refuse "unknown-key-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1,"foo":1}}' "unknown key"
expect_refuse "bad-enabled-refused" '{"spec":{"enabled":"yes","num_speculative_tokens":1}}' "must be a boolean"
expect_refuse "unknown-top-key-refused" '{"version":1,"arm":"dflash","spec":{"enabled":true,"num_speculative_tokens":1}}' \
  "unknown top-level key(s): arm"
expect_refuse "not-an-object-refused" '[1]' "must be a JSON object"

if [[ ${fails} -eq 0 ]]; then
  echo "OK: all spec-declaration envelope cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
