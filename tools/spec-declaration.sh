#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the serve-boot speculative config
# from it. tools/serve-up.sh, .github/workflows/benchmark.yml and
# tools/ranked-box-preflight.sh ALL resolve the serve spec THROUGH this one
# script, so the fail-closed validation and the derivation can never drift
# between the arm gate and the serve that actually boots.
#
# NOT AN EDITABLE PATH. benchmark.json editablePaths is
# {vllm, harness, mtp-head.manifest.json}. The DECLARATION
# (mtp-head.manifest.json) is editable -- a submission EXPRESSES a value there --
# but this DERIVATION is trusted, so a submission cannot rewrite how the value is
# interpreted or relax the envelope.
#
# WHY THIS EXISTS. vLLM fixes its speculative config when the engine is BUILT
# (`speculative_config.num_speculative_tokens`), so the draft depth cannot ride
# the Engine Protocol wire per request: the leg's resident is booted at the
# declared depth (RESIDENT_SPEC_DEPTH, exported by tools/serve-up.sh). This
# script is the declaration -> engine-config bridge. The echo half is in the
# adapter (harness/protocol-adapter: `resolve_spec` -> `effective_spec`), which
# resolves a requested depth against the envelope and echoes what ran; benchd
# seals that echo.
#
# THE DRAFTER IS THE CONTRACT'S, NEVER THE SUBMISSION'S. The fixture's
# `mtp_head` names the one drafter this track can run: source "pinned", a
# repository, a full revision and the sha256 of every file. The organizer
# stages it on the box and tools/serve-up.sh mounts it read-only. A declaration
# can only SWITCH IT ON and PICK A DEPTH. It cannot name a path, a repository, a
# revision or a file of its own: every such key is refused, and a declaration
# that restates the pin must restate it EXACTLY.
#
# THE DECLARATION, in mtp-head.manifest.json:
#
#   {
#     "version": 1,
#     "source": "pinned",                                 # optional; = mtp_head.source
#     "upstream_model_id": "<mtp_head.upstream_model_id>", # with source: required, exact
#     "upstream_revision": "<mtp_head.upstream_revision>", # with source: required, exact
#     "sha256": { "<file>": "<sha256>", ... },            # with source: required, exact
#     "spec": { "decoder": "dflash", "enabled": <bool>, "num_speculative_tokens": <int> }
#   }
#
# ABSENT `spec` block, `enabled: false`, or `num_speculative_tokens: 0` => SERIAL
# (RESIDENT_SPEC_DEPTH=0), the launch-reference behaviour. `enabled: true` with N
# in the contract's permitted_draft_depths => the resident is built with the
# declared decoder (default: the contract's mtp_head.decoder) at N speculative
# tokens. `max_bytes` and `bytes` are recorded, not read.
#
# MODE AND DECODER ARE TWO THINGS. The contract's allowed_modes names benchd's
# WIRE modes. On the CUDA Engine Protocol wire a speculative leg is mode "mtp"
# at depth N: benchd --mtp-depth N, the adapter's effective_spec, the serve
# label mtp<N> and speculative_oracles.<prompt>.mtp<N> all say "mtp". So a
# speculative declaration needs "mtp" in allowed_modes. The DECODER that serves
# that leg is the contract's mtp_head.decoder ("dflash" on this track: the
# pinned DFlash drafter); `decoder` prints it and tools/serve-up.sh boots it.
# `describe` prints "serial" or "mtp<N>", the wire label.
#
# VALIDATION is fail-closed. A REFUSAL (exit 1, message on stderr) for:
#   * a declaration that is not a JSON object, or carries an unknown top-level
#     key -- a weight path, a model directory or an own drafter by any name;
#   * a `source` other than the contract's mtp_head.source -- "remote",
#     "in_branch" and "none" by name on this track;
#   * a "pinned" source whose upstream_model_id, upstream_revision or sha256
#     map is missing or differs in any way from the contract's pin, or pin keys
#     stated without a source;
#   * a `spec` block that is not an object, or carries an unknown key;
#   * a `decoder` that is not a string or not the contract's mtp_head.decoder;
#   * an enabled declaration on a contract whose allowed_modes lacks "mtp";
#   * `enabled` that is not a boolean, or `num_speculative_tokens` not an integer;
#   * a value outside the structural range 0..8 (the structural sanity ceiling);
#   * an ENABLED depth that is not in the contract's permitted_draft_depths
#     (the same envelope harness/protocol-adapter/src/session_backend.rs mirrors
#     as MTP_MIN_DEPTH..=MTP_MAX_DEPTH). An out-of-envelope depth is REFUSED,
#     never clamped.
#
# Usage:
#   tools/spec-declaration.sh speculative   # 0 (serial) or 1 (speculative)
#   tools/spec-declaration.sh decoder       # the decoder a speculative leg runs (e.g. dflash)
#   tools/spec-declaration.sh draft-len     # num_speculative_tokens (0 = serial)
#   tools/spec-declaration.sh describe      # "serial" or "mtp<N>" (identity label)
#   tools/spec-declaration.sh validate      # exit 0 if valid, else refuse
# Every subcommand validates the declaration first and refuses on an invalid one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json}"

# The structural sanity ceiling for a declared draft length. The contract's
# permitted_draft_depths is the tighter, AUTHORITATIVE set enforced below when
# the value is enabled; this is the outer type/range guard around it.
SPEC_MAX_TOKENS=8

fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"
[[ -f "${CONTRACT}" ]] || fail "the track contract is missing: ${CONTRACT}"

# --- the contract's drafter -------------------------------------------------
contract_source="$(jq -r '.mtp_head.source // "pinned"' "${CONTRACT}")"
contract_decoder="$(jq -r '.mtp_head.decoder // "mtp"' "${CONTRACT}")"
# A speculative leg rides the wire as mode "mtp" (see the header).
speculative_mode_allowed="$(jq -r '(.allowed_modes // []) | index("mtp") != null' "${CONTRACT}")"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist.
enabled="false"
raw_tokens="0"
decoder="${contract_decoder}"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"
  [[ "$(jq -r 'type' "${MANIFEST}")" == "object" ]] \
    || fail "mtp-head.manifest.json must be a JSON object with keys {version, source, upstream_model_id, upstream_revision, sha256, max_bytes, bytes, spec}"

  # A typo'd or invented top-level key must not read as its default. A key that
  # would point the engine at weights -- a path, a model directory, a drafter of
  # the submission's own -- is the case that matters: the drafter is the
  # contract's pinned one, staged by the organizer, and nothing else is loaded.
  unknown_top="$(jq -r 'keys[] | select(IN("version","source","upstream_model_id","upstream_revision","sha256","max_bytes","bytes","spec") | not)' "${MANIFEST}")"
  [[ -z "${unknown_top}" ]] \
    || fail "the declaration carries unknown top-level key(s): $(printf '%s' "${unknown_top}" | tr '\n' ' '); allowed keys are version, source, upstream_model_id, upstream_revision, sha256, max_bytes, bytes, spec. A declaration cannot name weights or a path: the drafter is the contract's pinned one (fixture mtp_head)"

  has_pin_keys="$(jq -r '[has("upstream_model_id"), has("upstream_revision"), has("sha256")] | any' "${MANIFEST}")"
  if [[ "$(jq -r 'has("source")' "${MANIFEST}")" == "true" ]]; then
    declared_source="$(jq -r '.source | if type == "string" then . else tojson end' "${MANIFEST}")"
    [[ "${declared_source}" == "${contract_source}" ]] \
      || fail "source \"${declared_source}\" is not accepted; the contract's mtp_head.source is \"${contract_source}\", so the only accepted source is \"${contract_source}\""
    if [[ "${declared_source}" == "pinned" ]]; then
      # THE PIN, RESTATED EXACTLY: the repository, the full revision and the
      # sha256 of every file, no more and no fewer files.
      want_repo="$(jq -r '.mtp_head.upstream_model_id // empty' "${CONTRACT}")"
      want_rev="$(jq -r '.mtp_head.upstream_revision // empty' "${CONTRACT}")"
      want_sha="$(jq -c '[(.mtp_head.files // [])[] | {key: .path, value: .sha256}] | from_entries' "${CONTRACT}")"
      [[ -n "${want_repo}" && -n "${want_rev}" && "${want_sha}" != "{}" ]] \
        || fail "the contract's mtp_head is source \"pinned\" but pins no upstream_model_id, upstream_revision or files"
      got_repo="$(jq -r '.upstream_model_id | if type == "string" then . else tojson end' "${MANIFEST}")"
      [[ "${got_repo}" == "${want_repo}" ]] \
        || fail "source \"pinned\" names upstream_model_id \"${got_repo}\"; the contract pins \"${want_repo}\" and accepts no other drafter"
      got_rev="$(jq -r '.upstream_revision | if type == "string" then . else tojson end' "${MANIFEST}")"
      [[ "${got_rev}" == "${want_rev}" ]] \
        || fail "source \"pinned\" names upstream_revision \"${got_rev}\"; the contract pins revision ${want_rev} (the full 40-hex sha) and accepts no other"
      [[ "$(jq -r '.sha256 | type' "${MANIFEST}")" == "object" ]] \
        || fail "source \"pinned\" needs sha256 as an object {file: sha256} equal to the contract's mtp_head.files"
      jq -e --argjson want "${want_sha}" '.sha256 == $want' "${MANIFEST}" >/dev/null \
        || fail "source \"pinned\" declares sha256 $(jq -c '.sha256' "${MANIFEST}"); the contract pins exactly ${want_sha} and accepts no other file or digest"
    elif [[ "${has_pin_keys}" == "true" ]]; then
      fail "upstream_model_id, upstream_revision and sha256 are only accepted with source \"pinned\""
    fi
  elif [[ "${has_pin_keys}" == "true" ]]; then
    fail "upstream_model_id, upstream_revision and sha256 are only accepted with source \"${contract_source}\" stated beside them"
  fi

  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {decoder, enabled, num_speculative_tokens}"
    unknown="$(jq -r '.spec | keys[] | select(IN("decoder","enabled","num_speculative_tokens") | not)' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are decoder, enabled, num_speculative_tokens. A declaration cannot name weights or a path: the drafter is the contract's pinned one (fixture mtp_head)"
    if [[ "$(jq -r '.spec | has("decoder")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.decoder | type' "${MANIFEST}")" == "string" ]] \
        || fail "spec.decoder must be a string (got $(jq -r '.spec.decoder | type' "${MANIFEST}"))"
      decoder="$(jq -r '.spec.decoder' "${MANIFEST}")"
      [[ "${decoder}" == "${contract_decoder}" ]] \
        || fail "spec.decoder='${decoder}' is not this track's decoder; the contract's mtp_head.decoder is '${contract_decoder}', the only one a leg can run"
    fi
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 needs the wire mode mtp in allowed_modes and a CONTRACT-permitted depth.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  [[ "${speculative_mode_allowed}" == "true" ]] \
    || fail "the declaration enables speculation, but the contract's allowed_modes $(jq -c '.allowed_modes // []' "${CONTRACT}") does not name the wire mode mtp; this track runs serial only"
  if jq -e '.mtp_head.permitted_draft_depths | arrays' >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" '(.mtp_head.permitted_draft_depths | index($n)) != null' "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r '.mtp_head.permitted_draft_depths | map(tostring) | join(", ")' "${CONTRACT}")"
      fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted draft depth (permitted_draft_depths: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  decoder)     echo "${decoder}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "mtp${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|decoder|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
