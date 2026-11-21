#!/usr/bin/env bash
# check-warmup-form.sh -- refuse a warm-up golden whose token form tells it
# apart from the live goldens.
#
# WHY. Every warm-up pass runs the never-scored warm-up prompt (the prompt-reuse
# guard). If its prefill ids look different from the scored prompts -- a chat
# header the live prompts do not have, or another length -- an engine can tell
# the warm-up from a scored prompt by its tokens alone and treat the two
# differently. The live pool is the raw-continuation form: the first
# seed_tokens ids of a text, with no chat template and no special ids.
#
# WHAT IT CHECKS, on benchmark.prefill_prompt_tokens of each golden:
#   1. The warm-up's prefill length equals the prefill length of every live
#      golden.
#   2. The warm-up holds no added token id (a special or chat-template id, read
#      from the pinned tokenizer.json `added_tokens`) that none of the live
#      goldens hold.
# The tokenizer.json is the snapshot's own file, checked against the fixture's
# target.files pin first. Nothing is fetched and nothing is written.
#
# Usage:  tools/check-warmup-form.sh --fixture FILE --snapshot DIR \
#           --warmup WARMUP.golden.json LIVE.golden.json [LIVE.golden.json ...]
#
# Exit codes
#   0  the warm-up has the live form
#   1  refused: the warm-up form differs, or an input is not readable or valid
#   2  usage error
set -euo pipefail

FIXTURE="" SNAPSHOT="" WARMUP=""
LIVE=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fixture)  FIXTURE="${2:-}"; shift 2 ;;
    --snapshot) SNAPSHOT="${2:-}"; shift 2 ;;
    --warmup)   WARMUP="${2:-}"; shift 2 ;;
    -h|--help)  sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "check-warmup-form: unknown argument $1" >&2; exit 2 ;;
    *)  LIVE+=("$1"); shift ;;
  esac
done
[ -n "${FIXTURE}" ] && [ -n "${SNAPSHOT}" ] && [ -n "${WARMUP}" ] && [ "${#LIVE[@]}" -gt 0 ] || {
  echo "usage: check-warmup-form.sh --fixture FILE --snapshot DIR --warmup FILE LIVE..." >&2
  exit 2
}

refuse() { echo "check-warmup-form: REFUSE: $*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || refuse "jq is required"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# The pinned tokenizer.json: its digest must equal the fixture pin.
TOKENIZER="${SNAPSHOT}/tokenizer.json"
[ -r "${TOKENIZER}" ] || refuse "cannot read ${TOKENIZER}"
want_tok="$(jq -r '[.target.files[]? | select(.path == "tokenizer.json") | .sha256][0] // ""' "${FIXTURE}")"
[[ "${want_tok}" =~ ^[0-9a-f]{64}$ ]] || refuse "${FIXTURE} pins no tokenizer.json sha256"
have_tok="$(sha256_of "${TOKENIZER}")"
[ "${have_tok}" = "${want_tok}" ] || refuse "${TOKENIZER} is sha256 ${have_tok}; the fixture pins ${want_tok}"
added="$(jq -c '[.added_tokens[]? | .id | select(type == "number")] | unique' "${TOKENIZER}")" \
  || refuse "${TOKENIZER} is not valid JSON"

# prefill_ids FILE: the golden's prefill ids as a JSON array, or a refusal.
prefill_ids() {
  [ -r "$1" ] || refuse "cannot read the golden $1"
  jq -ce '.benchmark.prefill_prompt_tokens
          | if type == "array" and length > 0 and all(.[]; type == "number") then . else empty end' "$1" \
    2>/dev/null || refuse "$1 has no benchmark.prefill_prompt_tokens id array"
}

warm="$(prefill_ids "${WARMUP}")"
warm_len="$(jq 'length' <<<"${warm}")"
live_added='[]'
for golden in "${LIVE[@]}"; do
  ids="$(prefill_ids "${golden}")"
  len="$(jq 'length' <<<"${ids}")"
  [ "${len}" = "${warm_len}" ] \
    || refuse "the warm-up prefill is ${warm_len} ids but the live golden ${golden##*/} has ${len}; the warm-up must have the live length"
  live_added="$(jq -cn --argjson a "${live_added}" --argjson ids "${ids}" --argjson add "${added}" \
    '($a + [$ids[] | select(. as $i | $add | index($i))]) | unique')"
done

extra="$(jq -cn --argjson w "${warm}" --argjson add "${added}" --argjson live "${live_added}" \
  '[$w[] | select(. as $i | ($add | index($i)) and (($live | index($i)) | not))] | unique')"
if [ "${extra}" != "[]" ]; then
  refuse "the warm-up prefill holds added token id(s) ${extra} (special or chat-template ids) that no live golden holds; record the warm-up in the live form (raw continuation, no chat template)"
fi
echo "check-warmup-form: ok: the warm-up prefill is ${warm_len} ids like the ${#LIVE[@]} live golden(s) and holds no added token id they lack"
