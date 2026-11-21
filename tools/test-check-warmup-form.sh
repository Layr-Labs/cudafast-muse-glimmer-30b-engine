#!/usr/bin/env bash
# test-check-warmup-form.sh -- tools/check-warmup-form.sh refuses a warm-up
# golden whose token form differs from the live goldens, and passes one in the
# live (raw-continuation) form.
#
# Hermetic: no GPU, no network. The tokenizer and the goldens are fakes made
# here; the fixture is a copy whose tokenizer.json pin names the fake.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
CHECK="${REPO_ROOT}/tools/check-warmup-form.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok    $*"; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# A fake tokenizer: ids 900-903 are added tokens (a chat header and tail).
SNAP="${WORK}/snapshot"; mkdir -p "${SNAP}"
jq -n '{added_tokens: [
  {id: 900, content: "<|start|>", special: true},
  {id: 901, content: "<|end|>", special: true},
  {id: 902, content: "<|user|>", special: true},
  {id: 903, content: "<think>", special: false}],
  model: {type: "BPE"}}' > "${SNAP}/tokenizer.json"
FIXTURE="${WORK}/fixture.json"
jq --arg s "$(sha256_of "${SNAP}/tokenizer.json")" \
  '.target.files = [(.target.files[] | if .path == "tokenizer.json" then .sha256 = $s else . end)]' \
  "${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json" > "${FIXTURE}"

# golden FILE START LEN [PREFIX_IDS_JSON] [SUFFIX_IDS_JSON]: a golden whose
# prefill is PREFIX + plain ids START.. + SUFFIX, LEN ids in all.
golden() {
  local file="$1" start="$2" len="$3" pre="${4:-[]}" suf="${5:-[]}"
  jq -n --argjson s "${start}" --argjson n "${len}" --argjson pre "${pre}" --argjson suf "${suf}" \
    '{version: 1, benchmark: {prefill_prompt_tokens:
       ($pre + [range($s; $s + $n - ($pre | length) - ($suf | length))] + $suf)}}' > "${file}"
}
G="${WORK}/goldens"; mkdir -p "${G}"
golden "${G}/pool-9.golden.json" 10 512
golden "${G}/pool-10.golden.json" 100 512
golden "${G}/pool-11.golden.json" 200 512
LIVE=("${G}/pool-9.golden.json" "${G}/pool-10.golden.json" "${G}/pool-11.golden.json")

run() { "${CHECK}" --fixture "${FIXTURE}" --snapshot "${SNAP}" --warmup "$1" "${LIVE[@]}" > "${WORK}/out" 2>&1; }

# Case 1: a raw-form warm-up (plain ids, 512 long) passes.
golden "${G}/raw.json" 300 512
if run "${G}/raw.json"; then ok "case 1: a raw-form warm-up passes"
else fail "case 1: a raw-form warm-up was refused: $(cat "${WORK}/out")"; fi

# Case 2: a chat-form warm-up (chat header and tail ids, 512 long) is refused.
golden "${G}/chat.json" 300 512 '[900,902,13]' '[901,900,903]'
if run "${G}/chat.json"; then fail "case 2: a chat-form warm-up passed"
elif grep -q 'added token id(s) \[900,901,902,903\]' "${WORK}/out"; then
  ok "case 2: a chat-form warm-up is refused and its chat ids are named"
else fail "case 2: wrong refusal: $(cat "${WORK}/out")"; fi

# Case 3: a raw warm-up of another length is refused.
golden "${G}/short.json" 300 500
if run "${G}/short.json"; then fail "case 3: a 500-id warm-up passed"
elif grep -q 'the warm-up prefill is 500 ids but the live golden pool-9.golden.json has 512' "${WORK}/out"; then
  ok "case 3: a warm-up of another length is refused by name"
else fail "case 3: wrong refusal: $(cat "${WORK}/out")"; fi

# Case 4: an added id that a live golden also holds does not mark the warm-up.
golden "${G}/pool-11.golden.json" 200 512 '[903]'
golden "${G}/think.json" 300 512 '[903]'
if run "${G}/think.json"; then ok "case 4: an added id the live pool also holds passes"
else fail "case 4: refused an id a live golden holds: $(cat "${WORK}/out")"; fi
golden "${G}/pool-11.golden.json" 200 512

# Case 5: a tokenizer.json that is not the pinned one is refused.
cp "${SNAP}/tokenizer.json" "${WORK}/tokenizer.keep"
echo ' ' >> "${SNAP}/tokenizer.json"
if run "${G}/raw.json"; then fail "case 5: an unpinned tokenizer.json passed"
elif grep -q 'the fixture pins' "${WORK}/out"; then ok "case 5: an unpinned tokenizer.json is refused"
else fail "case 5: wrong refusal: $(cat "${WORK}/out")"; fi
cp "${WORK}/tokenizer.keep" "${SNAP}/tokenizer.json"

# Case 6: a golden with no prefill ids is refused.
echo '{}' > "${G}/empty.json"
if run "${G}/empty.json"; then fail "case 6: a golden with no prefill ids passed"
elif grep -q 'has no benchmark.prefill_prompt_tokens id array' "${WORK}/out"; then
  ok "case 6: a golden with no prefill ids is refused"
else fail "case 6: wrong refusal: $(cat "${WORK}/out")"; fi

if [ "${failures}" -ne 0 ]; then
  echo "test-check-warmup-form.sh: ${failures} case(s) failed" >&2
  exit 1
fi
echo "test-check-warmup-form.sh: all 6 cases passed"
