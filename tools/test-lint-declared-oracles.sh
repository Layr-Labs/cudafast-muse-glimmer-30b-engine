#!/usr/bin/env bash
# test-lint-declared-oracles.sh -- tools/lint-benchmark-manifest.py requires a
# per-depth oracle ONLY for the depth the tree's declaration enables.
#
# Asserts, on a scratch root (every entry a link to this checkout, except
# mtp-head.manifest.json, which each case writes):
#   1. a serial declaration passes with no per-depth oracle pinned;
#   2. an absent declaration passes the same way (absent = serial);
#   3. a declaration that enables mtp3 with no pinned mtp3 oracle fails, once per
#      live golden, naming the declared depth;
#   4. the same declaration passes when each live golden pins its mtp3 oracle
#      (and no other depth);
#   5. an invalid declaration (depth 7, outside permitted_draft_depths) fails
#      rather than reading as serial.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FIXTURE_REL="$(jq -r '.contractPath' "${ROOT_DIR}/benchmark.json")"
mapfile -t NAMES < <(jq -r '.live_goldens[]' "${ROOT_DIR}/${FIXTURE_REL}")

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

R="${WORK}/root"
mkdir -p "${R}"
for entry in "${ROOT_DIR}"/* "${ROOT_DIR}"/.github "${ROOT_DIR}"/.gitmodules; do
  [[ -e "${entry}" ]] || continue
  [[ "$(basename "${entry}")" == mtp-head.manifest.json ]] && continue
  ln -s "${entry}" "${R}/$(basename "${entry}")"
done

# The serial declaration each case starts from: the contract's pin, drafter
# off. Not the shipped file: that is the base's, which may switch it on.
SERIAL_DECL="${WORK}/serial.manifest.json"
jq -f "${ROOT_DIR}/tools/testlib/serial-declaration.jq" "${ROOT_DIR}/${FIXTURE_REL}" > "${SERIAL_DECL}"

# lint_with DECL_JQ [FIXTURE_JQ] -- DECL_JQ edits the serial declaration
# ("absent" writes none); FIXTURE_JQ edits a copy of the fixture.
lint_with() {
  if [[ "$1" == absent ]]; then
    rm -f "${R}/mtp-head.manifest.json"
  else
    jq "$1" "${SERIAL_DECL}" > "${R}/mtp-head.manifest.json"
  fi
  jq "${2:-.}" "${ROOT_DIR}/${FIXTURE_REL}" > "${WORK}/fixture.json"
  jq --arg c "${WORK}/fixture.json" '.contractPath = $c' "${ROOT_DIR}/benchmark.json" > "${WORK}/benchmark.json"
  python3 "${ROOT_DIR}/tools/lint-benchmark-manifest.py" --repo-root "${R}" --manifest "${WORK}/benchmark.json" 2>&1
}

out="$(lint_with '.')"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "no per-depth oracle needed (the declaration is serial)" <<<"${out}" \
  && ok "case 1: the serial declaration passes with no per-depth oracle pinned" \
  || fail "case 1: rc=${rc}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

out="$(lint_with absent)"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "no per-depth oracle needed (the declaration is serial)" <<<"${out}" \
  && ok "case 2: an absent declaration passes as serial" \
  || fail "case 2: rc=${rc}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

ON3='.spec.enabled = true | .spec.num_speculative_tokens = 3'
out="$(lint_with "${ON3}" '.speculative_oracles = {}')"; rc=$?
n="$(grep -c "has no pinned oracle speculative_oracles\.[a-z0-9_-]*\.mtp3 at .*; the declaration enables mtp3, and a declared depth with no pinned oracle cannot be scored" <<<"${out}")"
other="$(grep -E '^FAIL' <<<"${out}" | grep -vc 'mtp3' || true)"
[[ "${rc}" -ne 0 && "${n}" == "${#NAMES[@]}" && "${other}" == 0 ]] \
  && ok "case 3: mtp3 declared with no pinned mtp3 oracle fails once per live golden (${n}), naming the depth" \
  || fail "case 3: rc=${rc} mtp3 findings=${n} (want ${#NAMES[@]}) other=${other}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

PIN3="$(jq -nc --args '$ARGS.positional' "${NAMES[@]}")"
out="$(lint_with "${ON3}" "(.timed_prompt_pool[0].r2_path | sub(\"/[^/]*\$\"; \"\")) as \$p
  | .speculative_oracles = (reduce ${PIN3}[] as \$n ({}; .[\$n] = {mtp3: {r2_path: \"\(\$p)/\(\$n).mtp3.golden.json\", sha256: (\"a\" * 64), bytes: 1}}))")"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "a pinned per-depth oracle for the declared mtp3" <<<"${out}" \
  && ok "case 4: mtp3 declared with every mtp3 oracle pinned (no other depth) passes" \
  || fail "case 4: rc=${rc}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

out="$(lint_with '.spec.enabled = true | .spec.num_speculative_tokens = 7')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "the speculative-decode declaration is invalid: .*not a contract-permitted draft depth" <<<"${out}" \
  && ok "case 5: an invalid declaration fails, never read as serial" \
  || fail "case 5: rc=${rc}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-lint-declared-oracles.sh: all 5 cases passed"
else
  echo "test-lint-declared-oracles.sh: ${failures} case(s) FAILED" >&2
  exit 1
fi
