#!/usr/bin/env bash
# test-lint-scoring-limits.sh -- tools/lint-benchmark-manifest.py compares the
# decode and prefill ceilings across benchmark.json and the track fixture.
#
# Asserts, on scratch copies of the two files (the repository is read only):
#   1. the files as committed pass;
#   2. a fixture with no decode_speedup_ceiling fails: benchd applies no
#      ceiling the fixture does not declare;
#   3. a fixture ceiling that differs from benchmark.json's fails;
#   4. a fixture ceiling at or below 1 fails;
#   5-7. the same three refusals for prefill_speedup_ceiling;
#   8-9. a fixture with no seed_window_max_ratio, or one at 1, fails.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FIXTURE_REL="$(jq -r '.contractPath' "${ROOT_DIR}/benchmark.json")"
CEILING="$(jq -r '.decode_speedup_ceiling' "${ROOT_DIR}/${FIXTURE_REL}")"
PREFILL_CEILING="$(jq -r '.prefill_speedup_ceiling' "${ROOT_DIR}/${FIXTURE_REL}")"
SEED_RATIO="$(jq -r '.seed_window_max_ratio' "${ROOT_DIR}/${FIXTURE_REL}")"

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

# lint_with JQ -- lint benchmark.json against the fixture edited by JQ.
# The fixture copy carries a stand-in pin for each per-depth oracle the box has
# not recorded yet (tools/testlib/standin-oracles.jq): this suite is about the
# scoring limits, and the lint run on the committed tree reports that gap.
lint_with() {
  jq -f "${ROOT_DIR}/tools/testlib/standin-oracles.jq" "${ROOT_DIR}/${FIXTURE_REL}" | jq "$1" > "${WORK}/fixture.json"
  jq --arg c "${WORK}/fixture.json" '.contractPath = $c' "${ROOT_DIR}/benchmark.json" > "${WORK}/benchmark.json"
  python3 "${ROOT_DIR}/tools/lint-benchmark-manifest.py" --manifest "${WORK}/benchmark.json" 2>&1
}

out="$(lint_with '.')"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "states the fixture's decode_speedup_ceiling = ${CEILING}" <<<"${out}" \
  && grep -q "states the fixture's prefill_speedup_ceiling = ${PREFILL_CEILING}" <<<"${out}" \
  && grep -q "the early-decode limit is ${SEED_RATIO}" <<<"${out}" \
  && ok "case 1: the committed files (stand-in oracle pins) pass and state the same ceilings" \
  || fail "case 1: rc=${rc}: $(grep -E '^FAIL' <<<"${out}" | head -3)"

out="$(lint_with 'del(.decode_speedup_ceiling)')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract decode_speedup_ceiling: None" <<<"${out}" \
  && ok "case 2: a fixture with no decode ceiling fails" \
  || fail "case 2: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with ".decode_speedup_ceiling = (${CEILING} + 1)")"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "scoring.decodeSpeedupCeiling = ${CEILING} disagrees with contract decode_speedup_ceiling" <<<"${out}" \
  && ok "case 3: a fixture ceiling that differs from benchmark.json fails" \
  || fail "case 3: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with '.decode_speedup_ceiling = 1')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract decode_speedup_ceiling: 1 " <<<"${out}" \
  && ok "case 4: a fixture ceiling at 1 fails" \
  || fail "case 4: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with 'del(.prefill_speedup_ceiling)')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract prefill_speedup_ceiling: None" <<<"${out}" \
  && ok "case 5: a fixture without a prefill ceiling fails" \
  || fail "case 5: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with ".prefill_speedup_ceiling = (${PREFILL_CEILING} + 1)")"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "scoring.prefillSpeedupCeiling = ${PREFILL_CEILING} disagrees with contract prefill_speedup_ceiling" <<<"${out}" \
  && ok "case 6: a fixture prefill ceiling that differs from benchmark.json fails" \
  || fail "case 6: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with '.prefill_speedup_ceiling = 1')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract prefill_speedup_ceiling: 1 " <<<"${out}" \
  && ok "case 7: a fixture prefill ceiling at 1 fails" \
  || fail "case 7: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with 'del(.seed_window_max_ratio)')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract seed_window_max_ratio: None" <<<"${out}" \
  && ok "case 8: a fixture with no early-decode limit fails" \
  || fail "case 8: rc=${rc}: $(tail -3 <<<"${out}")"

out="$(lint_with '.seed_window_max_ratio = 1')"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "contract seed_window_max_ratio: 1 " <<<"${out}" \
  && ok "case 9: an early-decode limit at 1 fails" \
  || fail "case 9: rc=${rc}: $(tail -3 <<<"${out}")"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-lint-scoring-limits: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-lint-scoring-limits: all cases passed"
