#!/usr/bin/env bash
# test-optional-editable-path.sh -- a dispatched tree WITHOUT the optional
# mtp-head.manifest.json passes the edit-surface checks, as the scoring reader
# accepts it (benchmark.json optionalEditablePaths; docs/ci-coverage.md).
#
# Asserts, on a copy of this checkout's tracked files (the checkout is only
# read):
#   1. without mtp-head.manifest.json: tools/lint-benchmark-manifest.py passes,
#      .github/scripts/submission-static-review-checks.sh whole-surface passes,
#      tools/spec-declaration.sh derives serial, and the CI checks that read
#      the declaration (tools/check-doc-consistency.py,
#      tools/test-live-golden-rotation.sh) pass;
#   2. with an invalid mtp-head.manifest.json: tools/spec-declaration.sh still
#      refuses it;
#   3. without a REQUIRED editable path (harness/): the lint still fails.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

T="${WORK}/tree"
mkdir -p "${T}"
(cd "${ROOT_DIR}" && git ls-files -z | tar --null -T - -cf -) | tar -xf - -C "${T}" \
  || { echo "FAIL: cannot copy the tree" >&2; exit 1; }
rm -f "${T}/mtp-head.manifest.json"
# A stand-in pin for each per-depth oracle the box has not recorded yet
# (tools/testlib/standin-oracles.jq): this suite is about the optional path;
# the lint run on the committed tree reports that gap.
FIX="${T}/$(jq -r .contractPath "${T}/benchmark.json")"
jq -f "${ROOT_DIR}/tools/testlib/standin-oracles.jq" "${FIX}" > "${WORK}/fixture.json" && cat "${WORK}/fixture.json" > "${FIX}"

lint() { python3 "${T}/tools/lint-benchmark-manifest.py" --repo-root "${T}" --gitlink-targets report > "${WORK}/lint.out" 2>&1; }
review() {
  (cd "${T}" && CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
    MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR="${WORK}/review" \
    .github/scripts/submission-static-review-checks.sh) > "${WORK}/review.out" 2>&1
}

# --- case 1: the optional declaration is absent ------------------------------
lint && ok "case 1: the manifest lint passes a tree without mtp-head.manifest.json" \
  || fail "case 1: lint: $(grep -i fail "${WORK}/lint.out" | head -3)"
review && grep -q "optional editable path mtp-head.manifest.json is absent" "${WORK}/review.out" \
  && ok "case 1: the whole-surface static review passes it and says the optional path is absent" \
  || fail "case 1: review: $(tail -3 "${WORK}/review.out")"
[[ "$("${T}/tools/spec-declaration.sh" describe 2>&1)" == serial ]] \
  && ok "case 1: the scoring reader derives serial" \
  || fail "case 1: spec-declaration.sh describe: $("${T}/tools/spec-declaration.sh" describe 2>&1)"
# check-doc-consistency.py lists the tracked files (git ls-files): index the
# copy in a repository of its own for this check only.
{ git -C "${T}" init -q && git -C "${T}" add -A && python3 -B "${T}/tools/check-doc-consistency.py"; } > "${WORK}/doc.out" 2>&1 \
  && ok "case 1: tools/check-doc-consistency.py passes a tree without mtp-head.manifest.json" \
  || fail "case 1: check-doc-consistency: $(tail -3 "${WORK}/doc.out")"
rm -rf "${T}/.git"
"${T}/tools/test-live-golden-rotation.sh" > "${WORK}/rotation.out" 2>&1 \
  && ok "case 1: tools/test-live-golden-rotation.sh passes a tree without mtp-head.manifest.json" \
  || fail "case 1: test-live-golden-rotation: $(grep -E '^FAIL' "${WORK}/rotation.out" | head -3)"

# --- case 2: an invalid declaration still fails ------------------------------
printf '{"source": "none", "spec": {"enabled": "yes"}}\n' > "${T}/mtp-head.manifest.json"
if out="$("${T}/tools/spec-declaration.sh" describe 2>&1)"; then
  fail "case 2: an invalid declaration was accepted: ${out}"
else
  grep -q REFUSING <<<"${out}" && ok "case 2: the scoring reader refuses an invalid declaration" \
    || fail "case 2: no refusal message: ${out}"
fi
printf 'not json\n' > "${T}/mtp-head.manifest.json"
"${T}/tools/spec-declaration.sh" describe >/dev/null 2>&1 \
  && fail "case 2: a non-JSON declaration was accepted" \
  || ok "case 2: the scoring reader refuses a non-JSON declaration"
rm -f "${T}/mtp-head.manifest.json"

# --- case 3: a required editable path is still required ----------------------
rm -rf "${T}/harness"
if lint; then
  fail "case 3: the lint passed a tree without harness/"
else
  grep -q "editablePaths: harness does not exist in the tree" "${WORK}/lint.out" \
    && ok "case 3: the lint still refuses a tree without a required editable path" \
    || fail "case 3: lint failed for another reason: $(grep -i fail "${WORK}/lint.out" | head -3)"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-optional-editable-path.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
