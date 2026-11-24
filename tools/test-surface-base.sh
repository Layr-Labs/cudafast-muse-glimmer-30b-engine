#!/usr/bin/env bash
# test-surface-base.sh -- the surface-check job of .github/workflows/benchmark.yml
# judges a dispatched commit against the TRACK's protected branch (job env
# TRACK_PROTECTED_BRANCH), never the repository default branch: in the DEV
# repository the default branch serves another row, so a default-branch-based
# check reads every trusted file of this track as an entry edit.
#
# The two base steps ("Candidate diff stays inside the protected base's
# editable surface" and "Entry diff carries no weights and names no other
# loader") are cut from the workflow and run as written, with the trusted
# gates (.github/scripts/*) and this repository's benchmark.json, on a scratch
# repository shaped like the DEV checkout: origin/<default branch> (another row) and
# origin/<protected branch> that adds trusted files on top of it.
#
# Cases:
#   0. wiring: the job env names the protected branch; neither step reads the
#      repository default branch;
#   1. a baseline dispatch of the protected branch tip passes (empty diff);
#   2. an entry built on the protected branch that edits an editable file passes;
#   3. an entry that edits tools/* (a trusted file) fails, naming it;
#   4. the protected branch missing from the checkout fails closed;
#   5. a dispatched commit with no history in common with it fails closed;
#   6. control (the 2026-11-22 fault): the same steps judged against the default
#      branch refuse the baseline tip, so cases 1-2 pass for the right reason.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKFLOW="${ROOT_DIR}/.github/workflows/benchmark.yml"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }
g() { git -C "$1" -c user.name=t -c user.email=t@e -c commit.gpgsign=false "${@:2}"; }

# --- the workflow pieces under test ------------------------------------------
JOB="$(awk '/^  surface-check:/{f=1; print; next} f && /^  [a-z][a-z-]*:/{exit} f' "${WORKFLOW}")"
BRANCH="$(awk '/^    env:/{e=1; next} e && /^    [a-z]/{exit} e && /^      TRACK_PROTECTED_BRANCH:/{print $2; exit}' <<<"${JOB}")"
# step_block NAME -- the run block of the named step, de-indented.
step_block() {
  awk -v name="- name: $1" 'index($0, name) {s = 1; next}
    s && /^        run: \|/ {p = 1; next}
    p && /^      - / {exit}
    p {print}' <<<"${JOB}" | sed 's/^          //'
}
SURFACE="$(step_block "Candidate diff stays inside the protected base's editable surface")"
WEIGHTS="$(step_block "Entry diff carries no weights and names no other loader")"

# 0 -- wiring. The DEV checkout this models: the default branch is serial-row
# (the other row), and the track's protected branch is main.
DEFAULT=serial-row
TRACK=main
if [[ "${BRANCH}" == "${TRACK}" && "${BRANCH}" != "${DEFAULT}" ]] \
  && grep -qF 'base_ref="refs/remotes/origin/${TRACK_PROTECTED_BRANCH}"' <<<"${SURFACE}" \
  && grep -qF '"${trusted}/enforce-modifiable-surface.sh"' <<<"${SURFACE}" \
  && grep -qF 'git merge-base "refs/remotes/origin/${TRACK_PROTECTED_BRANCH}" "${GITHUB_SHA}"' <<<"${WEIGHTS}" \
  && ! grep -qiE 'default_branch|DEFAULT_BRANCH' <<<"${JOB}"; then
  ok "case 0: surface-check judges against TRACK_PROTECTED_BRANCH=${BRANCH}, never the default branch"
else
  fail "case 0: wiring: TRACK_PROTECTED_BRANCH='${BRANCH}' (want ${TRACK}), or a step still reads the default branch / not the track branch"
  echo "test-surface-base.sh: cannot run the cases without the wiring" >&2
  exit 1
fi

# --- a scratch repository shaped like the DEV checkout -----------------------
R="${WORK}/repo"
mkdir -p "${R}/.github/scripts" "${R}/tools" "${R}/harness"
cp "${ROOT_DIR}/.github/scripts/enforce-modifiable-surface.sh" "${ROOT_DIR}/.github/scripts/hardened-git.sh" \
  "${ROOT_DIR}/.github/scripts/entry-weight-check.py" "${R}/.github/scripts/"
cp "${ROOT_DIR}/benchmark.json" "${R}/"
printf 'serial row tool\n' > "${R}/tools/record-goldens.sh"
printf 'engine = 1\n' > "${R}/harness/engine.py"
g "${R}" init -q -b "${DEFAULT}"
g "${R}" add -A && g "${R}" commit -q -m "serial-row: the other row"
DEFAULT_TIP="$(g "${R}" rev-parse HEAD)"
# The protected branch: trusted-file changes on top of the other row's default
# branch (what the drafter track holds that serial-row does not).
g "${R}" checkout -q -b track
printf 'drafter track tool\n' > "${R}/tools/record-goldens.sh"
printf 'drafter-dir-check\n' > "${R}/tools/drafter-dir-check.sh"
printf '# drafter workflow\n' > "${R}/.github/workflow-note.txt"
g "${R}" add -A && g "${R}" commit -q -m "track: trusted files"
TIP="$(g "${R}" rev-parse HEAD)"
# An entry that edits an editable file only.
g "${R}" checkout -q -b entry-ok
printf 'engine = 2  # faster\n' > "${R}/harness/engine.py"
g "${R}" add -A && g "${R}" commit -q -m "entry: editable only"
ENTRY_OK="$(g "${R}" rev-parse HEAD)"
# An entry that also edits a trusted tool.
g "${R}" checkout -q -b entry-bad "${TIP}"
printf 'engine = 3\n' > "${R}/harness/engine.py"
printf 'patched by the entry\n' > "${R}/tools/record-goldens.sh"
g "${R}" add -A && g "${R}" commit -q -m "entry: edits tools/"
ENTRY_BAD="$(g "${R}" rev-parse HEAD)"
# A commit with no shared history.
g "${R}" checkout -q --orphan orphan
printf 'engine = 4\n' > "${R}/harness/engine.py"
g "${R}" add -A && g "${R}" commit -q -m "orphan"
ORPHAN="$(g "${R}" rev-parse HEAD)"
# What actions/checkout fetch-depth 0 leaves: every branch under origin/.
g "${R}" update-ref "refs/remotes/origin/${DEFAULT}" "${DEFAULT_TIP}"
g "${R}" update-ref "refs/remotes/origin/${BRANCH}" "${TIP}"

# run_job SHA [BASE_BRANCH] -- both base steps as written, in the checkout of
# SHA, against BASE_BRANCH (default: the workflow's value); sets rc, out.
run_job() {
  local tmp
  tmp="$(mktemp -d "${WORK}/runner-temp.XXXXXX")"
  g "${R}" checkout -q --detach "$1"
  out="$(cd "${R}" && env GITHUB_SHA="$1" RUNNER_TEMP="${tmp}" TRACK_PROTECTED_BRANCH="${2:-${BRANCH}}" \
      bash -c "set -euo pipefail
${SURFACE}
${WEIGHTS}" 2>&1)"
  rc=$?
}

run_job "${TIP}"
[[ ${rc} -eq 0 ]] && grep -q "protected base ${TIP} (merge base with ${BRANCH})" <<<"${out}" \
  && ok "case 1: a baseline dispatch of the ${BRANCH} tip passes with an empty diff" \
  || fail "case 1: rc=${rc}: $(tail -3 <<<"${out}")"

run_job "${ENTRY_OK}"
[[ ${rc} -eq 0 ]] && ok "case 2: an entry on ${BRANCH} that edits harness/ passes" \
  || fail "case 2: rc=${rc}: $(tail -3 <<<"${out}")"

run_job "${ENTRY_BAD}"
[[ ${rc} -ne 0 ]] && grep -q 'tools/record-goldens.sh is outside the modifiable surface' <<<"${out}" \
  && ok "case 3: an entry that edits tools/record-goldens.sh fails, naming it" \
  || fail "case 3: rc=${rc}: $(tail -3 <<<"${out}")"

g "${R}" update-ref -d "refs/remotes/origin/${BRANCH}"
run_job "${ENTRY_OK}"
[[ ${rc} -ne 0 ]] && grep -q "the track's protected branch ${BRANCH} is not in the checkout" <<<"${out}" \
  && ok "case 4: ${BRANCH} missing from the checkout fails closed" \
  || fail "case 4: rc=${rc}: $(tail -3 <<<"${out}")"
g "${R}" update-ref "refs/remotes/origin/${BRANCH}" "${TIP}"

run_job "${ORPHAN}"
[[ ${rc} -ne 0 ]] && grep -q "shares no history with ${BRANCH}" <<<"${out}" \
  && ok "case 5: a commit with no history in common with ${BRANCH} fails closed" \
  || fail "case 5: rc=${rc}: $(tail -3 <<<"${out}")"

run_job "${TIP}" "${DEFAULT}"
[[ ${rc} -ne 0 ]] && grep -q 'tools/record-goldens.sh is outside the modifiable surface' <<<"${out}" \
  && ok "case 6: control: judged against the other row's branch ${DEFAULT}, the ${BRANCH} tip is refused (the fault this guards)" \
  || fail "case 6: control: rc=${rc}: $(tail -3 <<<"${out}")"

if [[ ${failures} -eq 0 ]]; then
  echo "test-surface-base.sh: all 7 cases passed"
else
  echo "test-surface-base.sh: ${failures} case(s) FAILED" >&2
  exit 1
fi
