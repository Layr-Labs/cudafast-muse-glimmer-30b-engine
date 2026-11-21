#!/usr/bin/env bash
# ci-tests.sh -- discover every GPU-free test file and run each one
# (docs/ci-coverage.md). .github/workflows/ci.yml calls `run` once per suite,
# so a new test file runs in CI the day it is added: nothing lists it by hand.
#
# Suites (globs from the repository root):
#   tools     tools/test-*.sh, tools/test_*.py, tools/box-runner/test/test-*.sh
#   router    tools/router/test_*.py
#   resident  harness/vllm_resident/test_*.py (needs the release adapter built)
#
# How each file runs:
#   *.sh in any suite          ./<file> (it must be executable)
#   tools/test_*.py            python3 <file>
#   router / resident *.py     python3 -m unittest discover -s <dir> -p <file> -v
#
# THE EXCEPTIONS are tools/ci-test-exceptions.tsv: one `<file><TAB><reason>`
# line per test that needs a GPU (or anything else a hosted runner lacks); `#`
# starts a comment. A listed file is not run and its reason is printed. Every
# entry must name a discovered test file and give a reason, or `check` (and so
# every `run`, and tools/check-doc-consistency.py) fails: a stale entry would
# otherwise hide nothing and say something.
#
# Usage:
#   tools/ci-tests.sh discover <suite|all>   every test file of the suite, one per line
#   tools/ci-tests.sh list <suite|all>       the files `run` runs (discover minus exceptions)
#   tools/ci-tests.sh check                  verify the exception list against the files on disk
#   tools/ci-tests.sh run <suite>            run each listed file; print one PASS/FAIL line each
#
# Exit: 0 every test passed (or the list is valid); 1 a test failed or the
# exception list is wrong; 78 usage error.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCEPTIONS="${ROOT}/tools/ci-test-exceptions.tsv"
SUITES=(tools router resident)

usage() { echo "usage: $0 discover|list <suite|all> | check | run <suite>  (suites: ${SUITES[*]})" >&2; exit 78; }

suite_globs() {
  case "$1" in
    tools) printf '%s\n' 'tools/test-*.sh' 'tools/test_*.py' 'tools/box-runner/test/test-*.sh' ;;
    router) printf '%s\n' 'tools/router/test_*.py' ;;
    resident) printf '%s\n' 'harness/vllm_resident/test_*.py' ;;
    *) return 1 ;;
  esac
}

# valid_suite SUITE [all-ok] -- usage error unless SUITE is a suite (or `all`).
valid_suite() {
  [[ "$1" == all && "${2:-}" == all-ok ]] && return 0
  suite_globs "$1" >/dev/null || usage
}

# discover SUITE|all -- test files on disk, repository-relative, sorted.
discover() {
  local suite="$1" s g f
  local -a suites=("${suite}")
  [[ "${suite}" == all ]] && suites=("${SUITES[@]}")
  for s in "${suites[@]}"; do
    # One glob per line, read without expansion: each expands under ROOT
    # only, never against the caller's working directory.
    while IFS= read -r g; do
      for f in "${ROOT}"/${g}; do
        [[ -f "${f}" ]] && printf '%s\n' "${f#"${ROOT}"/}"
      done
    done < <(suite_globs "${s}")
  done | LC_ALL=C sort -u
}

# exception_entries -- "<file><TAB><reason>" per non-comment, non-blank line.
exception_entries() {
  [[ -f "${EXCEPTIONS}" ]] || return 0
  grep -v -E '^[[:space:]]*(#|$)' "${EXCEPTIONS}" || true
}

# check -- every exception names a discovered test file, once, with a reason.
check() {
  local all line file reason bad=0
  all="$(discover all)"
  [[ -f "${EXCEPTIONS}" ]] || { echo "ci-tests.sh: ${EXCEPTIONS#"${ROOT}"/} is missing (it may hold no entry, but it must exist)" >&2; return 1; }
  while IFS= read -r line; do
    file="${line%%$'\t'*}"
    reason="${line#*$'\t'}"
    if [[ "${line}" != *$'\t'* || -z "${reason//[[:space:]]/}" ]]; then
      echo "ci-tests.sh: exception '${file}' gives no reason (<file><TAB><reason>)" >&2; bad=1
    fi
    if ! grep -qxF -- "${file}" <<<"${all}"; then
      echo "ci-tests.sh: exception '${file}' is not a test file on disk (no suite discovers it)" >&2; bad=1
    fi
  done < <(exception_entries)
  local dup
  dup="$(exception_entries | cut -f1 | LC_ALL=C sort | uniq -d)"
  [[ -z "${dup}" ]] || { echo "ci-tests.sh: exception listed twice: ${dup}" >&2; bad=1; }
  return "${bad}"
}

# list SUITE|all -- discover minus exceptions.
list() {
  local excluded
  excluded="$(exception_entries | cut -f1)"
  discover "$1" | while IFS= read -r f; do
    grep -qxF -- "${f}" <<<"${excluded}" || printf '%s\n' "${f}"
  done
  return 0
}

# run_one SUITE FILE -- run one test file from the repository root.
run_one() {
  local suite="$1" file="$2"
  case "${file}" in
    *.sh)
      [[ -x "${ROOT}/${file}" ]] || { echo "ci-tests.sh: ${file} is not executable (chmod +x, and commit the mode)" >&2; return 1; }
      (cd "${ROOT}" && "./${file}")
      ;;
    *.py)
      if [[ "${suite}" == tools ]]; then
        (cd "${ROOT}" && python3 "${file}")
      else
        (cd "${ROOT}" && python3 -m unittest discover -s "$(dirname "${file}")" -p "$(basename "${file}")" -v)
      fi
      ;;
    *) echo "ci-tests.sh: no runner for ${file}" >&2; return 1 ;;
  esac
}

run_suite() {
  local suite="$1" f rc failed=() passed=0 line
  if ! check; then
    echo "ci-tests.sh: the exception list ${EXCEPTIONS#"${ROOT}"/} is wrong; refusing to run" >&2
    return 1
  fi
  while IFS= read -r line; do
    f="${line%%$'\t'*}"
    discover "${suite}" | grep -qxF -- "${f}" && echo "SKIP  ${f}: ${line#*$'\t'}"
  done < <(exception_entries)
  local -a files=()
  while IFS= read -r f; do files+=("${f}"); done < <(list "${suite}")
  [[ "${#files[@]}" -gt 0 ]] || { echo "ci-tests.sh: suite ${suite} discovered no test to run" >&2; return 1; }
  for f in "${files[@]}"; do
    [[ -n "${GITHUB_ACTIONS:-}" ]] && echo "::group::${f}"
    run_one "${suite}" "${f}" </dev/null
    rc=$?
    [[ -n "${GITHUB_ACTIONS:-}" ]] && echo "::endgroup::"
    if [[ "${rc}" -eq 0 ]]; then
      echo "PASS  ${f}"; passed=$((passed + 1))
    else
      echo "FAIL  ${f} (exit ${rc})"; failed+=("${f}")
    fi
  done
  echo "ci-tests.sh: suite ${suite}: ${passed} passed, ${#failed[@]} failed, of ${#files[@]}"
  [[ "${#failed[@]}" -eq 0 ]] || { printf 'ci-tests.sh: failed: %s\n' "${failed[@]}" >&2; return 1; }
}

case "${1:-}" in
  discover) [[ $# -eq 2 ]] || usage; valid_suite "$2" all-ok; discover "$2" ;;
  list) [[ $# -eq 2 ]] || usage; valid_suite "$2" all-ok; list "$2" ;;
  check) [[ $# -eq 1 ]] || usage; check && echo "ci-tests.sh: the exception list matches the files on disk ($(exception_entries | grep -c . || true) entries)" ;;
  run) [[ $# -eq 2 ]] || usage; valid_suite "$2"; run_suite "$2" ;;
  *) usage ;;
esac
