#!/usr/bin/env bash
# test-ci-tests.sh -- tools/ci-tests.sh, the test discovery CI runs
# (docs/ci-coverage.md). It runs a COPY of the tool in a scratch repository
# root, so no real suite runs from here.
#
# Asserts:
#   1. a new tools/test-*.sh, tools/test_*.py and box-runner test are
#      discovered and run with no list to edit, and a passing suite exits 0;
#   2. a failing test fails the suite, names the test, and the other tests
#      still run;
#   3. a listed exception is skipped with its reason printed;
#   4. an exception that names no test file on disk fails `check` and `run`;
#   5. an exception with no reason fails `check`;
#   6. a test file that is not executable fails by name;
#   7. the router suite runs each test_*.py through unittest, one file each;
#   8. the shipped exception list matches this repository's files.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

R="${WORK}/repo"
mkdir -p "${R}/tools/box-runner/test" "${R}/tools/router"
cp "${ROOT_DIR}/tools/ci-tests.sh" "${R}/tools/ci-tests.sh"
printf '# none\n' > "${R}/tools/ci-test-exceptions.tsv"
mk() { # mk FILE EXIT -- a test that records its run and exits EXIT
  printf '#!/usr/bin/env bash\necho "%s" >> "%s/ran"\nexit %s\n' "$1" "${WORK}" "$2" > "${R}/$1"
  chmod +x "${R}/$1"
}
mk tools/test-alpha.sh 0
mk tools/test-new-suite.sh 0
mk tools/box-runner/test/test-box.sh 0
printf 'open("%s/ran", "a").write("tools/test_py.py\\n")\n' "${WORK}" > "${R}/tools/test_py.py"
# Not tests: a helper and a mock beside them must not run.
mk tools/helper.sh 0
mk tools/box-runner/test/mock-run.sh 0

run() { : > "${WORK}/ran"; "${R}/tools/ci-tests.sh" "$@" > "${WORK}/out" 2>&1; }

# 1.
run run tools; rc=$?
want="tools/box-runner/test/test-box.sh
tools/test-alpha.sh
tools/test-new-suite.sh
tools/test_py.py"
[[ "${rc}" -eq 0 && "$(sort "${WORK}/ran")" == "${want}" ]] && grep -q "suite tools: 4 passed, 0 failed, of 4" "${WORK}/out" \
  && ok "case 1: every test file is discovered and run, helpers and mocks are not" \
  || fail "case 1: rc=${rc}, ran: $(tr '\n' ' ' < "${WORK}/ran"); $(tail -3 "${WORK}/out")"

# 2.
mk tools/test-broken.sh 3
run run tools; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "^FAIL  tools/test-broken.sh (exit 3)" "${WORK}/out" && [[ "$(wc -l < "${WORK}/ran")" -eq 5 ]] \
  && ok "case 2: a failing test fails the suite by name and the rest still run" \
  || fail "case 2: rc=${rc}: $(tail -3 "${WORK}/out")"

# 3.
printf 'tools/test-broken.sh\tneeds a GPU (test fixture)\n' >> "${R}/tools/ci-test-exceptions.tsv"
run run tools; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "^SKIP  tools/test-broken.sh: needs a GPU (test fixture)" "${WORK}/out" \
  && ! grep -qx tools/test-broken.sh "${WORK}/ran" \
  && ok "case 3: a listed exception is skipped with its reason" || fail "case 3: rc=${rc}: $(tail -3 "${WORK}/out")"

# 4.
printf 'tools/test-gone.sh\tneeds a GPU\n' >> "${R}/tools/ci-test-exceptions.tsv"
run check; rc_check=$?
run run tools; rc=$?
[[ "${rc_check}" -eq 1 && "${rc}" -eq 1 && ! -s "${WORK}/ran" ]] && grep -q "exception 'tools/test-gone.sh' is not a test file on disk" "${WORK}/out" \
  && ok "case 4: a stale exception fails check and run (nothing runs)" || fail "case 4: check=${rc_check} run=${rc}: $(tail -3 "${WORK}/out")"

# 5.
printf '# none\ntools/test-broken.sh\n' > "${R}/tools/ci-test-exceptions.tsv"
run check; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "exception 'tools/test-broken.sh' gives no reason" "${WORK}/out" \
  && ok "case 5: an exception with no reason fails check" || fail "case 5: rc=${rc}: $(tail -3 "${WORK}/out")"
printf '# none\n' > "${R}/tools/ci-test-exceptions.tsv"
rm -f "${R}/tools/test-broken.sh"

# 6.
chmod -x "${R}/tools/test-alpha.sh"
run run tools; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "tools/test-alpha.sh is not executable" "${WORK}/out" && grep -q "^FAIL  tools/test-alpha.sh" "${WORK}/out" \
  && ok "case 6: a test file that is not executable fails by name" || fail "case 6: rc=${rc}: $(tail -3 "${WORK}/out")"
chmod +x "${R}/tools/test-alpha.sh"

# 7.
for n in one two; do
  printf 'import unittest\nclass T(unittest.TestCase):\n    def test_%s(self):\n        open("%s/ran", "a").write("router %s\\n")\n' \
    "${n}" "${WORK}" "${n}" > "${R}/tools/router/test_${n}.py"
done
run run router; rc=$?
[[ "${rc}" -eq 0 && "$(sort "${WORK}/ran" | tr '\n' ' ')" == "router one router two " ]] \
  && grep -q "^PASS  tools/router/test_one.py" "${WORK}/out" && grep -q "^PASS  tools/router/test_two.py" "${WORK}/out" \
  && ok "case 7: the router suite runs each test file through unittest" || fail "case 7: rc=${rc}: $(tail -3 "${WORK}/out")"

# 8.
out="$("${ROOT_DIR}/tools/ci-tests.sh" check 2>&1)"; rc=$?
[[ "${rc}" -eq 0 ]] && ok "case 8: the shipped exception list matches this repository's files" || fail "case 8: ${out}"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-ci-tests: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-ci-tests: all cases passed"
