#!/usr/bin/env bash
# test-sandbox-kill.sh -- tools/sandbox-kill.sh tells "no process left" from
# "processes survived" from "pgrep could not look".
#
# Asserts, with sudo and pgrep stubbed (nothing is killed):
#   1. pgrep exit 1 (no match) after the kill passes, after one kill;
#   2. a process that survives every kill fails with exit 1 and names its pid;
#   3. a pgrep error (exit 2 or above) fails with exit 2 -- it is never read
#      as an empty sandbox;
#   4. a process that is gone on the second look passes.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin"
cat > "${WORK}/bin/sudo" <<STUB
#!/usr/bin/env bash
echo "sudo \$*" >> "${WORK}/calls"
STUB
# The stub pgrep answers from STUB_PGREP: a space-separated list of exit codes,
# one per call; exit 0 prints a pid.
cat > "${WORK}/bin/pgrep" <<STUB
#!/usr/bin/env bash
n=\$(grep -c '^pgrep' "${WORK}/calls" 2>/dev/null)
echo "pgrep \$*" >> "${WORK}/calls"
read -ra codes <<<"\${STUB_PGREP}"
code="\${codes[\$n]:-\${codes[-1]}}"
[[ "\${code}" == 0 ]] && echo 4242
exit "\${code}"
STUB
chmod +x "${WORK}/bin/"*
run() { # run CODES -- prints the exit status, keeps stderr in ${WORK}/err
  : > "${WORK}/calls"
  PATH="${WORK}/bin:${PATH}" STUB_PGREP="$1" "${ROOT_DIR}/tools/sandbox-kill.sh" sbx 2> "${WORK}/err"
  echo $?
}

rc="$(run "1")"
[[ "${rc}" == 0 && "$(grep -c '^sudo -n -u sbx -- kill -KILL -1' "${WORK}/calls")" == 1 ]] \
  && ok "case 1: no process after the kill passes" || fail "case 1: rc=${rc}, calls: $(tr '\n' ';' < "${WORK}/calls")"

rc="$(run "0")"
[[ "${rc}" == 1 ]] && grep -q "survived kill -KILL -1: 4242" "${WORK}/err" \
  && ok "case 2: a surviving process fails and is named" || fail "case 2: rc=${rc}, err: $(cat "${WORK}/err")"

rc="$(run "2")"
[[ "${rc}" == 2 ]] && grep -q "pgrep -u sbx failed (exit 2)" "${WORK}/err" \
  && ok "case 3: a pgrep error fails; it is not an empty sandbox" || fail "case 3: rc=${rc}, err: $(cat "${WORK}/err")"

rc="$(run "0 1")"
[[ "${rc}" == 0 && "$(grep -c '^pgrep' "${WORK}/calls")" == 2 ]] \
  && ok "case 4: a process gone on the second look passes" || fail "case 4: rc=${rc}"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-sandbox-kill: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-sandbox-kill: all cases passed"
