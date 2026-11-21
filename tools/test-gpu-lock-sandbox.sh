#!/usr/bin/env bash
# test-gpu-lock-sandbox.sh -- the box GPU lock the ranked GPU step takes is a
# file the sandbox user can neither write nor read (docs/sandbox-design.md 6a,
# "The GPU lock").
#
# The lock is runner state that tools/sandbox-wipe.sh never removes (it removes
# only what the sandbox user owns). A lock the sandbox user can write carries
# data from one leg or job into the next. This test runs the GPU step's own
# lock block, cut from .github/workflows/benchmark.yml, against a lock in a
# temp directory, with SANDBOX_USER=nobody and the REAL sudo, so ACLs and modes
# are the kernel's answer (tools/testlib/real-sandbox-user.sh; SKIP without it).
#
# Cases (a stub entry, as the sandbox user, tries to write a marker into the
# lock and to read it after the block ran):
#   1. no lock yet: the block makes it; the step holds it; the entry can
#      neither write nor read it;
#   2. a lock left 0666 by an earlier job (every box today): the same;
#   3. a lock the sandbox user made itself (0666, its own): the block refuses
#      and the step does not go on to time anything.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-gpu-lock-sandbox.sh
trap sbx_cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

# The GPU step's lock block, from its marker comment to "box GPU lock held".
BLOCK="$(awk '/# -- box-wide GPU lock/ {p = 1} p {print} p && /echo "box GPU lock held"/ {exit}' \
  "${ROOT_DIR}/.github/workflows/benchmark.yml" | sed 's/^          //')"
grep -q 'box GPU lock held' <<<"${BLOCK}" || { echo "FAIL: the GPU step's lock block is not in benchmark.yml" >&2; exit 1; }

# run_block LOCK -- the block, with the lock path moved to LOCK, from the repo
# root (as the step runs), with SANDBOX_USER set as the "Sandbox names" step
# sets it. Prints HELD when the step got past the lock.
run_block() {
  local block="${BLOCK//\/tmp\/mtplx-gpu-exclusive.lock/$1}"
  (cd "${ROOT_DIR}" && SANDBOX_USER="${SBX}" bash -c "set -euo pipefail
${block}
echo HELD")
}
# entry_reaches LOCK -- what a stub entry can do to the lock as the sandbox user.
entry_writes() { sbx sh -c 'printf leg1-marker >> "$1"' sh "$1" 2>/dev/null; }
entry_reads() { sbx cat -- "$1" >/dev/null 2>&1; }

check_sealed() { # check_sealed CASE LOCK OUT
  local c="$1" lock="$2" out="$3"
  grep -qx HELD <<<"${out}" && ok "${c}: the step holds the lock" || fail "${c}: the step did not get past the lock: ${out}"
  if entry_writes "${lock}"; then fail "${c}: the sandbox user wrote into the GPU lock"; else ok "${c}: the sandbox user cannot write the lock"; fi
  if entry_reads "${lock}"; then fail "${c}: the sandbox user can read the GPU lock"; else ok "${c}: the sandbox user cannot read the lock"; fi
  [[ ! -s "${lock}" ]] && ok "${c}: no marker in the lock" || fail "${c}: the lock holds $(wc -c < "${lock}") byte(s) the sandbox user wrote"
}

# --- case 1: no lock yet -----------------------------------------------------
mkdir -m 1777 "${SBX_WORK}/c1"
out="$(run_block "${SBX_WORK}/c1/gpu.lock" 2>&1)"
check_sealed "case 1" "${SBX_WORK}/c1/gpu.lock" "${out}"

# --- case 2: a 0666 lock an earlier job left --------------------------------
mkdir -m 1777 "${SBX_WORK}/c2"
( umask 0000; : > "${SBX_WORK}/c2/gpu.lock" )
out="$(run_block "${SBX_WORK}/c2/gpu.lock" 2>&1)"
check_sealed "case 2" "${SBX_WORK}/c2/gpu.lock" "${out}"

# --- case 3: the sandbox user made the lock ---------------------------------
mkdir -m 1777 "${SBX_WORK}/c3"
sbx sh -c 'umask 0000; : > "$1"' sh "${SBX_WORK}/c3/gpu.lock"
out="$(run_block "${SBX_WORK}/c3/gpu.lock" 2>&1)"; rc=$?
if [[ "${rc}" -ne 0 ]] && ! grep -qx HELD <<<"${out}" && grep -q 'REFUSING' <<<"${out}"; then
  ok "case 3: a sandbox-owned lock is refused before anything is timed"
else
  fail "case 3: a sandbox-owned lock was not refused (rc=${rc}): ${out}"
fi

if [[ "${failures}" -gt 0 ]]; then echo "FAIL: ${failures} failure(s)" >&2; exit 1; fi
echo "PASS: test-gpu-lock-sandbox.sh"
