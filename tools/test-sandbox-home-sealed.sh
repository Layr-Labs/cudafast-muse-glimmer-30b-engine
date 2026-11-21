#!/usr/bin/env bash
# test-sandbox-home-sealed.sh -- the sandbox user's permanent account home
# carries nothing from one leg to the next (docs/sandbox-design.md 6a, "The
# account home").
#
# The wipe empties the home, but the home DIRECTORY ITSELF stays. A sandbox
# user that owns it can store data in its metadata: its times (touch -d), its
# mode, its user.* extended attributes. The next leg (the same account) reads
# them back. This test runs the REAL tools/sandbox-wipe.sh with
# SANDBOX_USER=nobody and the real sudo (tools/testlib/real-sandbox-user.sh;
# SKIP without it).
#
# Cases:
#   1. the home the sandbox user owns (what setup made before this fix): leg 1
#      stores a marker in the home's mtime and, where the filesystem takes one,
#      in a user.* attribute; then the wipe runs; leg 2 looks for the marker.
#      Pass = the wipe refuses (the run stops), or the marker is gone.
#   2. a home the sandbox user does not own and cannot enter (what setup makes
#      now; root-owned on a box, this account's here): the wipe passes, and the
#      sandbox user can neither set the home's times, mode or attributes nor
#      list or write it.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-sandbox-home-sealed.sh
trap sbx_cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }
MARK=1234567890

mkdir -m 1777 "${SBX_WORK}/tmp" "${SBX_WORK}/var"
printf '%s\n' "${SBX_WORK}/tmp" > "${SBX_WORK}/wipe-dirs"
wipe() { # wipe HOME
  env SANDBOX_USER="${SBX}" SANDBOX_WIPE_LIST="${SBX_WORK}/wipe-dirs" SANDBOX_WIPE_HOME="$1" \
    "${ROOT_DIR}/tools/sandbox-wipe.sh" 2>&1
}
set_xattr() { sbx python3 -c 'import os, sys; os.setxattr(sys.argv[1], "user.leak", b"leg1")' "$1" 2>/dev/null; }
get_xattr() { sbx python3 -c 'import os, sys; print(os.getxattr(sys.argv[1], "user.leak").decode())' "$1" 2>/dev/null; }

# --- case 1: a home the sandbox user owns -----------------------------------
H1="${SBX_WORK}/var/home1"
sbx mkdir -m 0700 "${H1}"
# leg 1 (entry code, as the sandbox user)
sbx touch -d "@${MARK}" "${H1}" || fail "case 1: setup: the sandbox user could not set its own home's mtime"
xattr=0; set_xattr "${H1}" && xattr=1
out="$(wipe "${H1}")"; rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "case 1: the wipe refuses a home the sandbox user owns (rc=${rc})"
  grep -q 'account home' <<<"${out}" && ok "case 1: the refusal names the account home" \
    || fail "case 1: the wipe failed for another reason: ${out}"
else
  # leg 2 (the same account) reads the home's metadata back
  got="$(sbx stat -c %Y "${H1}" 2>/dev/null)"
  [[ "${got}" != "${MARK}" ]] && ok "case 1: the mtime marker is gone" \
    || fail "case 1: the wipe passed and the next leg reads leg 1's mtime marker ${got}"
  if [[ "${xattr}" == 1 ]]; then
    [[ "$(get_xattr "${H1}")" != leg1 ]] && ok "case 1: the user.* marker is gone" \
      || fail "case 1: the wipe passed and the next leg reads leg 1's user.leak attribute"
  fi
fi

# --- case 2: a home the sandbox user does not own ---------------------------
H2="${SBX_WORK}/var/home2"
mkdir -m 0700 "${H2}"
out="$(wipe "${H2}")"; rc=$?
[[ "${rc}" -eq 0 ]] && ok "case 2: the wipe passes with a sealed home" || fail "case 2: the wipe failed (rc=${rc}): ${out}"
sbx touch -d "@${MARK}" "${H2}" 2>/dev/null && fail "case 2: the sandbox user set the sealed home's mtime" \
  || ok "case 2: the sandbox user cannot set the home's times"
sbx chmod 0777 "${H2}" 2>/dev/null && fail "case 2: the sandbox user changed the sealed home's mode" \
  || ok "case 2: the sandbox user cannot change the home's mode"
set_xattr "${H2}" && fail "case 2: the sandbox user set a user.* attribute on the sealed home" \
  || ok "case 2: the sandbox user cannot set an attribute on the home"
sbx ls "${H2}" >/dev/null 2>&1 && fail "case 2: the sandbox user can list the sealed home" \
  || ok "case 2: the sandbox user cannot list the home"
sbx sh -c ': > "$1/x"' sh "${H2}" 2>/dev/null && fail "case 2: the sandbox user wrote into the sealed home" \
  || ok "case 2: the sandbox user cannot write into the home"

if [[ "${failures}" -gt 0 ]]; then echo "FAIL: ${failures} failure(s)" >&2; exit 1; fi
echo "PASS: test-sandbox-home-sealed.sh"
