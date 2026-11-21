#!/usr/bin/env bash
# test-sandbox-writable-roots.sh -- every place the sandbox user can write is
# checked AS that user: the wipe's listed roots at every wipe, the whole box at
# every preflight (docs/sandbox-design.md 6a, "Writable places").
#
# tools/sandbox-wipe.sh used to test each listed directory as the RUNNER and
# skip one the runner could not stat, and the wipe list was found once, at
# setup, from world-writable modes only. A directory the sandbox user reaches
# through a path the runner cannot, or writes through a group or an ACL entry,
# then kept data from one leg to the next. This test runs the REAL wipe and the
# REAL tools/sandbox-writable-inventory.sh with SANDBOX_USER=nobody and the real
# sudo (tools/testlib/real-sandbox-user.sh; SKIP without it).
#
# Cases:
#   1. a listed directory under a parent only the sandbox user can enter: leg 1
#      plants a file; after the wipe, the file is gone (or the wipe refused);
#   2. a listed directory that is missing: the wipe refuses;
#   3. the inventory refuses a directory the sandbox user can write through an
#      ACL entry and that is not in the wipe list;
#   4. the inventory refuses a file the sandbox user can write but does not own
#      (the wipe never removes it);
#   5. the inventory refuses a writable directory hidden in a directory the
#      sandbox user can enter but not list (the runner can list it);
#   6. a box whose only writable place is listed passes the inventory.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-sandbox-writable-roots.sh
trap sbx_cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }
W="${SBX_WORK}"

wipe() { # wipe LIST
  env SANDBOX_USER="${SBX}" SANDBOX_WIPE_LIST="$1" SANDBOX_WIPE_HOME="${W}/no-home" \
    "${ROOT_DIR}/tools/sandbox-wipe.sh" 2>&1
}
inventory() { # inventory ROOT LIST
  env SANDBOX_USER="${SBX}" SANDBOX_WIPE_LIST="$2" SANDBOX_INVENTORY_ROOTS="$1" SANDBOX_ROOT="${W}/no-sandbox-root" \
    "${ROOT_DIR}/tools/sandbox-writable-inventory.sh" check 2>&1
}

# --- case 1: a listed root the runner cannot stat ---------------------------
mkdir -m 1777 "${W}/c1"
sbx mkdir -m 0700 "${W}/c1/only-sbx"
sbx mkdir -m 0777 "${W}/c1/only-sbx/tmp"
printf '%s\n' "${W}/c1/only-sbx/tmp" > "${W}/list1"
[[ -d "${W}/c1/only-sbx/tmp" ]] && fail "case 1: setup: the runner can stat the listed directory" || true
sbx sh -c 'echo leg1 > "$1/planted"' sh "${W}/c1/only-sbx/tmp"   # leg 1
out="$(wipe "${W}/list1")"; rc=$?
if [[ "${rc}" -eq 0 ]] && sbx test -e "${W}/c1/only-sbx/tmp/planted"; then
  fail "case 1: the wipe passed and the next leg still finds leg 1's file under a root the runner cannot stat"
else
  ok "case 1: leg 1's file is gone, or the wipe refused (rc=${rc})"
fi

# --- case 2: a listed root that is missing ----------------------------------
printf '%s\n' "${W}/c1/only-sbx/tmp" "${W}/no-such-dir" > "${W}/list2"
out="$(wipe "${W}/list2")"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q 'missing' <<<"${out}" \
  && ok "case 2: a missing listed directory fails the wipe" \
  || fail "case 2: the wipe did not refuse a missing listed directory (rc=${rc}): ${out}"

# --- cases 3-6: the inventory -----------------------------------------------
# root R: a listed 1777 temp dir, plus one unlisted place per case.
mkroot() { mkdir -m 0755 "$1"; mkdir -m 1777 "$1/tmp"; printf '%s\n' "$1/tmp" > "$1.list"; }
mkroot "${W}/r3"; mkdir -m 0755 "${W}/r3/acl-dir"; setfacl -m "u:${SBX}:rwx" "${W}/r3/acl-dir"
out="$(inventory "${W}/r3" "${W}/r3.list")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q 'REFUSING' <<<"${out}" && grep -q 'acl-dir' <<<"${out}" \
  && ok "case 3: an ACL-writable directory outside the wipe list is refused" \
  || fail "case 3: rc=${rc}: ${out}"

mkroot "${W}/r4"; ( umask 0000; : > "${W}/r4/tmp/runner-file" )
out="$(inventory "${W}/r4" "${W}/r4.list")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q 'file(s) it does not own' <<<"${out}" \
  && ok "case 4: a sandbox-writable file it does not own is refused" \
  || fail "case 4: rc=${rc}: ${out}"

mkroot "${W}/r5"; mkdir -m 0711 "${W}/r5/blind"; mkdir -m 0777 "${W}/r5/blind/hidden"
out="$(inventory "${W}/r5" "${W}/r5.list")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q 'hidden' <<<"${out}" \
  && ok "case 5: a writable directory inside a directory the sandbox user cannot list is refused" \
  || fail "case 5: rc=${rc}: ${out}"

mkroot "${W}/r6"
out="$(inventory "${W}/r6" "${W}/r6.list")"; rc=$?
[[ "${rc}" -eq 0 ]] && ok "case 6: a box whose only writable place is listed passes" || fail "case 6: rc=${rc}: ${out}"

if [[ "${failures}" -gt 0 ]]; then echo "FAIL: ${failures} failure(s)" >&2; exit 1; fi
echo "PASS: test-sandbox-writable-roots.sh"
