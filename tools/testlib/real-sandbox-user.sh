#!/usr/bin/env bash
# real-sandbox-user.sh -- sourced by the tests that need the KERNEL's answer to
# "can the sandbox user reach this?" (ACLs, groups, modes), not a stub's.
#
# They run the trusted scripts with SANDBOX_USER=nobody through the real sudo.
# That needs a passwordless `sudo -n -u nobody` (a hosted CI runner and the CI
# container have one; a ranked box does not, its runner may only drop to its
# own sandbox user), setfacl, and no process running as nobody (the code under
# test may run `kill -KILL -1` as that user). Without all three the test prints
# one SKIP line and exits 0.
#
# real_sandbox_user_or_skip NAME -- sets SBX (the user) and SBX_WORK (a fresh
# directory, mode 0755 so SBX can reach what a case puts in it), or skips.
# sbx CMD... -- run CMD as SBX.
# sbx_cleanup -- remove SBX_WORK, the parts SBX owns included.
real_sandbox_user_or_skip() {
  local name="$1"
  SBX=nobody
  if ! sudo -n -u "${SBX}" -- true >/dev/null 2>&1 || ! command -v setfacl >/dev/null 2>&1 \
     || pgrep -u "${SBX}" >/dev/null 2>&1; then
    echo "SKIP: ${name}: needs a passwordless 'sudo -n -u ${SBX}', setfacl and no ${SBX} process (a hosted CI runner or the CI container)"
    exit 0
  fi
  SBX_WORK="$(mktemp -d "${TMPDIR:-/tmp}/sbxreal.XXXXXX")" || exit 1
  SBX_WORK="$(cd "${SBX_WORK}" && pwd -P)"
  chmod 0755 "${SBX_WORK}"
  if ! setfacl -m "u:${SBX}:---" "${SBX_WORK}" 2>/dev/null; then
    rmdir "${SBX_WORK}"
    echo "SKIP: ${name}: the filesystem under ${TMPDIR:-/tmp} takes no ACL"
    exit 0
  fi
  setfacl -b "${SBX_WORK}"
}
sbx() { sudo -n -u "${SBX}" -- "$@"; }
sbx_cleanup() {
  [[ -n "${SBX_WORK:-}" && -d "${SBX_WORK}" ]] || return 0
  sbx chmod -R u+rwX "${SBX_WORK}" >/dev/null 2>&1
  sbx find "${SBX_WORK}" -mindepth 1 -user "${SBX}" -delete >/dev/null 2>&1
  chmod -R u+rwX "${SBX_WORK}" 2>/dev/null
  rm -rf "${SBX_WORK}"
}
