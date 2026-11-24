#!/usr/bin/env bash
# test-sandbox-access-no-write.sh -- tools/sandbox-access.sh no-write asks the
# kernel, as the sandbox user, whether that user can write a path, and a path
# the user cannot reach at all passes (docs/sandbox-design.md 6a).
#
# The pinned drafter is staged under a runner directory the sandbox user may
# not search (the box denies it every entry of the runner's home but the
# checkout path); docker bind-mounts the drafter files into the container, so
# the user never walks the host path. serve-up.sh asks no-write on that
# directory before every sandboxed dflash leg: an unreachable directory must
# pass, and a reachable one is still walked in full, fail-closed.
#
# Real sudo, real ACLs (tools/testlib/real-sandbox-user.sh; SKIP without it).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-sandbox-access-no-write.sh
trap sbx_cleanup EXIT

ACCESS="${ROOT_DIR}/tools/sandbox-access.sh"
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }
# expect RC NAME PATH... -- no-write as the sandbox user must exit RC.
expect() {
  local want="$1" name="$2" rc; shift 2
  ( cd "${SBX_WORK}" && "${ACCESS}" no-write "${SBX}" "$@" ) >/dev/null 2>&1; rc=$?
  if [[ "${rc}" == "${want}" ]]; then ok "${name} (exit ${rc})"; else fail "${name}: exit ${rc}, want ${want}"; fi
}

W="${SBX_WORK}"
# A read-only tree the user can reach: pass.
mkdir -p "${W}/ro/sub"; echo x > "${W}/ro/sub/f"; chmod 0755 "${W}/ro" "${W}/ro/sub"; chmod 0644 "${W}/ro/sub/f"
expect 0 "reachable read-only tree passes" "${W}/ro"
# A reachable file the user can write: refused.
echo x > "${W}/open"; chmod 0666 "${W}/open"
expect 1 "reachable writable file is refused" "${W}/open"
# An ACL write grant below a reachable directory: refused.
mkdir "${W}/acl"; echo x > "${W}/acl/f"; chmod 0755 "${W}/acl"; chmod 0644 "${W}/acl/f"
setfacl -m "u:${SBX}:rw" "${W}/acl/f"
expect 1 "ACL write grant below a reachable directory is refused" "${W}/acl"
# The staged-drafter shape: a directory under a parent the user may not search
# (u:SBX:---), with files the user could read or even write by mode. The user
# cannot name anything in it, so it passes.
mkdir -p "${W}/private/drafter"; echo x > "${W}/private/drafter/model.safetensors"
chmod 0755 "${W}/private" "${W}/private/drafter"; chmod 0666 "${W}/private/drafter/model.safetensors"
setfacl -m "u:${SBX}:---" "${W}/private"
expect 0 "directory under a parent the user cannot search passes" "${W}/private/drafter"
# Unreachable plus reachable in one call: the reachable one is still checked.
expect 1 "unreachable path does not hide a reachable writable one" "${W}/private/drafter" "${W}/open"
expect 0 "unreachable path and a reachable read-only one pass" "${W}/private/drafter" "${W}/ro"
# A missing path is never a pass.
expect 2 "missing path is not proven" "${W}/no-such-dir"
# Below a reachable directory, a subdirectory the user cannot list is not proven.
mkdir -p "${W}/half/locked"; chmod 0755 "${W}/half"; chmod 0700 "${W}/half/locked"
expect 2 "unlistable directory below a reachable one is not proven" "${W}/half"

(( failures == 0 )) || { echo "FAIL: test-sandbox-access-no-write.sh (${failures} failure(s))" >&2; exit 1; }
echo "PASS: test-sandbox-access-no-write.sh"
