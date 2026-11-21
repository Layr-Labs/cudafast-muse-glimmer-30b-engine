#!/usr/bin/env bash
#
# sandbox-wipe.sh -- remove everything the sandbox user left behind outside the
# per-job directory, so nothing survives from one job or leg into the next
# (docs/sandbox-design.md section 5 item 6 and section 6a). TRUSTED: not in
# editablePaths. The workflow runs it at job start and in the cleanup step;
# tools/serve-up.sh --stop runs it after every candidate leg. INSIDE the per-job
# directory, each candidate leg's own storage is cleared by
# tools/run-engine-sandboxed.sh and tools/serve-up.sh --stop
# (tools/sandbox-clear-dirs.sh; "per-leg storage" in that doc).
#
#   1. kills every sandbox process (nothing may recreate what is deleted);
#   2. as the sandbox user, reopens (chmod u+rwx) every directory it owns under
#      each directory in the box's wipe list, and deletes every file and
#      directory it owns there (never crossing a mount). The list is every
#      directory on the box the sandbox user can write (/tmp, /var/tmp,
#      /dev/shm, /run/lock, /var/crash, /dev/mqueue, ...), recorded once by
#      tools/box-runner/setup-sandbox-user.sh in a root-owned file. A POSIX
#      message queue is a file under /dev/mqueue, so it goes the same way.
#      Each listed directory is first checked AS the sandbox user: it must be
#      a directory that user can list and enter. One that is missing, or that
#      only the runner cannot see, is not skipped: it fails the wipe (a
#      directory the runner cannot stat may still be one the sandbox user
#      writes). tools/sandbox-writable-inventory.sh, run by the ranked
#      preflight, re-takes the whole list as the sandbox user before each job;
#   3. as the sandbox user, removes every SysV shared memory segment, semaphore
#      set and message queue it owns;
#   4. verifies nothing the sandbox user owns is left in any of those places,
#      and exits non-zero if something is. Only counts are printed: a name
#      could carry prompt text;
#   5. verifies the sandbox user's permanent account home is SEALED: absent,
#      or a directory that user does not own and can neither read, enter nor
#      write, with no user.* attribute it can read (tools/sandbox-access.sh
#      sealed-home, asked as that user). Emptying the home is not enough: the
#      directory itself stays, and an owner can keep data in its times, mode
#      and attributes. tools/box-runner/setup-sandbox-user.sh makes it
#      root-owned 0700, and nothing gives that user HOME there: every process
#      it runs gets a per-leg or per-job HOME. A home that is not sealed fails
#      the wipe.
#
# IT FAILS CLOSED. Every walk runs as the sandbox user, which the next leg runs
# as too, so "the sandbox user cannot reach it" is exactly "the next leg cannot
# read it". Each walk skips ONE kind of path: a directory another user owns
# that the sandbox user cannot enter (no x). Nothing in it is reachable by the
# sandbox user, and it cannot change that directory's mode. Every other problem
# fails the wipe: a directory it can enter but not list, a directory it owns but
# cannot reopen, a find, chmod, rm, ipcs or ipcrm that exits non-zero. A failed
# wipe fails its caller: serve-up.sh --stop after a candidate leg (benchd seals
# the run as a failed score, LEG-TEARDOWN-FAILED), the scorer's reset before
# each attempt (no score is sealed), the job's start and cleanup steps. The
# error output of find and rm is dropped (it names paths); only the directory
# from the wipe list and a count are printed.
#
# A missing or empty wipe list is a failure: the box was never set up.
#
# Env: SANDBOX_USER (default nemosub) and SANDBOX_ETC (default
# /etc/muse-glimmer-sandbox), the names tools/box-runner/setup-sandbox-user.sh
# uses. SANDBOX_WIPE_LIST (default $SANDBOX_ETC/wipe-dirs) and
# SANDBOX_WIPE_HOME (default /var/lib/$SANDBOX_USER) are for tests.
# No -e: every step's status is checked explicitly, and any failure sets the
# exit status (a name that vanishes mid-walk is not a failure: the kill ran
# first, so nothing of the sandbox user is still changing).
set -uo pipefail

SANDBOX_USER="${SANDBOX_USER:-nemosub}"
WIPE_LIST="${SANDBOX_WIPE_LIST:-${SANDBOX_ETC:-/etc/muse-glimmer-sandbox}/wipe-dirs}"
[[ -f "${WIPE_LIST}" && -r "${WIPE_LIST}" ]] \
  || { echo "sandbox-wipe.sh: no wipe list ${WIPE_LIST}; run tools/box-runner/setup-sandbox-user.sh as root" >&2; exit 1; }
# One absolute path per line; anything else in the list is ignored.
mapfile -t DIRS < <(grep '^/' "${WIPE_LIST}")
[[ "${#DIRS[@]}" -gt 0 ]] || { echo "sandbox-wipe.sh: the wipe list ${WIPE_LIST} names no directory" >&2; exit 1; }
ACCOUNT_HOME="${SANDBOX_WIPE_HOME:-/var/lib/${SANDBOX_USER}}"
as_sandbox() { sudo -n -u "${SANDBOX_USER}" -- "$@"; }

as_sandbox true || { echo "sandbox-wipe.sh: 'sudo -n -u ${SANDBOX_USER}' fails; cannot wipe" >&2; exit 1; }
as_sandbox kill -KILL -1 >/dev/null 2>&1

# walk DIR MINDEPTH EXPR... -- as the sandbox user, find under DIR, never
# following a symlink (-P) or crossing a mount (-xdev). The only directory
# skipped is one another user owns that the sandbox user cannot enter (header,
# "fails closed"); find exits non-zero on any other error.
walk() {
  local d="$1" min="$2"; shift 2
  as_sandbox find -P "${d}" -xdev -ignore_readdir_race -mindepth "${min}" \
    \( -type d ! -user "${SANDBOX_USER}" ! -executable -prune \) -o "$@"
}
bad=0
broken() { echo "sandbox-wipe.sh: $*; the wipe fails" >&2; bad=1; }

# ipc_ids FLAG -- ids of the SysV objects of one kind (-m, -s, -q) the sandbox
# user owns. ipcs prints: key id owner perms ...
ipc_ids() { ipcs "$1" | awk -v u="${SANDBOX_USER}" '$1 ~ /^0x/ && $3 == u { print $2 }'; }

# The listed directories the sandbox user can list and enter, decided AS that
# user (one shell, which prints a line per good one and a last line to prove
# it finished). Every other listed directory fails the wipe.
# shellcheck disable=SC2016  # expands in the inner shell
probe="$(as_sandbox sh -c 'for d do [ -d "$d" ] && [ ! -L "$d" ] && [ -r "$d" ] && [ -x "$d" ] && printf "ok %s\n" "$d"; done; echo done' sh "${DIRS[@]}" 2>/dev/null)"
[[ "$(tail -1 <<<"${probe}")" == "done" ]] || { echo "sandbox-wipe.sh: the root check as ${SANDBOX_USER} did not finish; the wipe fails" >&2; exit 1; }
declare -A REACHABLE=()
while IFS= read -r line; do [[ "${line}" == "ok "* ]] && REACHABLE["${line#ok }"]=1; done <<<"${probe}"
for d in "${DIRS[@]}"; do
  [[ -n "${REACHABLE[${d}]:-}" ]] || broken "the listed directory ${d} is missing, or ${SANDBOX_USER} cannot list and enter it"
done

for d in "${DIRS[@]}"; do
  [[ -n "${REACHABLE[${d}]:-}" ]] || continue
  # Reopen first, the start directory included: each directory is reopened
  # before find reads it. A dot is printed for each chmod that fails.
  n="$(walk "${d}" 0 -type d -user "${SANDBOX_USER}" \( -exec chmod u+rwx {} \; -o -printf . \) 2>/dev/null | wc -c)" \
    || broken "cannot walk ${d} to reopen ${SANDBOX_USER}'s directories"
  [[ "${n}" -eq 0 ]] || broken "${n} ${SANDBOX_USER}-owned directories under ${d} could not be reopened"
  # Each top-most entry the sandbox user owns goes whole (-prune: find does not
  # walk into what rm removes); rm's failure is find's exit status.
  walk "${d}" 1 -user "${SANDBOX_USER}" -prune -exec rm -rf --one-file-system -- {} + >/dev/null 2>&1 \
    || broken "the delete under ${d} did not complete"
done
for kind in -m -s -q; do
  ids="$(ipc_ids "${kind}" 2>/dev/null)" || { broken "ipcs ${kind} failed"; continue; }
  for id in ${ids}; do
    as_sandbox ipcrm "${kind}" "${id}" >/dev/null 2>&1 || broken "ipcrm ${kind} failed for one object"
  done
done

for d in "${DIRS[@]}"; do
  [[ -n "${REACHABLE[${d}]:-}" ]] || continue
  n="$(walk "${d}" 1 -user "${SANDBOX_USER}" -printf . 2>/dev/null | wc -c)" \
    || { broken "cannot verify ${d} (a path in it cannot be read)"; continue; }
  [[ "${n}" -eq 0 ]] || broken "${n} ${SANDBOX_USER}-owned path(s) remain under ${d}"
done
for kind in -m -s -q; do
  ids="$(ipc_ids "${kind}" 2>/dev/null)" || { broken "ipcs ${kind} failed"; continue; }
  n="$(wc -w <<<"${ids}")"
  [[ "${n}" -eq 0 ]] || broken "${n} ${SANDBOX_USER}-owned SysV IPC object(s) (ipcs ${kind}) remain"
done
"$(dirname -- "${BASH_SOURCE[0]}")/sandbox-access.sh" sealed-home "${SANDBOX_USER}" "${ACCOUNT_HOME}" \
  || broken "the account home ${ACCOUNT_HOME} is not sealed (above)"
[[ "${bad}" -eq 0 ]] || exit 1
echo "sandbox-wipe.sh: nothing ${SANDBOX_USER}-owned left in ${#DIRS[@]} directories or SysV IPC; the account home is sealed"
