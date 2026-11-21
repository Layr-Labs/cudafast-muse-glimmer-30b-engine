#!/usr/bin/env bash
# gpu-lock-guard.sh LOCK -- make the box GPU lock a file the sandbox user can
# neither write nor read, and refuse scoring when it is not
# (docs/sandbox-design.md 6a, "The GPU lock"). TRUSTED: not in editablePaths.
#
# THE LOCK IS RUNNER STATE THE WIPE DOES NOT TOUCH. tools/sandbox-wipe.sh
# removes only what the sandbox user owns, so a runner-owned lock that the
# sandbox user can write would carry data from one leg (or job) into the next.
# The lock stays at its shared path: spark-gate, tools/calibrate-box.sh,
# tools/pool-refresh.sh and a second runner on the box take the same file, and
# a second path would end the box-wide exclusion. What changes is who can open
# it:
#
#   1. LOCK must be a regular file (never a symlink, never a directory). When it
#      is absent it is created here, exclusively, mode 0666 as before, so every
#      account that takes the lock can still open it.
#   2. When this account owns LOCK, it sets the ACL entry u:SANDBOX_USER:---
#      (the sandbox user cannot open it at all; nobody else changes).
#   3. As the sandbox user (tools/sandbox-access.sh), LOCK must be neither
#      writable nor readable. Anything else -- a lock the sandbox user owns, a
#      lock another account owns that the sandbox user can open, a failed
#      setfacl, a failed probe -- is a refusal (exit 1) and nothing is timed.
#
# The ranked workflow runs it in "Sandbox start" (before any participant code
# runs, so the lock is made here and not by the sandbox user) and again in the
# GPU step, right before flock.
#
# Env: SANDBOX_USER (required).
# Exit: 0 the sandbox user cannot open LOCK; 1 refused; 78 usage error.
set -uo pipefail

lock="${1:-}"
[[ -n "${lock}" && $# -eq 1 ]] || { echo "usage: gpu-lock-guard.sh LOCK" >&2; exit 78; }
user="${SANDBOX_USER:-}"
[[ -n "${user}" ]] || { echo "usage: gpu-lock-guard.sh needs SANDBOX_USER" >&2; exit 78; }
ACCESS="$(dirname -- "${BASH_SOURCE[0]}")/sandbox-access.sh"
refuse() { echo "::error::gpu-lock-guard.sh: REFUSING -- $*" >&2; exit 1; }

if [[ ! -e "${lock}" && ! -L "${lock}" ]]; then
  # noclobber: O_EXCL, so a name planted between the test and here is not adopted.
  ( umask 0000; set -C; : > "${lock}" ) 2>/dev/null \
    || [[ -e "${lock}" ]] || refuse "cannot create the GPU lock ${lock}"
fi
[[ -f "${lock}" && ! -L "${lock}" ]] || refuse "the GPU lock ${lock} is not a regular file"
owner="$(stat -c %u -- "${lock}" 2>/dev/null)" || refuse "cannot stat the GPU lock ${lock}"
if [[ "${owner}" == "$(id -u)" ]]; then
  setfacl -m "u:${user}:---" -- "${lock}" \
    || refuse "cannot deny ${user} the GPU lock ${lock} (setfacl, package acl, is required)"
fi
"${ACCESS}" no-write "${user}" "${lock}" && "${ACCESS}" no-read "${user}" "${lock}" \
  || refuse "the sandbox user ${user} can open the GPU lock ${lock} (owner uid ${owner}); its owner must run 'setfacl -m u:${user}:--- ${lock}', or remove it so this job creates it"
echo "gpu-lock-guard.sh: ${user} can neither read nor write the GPU lock ${lock}"
