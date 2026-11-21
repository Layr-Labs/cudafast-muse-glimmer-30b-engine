#!/usr/bin/env bash
# sandbox-grant-checkout.sh -- give the sandbox user read on this job's checkout
# and nothing else under the runner home (docs/sandbox-design.md 6a). The
# ranked workflow's "Sandbox start" step runs it as the runner account.
#
# What it sets, with setfacl:
#   * rX on GITHUB_WORKSPACE, recursively, and the same as the default ACL, so
#     every file the job writes there is readable by the sandbox user. -P: a
#     symlink in the checkout is never followed, so it cannot widen the grant.
#   * --x (traverse only) on each directory from the checkout's parent up to
#     the runner home, which is not world-readable.
#   * a deny on every entry beside that path (--x would still open any
#     world-readable entry whose name is guessed: .env, _diag, .credentials*,
#     ~/.cache), on RUNNER_TEMP, and on _work's _actions and _tool.
#
# ANOTHER RUNNER'S INSTALL IS DENIED LIKE ANY OTHER ENTRY. Each runner has its
# own sandbox user (tools/sandbox-names.sh; the workflow and the ranked
# preflight refuse a shared one), so a deny for this job's SANDBOX_USER on the
# other runner's directory leaves that runner's own sandbox user, and its jobs,
# untouched.
#
# Env (set by GitHub Actions and the "Sandbox names" step): SANDBOX_USER,
# GITHUB_WORKSPACE, RUNNER_WORKSPACE, RUNNER_TEMP, HOME.
set -euo pipefail

: "${SANDBOX_USER:?SANDBOX_USER is unset}" "${GITHUB_WORKSPACE:?}" "${RUNNER_WORKSPACE:?}" "${RUNNER_TEMP:?}"

ws="$(realpath "${GITHUB_WORKSPACE}")"
home="$(realpath "${HOME}")"
if [[ "${ws}" != "${home}"/* ]]; then
  echo "::error::the checkout ${ws} is not under runner's home ${home}; refusing to grant ${SANDBOX_USER} access" >&2
  exit 1
fi
setfacl -R -P -m "u:${SANDBOX_USER}:rX" "${ws}"
setfacl -R -P -d -m "u:${SANDBOX_USER}:rX" "${ws}"

me="$(id -u)"
child="${ws}"
d="$(dirname "${ws}")"
while :; do
  setfacl -m "u:${SANDBOX_USER}:--x" "${d}"
  # Symlinks skipped: setfacl would change their target. Entries runner does
  # not own it cannot set an ACL on; setup-sandbox-user.sh (root) denied those
  # once.
  while IFS= read -r -d '' e; do
    setfacl -m "u:${SANDBOX_USER}:---" "${e}"
  done < <(find "${d}" -mindepth 1 -maxdepth 1 ! -type l -user "${me}" ! -path "${child}" -print0)
  [[ "${d}" == "${home}" ]] && break
  child="${d}"
  d="$(dirname "${d}")"
done
setfacl -m "u:${SANDBOX_USER}:---" "${RUNNER_TEMP}"
work="$(dirname "$(realpath "${RUNNER_WORKSPACE}")")"
for sib in _actions _tool; do
  if [[ -d "${work}/${sib}" ]]; then setfacl -m "u:${SANDBOX_USER}:---" "${work}/${sib}"; fi
done
