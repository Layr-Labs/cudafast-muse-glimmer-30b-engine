#!/usr/bin/env bash
# sandbox-clear-dirs.sh USER DIR... -- empty each DIR of everything in it and
# verify that nothing is left (docs/sandbox-design.md section 6a, "per-leg
# storage"). TRUSTED: not in editablePaths. Each DIR itself stays.
#
# USER (the sandbox user) may have made its own directories unreadable, so it
# first reopens them, then deletes what it owns; the caller deletes the rest.
# -P never follows a symlink out of DIR. Only the count of what is left decides
# the exit status, and only DIR's last name and that count are printed: an
# entry name could carry prompt text.
#
# Callers: tools/run-engine-sandboxed.sh after each candidate leg,
# tools/serve-up.sh --stop for a sandboxed leg's resident directory, and
# tools/engine/build.sh once the build outputs are copied into .build/.
#
# Exit 0: every DIR is empty (a DIR that does not exist is skipped).
# 1: something is left in a DIR. 78: usage error.
set -uo pipefail

[[ $# -ge 2 ]] || { echo "usage: sandbox-clear-dirs.sh USER DIR..." >&2; exit 78; }
user="$1"; shift
rc=0
for dir in "$@"; do
  [[ -d "${dir}" && ! -L "${dir}" ]] || continue
  sudo -n -u "${user}" -- chmod -R u+rwX "${dir}" >/dev/null 2>&1 || true
  sudo -n -u "${user}" -- find -P "${dir}" -mindepth 1 -delete >/dev/null 2>&1 || true
  find -P "${dir}" -mindepth 1 -delete >/dev/null 2>&1 || true
  # A failed look is not an empty directory: it fails too.
  count="$(find -P "${dir}" -mindepth 1 -printf . 2>/dev/null | wc -c)" \
    || { echo "sandbox-clear-dirs.sh: cannot list $(basename -- "${dir}") after cleanup" >&2; rc=1; continue; }
  if (( count != 0 )); then
    echo "sandbox-clear-dirs.sh: $(basename -- "${dir}") is not empty after cleanup (${count} leftover entries)" >&2
    rc=1
  fi
done
exit "${rc}"
