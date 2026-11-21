#!/usr/bin/env bash
# sandbox-kill.sh USER -- kill every process USER owns and verify none is left.
#
# The one kill-and-verify loop of the sandbox (docs/sandbox-design.md 6a). The
# ranked workflow's cleanup step, tools/muse-glimmer-measure-and-score.sh and
# tools/serve-up.sh call it, so the three cannot drift apart.
#
# `kill -1` as USER reaches EVERY process USER owns on the box, so USER must
# belong to this runner alone. Each runner has its own sandbox user
# (tools/sandbox-names.sh); the ranked job and the preflight refuse a box where
# another runner shares it (docs/sandbox-design.md 6b).
#
# kill -1's status means nothing when there is nothing left to kill, so it is
# ignored; pgrep decides. pgrep exits 0 when a process matches, 1 when none
# does, and above 1 when it could not look. A failed lookup is NOT an empty
# result: it proves nothing about the sandbox, so it fails here.
#
# Exit 0: no process of USER is left. 1: processes survived ten kills (their
# pids are on stderr). 2: pgrep failed, so the sandbox state is unknown.
set -uo pipefail

user="${1:?usage: sandbox-kill.sh USER}"
left=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sudo -n -u "${user}" -- kill -KILL -1 >/dev/null 2>&1
  left="$(pgrep -u "${user}")"
  rc=$?
  case "${rc}" in
    0) ;;
    1) exit 0 ;;
    *)
      echo "sandbox-kill.sh: pgrep -u ${user} failed (exit ${rc}); cannot verify that no ${user} process is left" >&2
      exit 2
      ;;
  esac
  sleep 0.5
done
echo "sandbox-kill.sh: ${user} processes survived kill -KILL -1: $(tr '\n' ' ' <<<"${left}")" >&2
exit 1
