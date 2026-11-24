#!/usr/bin/env bash
# sandbox-access.sh -- ask the kernel, AS the sandbox user, what that user can
# reach (docs/sandbox-design.md 6a, "Checked as the sandbox user"). TRUSTED:
# not in editablePaths.
#
# Every probe runs through the runner's existing sudo drop to the sandbox user
# (`sudo -n -u USER`, tools/box-runner/setup-sandbox-user.sh), so the answer
# includes that user's groups, every ACL entry and every parent directory on
# the way. A mode check done as the runner would miss an ACL or a group grant.
#
#   sandbox-access.sh no-write USER PATH...
#       exit 0 when USER can write no PATH and, for a directory, nothing under
#       it (find -writable, which asks access(2) as USER). A PATH USER cannot
#       reach at all (a directory above it USER may not search, so USER cannot
#       even stat it) passes: USER cannot write what it cannot name. Every
#       PATH must exist for the caller (a missing one is exit 2), and below a
#       PATH USER can reach, a find error means the tree was not fully
#       checked, and that is exit 2, not a pass.
#   sandbox-access.sh no-read USER PATH...
#       exit 0 when USER can read none of the PATHs and enter none that is a
#       directory (a directory USER can enter but not list still opens every
#       entry whose name is guessed). A PATH USER cannot even reach passes:
#       that is the point.
#   sandbox-access.sh sealed-home USER HOME
#       exit 0 when USER's permanent account home cannot hold anything from
#       one leg for the next: HOME is absent, or it is a directory USER does
#       not own (so USER cannot chmod it, touch its times or set an extended
#       attribute on it) that USER can neither read, enter nor write. When
#       getfattr is present, the user.* attributes USER can read on HOME must
#       also be none. A symlink is refused.
#
# Exit 0 pass; 1 USER can reach something (only a count is printed: a staged
# name could carry participant text); 2 the probe itself failed (the sudo drop
# fails, or the probe did not finish), so nothing is proven; 78 usage error.
# A caller treats every non-zero exit as a refusal.
set -uo pipefail

usage() { echo "usage: sandbox-access.sh no-write|no-read USER PATH... | sealed-home USER HOME" >&2; exit 78; }
[[ $# -ge 3 ]] || usage
mode="$1" user="$2"; shift 2
say() { echo "sandbox-access.sh: $*" >&2; }

# reach_count r|rw PATH... -- how many PATHs USER can read, enter (a directory)
# or, with rw, write. One shell as USER tests them all; its last line proves it
# finished, so a failed probe is never a count of 0.
reach_count() {
  local want="$1" out n; shift
  # shellcheck disable=SC2016  # expands in the inner shell
  out="$(sudo -n -u "${user}" -- sh -c 'w="$1"; shift; n=0
for p do
  if [ -r "$p" ] || { [ -d "$p" ] && [ -x "$p" ]; } || { [ "$w" = rw ] && [ -w "$p" ]; }; then n=$((n + 1)); fi
done
echo "probed $n"' sh "${want}" "$@" 2>/dev/null)"
  n="$(sed -n 's/^probed \([0-9][0-9]*\)$/\1/p' <<<"${out}" | tail -1)"
  [[ -n "${n}" ]] || { say "the probe as ${user} did not finish; nothing is proven"; return 2; }
  printf '%s\n' "${n}"
}

# writable_count PATH... -- how many entries USER can write in the PATHs and
# under them (find -writable asks access(2) as USER). Fails when the walk does
# not finish (a directory USER cannot list or reach).
# -P: a symlink is never followed (its target is checked where it lives).
# -xdev: a mount below PATH is not this tree's.
writable_count() {
  sudo -n -u "${user}" -- find -P "$@" -xdev -writable -printf . 2>/dev/null | wc -c
}

# reachable PATH... -- set WALK to the PATHs USER can reach at all (an lstat as
# USER succeeds; it fails when a directory above the PATH is closed to USER).
# One shell as USER answers for every PATH in order; its last line proves it
# finished, so a failed probe is never "nothing reachable".
reachable() {
  local out i
  local -a ans
  # shellcheck disable=SC2016  # expands in the inner shell
  out="$(sudo -n -u "${user}" -- sh -c 'for p do
  if [ -e "$p" ] || [ -L "$p" ]; then echo y; else echo n; fi
done
echo reached' sh "$@" 2>/dev/null)"
  mapfile -t ans <<<"${out}"
  [[ ${#ans[@]} -eq $(( $# + 1 )) && "${ans[$#]}" == reached ]] || return 1
  WALK=()
  for (( i = 0; i < $#; i++ )); do
    [[ "${ans[i]}" == y ]] && WALK+=("${@:i+1:1}")
  done
  return 0
}

sudo -n -u "${user}" -- true >/dev/null 2>&1 \
  || { say "'sudo -n -u ${user}' fails; cannot check what ${user} can reach"; exit 2; }

case "${mode}" in
  no-write)
    for p in "$@"; do
      [[ -e "${p}" || -L "${p}" ]] || { say "a path given does not exist; nothing is proven"; exit 2; }
    done
    if ! n="$(writable_count "$@")"; then
      # The walk failed somewhere. A PATH USER cannot reach at all passes;
      # the PATHs it can reach are walked again, and must walk clean.
      reachable "$@" || { say "the probe as ${user} did not finish; nothing is proven"; exit 2; }
      if (( ${#WALK[@]} == $# )); then
        say "${user} cannot walk every path given (a directory it cannot list); nothing is proven"
        exit 2
      fi
      n=0
      if (( ${#WALK[@]} > 0 )) && ! n="$(writable_count "${WALK[@]}")"; then
        say "${user} cannot walk every path given (a directory it cannot list); nothing is proven"
        exit 2
      fi
    fi
    if (( n != 0 )); then
      say "${user} can write ${n} file(s) or directory(ies) under the $# path(s) given"
      exit 1
    fi
    ;;
  no-read)
    n="$(reach_count r "$@")" || exit 2
    if (( n != 0 )); then
      say "${user} can read or enter ${n} of the $# path(s) given"
      exit 1
    fi
    ;;
  sealed-home)
    [[ $# -eq 1 ]] || usage
    home="$1"
    [[ -e "${home}" || -L "${home}" ]] || exit 0
    [[ -d "${home}" && ! -L "${home}" ]] || { say "the account home ${home} is not a directory"; exit 1; }
    owner="$(stat -c %U -- "${home}" 2>/dev/null)" || { say "cannot stat the account home ${home}"; exit 2; }
    [[ "${owner}" != "${user}" ]] \
      || { say "${user} owns its account home ${home}, so it can change its mode, times and attributes; it must be root-owned (re-run tools/box-runner/setup-sandbox-user.sh as root)"; exit 1; }
    n="$(reach_count rw "${home}")" || exit 2
    (( n == 0 )) || { say "${user} can read, enter or write its account home ${home}"; exit 1; }
    if command -v getfattr >/dev/null 2>&1; then
      # USER cannot read HOME, so this lists nothing; it is the second proof.
      n="$(sudo -n -u "${user}" -- getfattr --absolute-names -d -m '^user\.' -- "${home}" 2>/dev/null | grep -c '^user\.')"
      (( n == 0 )) || { say "${user} can read ${n} user.* attribute(s) on its account home ${home}"; exit 1; }
    fi
    ;;
  *) usage ;;
esac
exit 0
