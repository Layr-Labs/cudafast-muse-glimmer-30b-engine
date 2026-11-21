#!/usr/bin/env bash
# sandbox-names.sh -- the sandbox names of ONE runner (docs/sandbox-design.md 6b).
#
# Every runner on a box has its own sandbox user, share group, sandbox root,
# config dir, sudoers file and units. Two runners must never share a sandbox
# user: the job cleanup kills with `kill -KILL -1` as that user
# (tools/sandbox-kill.sh) and the wipe removes every file it owns
# (tools/sandbox-wipe.sh), so a shared user lets one runner's job kill and wipe
# the other runner's job.
#
# Each name resolves in this order:
#   1. the environment (a job inherits its runner's actions-runner/.env; the
#      root setup gets values from `sudo env NAME=value ...`);
#   2. a NAME=value line in the runner dir's .env (the root setup runs outside
#      any job, so it reads the file itself);
#   3. for SUDOERS_NAME, EGRESS_UNIT and ACL_UNIT only: the sudoers file or
#      unit that tools/box-runner/setup-sandbox-user.sh installed for this
#      runner's SANDBOX_USER, when exactly one exists (a box set up with
#      explicit names whose .env carries only the four names a job needs keeps
#      its installed names, and its setup never writes a second set);
#   4. the default from the runner dir name: `actions-runner` gives the base
#      name below (a box with one runner sets nothing), any other name gives
#      the base name plus "-<suffix>" (`actions-runner-dev` -> `nemosub-dev`).
#
#   name          base name
#   SANDBOX_USER  nemosub
#   SHARE_GROUP   nemoshare
#   SANDBOX_ROOT  /var/lib/muse-glimmer/sandbox
#   SANDBOX_ETC   /etc/muse-glimmer-sandbox
#   SUDOERS_NAME  muse-glimmer-sandbox
#   EGRESS_UNIT   muse-glimmer-sandbox-egress.service
#   ACL_UNIT      muse-glimmer-sandbox-acl.service
#
# The runner dir: RUNNER_DIR; else, inside a job, the part of RUNNER_WORKSPACE
# (else GITHUB_WORKSPACE) before /_work/; else <home>/actions-runner, where
# <home> is SANDBOX_RUNNER_HOME, else HOME.
#
# Usage:
#   tools/sandbox-names.sh env           print NAME=value for each name (the
#                                        form GITHUB_ENV takes)
#   tools/sandbox-names.sh check-unique  exit 1 when another runner on this box
#                                        resolves any of the names to the same
#                                        value as this runner
#   . tools/sandbox-names.sh             define the functions only
#
# Another runner is a directory D that holds a D/.runner file, directly under
# this runner's home or directly under a sibling home (another directory beside
# this runner's home). Its names resolve from its own .env and its own dir name
# (steps 2 and 3); this job's environment says nothing about it. A sibling home
# this account cannot list is not searched. A runner whose .env exists but
# cannot be read is a refusal: its names are unknown.
#
# Exit: 0 ok; 1 a name is shared (check-unique); 78 usage error.

SANDBOX_NAME_KEYS=(SANDBOX_USER SHARE_GROUP SANDBOX_ROOT SANDBOX_ETC SUDOERS_NAME EGRESS_UNIT ACL_UNIT)

# sandbox_runner_dir [HOME] -- this runner's install dir (see the header).
# shellcheck disable=SC2120  # setup-sandbox-user.sh passes the runner account's HOME
sandbox_runner_dir() {
  local home="${1:-${SANDBOX_RUNNER_HOME:-${HOME:-}}}" d="${RUNNER_DIR:-}"
  local ws="${RUNNER_WORKSPACE:-${GITHUB_WORKSPACE:-}}"
  [[ -z "${d}" && "${ws}" == */_work/* ]] && d="${ws%%/_work/*}"
  printf '%s\n' "${d:-${home}/actions-runner}"
}

# sandbox_suffix RUNNER_DIR -- "" for actions-runner, else "-<suffix>": the dir
# name without a leading "actions-runner" and separator, lower case, every
# other character than a-z and 0-9 made "-", at most 22 characters (a user or
# group name stays within 32).
sandbox_suffix() {
  local b x
  b="$(basename -- "$1")"
  [[ "${b}" == actions-runner ]] && return 0
  x="${b#actions-runner}"
  x="${x#[-_.]}"
  x="$(printf '%s' "${x}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | tr -s '-')"
  x="${x#-}"
  x="${x%-}"
  [[ -n "${x}" ]] || x="r$(printf '%s' "${b}" | cksum | cut -d' ' -f1)"
  printf -- '-%s\n' "${x:0:22}"
}

# sandbox_default NAME SUFFIX -- the default value of NAME for that suffix.
sandbox_default() {
  local s="$2"
  case "$1" in
    SANDBOX_USER) printf 'nemosub%s\n' "${s}" ;;
    SHARE_GROUP) printf 'nemoshare%s\n' "${s}" ;;
    SANDBOX_ROOT) printf '/var/lib/muse-glimmer/sandbox%s\n' "${s}" ;;
    SANDBOX_ETC) printf '/etc/muse-glimmer-sandbox%s\n' "${s}" ;;
    SUDOERS_NAME) printf 'muse-glimmer-sandbox%s\n' "${s}" ;;
    EGRESS_UNIT) printf 'muse-glimmer-sandbox%s-egress.service\n' "${s}" ;;
    ACL_UNIT) printf 'muse-glimmer-sandbox%s-acl.service\n' "${s}" ;;
    *) return 1 ;;
  esac
}

# The lines setup-sandbox-user.sh writes for a sandbox user USER; step 3 finds
# an installed sudoers file or unit by them, so both sides use these functions.
sandbox_sudoers_rule() { printf 'ALL=(%s) NOPASSWD:SETENV: ALL\n' "$1"; }
sandbox_egress_description() { printf 'Description=Reject all %s egress (re-applied every boot)\n' "$1"; }
sandbox_acl_description() {
  printf 'Description=Deny %s the host resolver sockets (/run is tmpfs; re-applied every boot)\n' "$1"
}

# sandbox_installed NAME USER -- step 3: the name of the one sudoers file
# (a "<runner> <sandbox_sudoers_rule USER>" line; sudo skips names with a dot
# or a trailing ~) or unit (its Description line) installed for USER, else
# nothing. A job cannot read /etc/sudoers.d, so it gets step 4 for
# SUDOERS_NAME; only the root setup and --check read that file.
sandbox_installed() {
  local u="$2" f b line
  local -a hits=()
  case "$1" in
    SUDOERS_NAME)
      line=" $(sandbox_sudoers_rule "${u}")"
      for f in /etc/sudoers.d/*; do
        b="${f##*/}"
        [[ -f "${f}" && -r "${f}" && "${b}" != *.* && "${b}" != *~ ]] || continue
        awk -v s="${line}" 'index($0, " ") > 1 && substr($0, index($0, " ")) == s { found = 1 } END { exit !found }' \
          "${f}" 2>/dev/null && hits+=("${b}")
      done ;;
    EGRESS_UNIT|ACL_UNIT)
      if [[ "$1" == EGRESS_UNIT ]]; then line="$(sandbox_egress_description "${u}")"; else line="$(sandbox_acl_description "${u}")"; fi
      for f in /etc/systemd/system/*.service; do
        [[ -f "${f}" ]] && grep -qxF -- "${line}" "${f}" 2>/dev/null && hits+=("${f##*/}")
      done ;;
    *) return 0 ;;
  esac
  [[ "${#hits[@]}" -eq 1 ]] && printf '%s\n' "${hits[0]}"
  return 0
}

# sandbox_env_file_value FILE NAME -- the last NAME=value line of FILE (quotes
# stripped), or nothing.
sandbox_env_file_value() {
  local v
  [[ -r "$1" ]] || return 0
  v="$(sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1)"
  v="${v%\"}"
  v="${v#\"}"
  printf '%s' "${v}"
}

# sandbox_name RUNNER_DIR NAME [USE_ENV] -- NAME for the runner in RUNNER_DIR.
# USE_ENV=0 skips step 1 (for another runner, whose environment is not ours).
sandbox_name() {
  local dir="$1" name="$2" use_env="${3:-1}" v=""
  [[ "${use_env}" == 1 ]] && v="${!name:-}"
  [[ -n "${v}" ]] || v="$(sandbox_env_file_value "${dir}/.env" "${name}")"
  case "${name}" in
    SUDOERS_NAME|EGRESS_UNIT|ACL_UNIT)
      [[ -n "${v}" ]] || v="$(sandbox_installed "${name}" "$(sandbox_name "${dir}" SANDBOX_USER "${use_env}")")" ;;
  esac
  [[ -n "${v}" ]] || v="$(sandbox_default "${name}" "$(sandbox_suffix "${dir}")")"
  printf '%s\n' "${v}"
}

# sandbox_names_env -- NAME=value per name for this runner.
sandbox_names_env() {
  local dir n
  dir="$(sandbox_runner_dir)"
  for n in "${SANDBOX_NAME_KEYS[@]}"; do
    printf '%s=%s\n' "${n}" "$(sandbox_name "${dir}" "${n}")"
  done
}

# sandbox_check_unique -- refuse when another runner on this box resolves a
# name to this runner's value. Prints one line per clash on stderr.
sandbox_check_unique() {
  local own own_real home parent f d d_real n mine theirs rc=0 others=0
  own="$(sandbox_runner_dir)"
  own_real="$(cd "${own}" 2>/dev/null && pwd -P || printf '%s' "${own}")"
  home="$(dirname -- "${own}")"
  parent="$(dirname -- "${home}")"
  local -a mine_all=()
  for n in "${SANDBOX_NAME_KEYS[@]}"; do mine_all+=("$(sandbox_name "${own}" "${n}")"); done
  for f in "${parent}"/*/*/.runner; do
    [[ -e "${f}" ]] || continue
    d="$(dirname -- "${f}")"
    d_real="$(cd "${d}" 2>/dev/null && pwd -P || printf '%s' "${d}")"
    [[ "${d_real}" == "${own_real}" ]] && continue
    others=$((others + 1))
    if [[ -e "${d}/.env" && ! -r "${d}/.env" ]]; then
      echo "sandbox-names.sh: REFUSING -- runner ${d} has a .env this account cannot read, so its sandbox names are unknown; make it readable or move that runner" >&2
      rc=1
      continue
    fi
    local i
    for i in "${!SANDBOX_NAME_KEYS[@]}"; do
      n="${SANDBOX_NAME_KEYS[${i}]}"
      mine="${mine_all[${i}]}"
      theirs="$(sandbox_name "${d}" "${n}" 0)"
      if [[ "${mine}" == "${theirs}" ]]; then
        echo "sandbox-names.sh: REFUSING -- runner ${d} also uses ${n}=${theirs} (this runner: ${own}); each runner needs its own, or one runner's job can kill and wipe the other's (set ${n} in one runner's .env, or let it default from the runner dir name)" >&2
        rc=1
      fi
    done
  done
  if [[ "${rc}" -eq 0 ]]; then
    echo "sandbox-names.sh: ${own} uses SANDBOX_USER=${mine_all[0]}; no other runner on this box (${others} found) shares a sandbox name"
  fi
  return "${rc}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  case "${1:-}" in
    env) sandbox_names_env ;;
    check-unique) sandbox_check_unique ;;
    *) echo "usage: $0 env|check-unique" >&2; exit 78 ;;
  esac
fi
