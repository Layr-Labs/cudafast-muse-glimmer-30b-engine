#!/usr/bin/env bash
# tools/box/converge.sh -- make a CUDA ranked box match its box file, or prove
# that it does.
#
#   sudo tools/box/converge.sh <box> [--secrets-stdin]   apply
#   sudo tools/box/converge.sh <box> --check             report only
#        ... --unit <name>                               only the named unit(s)
#
# <box> names tools/box/<box>.env. That file holds VALUES ONLY (see
# load_box_file). Run this script from the engine checkout that the box file
# names in BOX_ENGINE_DIR: that checkout is the engine commit of the box, and
# its fixture and its tools/fetch-benchd.sh give the pins the units check.
#
# Every unit prints one line per item:
#   OK            the box matches the box file; nothing done
#   CONVERGED     the box differed and apply changed it
#   WOULD-CHANGE  the box differs; --check changed nothing
#   FAIL          the box differs and this script cannot fix it (the line says
#                 what to do), or a change failed
#   NOTE          information only
# --check changes nothing on the box. It exits 0 only when every item is OK.
# Apply exits 0 when no item is FAIL. A second apply on a converged box prints
# no CONVERGED line.
#
# THE UNITS, in order. The engine tools do the work; this script only puts
# their inputs in place and runs them with the box values.
#   host            the hostname is BOX_HOST; apply refuses while a job runs
#   account         the runner account, its groups, the GPU lock group
#   gpu-lock        BOX_GPU_LOCK root:<lock group> 0660, re-made at boot by a
#                   tmpfiles.d rule
#   dirs            the staging root and the resident socket dir (0700)
#   rust            the runner account's own rustup toolchain (the workflow
#                   needs cargo on PATH; setup-sandbox-user.sh copies it)
#   engine          this checkout is BOX_ENGINE_DIR, clean, at the mirror's main
#   benchd          the benchd pair, sha256 + bytes = the box file
#                   (tools/fetch-benchd.sh)
#   weights         every pinned target file in BOX_SNAPSHOT_DIR, bytes and
#                   sha256, root-owned and read-only
#   image           the pinned engine container, by digest
#   goldens         every pinned golden in BOX_GOLDEN_DIR (tools/fetch-goldens.sh)
#   reference       the reference tree at baseline_reference_commit, built
#                   (tools/stage-baseline-workspace.sh)
#   runner-install  the actions-runner release (pin of tools/box-runner/install.sh)
#   runner-register the registration: name, repository, labels (config.sh)
#   runner-env      actions-runner/.env, rendered from the box file, 0600
#   calibration     the calibration file exists (tools/calibrate-box.sh)
#   sandbox         the sandbox user (tools/box-runner/setup-sandbox-user.sh)
#   service         the runner's systemd service: installed, enabled, active
#   preflight       tools/ranked-box-preflight.sh passes as the runner, with
#                   the runner .env (the check a ranked job runs first)
#
# CREDENTIALS. No credential is written on the box. Three units can need one:
# the mirror fetch and the benchd fetch (a GitHub token that can read the
# repositories), the registration (the same token makes a one-hour
# registration token), and the goldens (the R2 values). Apply reads them ONLY
# with --secrets-stdin, as KEY=VALUE lines on stdin (GITHUB_TOKEN,
# R2_BUCKET_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY), keeps them in
# this process, and gives each one only to the command that needs it. A box
# that is already converged needs none.
#
# THE CALIBRATION IS NOT AN UPDATE. Apply runs tools/calibrate-box.sh only when
# the calibration file is absent (about 35 minutes; the tool takes the GPU lock
# itself). A file that the preflight refuses is reported; remove it
# deliberately, then apply. The reference tree is the same: a tree at another
# commit is reported, never reset.
#
# Another tenant on the box: the units in BOX_UNTOUCHED_UNITS are read and
# reported, never changed. Every name this script writes comes from the box
# file.
set -uo pipefail

TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
REPO_DIR="$(cd -- "${TOOL_DIR}/../.." >/dev/null && pwd -P)"

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

BOX=""
CHECK=0
SECRETS_STDIN=0
ONLY_UNITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --secrets-stdin) SECRETS_STDIN=1 ;;
    --unit) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; ONLY_UNITS+=("$2"); shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "converge.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) [[ -z "${BOX}" ]] || { echo "converge.sh: one box only" >&2; exit 2; }; BOX="$1" ;;
  esac
  shift
done
[[ -n "${BOX}" ]] || { usage >&2; exit 2; }
[[ "${CHECK}" -eq 1 && "${SECRETS_STDIN}" -eq 1 ]] && { echo "converge.sh: --check reads no secret" >&2; exit 2; }

# ---------------------------------------------------------------------------
# status lines
N_OK=0; N_CONV=0; N_WOULD=0; N_FAIL=0
UNIT="-"
line() { printf '  %-13s %-16s %s\n' "$1" "${UNIT}" "$2"; }
ok()   { N_OK=$((N_OK + 1)); line OK "$*"; }
conv() { N_CONV=$((N_CONV + 1)); line CONVERGED "$*"; }
would(){ N_WOULD=$((N_WOULD + 1)); line WOULD-CHANGE "$*"; }
bad()  { N_FAIL=$((N_FAIL + 1)); line FAIL "$*"; }
note() { line NOTE "$*"; }
die()  { echo "converge.sh: $*" >&2; exit 2; }

# change <description> <command...>: --check reports, apply runs the command.
change() {
  local what="$1"; shift
  if [[ "${CHECK}" -eq 1 ]]; then would "${what}"; return 1; fi
  if "$@"; then conv "${what}"; return 0; fi
  bad "${what}: the change failed"; return 1
}

# ---------------------------------------------------------------------------
# the box file: values only
BOX_FILE="${BOX_FILE:-${TOOL_DIR}/${BOX}.env}"
[[ -f "${BOX_FILE}" ]] || die "no box file ${BOX_FILE}; known boxes: $(cd "${TOOL_DIR}" && ls -- *.env 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')"
load_box_file() {
  local n=0 l
  while IFS= read -r l || [[ -n "${l}" ]]; do
    n=$((n + 1))
    [[ -z "${l}" || "${l}" =~ ^# ]] && continue
    [[ "${l}" =~ ^BOX_[A-Z0-9_]+=\"[A-Za-z0-9_./:@,\ +=-]*\"$ ]] \
      || die "${BOX_FILE}:${n} is not a comment, a blank line or BOX_<KEY>=\"<value>\""
  done < "${BOX_FILE}"
  # shellcheck source=/dev/null  # values only, checked above
  . "${BOX_FILE}"
}
load_box_file

need() {
  local k
  for k in "$@"; do [[ -n "${!k:-}" ]] || die "${BOX_FILE} sets no ${k}"; done
}
need BOX_HOST BOX_TRACK_ID BOX_TRACK_FIXTURE BOX_GOLDEN_ENV BOX_RUNNER_USER BOX_RUNNER_HOME \
  BOX_RUST_TOOLCHAIN BOX_STAGING_ROOT BOX_MIRROR_DIR BOX_MIRROR_URL BOX_ENGINE_DIR \
  BOX_BENCHD_BIN_DIR BOX_BENCHD_SHA256 BOX_BENCHD_BYTES BOX_SNAPSHOT_DIR BOX_GOLDEN_DIR \
  BOX_BASELINE_WORKSPACE BOX_BASELINE_CALIBRATION BOX_ACTIONS_RUNNER_DIR BOX_RUNNER_REPO_URL \
  BOX_RUNNER_NAME BOX_RUNNER_LABELS BOX_GPU_LOCK BOX_LOCK_GROUP BOX_SANDBOX_USER BOX_SHARE_GROUP \
  BOX_SANDBOX_ROOT BOX_SANDBOX_ETC BOX_SUDOERS_NAME BOX_EGRESS_UNIT BOX_ACL_UNIT BOX_RUST_DIR \
  BOX_SERVE_UP_LOG_DIR BOX_SERVE_UP_SOCKET_DIR BOX_SERVE_UP_CONTAINER_PREFIX BOX_REFERENCE_BUILT

RU="${BOX_RUNNER_USER}"
RH="${BOX_RUNNER_HOME}"
FIXTURE="${REPO_DIR}/${BOX_TRACK_FIXTURE}"
[[ -r "${FIXTURE}" ]] || die "no track fixture ${FIXTURE}"
command -v jq >/dev/null 2>&1 || die "jq is required"
REPO_SLUG="${BOX_RUNNER_REPO_URL#https://github.com/}"
SERVICE="actions.runner.${REPO_SLUG/\//-}.${BOX_RUNNER_NAME}.service"
SYSTEMD_DIR="${CONVERGE_SYSTEMD_DIR:-/etc/systemd/system}"
TMPFILES_DIR="${CONVERGE_TMPFILES_DIR:-/etc/tmpfiles.d}"
ENV_FILE="${BOX_ACTIONS_RUNNER_DIR}/.env"

IS_ROOT=0; [[ "$(id -u)" -eq 0 ]] && IS_ROOT=1
[[ "${IS_ROOT}" -eq 1 ]] || die "run as root: sudo $0 ${BOX}$([[ "${CHECK}" -eq 1 ]] && echo ' --check')"

# secrets: KEY=VALUE lines on stdin, kept in this process only
GITHUB_TOKEN_IN=""; R2_ENDPOINT_IN=""; R2_ID_IN=""; R2_SECRET_IN=""
if [[ "${SECRETS_STDIN}" -eq 1 ]]; then
  while IFS= read -r l || [[ -n "${l}" ]]; do
    case "${l}" in
      GITHUB_TOKEN=*) GITHUB_TOKEN_IN="${l#*=}" ;;
      R2_BUCKET_ENDPOINT=*) R2_ENDPOINT_IN="${l#*=}" ;;
      R2_ACCESS_KEY_ID=*) R2_ID_IN="${l#*=}" ;;
      R2_SECRET_ACCESS_KEY=*) R2_SECRET_IN="${l#*=}" ;;
      '') ;;
      *) die "--secrets-stdin: unknown key ${l%%=*}" ;;
    esac
  done
fi

# as_runner CMD...: run as the runner account, in its home, with a clean env.
as_runner() {
  runuser -u "${RU}" -- env -i HOME="${RH}" USER="${RU}" LOGNAME="${RU}" \
    PATH="${RH}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 "$@"
}
# runner_git DIR ARGS...: git as the runner account (it owns the trees).
runner_git() { local d="$1"; shift; as_runner git -C "${d}" "$@"; }
# mirror_git DIR ARGS...: one git command as the runner account. With a
# GITHUB_TOKEN from --secrets-stdin, the token is an http header in that
# process's environment only; it is never in a git config file.
mirror_git() {
  local d="$1"; shift
  if [[ -z "${GITHUB_TOKEN_IN}" ]]; then
    as_runner env GIT_TERMINAL_PROMPT=0 git -C "${d}" "$@"; return
  fi
  printf '%s\n' "${GITHUB_TOKEN_IN}" | as_runner bash -c \
    'IFS= read -r t; d="$1"; shift; GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraheader GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf "x-access-token:%s" "${t}" | base64 -w0)" exec git -C "${d}" "$@"' \
    _ "${d}" "$@"
}

# The runner .env, rendered from the box file. The order is the order the
# workflow and the engine tools read; LANG and PATH first, as the runner's own
# env.sh writes them.
render_env() {
  cat <<EOF
LANG=en_US.utf8
PATH=${RH}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MLXFAST_TARGET_SNAPSHOT_DIR=${BOX_SNAPSHOT_DIR}
BENCHD_BIN_DIR=${BOX_BENCHD_BIN_DIR}
${BOX_GOLDEN_ENV}=${BOX_GOLDEN_DIR}
MLXFAST_BASELINE_WORKSPACE=${BOX_BASELINE_WORKSPACE}
MLXFAST_BASELINE_CALIBRATION=${BOX_BASELINE_CALIBRATION}
RUNNER_USER=${RU}
SANDBOX_USER=${BOX_SANDBOX_USER}
SHARE_GROUP=${BOX_SHARE_GROUP}
SANDBOX_ROOT=${BOX_SANDBOX_ROOT}
SANDBOX_ETC=${BOX_SANDBOX_ETC}
SUDOERS_NAME=${BOX_SUDOERS_NAME}
EGRESS_UNIT=${BOX_EGRESS_UNIT}
ACL_UNIT=${BOX_ACL_UNIT}
RUST_DIR=${BOX_RUST_DIR}
SERVE_UP_LOG_DIR=${BOX_SERVE_UP_LOG_DIR}
SERVE_UP_SOCKET_DIR=${BOX_SERVE_UP_SOCKET_DIR}
SERVE_UP_CONTAINER_PREFIX=${BOX_SERVE_UP_CONTAINER_PREFIX}
EOF
  # A second runner of the account names its own install (setup-sandbox-user.sh).
  [[ "${BOX_ACTIONS_RUNNER_DIR}" == "${RH}/actions-runner" ]] || printf 'RUNNER_DIR=%s\n' "${BOX_ACTIONS_RUNNER_DIR}"
  # BOX_ENV_EXTRA: KEY=VALUE pairs, space separated, that the engine's own tools read.
  local kv
  for kv in ${BOX_ENV_EXTRA:-}; do printf '%s\n' "${kv}"; done
}
WORK="$(mktemp -d /tmp/box-converge.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT
chmod 0711 "${WORK}"
RENDERED_ENV="${WORK}/runner.env"
render_env > "${RENDERED_ENV}"

# with_runner_env CMD...: as the runner, with the rendered .env exported (the
# environment a ranked job and the engine tools see).
with_runner_env() {
  local copy="${WORK}/env.${RANDOM}"
  install -m 0400 -o "${RU}" /dev/null "${copy}" 2>/dev/null || return 1
  cat "${RENDERED_ENV}" > "${copy}"
  as_runner bash -c 'set -a; . "$1"; set +a; shift; exec "$@"' _ "${copy}" "$@"
  local rc=$?
  rm -f "${copy}"
  return "${rc}"
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
bytes_of() { stat -c %s "$1"; }
mode_owner() { stat -c '%a %U:%G' "$1" 2>/dev/null; }
user_exists() { id -u "$1" >/dev/null 2>&1; }

selected() {
  [[ ${#ONLY_UNITS[@]} -eq 0 ]] && return 0
  local u; for u in "${ONLY_UNITS[@]}"; do [[ "${u}" == "$1" ]] && return 0; done
  return 1
}

ENV_CHANGED=0
STOP_AFTER_UNIT=0

# ---------------------------------------------------------------------------
unit_host() {
  local h; h="$(hostname -s 2>/dev/null || hostname)"
  [[ "${h}" == "${BOX_HOST}" ]] || { bad "hostname is ${h}, the box file names ${BOX_HOST}: wrong box file"; return 1; }
  ok "hostname ${h}; box file ${BOX_FILE##*/}; track ${BOX_TRACK_ID}"
  if user_exists "${RU}" && pgrep -u "${RU}" -f 'Runner.Worker|calibrate-box.sh' >/dev/null 2>&1; then
    if [[ "${CHECK}" -eq 1 ]]; then note "a job or a calibration of ${RU} is running; this check reads a box in use"
    else bad "a job or a calibration of ${RU} is running; apply at a quiet moment"; return 1; fi
  else
    ok "no job and no calibration of ${RU} is running"
  fi
  local u st
  for u in ${BOX_UNTOUCHED_UNITS:-}; do
    st="$(systemctl is-active "${u}" 2>/dev/null || true)"
    note "untouched: ${u} is ${st:-unknown}"
  done
}

unit_account() {
  if getent group "${BOX_LOCK_GROUP}" >/dev/null; then ok "group ${BOX_LOCK_GROUP} exists"
  else change "group ${BOX_LOCK_GROUP}: create" groupadd "${BOX_LOCK_GROUP}"; fi
  if user_exists "${RU}"; then ok "account ${RU} exists"
  else change "account ${RU}: create (home ${RH}, bash)" useradd -m -d "${RH}" -s /bin/bash "${RU}" || return 1; fi
  if [[ -d "${RH}" && "$(stat -c %U "${RH}")" == "${RU}" ]]; then ok "home ${RH} owned by ${RU}"
  else change "home ${RH}: owner ${RU}" install -d -o "${RU}" -g "${RU}" -m 0750 "${RH}"; fi
  local g
  for g in $(printf '%s\n' ${BOX_RUNNER_GROUPS:-} "${BOX_LOCK_GROUP}" | awk '!seen[$0]++'); do
    if id -nG "${RU}" 2>/dev/null | tr ' ' '\n' | grep -qx "${g}"; then ok "${RU} is in group ${g}"
    elif getent group "${g}" >/dev/null; then change "${RU}: join group ${g}" usermod -aG "${g}" "${RU}"
    else bad "group ${g} does not exist (install its package first: docker -> the NVIDIA container stack)"; fi
  done
}

unit_gpu_lock() {
  local rule="f ${BOX_GPU_LOCK} 0660 root ${BOX_LOCK_GROUP} -"
  local f="${TMPFILES_DIR}/mtplx-gpu-lock.conf"
  if grep -rqxF -- "${rule}" "${TMPFILES_DIR}" 2>/dev/null; then ok "tmpfiles.d rule re-makes ${BOX_GPU_LOCK} at boot"
  else change "tmpfiles.d rule ${f}" bash -c 'printf "%s\n" "$1" > "$2" && chmod 0644 "$2"' _ "${rule}" "${f}"; fi
  if [[ "$(mode_owner "${BOX_GPU_LOCK}")" == "660 root:${BOX_LOCK_GROUP}" ]]; then ok "${BOX_GPU_LOCK} 0660 root:${BOX_LOCK_GROUP}"
  elif [[ -e "${BOX_GPU_LOCK}" ]]; then
    change "${BOX_GPU_LOCK}: 0660 root:${BOX_LOCK_GROUP} (was $(mode_owner "${BOX_GPU_LOCK}"))" \
      bash -c 'chown "root:$2" "$1" && chmod 0660 "$1"' _ "${BOX_GPU_LOCK}" "${BOX_LOCK_GROUP}"
  else change "${BOX_GPU_LOCK}: create 0660 root:${BOX_LOCK_GROUP}" install -m 0660 -o root -g "${BOX_LOCK_GROUP}" /dev/null "${BOX_GPU_LOCK}"; fi
}

# want_dir PATH MODE OWNER:GROUP
want_dir() {
  local p="$1" m="$2" o="$3"
  if [[ -d "${p}" && "$(mode_owner "${p}")" == "${m} ${o}" ]]; then ok "${p} ${m} ${o}"
  else change "${p}: dir ${m} ${o}$([[ -e "${p}" ]] && echo " (was $(mode_owner "${p}"))")" \
    install -d -m "${m}" -o "${o%%:*}" -g "${o##*:}" "${p}"; fi
}

unit_dirs() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  if [[ -d "${BOX_STAGING_ROOT}" && "$(stat -c %U "${BOX_STAGING_ROOT}")" == "${RU}" ]]; then ok "staging root ${BOX_STAGING_ROOT}"
  else change "staging root ${BOX_STAGING_ROOT}" install -d -m 0775 -o "${RU}" -g "${RU}" "${BOX_STAGING_ROOT}"; fi
  install_parent_dirs "${BOX_SERVE_UP_SOCKET_DIR}"
  want_dir "${BOX_SERVE_UP_SOCKET_DIR}" 700 "${RU}:${RU}"
  # Entries in the staging root that the box file does not declare: reported.
  local e known
  known=" ${BOX_MIRROR_DIR##*/} ${BOX_ENGINE_DIR##*/} ${BOX_BENCHD_BIN_DIR##*/} ${BOX_GOLDEN_DIR##*/} $(basename "$(dirname "${BOX_BASELINE_WORKSPACE}")") "
  if [[ -d "${BOX_STAGING_ROOT}" ]]; then
    for e in "${BOX_STAGING_ROOT}"/*; do
      [[ -e "${e}" ]] || continue
      [[ "${known}" == *" ${e##*/} "* ]] || note "not in the box file (left as is): ${e}"
    done
  fi
}
# install_parent_dirs PATH: root-owned 0755 parents for a path under /srv or /var/lib.
install_parent_dirs() {
  local parent; parent="$(dirname "$1")"
  [[ -d "${parent}" ]] && return 0
  change "${parent}: dir 755 root:root" install -d -m 0755 -o root -g root "${parent}"
}

unit_rust() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  local have
  have="$(as_runner bash -c 'cd / && rustup show active-toolchain 2>/dev/null' | awk '{print $1}')"
  if [[ "${have}" == "${BOX_RUST_TOOLCHAIN}-"* ]]; then ok "${RU} rustup toolchain ${have}"; return 0; fi
  if [[ -x "${RH}/.cargo/bin/rustup" ]]; then
    change "${RU} rustup toolchain ${BOX_RUST_TOOLCHAIN} (was ${have:-none})" \
      as_runner bash -c 'cd / && rustup toolchain install "$1" --profile minimal && rustup default "$1"' _ "${BOX_RUST_TOOLCHAIN}"
  else
    change "${RU} rustup + toolchain ${BOX_RUST_TOOLCHAIN}" install_rustup
  fi
}
# rustup-init from static.rust-lang.org, checked against its published sha256.
install_rustup() {
  local triple url want got
  triple="$(uname -m)-unknown-linux-gnu"
  url="https://static.rust-lang.org/rustup/dist/${triple}/rustup-init"
  curl -fsSL --proto '=https' -o "${WORK}/rustup-init" "${url}" || return 1
  want="$(curl -fsSL --proto '=https' "${url}.sha256" | awk '{print $1}')"
  got="$(sha256_of "${WORK}/rustup-init")"
  [[ -n "${want}" && "${want}" == "${got}" ]] || { echo "rustup-init sha256 ${got} != published ${want}" >&2; return 1; }
  chmod 0755 "${WORK}/rustup-init"
  as_runner bash -c 'cd / && "$1" -y --no-modify-path --profile minimal --default-toolchain "$2"' _ "${WORK}/rustup-init" "${BOX_RUST_TOOLCHAIN}"
}

unit_engine() {
  if [[ "${REPO_DIR}" != "${BOX_ENGINE_DIR}" ]]; then
    # The first run on a fresh box comes from any checkout: it makes the
    # mirror and the box engine checkout, and every later run comes from there.
    if [[ -e "${BOX_ENGINE_DIR}" ]]; then
      bad "this script runs from ${REPO_DIR}; the box engine checkout is ${BOX_ENGINE_DIR}: run it from there"; return 1
    fi
    if [[ "${CHECK}" -eq 1 ]]; then would "engine: make the mirror ${BOX_MIRROR_DIR} and the checkout ${BOX_ENGINE_DIR} at the mirror's main"; return 1; fi
    user_exists "${RU}" || { bad "engine: no account ${RU}"; return 1; }
    if [[ ! -d "${BOX_MIRROR_DIR}" ]]; then
      install -d -m 0775 -o "${RU}" -g "${RU}" "$(dirname "${BOX_MIRROR_DIR}")"
      mirror_git "$(dirname "${BOX_MIRROR_DIR}")" clone -q --mirror "${BOX_MIRROR_URL}" "${BOX_MIRROR_DIR}" \
        || { bad "engine: cannot clone ${BOX_MIRROR_URL} (a private repository needs --secrets-stdin and a GITHUB_TOKEN)"; return 1; }
    fi
    if runner_git "$(dirname "${BOX_ENGINE_DIR}")" clone -q "${BOX_MIRROR_DIR}" "${BOX_ENGINE_DIR}" \
        && runner_git "${BOX_ENGINE_DIR}" checkout -q --detach origin/main; then
      conv "engine: mirror + checkout at $(runner_git "${BOX_ENGINE_DIR}" rev-parse --short=12 HEAD)"
    else bad "engine: clone of the mirror failed"; return 1; fi
    note "run every later apply and check from ${BOX_ENGINE_DIR}: sudo ${BOX_ENGINE_DIR}/tools/box/converge.sh ${BOX}"
    STOP_AFTER_UNIT=1
    return 0
  fi
  local head main dirty before
  if [[ "${CHECK}" -eq 0 && -n "${GITHUB_TOKEN_IN}" ]]; then
    # With a token, apply first brings the mirror to GitHub's main.
    before="$(runner_git "${BOX_MIRROR_DIR}" rev-parse refs/heads/main 2>/dev/null)"
    if mirror_git "${BOX_MIRROR_DIR}" fetch -q --prune origin >/dev/null 2>&1; then
      main="$(runner_git "${BOX_MIRROR_DIR}" rev-parse refs/heads/main 2>/dev/null)"
      [[ "${main}" == "${before}" ]] && ok "mirror main ${main:0:12} = GitHub" || conv "mirror main ${before:0:12} -> ${main:0:12}"
    else bad "mirror fetch from ${BOX_MIRROR_URL} failed"; fi
  fi
  head="$(runner_git "${REPO_DIR}" rev-parse HEAD 2>/dev/null)"
  main="$(runner_git "${BOX_MIRROR_DIR}" rev-parse refs/heads/main 2>/dev/null)"
  dirty="$(runner_git "${REPO_DIR}" status --porcelain --untracked-files=no 2>/dev/null)"
  [[ -z "${dirty}" ]] && ok "engine checkout is clean" \
    || bad "engine checkout has changes to tracked files: commit nothing on a box; restore it"
  if [[ -n "${head}" && "${head}" == "${main}" ]]; then ok "engine checkout at ${head:0:12} = mirror main"
  else
    bad "engine checkout at ${head:0:12}, mirror main ${main:0:12}: advance it with 'sudo -u ${RU} git -C ${BOX_ENGINE_DIR} fetch -q origin && sudo -u ${RU} git -C ${BOX_ENGINE_DIR} checkout -q --detach origin/main', then run this script again"
  fi
  local url; url="$(runner_git "${BOX_MIRROR_DIR}" config --get remote.origin.url 2>/dev/null)"
  [[ "${url}" == "${BOX_MIRROR_URL}" ]] && ok "mirror ${BOX_MIRROR_DIR} of ${BOX_MIRROR_URL}" \
    || bad "mirror remote is '${url}', the box file names ${BOX_MIRROR_URL}"
}

# mirror_has COMMIT: the commit is in the box mirror; apply fetches it once.
mirror_has() {
  runner_git "${BOX_MIRROR_DIR}" cat-file -e "$1^{commit}" 2>/dev/null && return 0
  [[ "${CHECK}" -eq 1 ]] && return 1
  mirror_git "${BOX_MIRROR_DIR}" fetch -q --prune origin >/dev/null 2>&1 || return 1
  runner_git "${BOX_MIRROR_DIR}" cat-file -e "$1^{commit}" 2>/dev/null
}

unit_benchd() {
  local b="${BOX_BENCHD_BIN_DIR}/benchd" m="${BOX_BENCHD_BIN_DIR}/benchd.manifest.json"
  local pin_src; pin_src="$(sed -n 's/^PINNED_SOURCE_COMMIT="\([0-9a-f]\{40\}\)"$/\1/p' "${REPO_DIR}/tools/fetch-benchd.sh")"
  if [[ -f "${b}" && "$(bytes_of "${b}")" == "${BOX_BENCHD_BYTES}" && "$(sha256_of "${b}")" == "${BOX_BENCHD_SHA256}" ]] \
      && [[ "$(jq -r '.sha256' "${m}" 2>/dev/null)" == "${BOX_BENCHD_SHA256}" ]]; then
    ok "benchd sha256 ${BOX_BENCHD_SHA256:0:12} ${BOX_BENCHD_BYTES} B, manifest agrees (source $(jq -r '.source_commit[0:12]' "${m}"))"
  else
    local why="absent"; [[ -f "${b}" ]] && why="sha256 $(sha256_of "${b}" | cut -c1-12)"
    if [[ "${CHECK}" -eq 1 ]]; then would "benchd: stage ${BOX_BENCHD_SHA256:0:12} (now ${why})"
    elif [[ -z "${GITHUB_TOKEN_IN}" ]]; then bad "benchd is ${why}: apply with --secrets-stdin and a GITHUB_TOKEN that reads the dist channel"
    else
      install -d -m 0775 -o "${RU}" -g "${RU}" "${BOX_BENCHD_BIN_DIR}"
      if printf '%s\n' "${GITHUB_TOKEN_IN}" | as_runner bash -c 'IFS= read -r BENCHD_DIST_TOKEN; export BENCHD_DIST_TOKEN BENCHD_REFRESH=1 BENCHD_BIN_DIR="$1"; cd "$2" && tools/fetch-benchd.sh >/dev/null' _ "${BOX_BENCHD_BIN_DIR}" "${REPO_DIR}" \
          && [[ "$(sha256_of "${b}")" == "${BOX_BENCHD_SHA256}" ]]; then
        conv "benchd staged by tools/fetch-benchd.sh: sha256 ${BOX_BENCHD_SHA256:0:12}"
      else bad "tools/fetch-benchd.sh did not stage sha256 ${BOX_BENCHD_SHA256}: the box file and the fetch-benchd.sh pin disagree?"; fi
    fi
  fi
  if [[ -f "${m}" ]]; then
    [[ "$(jq -r '.source_commit' "${m}")" == "${pin_src}" ]] && ok "benchd source = the tools/fetch-benchd.sh pin ${pin_src:0:12}" \
      || bad "benchd source $(jq -r '.source_commit[0:12]' "${m}") != the tools/fetch-benchd.sh pin ${pin_src:0:12}: update the box file pin"
  fi
}

unit_weights() {
  local pins n=0 missing=() wrong=() rel bytes sha f
  # A file with a "source" ("<repo>@<revision>") comes from that repository;
  # every other file from the target's upstream_model_id @ upstream_revision.
  pins="$(jq -r '.target.files[] | [.path, (.bytes|tostring), .sha256, (.source // "")] | @tsv' "${FIXTURE}")"
  [[ -n "${pins}" ]] || { bad "the fixture pins no target files"; return 1; }
  install_parent_dirs "$(dirname "${BOX_SNAPSHOT_DIR}")"
  want_dir "$(dirname "${BOX_SNAPSHOT_DIR}")" 755 root:root
  want_dir "${BOX_SNAPSHOT_DIR}" 755 root:root
  # Bytes first; then every sha256, 8 at a time.
  while IFS=$'\t' read -r rel bytes sha _; do
    n=$((n + 1)); f="${BOX_SNAPSHOT_DIR}/${rel}"
    if [[ ! -f "${f}" || "$(bytes_of "${f}")" != "${bytes}" ]]; then missing+=("${rel}"); continue; fi
    printf '%s\t%s\n' "${f}" "${sha}"
  done <<< "${pins}" > "${WORK}/weights.want"
  cut -f1 "${WORK}/weights.want" | xargs -r -d '\n' -P 8 -n 1 sha256sum > "${WORK}/weights.got"
  mapfile -t wrong < <(awk -F'\t' 'NR == FNR { want[$1] = $2; next }
    { f = substr($0, 67); if (want[f] != substr($0, 1, 64)) print f; delete want[f] }
    END { for (f in want) print f }' "${WORK}/weights.want" "${WORK}/weights.got" | sed "s#^${BOX_SNAPSHOT_DIR}/##")
  if [[ ${#missing[@]} -eq 0 && ${#wrong[@]} -eq 0 ]]; then
    ok "target snapshot: ${n} pinned files, bytes + sha256"
  else
    local base src url; base="https://huggingface.co/$(jq -r '.target.upstream_model_id' "${FIXTURE}")/resolve/$(jq -r '.target.upstream_revision' "${FIXTURE}")"
    if [[ "${CHECK}" -eq 1 ]]; then would "target snapshot: fetch ${#missing[@]} missing, ${#wrong[@]} wrong of ${n} from the fixture's upstream"
    else
      local r ok_all=1
      for r in "${wrong[@]}"; do rm -f "${BOX_SNAPSHOT_DIR}/${r}"; done
      while IFS=$'\t' read -r rel bytes sha src; do
        f="${BOX_SNAPSHOT_DIR}/${rel}"
        [[ -f "${f}" && "$(bytes_of "${f}")" == "${bytes}" ]] && continue
        url="${base}/${rel}"
        [[ -n "${src}" ]] && url="https://huggingface.co/${src%@*}/resolve/${src##*@}/${rel}"
        [[ -n "${BOX_WEIGHTS_BASE_URL:-}" ]] && url="${BOX_WEIGHTS_BASE_URL}/${rel}"
        curl -fL --retry 5 --retry-delay 5 -C - -s -o "${f}.partial" "${url}" \
          && [[ "$(bytes_of "${f}.partial")" == "${bytes}" && "$(sha256_of "${f}.partial")" == "${sha}" ]] \
          && chmod 0644 "${f}.partial" && mv -f "${f}.partial" "${f}" \
          || { rm -f "${f}.partial"; ok_all=0; bad "target file ${rel}: download or pin check failed"; }
      done <<< "${pins}"
      [[ "${ok_all}" -eq 1 ]] && conv "target snapshot: fetched ${#missing[@]} + ${#wrong[@]} files, every pin verified"
    fi
  fi
  local w; w="$(find "${BOX_SNAPSHOT_DIR}" -mindepth 1 \( ! -user root -o -perm /022 \) -print -quit 2>/dev/null)"
  [[ -z "${w}" ]] && ok "target snapshot is root-owned and writable by root only" \
    || change "target snapshot: root:root, files 0644, dirs 0755" \
      bash -c 'chown -R root:root "$1" && find "$1" -type f -exec chmod 0644 {} + && find "$1" -type d -exec chmod 0755 {} +' _ "${BOX_SNAPSHOT_DIR}"
}

unit_image() {
  local img dig ref
  img="$(jq -r '.serve_configuration.container_image' "${FIXTURE}")"
  dig="$(jq -r '.serve_configuration.container_digest // empty' "${FIXTURE}")"
  if [[ -n "${dig}" ]]; then
    # A pinned public image: pulled by digest.
    ref="${img%%:*}@${dig}"
    if docker image inspect "${ref}" >/dev/null 2>&1; then ok "engine image ${img} ${dig:0:19}"
    else change "engine image: docker pull ${ref}" bash -c 'docker pull -q "$1" >/dev/null' _ "${ref}"; fi
    return 0
  fi
  # A box-built image: tools/engine/build-image.sh builds it and writes its Id
  # to BOX_IMAGE_ID_FILE, which the ranked job reads.
  [[ -n "${BOX_IMAGE_ID_FILE:-}" ]] || { bad "the fixture pins no container_digest and the box file sets no BOX_IMAGE_ID_FILE"; return 1; }
  local id got
  id="$(tr -d '[:space:]' < "${BOX_IMAGE_ID_FILE}" 2>/dev/null)"
  got="$(docker image inspect --format '{{.Id}}' "${id:-none}" 2>/dev/null)"
  if [[ "${id}" =~ ^sha256:[0-9a-f]{64}$ && "${got}" == "${id}" ]]; then
    ok "engine image ${img} built on this box, Id ${id:7:12} = ${BOX_IMAGE_ID_FILE}"
  else
    install_parent_dirs "${BOX_IMAGE_ID_FILE}"
    change "engine image: tools/engine/build-image.sh ${BOX_IMAGE_ID_FILE}" \
      bash -c 'cd "$1" && tools/engine/build-image.sh "$2" >/dev/null && chmod 0644 "$2"' _ "${REPO_DIR}" "${BOX_IMAGE_ID_FILE}"
  fi
}

# pending <what>: an item that waits for other work (the track's goldens):
# --check reports it as a difference; apply reports it and goes on.
pending() {
  if [[ "${CHECK}" -eq 1 ]]; then would "pending: $*"; else note "pending: $*"; fi
}
GOLDENS_READY=1
unit_goldens() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  if [[ "$(jq -r '.golden_pins_provisional // false' "${FIXTURE}")" == "true" ]]; then
    GOLDENS_READY=0
    pending "the fixture's golden pins are provisional; record the goldens (tools/record-goldens.sh), pin them, then apply"
    return 0
  fi
  local pins n=0 bad_n=0 key sha bytes f
  pins="$(jq -r '
    [ (.timed_prompt_pool // [])[],
      ((.speculative_oracles // {}) | to_entries[] | .value | to_entries[] | .value) ]
    | map(select(type == "object" and (.r2_path | type) == "string"))
    | unique_by(.r2_path)[] | [.r2_path, .sha256, (.bytes | tostring)] | @tsv' "${FIXTURE}")"
  while IFS=$'\t' read -r key sha bytes; do
    [[ -n "${key}" ]] || continue
    n=$((n + 1)); f="${BOX_GOLDEN_DIR}/${key##*/}"
    [[ -f "${f}" && "$(bytes_of "${f}")" == "${bytes}" && "$(sha256_of "${f}")" == "${sha}" ]] || bad_n=$((bad_n + 1))
  done <<< "${pins}"
  [[ "${bad_n}" -eq 0 ]] || GOLDENS_READY=0
  if [[ "${bad_n}" -eq 0 ]]; then ok "goldens: ${n} pinned files, bytes + sha256"
  elif [[ "${CHECK}" -eq 1 ]]; then would "goldens: stage ${bad_n} of ${n} (tools/fetch-goldens.sh --all)"
  elif [[ -z "${R2_ENDPOINT_IN}" || -z "${R2_ID_IN}" || -z "${R2_SECRET_IN}" ]]; then
    bad "goldens: ${bad_n} of ${n} not staged: apply with --secrets-stdin and the R2 values"
  else
    install -d -m 0750 -o "${RU}" -g "${RU}" "${BOX_GOLDEN_DIR}"
    if printf '%s\n%s\n%s\n' "${R2_ENDPOINT_IN}" "${R2_ID_IN}" "${R2_SECRET_IN}" | as_runner bash -c \
        'IFS= read -r R2_BUCKET_ENDPOINT; IFS= read -r R2_ACCESS_KEY_ID; IFS= read -r R2_SECRET_ACCESS_KEY; export R2_BUCKET_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; cd "$1" && tools/fetch-goldens.sh --all --out "$2" >/dev/null 2>&1' _ "${REPO_DIR}" "${BOX_GOLDEN_DIR}"; then
      conv "goldens: tools/fetch-goldens.sh --all staged ${bad_n} of ${n}"
      GOLDENS_READY=1
    else bad "goldens: tools/fetch-goldens.sh --all failed"; fi
  fi
  [[ -d "${BOX_GOLDEN_DIR}" ]] || return 0
  want_dir "${BOX_GOLDEN_DIR}" 750 "${RU}:${RU}"
  local loose; loose="$(find "${BOX_GOLDEN_DIR}" -maxdepth 1 -type f ! -perm 0440 -print -quit)"
  [[ -z "${loose}" ]] && ok "goldens are 0440" \
    || change "goldens: files 0440 ${RU}:${RU}" bash -c 'find "$1" -maxdepth 1 -type f -exec chown "$2:$2" {} + -exec chmod 0440 {} +' _ "${BOX_GOLDEN_DIR}" "${RU}"
}

unit_reference() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  local want ws="${BOX_BASELINE_WORKSPACE}" have
  want="$(jq -r '.baseline_reference_commit' "${FIXTURE}")"
  if [[ -d "${ws}" ]]; then
    have="$(runner_git "${ws}" rev-parse HEAD 2>/dev/null)"
    if [[ "${have}" != "${want}" ]]; then
      bad "reference tree at ${have:0:12}, the fixture pins ${want:0:12}: declare a new BOX_BASELINE_WORKSPACE path (a tree is never reset in place)"; return 1
    fi
    runner_git "${ws}" diff --quiet HEAD 2>/dev/null && ok "reference tree at ${want:0:12}, clean" \
      || bad "the reference tree has changes: remove ${ws} deliberately and apply"
    local rel built=1
    for rel in ${BOX_REFERENCE_BUILT}; do [[ -f "${ws}/${rel}" ]] || built=0; done
    [[ "${built}" -eq 1 ]] && ok "reference tree is built" || bad "the reference tree is not built: remove ${ws} deliberately and apply"
    return 0
  fi
  if ! mirror_has "${want}"; then
    if [[ "${CHECK}" -eq 1 ]]; then would "reference tree: stage ${want:0:12} (the mirror lacks it: apply fetches it)"
    else bad "the mirror lacks ${want:0:12}: apply with --secrets-stdin and a GITHUB_TOKEN"; fi
    return 1
  fi
  [[ "${CHECK}" -eq 1 || -d "$(dirname "${ws}")" ]] || install -d -m 0775 -o "${RU}" -g "${RU}" "$(dirname "${ws}")"
  change "reference tree: tools/stage-baseline-workspace.sh ${ws} at ${want:0:12}" \
    as_runner bash -c 'cd "$1" && tools/stage-baseline-workspace.sh "$2" --source "$3" >/dev/null' _ "${REPO_DIR}" "${ws}" "${BOX_MIRROR_DIR}"
}

# The actions-runner release pinned by tools/box-runner/install.sh (one pin).
unit_runner_install() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  if [[ -x "${BOX_ACTIONS_RUNNER_DIR}/config.sh" && -x "${BOX_ACTIONS_RUNNER_DIR}/run.sh" ]]; then
    ok "actions-runner in ${BOX_ACTIONS_RUNNER_DIR} (it updates itself; installed $(cat "${BOX_ACTIONS_RUNNER_DIR}/.box-runner-installed-version" 2>/dev/null || echo '?'))"
    return 0
  fi
  change "actions-runner: install the tools/box-runner/install.sh pin in ${BOX_ACTIONS_RUNNER_DIR}" install_actions_runner
}
install_actions_runner() {
  local inst="${REPO_DIR}/tools/box-runner/install.sh" ver tag sha tarball
  ver="$(sed -n 's/^ACTIONS_RUNNER_VERSION="\${ACTIONS_RUNNER_VERSION:-\([0-9.]*\)}"$/\1/p' "${inst}")"
  case "$(uname -m)" in aarch64|arm64) tag=linux-arm64 ;; x86_64) tag=linux-x64 ;; *) return 1 ;; esac
  sha="$(sed -n "s/^ *${tag}) printf '%s' \"\([0-9a-f]\{64\}\)\" ;;$/\1/p" "${inst}")"
  [[ -n "${ver}" && -n "${sha}" ]] || { echo "no actions-runner pin in ${inst}" >&2; return 1; }
  tarball="${WORK}/actions-runner.tar.gz"
  curl -fsSL --max-time 300 -o "${tarball}" \
    "https://github.com/actions/runner/releases/download/v${ver}/actions-runner-${tag}-${ver}.tar.gz" || return 1
  [[ "$(sha256_of "${tarball}")" == "${sha}" ]] || { echo "actions-runner tarball sha256 mismatch" >&2; return 1; }
  install -d -m 0755 -o "${RU}" -g "${RU}" "${BOX_ACTIONS_RUNNER_DIR}"
  as_runner tar -xzf "${tarball}" -C "${BOX_ACTIONS_RUNNER_DIR}" \
    && printf '%s' "${ver}" | as_runner tee "${BOX_ACTIONS_RUNNER_DIR}/.box-runner-installed-version" >/dev/null
}

unit_runner_register() {
  local r="${BOX_ACTIONS_RUNNER_DIR}/.runner" name url
  if [[ -f "${r}" ]]; then
    name="$(jq -r '.agentName' "${r}" 2>/dev/null)"; url="$(jq -r '.gitHubUrl' "${r}" 2>/dev/null)"
    if [[ "${name}" == "${BOX_RUNNER_NAME}" && "${url}" == "${BOX_RUNNER_REPO_URL}" ]]; then
      ok "registered as ${name} on ${url#https://github.com/}"
    else bad "registered as ${name} on ${url}, the box file names ${BOX_RUNNER_NAME} on ${BOX_RUNNER_REPO_URL}: remove the registration (config.sh remove) deliberately, then apply"; fi
    return 0
  fi
  [[ -x "${BOX_ACTIONS_RUNNER_DIR}/config.sh" ]] || { would "register: the runner is not installed yet"; return 1; }
  if [[ "${CHECK}" -eq 1 ]]; then would "register ${BOX_RUNNER_NAME} on ${REPO_SLUG}, labels ${BOX_RUNNER_LABELS}"; return 1; fi
  [[ -n "${GITHUB_TOKEN_IN}" ]] || { bad "register: apply with --secrets-stdin and a GITHUB_TOKEN with admin on ${REPO_SLUG}"; return 1; }
  local reg
  reg="$(curl -fsS -X POST -H "Authorization: Bearer ${GITHUB_TOKEN_IN}" -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/${REPO_SLUG}/actions/runners/registration-token" | jq -r '.token // empty')"
  [[ -n "${reg}" ]] || { bad "register: GitHub gave no registration token"; return 1; }
  if printf '%s\n' "${reg}" | as_runner bash -c 'IFS= read -r ACTIONS_RUNNER_INPUT_TOKEN; export ACTIONS_RUNNER_INPUT_TOKEN; cd "$1" && ./config.sh --unattended --url "$2" --name "$3" --labels "$4" --replace >/dev/null' \
      _ "${BOX_ACTIONS_RUNNER_DIR}" "${BOX_RUNNER_REPO_URL}" "${BOX_RUNNER_NAME}" "${BOX_RUNNER_LABELS}"; then
    conv "registered ${BOX_RUNNER_NAME} on ${REPO_SLUG}, labels self-hosted,Linux,$(uname -m | sed 's/aarch64/ARM64/;s/x86_64/X64/'),${BOX_RUNNER_LABELS}"
  else bad "register: config.sh failed"; fi
  reg=""
}

unit_runner_env() {
  [[ -d "${BOX_ACTIONS_RUNNER_DIR}" ]] || { would "runner .env: the runner is not installed yet"; return 1; }
  if [[ -f "${ENV_FILE}" ]] && cmp -s "${ENV_FILE}" "${RENDERED_ENV}"; then ok "runner .env = the box file ($(wc -l < "${RENDERED_ENV}") keys)"
  else
    local diffkeys=""
    [[ -f "${ENV_FILE}" ]] && diffkeys="$(diff <(sort "${ENV_FILE}") <(sort "${RENDERED_ENV}") | sed -n 's/^[<>] \([A-Z0-9_]*\)=.*/\1/p' | sort -u | tr '\n' ' ')"
    change "runner .env: write ${ENV_FILE}${diffkeys:+ (keys ${diffkeys% })}" \
      bash -c 'install -m 0600 -o "$3" -g "$3" "$1" "$2"' _ "${RENDERED_ENV}" "${ENV_FILE}" "${RU}" && ENV_CHANGED=1
  fi
  if [[ -f "${ENV_FILE}" && "$(mode_owner "${ENV_FILE}")" != "600 ${RU}:${RU}" ]]; then
    change "runner .env: 0600 ${RU}:${RU}" bash -c 'chown "$2:$2" "$1" && chmod 0600 "$1"' _ "${ENV_FILE}" "${RU}"
  fi
}

unit_calibration() {
  local c="${BOX_BASELINE_CALIBRATION}"
  if [[ -f "${c}" ]]; then
    ok "calibration ${c##*/}: box $(jq -r '.box // "?"' "${c}"), captured $(jq -r '.captured_at // "?"' "${c}"), sha256 $(sha256_of "${c}" | cut -c1-12) (the preflight unit checks the band)"
    return 0
  fi
  if [[ "${GOLDENS_READY}" -ne 1 ]]; then pending "calibration: waits for the goldens"; return 0; fi
  if [[ "${CHECK}" -eq 1 ]]; then would "calibration: run tools/calibrate-box.sh ${BOX_RUNNER_NAME} (about 35 min, takes the GPU lock)"; return 1; fi
  if with_runner_env bash -c 'cd "$1" && tools/calibrate-box.sh "$2" "$3" --lock-wait 7200' _ "${REPO_DIR}" "${BOX_RUNNER_NAME}" "${c}"; then
    conv "calibration: tools/calibrate-box.sh wrote ${c} (sha256 $(sha256_of "${c}" | cut -c1-12))"
  else bad "calibration: tools/calibrate-box.sh refused (its output above names why); a refusal is a result"; fi
}

sandbox_names() {
  printf '%s\n' RUNNER_USER="${RU}" SANDBOX_USER="${BOX_SANDBOX_USER}" SHARE_GROUP="${BOX_SHARE_GROUP}" \
    SANDBOX_ROOT="${BOX_SANDBOX_ROOT}" SANDBOX_ETC="${BOX_SANDBOX_ETC}" SUDOERS_NAME="${BOX_SUDOERS_NAME}" \
    EGRESS_UNIT="${BOX_EGRESS_UNIT}" ACL_UNIT="${BOX_ACL_UNIT}" RUST_DIR="${BOX_RUST_DIR}"
  [[ "${BOX_ACTIONS_RUNNER_DIR}" == "${RH}/actions-runner" ]] || printf '%s\n' RUNNER_DIR="${BOX_ACTIONS_RUNNER_DIR}"
}
unit_sandbox() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  local out
  if out="$(with_runner_env "${REPO_DIR}/tools/box-runner/setup-sandbox-user.sh" --check 2>&1)"; then
    ok "sandbox ${BOX_SANDBOX_USER}: setup-sandbox-user.sh --check passed ($(grep -c '^PASS' <<<"${out}") PASS)"
    return 0
  fi
  local fails; fails="$(grep -E '^FAIL' <<<"${out}" | cut -c1-120 | paste -sd ';' -)"
  if [[ "${CHECK}" -eq 1 ]]; then would "sandbox ${BOX_SANDBOX_USER}: run setup-sandbox-user.sh (${fails:-check failed})"; return 1; fi
  # setup-sandbox-user.sh reads SERVE_UP_LOG_DIR from the runner .env and ends with its own --check.
  mapfile -t names < <(sandbox_names)
  if env "${names[@]}" "${REPO_DIR}/tools/box-runner/setup-sandbox-user.sh" >"${WORK}/sandbox.log" 2>&1; then
    conv "sandbox ${BOX_SANDBOX_USER}: setup-sandbox-user.sh applied and its --check passed"
  else bad "sandbox: setup-sandbox-user.sh failed: $(grep -E '^FAIL|setup-sandbox-user.sh:' "${WORK}/sandbox.log" | tail -3 | paste -sd ';' -)"; fi
}

unit_service() {
  local unit="${SYSTEMD_DIR}/${SERVICE}"
  if [[ -f "${unit}" ]]; then
    if grep -qx "User=${RU}" "${unit}"; then ok "${SERVICE} installed, User=${RU}"
    else bad "${SERVICE} does not run as ${RU}: uninstall it with svc.sh deliberately, then apply"; fi
  elif [[ -f "${BOX_ACTIONS_RUNNER_DIR}/.runner" ]]; then
    change "${SERVICE}: svc.sh install ${RU}" bash -c 'cd "$1" && ./svc.sh install "$2" >/dev/null' _ "${BOX_ACTIONS_RUNNER_DIR}" "${RU}" || return 1
  else would "${SERVICE}: not registered yet"; return 1; fi
  if [[ "$(systemctl is-enabled "${SERVICE}" 2>/dev/null)" == "enabled" ]]; then ok "${SERVICE} enabled (starts at boot)"
  else change "${SERVICE}: enable" systemctl enable -q "${SERVICE}"; fi
  if [[ "$(systemctl is-active "${SERVICE}" 2>/dev/null)" == "active" ]]; then
    if [[ "${ENV_CHANGED}" -eq 1 ]]; then change "${SERVICE}: restart (the runner reads .env at start)" systemctl restart "${SERVICE}"
    else ok "${SERVICE} active"; fi
  else change "${SERVICE}: start" systemctl start "${SERVICE}"; fi
}

unit_preflight() {
  user_exists "${RU}" || { would "no account ${RU} yet"; return 1; }
  if [[ "${GOLDENS_READY}" -ne 1 ]]; then pending "tools/ranked-box-preflight.sh: waits for the goldens"; return 0; fi
  local out rc
  out="$(with_runner_env env RUNNER_NAME="${BOX_RUNNER_NAME}" MLXFAST_BOX_NAME="${BOX_RUNNER_NAME}" \
    bash -c 'cd "$1" && tools/ranked-box-preflight.sh' _ "${REPO_DIR}" 2>&1)"; rc=$?
  if [[ "${rc}" -eq 0 ]]; then ok "tools/ranked-box-preflight.sh: all $(grep -c '^ok ' <<<"${out}") checks passed as ${RU} with the runner .env"
  else bad "tools/ranked-box-preflight.sh: $(grep -m1 'REFUSING' <<<"${out}" | cut -c1-300)"; fi
}

# ---------------------------------------------------------------------------
echo "== tools/box/converge.sh ${BOX} $([[ "${CHECK}" -eq 1 ]] && echo --check || echo apply) at $(runner_git "${REPO_DIR}" rev-parse --short=12 HEAD 2>/dev/null || git -C "${REPO_DIR}" rev-parse --short=12 HEAD 2>/dev/null)"
UNITS=(host account gpu-lock dirs rust engine benchd weights image goldens reference
       runner-install runner-register runner-env calibration sandbox service preflight)
for u in "${ONLY_UNITS[@]}"; do
  [[ " ${UNITS[*]} " == *" ${u} "* ]] || die "unknown unit ${u}; units: ${UNITS[*]}"
done
for u in "${UNITS[@]}"; do
  selected "${u}" || continue
  UNIT="${u}"
  "unit_${u//-/_}"
  rc=$?
  # host refuses the rest of an apply while a job runs (or on the wrong box).
  if [[ "${u}" == host && "${rc}" -ne 0 && "${CHECK}" -eq 0 ]]; then break; fi
  [[ "${STOP_AFTER_UNIT}" -eq 1 ]] && break
done
GITHUB_TOKEN_IN=""; R2_ENDPOINT_IN=""; R2_ID_IN=""; R2_SECRET_IN=""

UNIT="summary"
note "ok=${N_OK} converged=${N_CONV} would-change=${N_WOULD} fail=${N_FAIL}"
if [[ "${CHECK}" -eq 1 ]]; then
  if [[ "${N_WOULD}" -eq 0 && "${N_FAIL}" -eq 0 ]]; then note "check: the box matches ${BOX_FILE##*/}"; exit 0; fi
  note "check: the box does NOT match ${BOX_FILE##*/}"; exit 1
fi
[[ "${N_FAIL}" -eq 0 ]]
