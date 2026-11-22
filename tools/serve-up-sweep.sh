#!/usr/bin/env bash
# serve-up-sweep.sh -- remove tools/serve-up.sh resident containers that
# outlived the job that booted them.
#
# WHY. tools/serve-up.sh runs the resident with `docker run --rm ... &` and
# removes the container in its EXIT trap. A cancelled job breaks both: the
# runner signals only the step shell, and at "Complete job" it SIGKILLs the
# orphaned serve-up.sh and its docker client ("Terminate orphan process").
# SIGKILL runs no trap, and a killed docker client does not stop its container:
# dockerd keeps it, on the GPU, with no owner. The sandbox cleanup step does not
# touch docker. Every later ranked run on the box then refuses at the GPU idle
# bookend ("a foreign compute process holds it: <pid>, VLLM::EngineCore").
#
# LABELS (set by tools/serve-up.sh inside a job):
#   mlxfast.serve-up.runner = <owner/repo>/<runner name>
#   mlxfast.serve-up.job    = <owner/repo>/<run id>-<run attempt>
# The repository is part of both: runners of two repositories on one box may
# carry the same runner name.
#
# Usage:
#   tools/serve-up-sweep.sh --job ID
#       Remove the serve-up containers whose job label is ID: the calling job's
#       own containers. Safe without the GPU lock: it never touches a container
#       of another job.
#   tools/serve-up-sweep.sh --locked --runner ID
#       ONLY for a caller that holds the box GPU lock
#       (/tmp/mtplx-gpu-exclusive.lock). Remove the serve-up containers whose
#       runner label is ID (an earlier job of this runner: a runner runs one job
#       at a time), and the unlabelled ones, but those only when no other
#       Runner.Worker is alive on the box. Unlabelled containers come from a
#       tree whose serve-up.sh predates the labels (the pinned reference tree
#       boots the serial-control leg) or from a run outside a job. Several
#       runners can share one box and one user, so an unlabelled container may
#       belong to another runner's live job; with no other Runner.Worker alive
#       there is no other job. A cancelled job's Runner.Worker can outlive its
#       killed orphans by a few seconds, so the sweep waits up to
#       SERVE_UP_SWEEP_WORKER_WAIT seconds (default 120) for other workers to
#       end before it leaves the unlabelled ones. A container labelled with
#       ANOTHER runner is never removed: that runner's own jobs sweep it.
#
# Only names of the exact serve-up.sh shape (<prefix><pid>-<UTC stamp>) are
# touched; any other container on the box is left alone. Containers this mode
# may not remove are listed and left; the GPU idle bookend decides about them.
#
# Exit: 0 every container this mode may remove is gone; 1 one survived, or
# docker or the process lookup failed; 78 usage error.
set -uo pipefail

PREFIX="${SERVE_UP_CONTAINER_PREFIX:-serve-up-}"
RUNNER_LABEL=mlxfast.serve-up.runner
JOB_LABEL=mlxfast.serve-up.job
WORKER_WAIT="${SERVE_UP_SWEEP_WORKER_WAIT:-120}"
WORKER_POLL="${SERVE_UP_SWEEP_WORKER_POLL:-5}"
log() { printf 'serve-up-sweep.sh: %s\n' "$*" >&2; }
usage() { log "usage: $0 --job ID | --locked --runner ID"; exit 78; }

job=""; runner=""; locked=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job) [[ $# -ge 2 && -n "$2" && -z "${job}" ]] || usage; job="$2"; shift 2 ;;
    --runner) [[ $# -ge 2 && -n "$2" && -z "${runner}" ]] || usage; runner="$2"; shift 2 ;;
    --locked) [[ "${locked}" -eq 0 ]] || usage; locked=1; shift ;;
    *) usage ;;
  esac
done
if [[ -n "${job}" ]]; then
  [[ -z "${runner}" && "${locked}" -eq 0 ]] || usage
else
  [[ -n "${runner}" && "${locked}" -eq 1 ]] || usage
fi
[[ "${WORKER_WAIT}" =~ ^[0-9]+$ && "${WORKER_POLL}" =~ ^[1-9][0-9]*$ ]] || usage

command -v docker >/dev/null 2>&1 || { log "docker is not on PATH; cannot verify that no resident container is left"; exit 1; }

# list_all -- "<name>|<job label>|<runner label>" for every serve-up.sh-shaped
# container, running or not. Return 2 when docker cannot list. The separator is
# not whitespace, so an empty label keeps its field (`read` would merge
# repeated tabs); the runner label is last, so a "|" in it stays in it.
list_all() {
  local out
  out="$(docker ps -a --filter "name=^${PREFIX}" \
    --format "{{.Names}}|{{.Label \"${JOB_LABEL}\"}}|{{.Label \"${RUNNER_LABEL}\"}}")" || return 2
  # The name filter is a pattern match; keep only the exact serve-up.sh shape.
  printf '%s\n' "${out}" | grep -E "^${PREFIX}[0-9]+-[0-9]{8}T[0-9]{6}Z[|]" || true
}

# other_worker_alive -- 0 when a Runner.Worker that is not an ancestor of this
# script is alive, 1 when none is, 2 when that cannot be told.
other_worker_alive() {
  local workers rc pid p
  workers="$(pgrep -x Runner.Worker)"; rc=$?
  [[ "${rc}" -le 1 ]] || return 2
  [[ -n "${workers}" ]] || return 1
  local -a ancestors=()
  p=$$
  while [[ -n "${p}" && "${p}" -gt 1 ]]; do
    ancestors+=("${p}")
    p="$(ps -o ppid= -p "${p}" 2>/dev/null | tr -d '[:space:]')"
  done
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ " ${ancestors[*]} " == *" ${pid} "* ]] || return 0
  done <<<"${workers}"
  return 1
}

sweep_unlabelled=0
# targets -- the names this mode removes, from list_all output on stdin.
# Other lines go to fd 3.
targets() {
  local name rlabel jlabel
  while IFS='|' read -r name jlabel rlabel; do
    [[ -n "${name}" ]] || continue
    if [[ -n "${job}" ]]; then
      if [[ "${jlabel}" == "${job}" ]]; then printf '%s\n' "${name}"; continue; fi
    elif [[ "${rlabel}" == "${runner}" ]] || [[ -z "${rlabel}" && -z "${jlabel}" && "${sweep_unlabelled}" -eq 1 ]]; then
      printf '%s\n' "${name}"; continue
    fi
    printf '%s (runner label "%s", job label "%s")\n' "${name}" "${rlabel}" "${jlabel}" >&3
  done
}

if ! all="$(list_all)"; then
  log "docker ps failed; the resident container state is unknown"
  exit 1
fi

if [[ "${locked}" -eq 1 ]] && grep -qE "^[^|]+[|][|]$" <<<"${all}"; then
  # Unlabelled containers exist: may they go?
  waited=0
  while :; do
    other_worker_alive; rc=$?
    if [[ "${rc}" -eq 1 ]]; then sweep_unlabelled=1; break; fi
    if [[ "${rc}" -ne 0 ]]; then
      log "pgrep failed; cannot tell whether another job is alive on this box"
      exit 1
    fi
    if [[ "${waited}" -ge "${WORKER_WAIT}" ]]; then
      log "another Runner.Worker is still alive after ${waited}s; unlabelled ${PREFIX}* containers are left alone"
      break
    fi
    sleep "${WORKER_POLL}"; waited=$((waited + WORKER_POLL))
  done
fi

kept_file="$(mktemp)" || { log "mktemp failed"; exit 1; }
trap 'rm -f "${kept_file}"' EXIT
names="$(targets <<<"${all}" 3>"${kept_file}")"
while IFS= read -r line; do
  [[ -z "${line}" ]] || log "left alone (not this mode's to remove): ${line}"
done < "${kept_file}"

if [[ -z "${names}" ]]; then
  log "no stale ${PREFIX}* container to remove (${job:+job ${job}}${runner:+locked, runner ${runner}})"
  exit 0
fi

rc=0
while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  info="$(docker inspect --format '{{.State.Status}} started={{.State.StartedAt}}' "${name}" 2>/dev/null || echo unknown)"
  log "removing stale resident container ${name} (${info})"
  docker rm -f "${name}" >/dev/null 2>&1 || { log "docker rm -f ${name} failed"; rc=1; }
done <<<"${names}"

if ! all="$(list_all)"; then
  log "docker ps failed after the sweep; cannot verify"
  exit 1
fi
left="$(targets <<<"${all}" 3>/dev/null)"
if [[ -n "${left}" ]]; then
  log "resident containers survived the sweep: $(tr '\n' ' ' <<<"${left}")"
  exit 1
fi
exit "${rc}"
