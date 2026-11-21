#!/usr/bin/env bash
#
# run-engine-sandboxed.sh -- the engine benchd spawns for the CANDIDATE leg when
# participant code runs as the sandbox user (docs/sandbox-design.md section 5
# item 4 and section 6a), through the shim tools/stage-cuda-engine.sh writes at
# the staged engine path. TRUSTED: not in editablePaths.
#
# It runs the real adapter as ${SANDBOX_USER} and nothing else changes:
#
#   * stdin and stdout are benchd's pipes, handed straight to the adapter. The
#     Engine Protocol is stdio, so nothing here reads, buffers or parses a
#     message; sudo adds a few milliseconds at spawn, outside every timed window.
#   * the adapter's STDERR goes to a runner-private 0600 file, never to the job
#     log. The engine sees the hidden prompts, and the job log must not become a
#     way to carry them out.
#   * NO UID-WIDE KILL HERE. The candidate's resident runs as the same user and
#     must outlive this adapter: the pinned benchd (3101418f95eb,
#     crates/benchd/src/official.rs) opens ONE adapter per leg for the warm-up,
#     timed prefill and timed decode (run_timed_window, :567) and keeps it for
#     correctness, but on the last pair it spawns a SECOND adapter on the same
#     resident for the pinned correctness golden after the first one exits
#     (finish_official, :2577-2583, through the spawn_timed fallback at
#     :1737-1743). A kill here would end that resident. Every sandbox process
#     is killed, and none may survive, at the confirmed LEG TEARDOWN instead:
#     tools/serve-up.sh --stop (tools/sandbox-kill.sh, then the per-leg clear
#     and tools/sandbox-wipe.sh), which benchd runs before the next leg boots
#     (candidate_leg.finish(), :1646-1648; LegServe's Drop, legserve.rs:214)
#     and seals as LEG-TEARDOWN-FAILED when it fails.
#   * the exit status is the adapter's.
#   * SIGTERM and SIGINT are forwarded to the adapter.
#   * the adapter runs under prlimit --fsize (64 MiB), --nproc and --as: no
#     file it writes, its stderr file included, grows past the fsize cap; a
#     write beyond it kills the writer with SIGXFSZ. Pipes are not files, so
#     stdio to benchd is unaffected. nproc/as bound fork and address-space bombs
#     (defaults 4096 and 16 GiB; override with ADAPTER_NPROC_LIMIT /
#     ADAPTER_AS_LIMIT bytes).
#   * PER-LEG STORAGE. benchd spawns this wrapper once per candidate leg, before
#     the leg's warm-up and timed phases (and once more on the same leg for the
#     pinned correctness golden, above). The adapter's HOME and TMPDIR is a
#     FRESH directory made here, $SANDBOX_ROOT/$SANDBOX_JOB/adapter-home/
#     leg.<tag>.XXXXXX, where <tag> is this leg's resident directory name
#     (sock/<tag>/, the run tag tools/serve-up.sh --boot makes with an
#     exclusive mkdir, one per leg); leg.XXXXXX when no such directory is
#     named. It is 2770; adapter-home itself is 2750, made by
#     tools/engine/build.sh, so the sandbox user cannot plant, rename or remove
#     a name in it: every name in adapter-home was made by this wrapper.
#     After the adapter, that directory is removed and the job's build scratch
#     dirs (cargo-home, target, kernels-out: nothing reads them at run time)
#     are emptied (tools/sandbox-clear-dirs.sh); a failed clear fails the leg.
#     tools/serve-up.sh --stop removes the leg's resident directory under sock/,
#     and tools/sandbox-wipe.sh clears the shared temp dirs.
#   * A KILLED ADAPTER OF THIS LEG. benchd ends a session by SIGKILL on this
#     wrapper (bench-runner transport.rs Drop), so the clear above never runs
#     for the first adapter when benchd drops it for the pinned correctness
#     worker. Before the check below, a leg.<tag>.XXXXXX directory of THIS
#     leg's own tag is that killed adapter's home: it is emptied with the same
#     tools/sandbox-clear-dirs.sh and removed, one line is logged
#     (LEG-ISOLATION-CLEARED <n> entries left by a killed leg), and the check
#     below then runs as before. That is no leak across legs: the earlier
#     adapter ran on the same resident, for the same candidate, in the same
#     leg, and its data is wiped before this adapter starts. A home of any
#     other tag, or of none, is not cleared: it refuses as before.
#   * LEG ISOLATION CHECK. Before the adapter starts, any entry in adapter-home,
#     cargo-home, target or kernels-out, or any sock/ entry other than this
#     leg's resident directory (the one benchd's socket names), is a file an
#     earlier leg may have left: the leg is REFUSED (LEG-ISOLATION-REFUSED,
#     exit 1) and the adapter never runs. benchd then seals the run as a
#     failed score. Only directory names and counts are printed.
#
# Env (the staged shim exports every one; benchd spawns it with a cleared env):
#   SANDBOX_USER        the sandbox user (nemosub)
#   SANDBOX_ENGINE_BIN  the real adapter, staged where the sandbox user can run
#                       it but not change it
#   SANDBOX_ROOT, SANDBOX_JOB  the per-job sandbox directory
#   SANDBOX_ENGINE_LOG_DIR  the job's private directory (RUNNER_TEMP at staging);
#                       the stderr file goes here. No fallback: benchd clears the
#                       environment, so the staged shim sets all of these.
# No -e: every step's status is checked explicitly (the adapter's comes from
# `wait`, and kill -1's is deliberately ignored; pgrep decides).
set -uo pipefail

die() { echo "run-engine-sandboxed.sh: $*" >&2; exit 1; }
# Largest file the adapter may write (bytes), its stderr file included.
ADAPTER_FSIZE=67108864
ADAPTER_NPROC_LIMIT="${ADAPTER_NPROC_LIMIT:-4096}"
ADAPTER_AS_LIMIT="${ADAPTER_AS_LIMIT:-17179869184}"

[[ -n "${SANDBOX_USER:-}" ]] || die "SANDBOX_USER is unset"
[[ -x "${SANDBOX_ENGINE_BIN:-}" ]] || die "SANDBOX_ENGINE_BIN is not an executable adapter: '${SANDBOX_ENGINE_BIN:-}'"
[[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || die "SANDBOX_ROOT or SANDBOX_JOB is unset"
job_dir="${SANDBOX_ROOT}/${SANDBOX_JOB}"
adapter_homes="${job_dir}/adapter-home"
[[ -d "${adapter_homes}" && ! -L "${adapter_homes}" ]] || die "the job's adapter-home ${adapter_homes} is missing (tools/engine/build.sh creates it)"
# The job dirs the sandbox user can write that hold no state a leg needs.
SCRATCH_DIRS=(cargo-home target kernels-out)
CLEAR_DIRS="$(dirname -- "${BASH_SOURCE[0]}")/sandbox-clear-dirs.sh"

log_dir="${SANDBOX_ENGINE_LOG_DIR:-}"
[[ -n "${log_dir}" && -d "${log_dir}" ]] || die "SANDBOX_ENGINE_LOG_DIR is not a directory: '${log_dir}'"

# This leg's resident directory: serve-up.sh's sock/<run tag>/vllm-resident.sock,
# named by the socket benchd injects. serve-up.sh resolves the socket dir with
# pwd -P, so the parents are compared physically. Nothing else may be in sock/.
# LEG_TAG names this leg's adapter homes (header): set only when every socket
# names the same existing, non-symlink directory there with a plain name.
own=()
tags=()
sock_real="$(cd -- "${job_dir}/sock" 2>/dev/null && pwd -P)"
for s in "${DS4_RESIDENT_SOCKET:-}" "${BENCH_WORKER_RESIDENT_SOCKET:-}"; do
  [[ -n "${sock_real}" && "$(basename -- "${s}")" == vllm-resident.sock ]] || continue
  leg_dir="$(dirname -- "${s}")"
  [[ "$(cd -- "$(dirname -- "${leg_dir}")" 2>/dev/null && pwd -P)" == "${sock_real}" ]] || continue
  own+=(! -name "$(basename -- "${leg_dir}")")
  tags+=("$(basename -- "${leg_dir}")")
done
LEG_TAG=""
for tag in "${tags[@]}"; do
  if [[ "${tag}" =~ ^[A-Za-z0-9_-]+$ && -d "${sock_real}/${tag}" && ! -L "${sock_real}/${tag}" \
        && ( -z "${LEG_TAG}" || "${LEG_TAG}" == "${tag}" ) ]]; then
    LEG_TAG="${tag}"
  else
    LEG_TAG=""; break
  fi
done

# A KILLED ADAPTER OF THIS LEG (header). Only directories whose whole name is
# this leg's home pattern are cleared; every other entry stays for the check.
if [[ -n "${LEG_TAG}" ]]; then
  stale=()
  while IFS= read -r -d '' p; do
    [[ "$(basename -- "${p}")" =~ ^leg\.${LEG_TAG}\.[A-Za-z0-9]{6}$ ]] && stale+=("${p}")
  done < <(find -P "${adapter_homes}" -mindepth 1 -maxdepth 1 -type d -name "leg.${LEG_TAG}.*" -print0 2>/dev/null)
  if (( ${#stale[@]} > 0 )); then
    n="$(find -P "${stale[@]}" -mindepth 1 -printf . 2>/dev/null | wc -c)"
    if "${CLEAR_DIRS}" "${SANDBOX_USER}" "${stale[@]}" && rmdir -- "${stale[@]}"; then
      echo "run-engine-sandboxed.sh: LEG-ISOLATION-CLEARED ${n} entries left by a killed leg (${#stale[@]} home(s) of an earlier adapter of this same leg, wiped before this adapter runs)" >&2
    else
      echo "::error::run-engine-sandboxed.sh: a killed adapter's home of this leg could not be cleared (above)" >&2
    fi
  fi
fi

# THE LEG ISOLATION CHECK (header). Entries are counted, never named.
refuse_leg() { echo "::error::run-engine-sandboxed.sh: LEG-ISOLATION-REFUSED: $1 holds $2 entr$( (( $2 == 1 )) && echo y || echo ies) an earlier leg may have left; the adapter does not run" >&2; exit 1; }
entries() { find -P "$1" -mindepth 1 -maxdepth 1 "${@:2}" -printf . 2>/dev/null | wc -c; }
for d in adapter-home "${SCRATCH_DIRS[@]}"; do
  [[ -d "${job_dir}/${d}" ]] || continue
  n="$(entries "${job_dir}/${d}")" || die "cannot list ${d} for the leg isolation check"
  (( n == 0 )) || refuse_leg "${d}" "${n}"
done
if [[ -n "${sock_real}" ]]; then
  n="$(entries "${job_dir}/sock" "${own[@]}")" || die "cannot list sock for the leg isolation check"
  (( n == 0 )) || refuse_leg sock "${n}"
fi

# A fresh home for this leg only: mktemp -d makes it exclusively (0700, the
# caller's), then the sandbox user gets write through the share group. Its name
# carries this leg's tag, so a later adapter of the same leg can recognise it.
adapter_home="$(mktemp -d "${adapter_homes}/leg.${LEG_TAG:+${LEG_TAG}.}XXXXXX")" || die "cannot create this leg's home under ${adapter_homes}"
chmod 2770 "${adapter_home}" || die "cannot open this leg's home ${adapter_home} to ${SANDBOX_USER}"
# mktemp creates the file 0600 and exclusively, so a name or symlink planted in a
# shared directory cannot redirect it.
log="$(mktemp "${log_dir}/engine-stderr.XXXXXX")" || die "cannot create the adapter's stderr file in ${log_dir}"
echo "run-engine-sandboxed.sh: the adapter runs as ${SANDBOX_USER}; its stderr goes to ${log}, not the job log" >&2

# In the background so this shell can forward signals while it waits. stdin is
# redirected explicitly: a non-interactive `&` would otherwise give it /dev/null.
# HOME/TMPDIR are set inside the command: glibc strips TMPDIR from any setuid
# program's environment (sudo included), so --preserve-env cannot carry it.
sudo -n -u "${SANDBOX_USER}" --preserve-env=DS4_RESIDENT_SOCKET,BENCH_WORKER_RESIDENT_SOCKET \
  -- prlimit --fsize="${ADAPTER_FSIZE}" --nproc="${ADAPTER_NPROC_LIMIT}" --as="${ADAPTER_AS_LIMIT}" \
  -- env HOME="${adapter_home}" TMPDIR="${adapter_home}" "${SANDBOX_ENGINE_BIN}" "$@" <&0 2>>"${log}" &
child=$!
forwarded=0
trap 'forwarded=1; kill -TERM "${child}" 2>/dev/null' TERM
trap 'forwarded=1; kill -INT "${child}" 2>/dev/null' INT

# A trapped signal returns `wait` early; wait again for the adapter's own status.
wait "${child}"
status=$?
while [[ "${forwarded}" == 1 ]]; do
  forwarded=0
  wait "${child}"
  status=$?
done

# PER-LEG STORAGE (header): nothing this leg wrote reaches the next one. The
# UID-wide kill is the leg teardown's (header); serve-up.sh --stop clears these
# directories again after it, so a process the adapter left cannot refill them
# for the next leg.
scratch=()
for d in "${SCRATCH_DIRS[@]}"; do scratch+=("${job_dir}/${d}"); done
if ! "${CLEAR_DIRS}" "${SANDBOX_USER}" "${adapter_home}" "${scratch[@]}" || ! rmdir -- "${adapter_home}"; then
  echo "::error::run-engine-sandboxed.sh: this leg's storage could not be cleared (above); the next leg refuses" >&2
  status=1
fi
exit "${status}"
