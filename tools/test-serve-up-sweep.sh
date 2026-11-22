#!/usr/bin/env bash
# test-serve-up-sweep.sh -- tools/serve-up-sweep.sh against a fake docker and
# a fake pgrep.
#
# Hermetic: no daemon, no GPU, no runner. The fake docker keeps one file per
# container (its labels, one k=v per line) and answers `ps -a --filter name=...
# --format`, `inspect --format` and `rm -f`, the only calls the sweep makes. The
# fake pgrep prints FAKE_PGREP_OUT and exits FAKE_PGREP_RC; while the file
# FAKE_PGREP_COUNT holds a number above 0, it lowers it by one and prints
# FAKE_PGREP_OUT, then prints nothing (a worker that ends during the wait).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP="${ROOT_DIR}/tools/serve-up-sweep.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin" "${WORK}/state"
cat > "${WORK}/bin/docker" <<'FAKE'
#!/usr/bin/env bash
state="${FAKE_DOCKER_STATE:?}"
case "$1" in
  ps)
    shift; name_re=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --filter) case "$2" in name=*) name_re="${2#name=}" ;; *) exit 9 ;; esac; shift 2 ;;
        *) shift ;;
      esac
    done
    for f in "${state}"/*; do
      [[ -e "${f}" ]] || continue
      n="$(basename "${f}")"
      [[ -z "${name_re}" ]] || [[ "${n}" =~ ${name_re} ]] || continue
      r="$(sed -n 's/^mlxfast\.serve-up\.runner=//p' "${f}")"
      j="$(sed -n 's/^mlxfast\.serve-up\.job=//p' "${f}")"
      printf '%s|%s|%s\n' "${n}" "${j}" "${r}"
    done ;;
  inspect) echo "running" ;;
  rm)
    [[ -z "${FAKE_DOCKER_RM_FAILS:-}" ]] || exit 1
    for n in "$@"; do [[ "${n}" == rm || "${n}" == -f ]] || rm -f "${state}/${n}"; done ;;
  *) exit 1 ;;
esac
FAKE
cat > "${WORK}/bin/pgrep" <<'FAKE'
#!/usr/bin/env bash
if [[ -n "${FAKE_PGREP_COUNT:-}" ]]; then
  n="$(cat "${FAKE_PGREP_COUNT}")"
  [[ "${n}" -gt 0 ]] || exit 1
  echo $((n - 1)) > "${FAKE_PGREP_COUNT}"
fi
[[ -z "${FAKE_PGREP_OUT:-}" ]] || printf '%s\n' "${FAKE_PGREP_OUT}"
exit "${FAKE_PGREP_RC:-1}"
FAKE
chmod +x "${WORK}/bin/docker" "${WORK}/bin/pgrep"
export PATH="${WORK}/bin:${PATH}" FAKE_DOCKER_STATE="${WORK}/state" SERVE_UP_SWEEP_WORKER_WAIT=0

mk() { printf '%s\n' "${@:2}" > "${WORK}/state/$1"; }
has() { [[ -e "${WORK}/state/$1" ]]; }
left() { find "${WORK}/state" -mindepth 1 -exec basename {} \; | sort | tr '\n' ' '; }
A=org/engine/runner-a          # this runner
B=org/engine-two/runner-a      # same runner name, another repository
OWN_JOB=serve-up-10-20261122T000000Z
OWN_OLD=serve-up-11-20261122T000000Z
OTHER=serve-up-22-20261122T000000Z
BARE=serve-up-33-20261122T000000Z
JOBONLY=serve-up-44-20261122T000000Z
box() {
  rm -f "${WORK}/state/"*
  mk "${OWN_JOB}" "mlxfast.serve-up.runner=${A}" mlxfast.serve-up.job=org/engine/100-1
  mk "${OWN_OLD}" "mlxfast.serve-up.runner=${A}" mlxfast.serve-up.job=org/engine/99-1
  mk "${OTHER}" "mlxfast.serve-up.runner=${B}" mlxfast.serve-up.job=org/engine-two/100-1
  mk "${BARE}"                                   # pre-label, or booted outside a job
  mk "${JOBONLY}" mlxfast.serve-up.job=org/engine/100-1   # empty runner label
  mk my-other-container "mlxfast.serve-up.runner=${A}"
  mk serve-up-notastamp
}
keeps_foreign() { has "${OTHER}" && has my-other-container && has serve-up-notastamp; }

# 1. --job removes only that job's containers, even when another repository's
#    job has the same run id and runner name.
box
if "${SWEEP}" --job org/engine/100-1 && ! has "${OWN_JOB}" && ! has "${JOBONLY}" && has "${OWN_OLD}" && has "${BARE}" \
    && keeps_foreign; then
  ok "--job removes only its own job's containers"
else
  fail "--job removed the wrong set; left: $(left)"
fi

# 2. --locked with no Runner.Worker alive: own runner's and unlabelled, never
#    another repository's runner of the same name.
box
if FAKE_PGREP_RC=1 "${SWEEP}" --locked --runner "${A}" && ! has "${OWN_JOB}" && ! has "${OWN_OLD}" \
    && ! has "${BARE}" && has "${JOBONLY}" && keeps_foreign; then
  ok "--locked with no other job removes own and unlabelled, never another runner's"
else
  fail "--locked (no worker) removed the wrong set; left: $(left)"
fi

# 3. --locked when the only Runner.Worker is this sweep's own ancestor.
box
if FAKE_PGREP_OUT="$$" FAKE_PGREP_RC=0 "${SWEEP}" --locked --runner "${A}" && ! has "${BARE}" \
    && ! has "${OWN_OLD}" && keeps_foreign; then
  ok "--locked does not count its own Runner.Worker as another job"
else
  fail "--locked (own worker) removed the wrong set; left: $(left)"
fi

# 4. --locked while another Runner.Worker stays alive leaves unlabelled ones.
box
other_pid=999999
while kill -0 "${other_pid}" 2>/dev/null; do other_pid=$((other_pid - 1)); done
if FAKE_PGREP_OUT="$(printf '%s\n%s' "$$" "${other_pid}")" FAKE_PGREP_RC=0 "${SWEEP}" --locked --runner "${A}" \
    && ! has "${OWN_OLD}" && has "${BARE}" && keeps_foreign; then
  ok "--locked leaves unlabelled containers while another job is alive"
else
  fail "--locked (other worker) removed the wrong set; left: $(left)"
fi

# 5. --locked waits for another Runner.Worker that ends (a cancelled job's
#    worker finishing), then removes the unlabelled ones.
box
echo 2 > "${WORK}/pgrep-count"
if FAKE_PGREP_COUNT="${WORK}/pgrep-count" FAKE_PGREP_OUT="${other_pid}" FAKE_PGREP_RC=0 \
    SERVE_UP_SWEEP_WORKER_WAIT=5 SERVE_UP_SWEEP_WORKER_POLL=1 "${SWEEP}" --locked --runner "${A}" \
    && ! has "${BARE}" && keeps_foreign; then
  ok "--locked waits for an ending Runner.Worker"
else
  fail "--locked (ending worker) removed the wrong set; left: $(left)"
fi

# 6. A pgrep error fails closed: exit 1 and nothing unlabelled removed.
box
if FAKE_PGREP_RC=3 "${SWEEP}" --locked --runner "${A}" >/dev/null 2>&1; then
  fail "a pgrep error passed"
elif has "${BARE}" && keeps_foreign; then
  ok "a pgrep error fails closed"
else
  fail "a pgrep error removed the wrong set; left: $(left)"
fi

# 7. Nothing to sweep is success.
rm -f "${WORK}/state/"*
if "${SWEEP}" --locked --runner "${A}" && "${SWEEP}" --job x/y/1-1; then ok "an empty box is exit 0"; else fail "an empty box did not exit 0"; fi

# 8. Usage errors.
for args in "" "--locked" "--runner a" "--runner" "--job" "--job a --locked" "--job a --runner b" \
    "--locked --runner a --runner b" "--locked --locked --runner a" "--job a extra"; do
  # shellcheck disable=SC2086 # word splitting of the case is the point
  "${SWEEP}" ${args} >/dev/null 2>&1; rc=$?
  [[ "${rc}" -eq 78 ]] || fail "'${args}' exited ${rc}, not 78"
done
"${SWEEP}" --job "" >/dev/null 2>&1; rc=$?
if [[ "${rc}" -eq 78 ]]; then ok "usage errors exit 78"; else fail "an empty job id exited ${rc}, not 78"; fi

# 9. A container that survives docker rm fails the sweep.
box
if FAKE_DOCKER_RM_FAILS=1 "${SWEEP}" --job org/engine/100-1 >/dev/null 2>&1; then
  fail "a surviving container passed"
else
  ok "a surviving container fails the sweep"
fi

# 10. A docker that cannot list fails closed.
cat > "${WORK}/bin/docker" <<'FAKE'
#!/usr/bin/env bash
exit 1
FAKE
if "${SWEEP}" --locked --runner "${A}" >/dev/null 2>&1; then fail "a failing docker ps passed"; else ok "a failing docker ps fails closed"; fi

if [[ "${failures}" -ne 0 ]]; then
  echo "test-serve-up-sweep.sh: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-serve-up-sweep.sh: all passed"
