#!/usr/bin/env bash
# test-resident-log-sandbox.sh -- a sandboxed candidate resident's log is never
# readable by the sandbox user (docs/sandbox-design.md 6a, "Resident logs").
#
# The resident log can carry the hidden prompts. With SERVE_UP_LOG_DIR unset,
# tools/serve-up.sh used to write it under the checkout, whose default ACL
# (tools/sandbox-grant-checkout.sh) grants the sandbox user read on every new
# file: umask 077 does not apply under a default ACL, so a 0600 log still came
# out readable to the next leg's adapter.
#
# The REAL serve-up.sh boots a sandboxed leg through stub docker (the resident
# runs on its fake backend as this user), as in test-serve-up-boot-stop.sh. The
# sandbox user is nobody; every access probe serve-up runs through sudo goes to
# the REAL sudo as nobody, so the kernel decides
# (tools/testlib/real-sandbox-user.sh; SKIP without it). The checkout copy gets
# the checkout ACLs the workflow sets.
#
# Cases:
#   1. SERVE_UP_LOG_DIR unset: the boot refuses, or nobody cannot read the log;
#   2. SERVE_UP_LOG_DIR names a directory whose default ACL grants nobody read:
#      the boot refuses, or nobody cannot read the log;
#   3. a private log dir (runner-only): the boot passes and nobody cannot read
#      the log.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-resident-log-sandbox.sh
# shellcheck source=tools/testlib/synthetic-window.sh
. "${ROOT_DIR}/tools/testlib/synthetic-window.sh"
synthetic_window_env
command -v python3 >/dev/null 2>&1 && command -v jq >/dev/null 2>&1 || { echo "FAIL: python3 and jq are required" >&2; exit 1; }

W="${SBX_WORK}"
SROOT="$(mktemp -d /tmp/rlsbx.XXXXXX)"   # short: the socket path holds 107 bytes
STUB="${W}/stub"
cleanup() {
  local f pid
  for f in "${STUB}"/container.*; do
    [[ -e "${f}" ]] || continue
    pid="$(cat "${f}")"; [[ -n "${pid}" ]] && kill -KILL "${pid}" 2>/dev/null
  done
  rm -rf "${SROOT}"
  sbx_cleanup
}
trap cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

synthetic_window_weights "${W}/weights" || { echo "FAIL: cannot write the synthetic window" >&2; exit 1; }
synthetic_window_meminfo "${W}/meminfo"
TREE="${W}/tree"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TREE}/harness" "${TREE}/.build/vllm-overlay/vllm" "${TREE}/.build/kernels" "${STUB}/bin"
cp "${ROOT_DIR}/tools/serve-up.sh" "${TREE}/tools/"
for f in sandbox-access.sh sandbox-wipe.sh sandbox-kill.sh sandbox-clear-dirs.sh wait-for-memory.sh; do
  [[ -f "${ROOT_DIR}/tools/${f}" ]] && cp "${ROOT_DIR}/tools/${f}" "${TREE}/tools/"
done
cp "${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json" "${TREE}/fixtures/"
cp -R "${ROOT_DIR}/harness/vllm_resident" "${TREE}/harness/"
# The checkout ACLs of a ranked job (tools/sandbox-grant-checkout.sh).
setfacl -R -m "u:${SBX}:rX" "${TREE}" && setfacl -R -d -m "u:${SBX}:rX" "${TREE}"

cat > "${STUB}/bin/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  image) exit 0 ;;
  run)
    shift; name=""; envs=()
    while [[ \$# -gt 0 ]]; do
      case "\$1" in
        --name) name="\$2"; shift 2 ;;
        --env) envs+=("\$2"); shift 2 ;;
        --entrypoint) shift 3; break ;;
        --rm|--init|--read-only) shift ;;
        *) shift 2 ;;
      esac
    done
    printf '%s\n' "\$\$" > "${STUB}/container.\${name}"
    exec env "\${envs[@]}" RESIDENT_BACKEND=fake python3 "\$@" ;;
  stop|rm) exit 0 ;;
esac
exit 1
EOF
# sudo: the access probes (tools/sandbox-access.sh) go to the REAL sudo as
# nobody; the health probe runs as this user (it owns the stub resident's
# socket); the kill is skipped.
cat > "${STUB}/bin/sudo" <<EOF
#!/usr/bin/env bash
[[ "\$1" == -n && "\$2" == -u && "\$4" == -- ]] || exit 97
shift 4
case "\$1" in
  true|sh|find|getfattr) exec /usr/bin/sudo -n -u "${SBX}" -- "\$@" ;;
  kill) exit 0 ;;
esac
exec "\$@"
EOF
cat > "${STUB}/bin/getent" <<EOF
#!/usr/bin/env bash
[[ "\$*" == "group nemoshare" ]] && { echo "nemoshare:x:$(id -g):"; exit 0; }
exec $(command -v getent) "\$@"
EOF
chmod 0755 "${STUB}/bin/"*

boot() { # boot JOB [VAR=VALUE...] -- one sandboxed --boot as job JOB; prints its output
  local job="$1"; shift
  mkdir -p "${SROOT}/${job}" && mkdir -m 2750 "${SROOT}/${job}/sock"
  env PATH="${STUB}/bin:${PATH}" SERVE_UP_LAUNCHER=docker SERVE_UP_WEIGHTS_DIR="${W}/weights" \
    SERVE_UP_HEALTH_TIMEOUT_S=60 SERVE_UP_MEMINFO="${W}/meminfo" \
    SERVE_UP_CONTAINER_USER="${SBX}" SANDBOX_ROOT="${SROOT}" SANDBOX_JOB="${job}" "$@" \
    "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${W}/socket-${job}.txt" 2>&1
}
# readable_logs DIR -- resident logs under DIR the sandbox user can read.
readable_logs() {
  local f n=0
  for f in "$1"/vllm-resident.*.log; do
    [[ -e "${f}" ]] || continue
    sbx cat -- "${f}" >/dev/null 2>&1 && n=$((n + 1))
  done
  echo "${n}"
}

# --- case 1: SERVE_UP_LOG_DIR unset ------------------------------------------
out="$(unset SERVE_UP_LOG_DIR; boot j1)"; rc=$?
if [[ "${rc}" -ne 0 ]]; then
  grep -q 'SERVE_UP_LOG_DIR' <<<"${out}" && ok "case 1: an unset SERVE_UP_LOG_DIR is refused at boot" \
    || fail "case 1: the boot failed for another reason: $(tail -3 <<<"${out}")"
else
  n="$(readable_logs "${TREE}/.build/resident")"
  [[ "${n}" -eq 0 ]] && ok "case 1: the sandbox user cannot read the resident log" \
    || fail "case 1: the sandbox user can read ${n} resident log(s) under the checkout"
fi

# --- case 2: a log dir whose default ACL grants the sandbox user read ---------
mkdir -m 0750 "${W}/acl-logs"; setfacl -m "u:${SBX}:rx" -m "d:u:${SBX}:r" "${W}/acl-logs"
out="$(boot j2 SERVE_UP_LOG_DIR="${W}/acl-logs")"; rc=$?
if [[ "${rc}" -ne 0 ]]; then
  grep -q 'can read' <<<"${out}" && ok "case 2: a resident log the sandbox user can read is refused at boot" \
    || fail "case 2: the boot failed for another reason: $(tail -3 <<<"${out}")"
else
  n="$(readable_logs "${W}/acl-logs")"
  [[ "${n}" -eq 0 ]] && ok "case 2: the sandbox user cannot read the resident log" \
    || fail "case 2: the sandbox user can read ${n} resident log(s)"
fi

# --- case 3: a private log dir ------------------------------------------------
mkdir -m 0750 "${W}/private-logs"
out="$(boot j3 SERVE_UP_LOG_DIR="${W}/private-logs")"; rc=$?
[[ "${rc}" -eq 0 ]] && ok "case 3: a sandboxed leg boots with a private log dir" \
  || fail "case 3: the boot failed (rc=${rc}): $(tail -5 <<<"${out}")"
n="$(readable_logs "${W}/private-logs")"
[[ "${n}" -eq 0 ]] && ok "case 3: the sandbox user cannot read the resident log" \
  || fail "case 3: the sandbox user can read ${n} resident log(s)"

if [[ "${failures}" -gt 0 ]]; then echo "FAIL: ${failures} failure(s)" >&2; exit 1; fi
echo "PASS: test-resident-log-sandbox.sh"
