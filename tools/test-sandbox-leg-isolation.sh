#!/usr/bin/env bash
# test-sandbox-leg-isolation.sh -- no file one candidate leg writes is readable
# by the next candidate leg (docs/sandbox-design.md section 6a, "per-leg
# storage"; docs/participant-contract.md 5.2.2).
#
# Every candidate leg spawns tools/run-engine-sandboxed.sh once, before its
# warm-up and timed phases. The properties it must keep:
#
#   * each leg's HOME and TMPDIR is a FRESH directory under the job's
#     adapter-home, never the one an earlier leg had;
#   * after the leg, that directory is gone and the job's build scratch dirs
#     (cargo-home, target, kernels-out) are empty again, so a file the adapter
#     wrote into any of them does not reach the next leg;
#   * before the adapter starts, any entry left in adapter-home, cargo-home,
#     target or kernels-out, or any sock/ entry other than this leg's resident
#     directory, REFUSES the leg (LEG-ISOLATION-REFUSED): the adapter never
#     runs, the wrapper exits non-zero, and only directory names and counts are
#     printed, never an entry name.
#   * when benchd SIGKILLs a leg's first wrapper and spawns a second adapter on
#     the same leg (the pinned correctness worker), the killed adapter's home
#     (adapter-home/leg.<this leg's tag>.XXXXXX) is wiped and removed before the
#     second adapter starts (LEG-ISOLATION-CLEARED); that adapter runs on an
#     empty HOME and reads nothing the first wrote. A home of another leg's
#     tag, any other name, a symlink, or a home a live process keeps writing
#     into still REFUSES the leg.
#
# Hermetic: sudo and pgrep are stubs on PATH (as in
# tools/test-run-engine-sandboxed.sh); the stub sudo runs the command as this
# user, so the "sandbox user" is this user.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="${ROOT_DIR}/tools/run-engine-sandboxed.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

JOB="${WORK}/sbx/9-1"
mkdir -p "${WORK}/bin" "${WORK}/logs" "${JOB}"
# The layout tools/engine/build.sh makes (modes do not matter to the stubs).
for d in vendor sock cargo-home target kernels-out adapter-home; do mkdir -p "${JOB}/${d}"; done

cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_SUDO_LOG}"
keep=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) shift ;;
    -u) shift 2 ;;
    --preserve-env=*) keep="${1#*=}"; shift ;;
    --) shift; break ;;
    *) echo "stub sudo: unexpected option $1" >&2; exit 97 ;;
  esac
done
[[ "$1" == kill ]] && exit 0
envs=("PATH=${PATH}")
for v in ${keep//,/ }; do
  [[ -n "${!v:-}" ]] && envs+=("${v}=${!v}")
done
exec env -i "${envs[@]}" "$@"
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "${WORK}/bin/pgrep"

# The fake candidate adapter. --plant writes "prompt" text into its HOME and
# into the job's target/ (it finds the job directory from its HOME, as a real
# adapter could). --probe reports every planted file it can read anywhere in
# the job directory, and its own HOME.
cat > "${WORK}/adapter" <<'EOF'
#!/usr/bin/env bash
job="${HOME%%/adapter-home*}"
case "$1" in
  --plant)
    echo "leg 1 prompt" > "${HOME}/stash"
    echo "leg 1 prompt" > "${job}/target/stash"
    echo "home=${HOME}"
    ;;
  --probe)
    echo "home=${HOME}"
    echo "homecount=$(find "${HOME}" -mindepth 1 | wc -l)"
    find "${job}" -name stash -type f -readable 2>/dev/null | sed 's/^/found=/'
    ;;
  --plant-and-hang)
    # A first adapter of a leg that benchd drops: it plants, then waits on
    # benchd's stdin until EOF. State goes next to the job dir, outside it.
    echo "leg 1 prompt" > "${HOME}/stash"
    echo planted > "${job%/*}/hang.state"
    while read -r _; do :; done
    echo exited > "${job%/*}/hang.state"
    ;;
esac
exit 0
EOF
chmod +x "${WORK}/bin/sudo" "${WORK}/bin/pgrep" "${WORK}/adapter"

# leg ARGS... -- one candidate leg through the wrapper; stdout to $out, stderr to
# $err, status to $status.
leg() {
  out="$(env PATH="${WORK}/bin:${PATH}" \
      SANDBOX_USER=nemosub SANDBOX_ENGINE_BIN="${WORK}/adapter" \
      SANDBOX_ENGINE_LOG_DIR="${WORK}/logs" SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=9-1 \
      DS4_RESIDENT_SOCKET="${RESIDENT_SOCKET:-}" BENCH_WORKER_RESIDENT_SOCKET="${RESIDENT_SOCKET:-}" \
      STUB_SUDO_LOG="${WORK}/sudo.log" \
      "${WRAPPER}" "$@" </dev/null 2>"${WORK}/err")"
  status=$?
  err="$(cat "${WORK}/err")"
}
adapter_ran() { grep -q -- "prlimit .* ${WORK}/adapter" "${WORK}/sudo.log"; }
empty_dir() { [[ -z "$(find "$1" -mindepth 1 -print -quit)" ]]; }

# --- case 1: what leg 1 writes, leg 2 cannot read ---------------------------
mkdir -p "${JOB}/sock/1-20261121T000000Z"
RESIDENT_SOCKET="${JOB}/sock/1-20261121T000000Z/vllm-resident.sock"
: > "${WORK}/sudo.log"
leg --plant
home1="$(sed -n 's/^home=//p' <<<"${out}")"
[[ "${status}" -eq 0 && -n "${home1}" ]] && adapter_ran \
  && ok "case 1: leg 1 runs and plants a file in its HOME and in target/" \
  || fail "case 1: leg 1 status ${status}, out '${out}', err '${err}'"
# serve-up.sh --stop removes the leg's resident directory; here, by hand.
rm -rf "${JOB}/sock/1-20261121T000000Z"
mkdir -p "${JOB}/sock/2-20261121T000100Z"
RESIDENT_SOCKET="${JOB}/sock/2-20261121T000100Z/vllm-resident.sock"
: > "${WORK}/sudo.log"
leg --probe
home2="$(sed -n 's/^home=//p' <<<"${out}")"
found="$(grep -c '^found=' <<<"${out}")"
[[ "${status}" -eq 0 && "${found}" -eq 0 ]] \
  && ok "case 1: leg 2 can read no file leg 1 wrote" \
  || fail "case 1: leg 2 status ${status}; it read ${found} file(s) leg 1 wrote: $(grep '^found=' <<<"${out}" | tr '\n' ' ')"
[[ -n "${home2}" && "${home2}" != "${home1}" && "${home2}" == "${JOB}/adapter-home/"* ]] \
  && ok "case 1: each leg gets a fresh HOME under adapter-home" \
  || fail "case 1: leg 1 HOME '${home1}', leg 2 HOME '${home2}'"
[[ ! -e "${home1}" ]] && empty_dir "${JOB}/adapter-home" && empty_dir "${JOB}/target" \
  && ok "case 1: after a leg its HOME is gone and target/ is empty" \
  || fail "case 1: after the legs: adapter-home '$(ls -A "${JOB}/adapter-home" | tr '\n' ' ')', target '$(ls -A "${JOB}/target" | tr '\n' ' ')'"
rm -rf "${JOB}/sock/2-20261121T000100Z"

# --- case 2: a leftover in a scratch dir refuses the leg ---------------------
for d in adapter-home cargo-home target kernels-out; do
  mkdir -p "${JOB}/sock/3-20261121T000200Z"
  RESIDENT_SOCKET="${JOB}/sock/3-20261121T000200Z/vllm-resident.sock"
  mkdir -p "${JOB}/${d}/zz-prompt-named-dir"
  echo "old prompt" > "${JOB}/${d}/zz-prompt-named-dir/zz-prompt-named-file"
  : > "${WORK}/sudo.log"
  leg --probe
  if [[ "${status}" -ne 0 ]] && ! adapter_ran && grep -q "LEG-ISOLATION-REFUSED: ${d} holds 1 entr" <<<"${err}" \
     && ! grep -q 'zz-prompt-named' <<<"${err}"; then
    ok "case 2: a leftover in ${d}/ refuses the leg before the adapter runs, by count only"
  else
    fail "case 2 (${d}): status ${status}, adapter ran: $(adapter_ran && echo yes || echo no), err '${err}'"
  fi
  rm -rf "${JOB:?}/${d}/zz-prompt-named-dir" "${JOB}/sock/3-20261121T000200Z"
done

# --- case 3: an earlier leg's resident directory refuses the leg -------------
mkdir -p "${JOB}/sock/4-20261121T000300Z/run/home" "${JOB}/sock/5-20261121T000400Z"
echo "old prompt" > "${JOB}/sock/4-20261121T000300Z/run/home/stash"
RESIDENT_SOCKET="${JOB}/sock/5-20261121T000400Z/vllm-resident.sock"
: > "${WORK}/sudo.log"
leg --probe
[[ "${status}" -ne 0 ]] && ! adapter_ran && grep -q "LEG-ISOLATION-REFUSED: sock holds 1 entr" <<<"${err}" \
  && ok "case 3: another leg's sock/<tag>/ refuses the leg" \
  || fail "case 3: status ${status}, adapter ran: $(adapter_ran && echo yes || echo no), err '${err}'"
rm -rf "${JOB}/sock/4-20261121T000300Z"
: > "${WORK}/sudo.log"
leg --probe
[[ "${status}" -eq 0 ]] && adapter_ran \
  && ok "case 3: this leg's own resident directory alone in sock/ is allowed" \
  || fail "case 3: status ${status}, err '${err}'"
rm -rf "${JOB}/sock/5-20261121T000400Z"

# wait_for CMD -- up to 10 s for CMD to succeed.
wait_for() { local _i; for _i in $(seq 100); do bash -c "$1" && return 0; sleep 0.1; done; return 1; }
STATE="${WORK}/sbx/hang.state"

# --- case 4: benchd SIGKILLs the first adapter's wrapper; a second adapter of
# the SAME leg runs, on a cleared, empty HOME (official.rs:2577-2583 drops the
# correctness session and spawns the pinned correctness worker; the Drop is a
# SIGKILL on this wrapper, bench-runner transport.rs:1177-1181) --------------
TAG4="6-20261121T000500Z"
mkdir -p "${JOB}/sock/${TAG4}"
RESIDENT_SOCKET="${JOB}/sock/${TAG4}/vllm-resident.sock"
mkfifo "${WORK}/stdin.fifo"
exec 8<>"${WORK}/stdin.fifo"   # benchd's write end of the adapter's stdin
rm -f "${STATE}"
: > "${WORK}/sudo.log"
env PATH="${WORK}/bin:${PATH}" \
    SANDBOX_USER=nemosub SANDBOX_ENGINE_BIN="${WORK}/adapter" \
    SANDBOX_ENGINE_LOG_DIR="${WORK}/logs" SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=9-1 \
    DS4_RESIDENT_SOCKET="${RESIDENT_SOCKET}" BENCH_WORKER_RESIDENT_SOCKET="${RESIDENT_SOCKET}" \
    STUB_SUDO_LOG="${WORK}/sudo.log" \
    "${WRAPPER}" --plant-and-hang <"${WORK}/stdin.fifo" >/dev/null 2>"${WORK}/err1" 8>&- &
wpid=$!
if wait_for "grep -qsx planted '${STATE}'"; then
  kill -KILL "${wpid}"; wait "${wpid}" 2>/dev/null
  exec 8>&-                     # benchd closes the pipe; the orphaned adapter exits
  wait_for "grep -qsx exited '${STATE}'" || fail "case 4: the first adapter did not exit after stdin closed"
  left="$(find "${JOB}/adapter-home" -mindepth 1 -maxdepth 1 | wc -l)"
  [[ "${left}" -eq 1 ]] \
    && ok "case 4: the killed wrapper left its leg home behind (the fault)" \
    || fail "case 4: after the kill adapter-home holds ${left} entries, expected 1"
  leg --probe
  home4="$(sed -n 's/^home=//p' <<<"${out}")"
  homecount="$(sed -n 's/^homecount=//p' <<<"${out}")"
  found="$(grep -c '^found=' <<<"${out}")"
  if [[ "${status}" -eq 0 ]] && grep -q 'LEG-ISOLATION-CLEARED 1 entries left by a killed leg' <<<"${err}" \
     && [[ "${homecount}" == 0 && "${found}" -eq 0 && "${home4}" == "${JOB}/adapter-home/leg.${TAG4}."* ]]; then
    ok "case 4: the second adapter of the leg runs (CLEARED) on an empty HOME and reads nothing the first wrote"
  else
    fail "case 4: second adapter status ${status}, home entries '${homecount}', read ${found} file(s), home '${home4}', err '${err}'"
  fi
  empty_dir "${JOB}/adapter-home" \
    && ok "case 4: after the second adapter adapter-home is empty" \
    || fail "case 4: adapter-home holds '$(ls -A "${JOB}/adapter-home" | tr '\n' ' ')'"
else
  kill -KILL "${wpid}" 2>/dev/null; exec 8>&-
  fail "case 4: the first adapter never planted; err '$(cat "${WORK}/err1")'"
fi
rm -rf "${JOB:?}/adapter-home/"* "${JOB}/sock/${TAG4}"

# --- case 5: a killed adapter's home is cleared ONLY by its own leg ----------
# Another leg's tag, a malformed name of this leg's tag, and a symlink of this
# leg's pattern all REFUSE: none is this leg's own earlier home.
TAG5="8-20261121T000700Z"
mkdir -p "${JOB}/sock/${TAG5}"
RESIDENT_SOCKET="${JOB}/sock/${TAG5}/vllm-resident.sock"
mkdir -p "${WORK}/elsewhere"; echo "old prompt" > "${WORK}/elsewhere/stash"
for name in "leg.7-20261121T000600Z.Ab12Cd" "leg.${TAG5}.Ab12Cd.x" "leg.${TAG5}" "leg.Ab12Cd" "symlink"; do
  if [[ "${name}" == symlink ]]; then
    ln -s "${WORK}/elsewhere" "${JOB}/adapter-home/leg.${TAG5}.Zz99Yy"
  else
    mkdir -p "${JOB}/adapter-home/${name}"; echo "old prompt" > "${JOB}/adapter-home/${name}/stash"
  fi
  : > "${WORK}/sudo.log"
  leg --probe
  if [[ "${status}" -ne 0 ]] && ! adapter_ran && grep -q 'LEG-ISOLATION-REFUSED: adapter-home holds 1 entry' <<<"${err}" \
     && ! grep -q 'LEG-ISOLATION-CLEARED' <<<"${err}" && ! grep -q 'Ab12Cd\|Zz99Yy' <<<"${err}"; then
    ok "case 5: a leftover '${name}' that is not this leg's own home refuses the leg"
  else
    fail "case 5 (${name}): status ${status}, adapter ran: $(adapter_ran && echo yes || echo no), err '${err}'"
  fi
  [[ -e "${WORK}/elsewhere/stash" ]] || fail "case 5 (${name}): the symlink target was touched"
  rm -rf "${JOB:?}/adapter-home/"*
done
rm -rf "${JOB}/sock/${TAG5}"

# --- case 6: a live sandbox-user process that keeps writing into this leg's
# killed home: the clear cannot finish, so the leg REFUSES ---------------------
TAG6="9-20261121T000800Z"
mkdir -p "${JOB}/sock/${TAG6}" "${JOB}/adapter-home/leg.${TAG6}.Live01"
RESIDENT_SOCKET="${JOB}/sock/${TAG6}/vllm-resident.sock"
writers=()
for w in 1 2 3; do
  ( d="${JOB}/adapter-home/leg.${TAG6}.Live01"; while [[ -d "${d}" ]]; do echo "leg 1 prompt" > "${d}/stash${w}" 2>/dev/null; done ) &
  writers+=("$!")
done
sleep 0.2
: > "${WORK}/sudo.log"
leg --probe
found="$(grep -c '^found=' <<<"${out}")"
if [[ "${status}" -ne 0 ]] && ! adapter_ran && grep -q 'LEG-ISOLATION-REFUSED: adapter-home holds 1 entry' <<<"${err}"; then
  ok "case 6: a live process holding this leg's killed home refuses the leg"
elif [[ "${status}" -eq 0 && "${found}" -eq 0 ]] && grep -q 'LEG-ISOLATION-CLEARED' <<<"${err}"; then
  # Only if all three writers were starved for the whole clear: the home is
  # gone, and still nothing it held is readable. Isolation holds either way.
  ok "case 6: the writers were starved, the home was removed, the adapter read nothing (race outcome)"
else
  fail "case 6: status ${status}, adapter ran: $(adapter_ran && echo yes || echo no), read ${found} file(s), err '${err}'"
fi
kill "${writers[@]}" 2>/dev/null; wait "${writers[@]}" 2>/dev/null
rm -rf "${JOB:?}/adapter-home/"* "${JOB}/sock/${TAG6}"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-leg-isolation.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
