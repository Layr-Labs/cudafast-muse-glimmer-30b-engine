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
    find "${job}" -name stash -type f -readable 2>/dev/null | sed 's/^/found=/'
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

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-leg-isolation.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
