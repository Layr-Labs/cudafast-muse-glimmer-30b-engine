#!/usr/bin/env bash
# test-live-golden-rotation.sh -- the per-job scored-prompt draw, hermetically.
#
# Covers tools/draw-live-golden.sh (uniform over the rotation pool, refusal of
# a pool name outside timed_prompt_pool, fallback to live_golden when no
# rotation is declared) and tools/nemotron35-measure-and-score.sh's use of the
# drawn name (refusal outside the pool, refusal when unset on a real run, the
# drawn prompt's serial golden and per-depth oracle handed to benchd with
# --live-prompt, and the no-rotation fallback). benchd is a stub that records
# its argv. No GPU, no network.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
DRAW="${REPO_ROOT}/tools/draw-live-golden.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok    $*"; }

mapfile -t NAMES < <(jq -r '.timed_prompt_pool[].r2_path | split("/")[-1] | sub("\\.golden\\.json$"; "")' "${FIXTURE}")

# A fixture copy whose rotation pool is the whole timed pool, with synthetic
# per-depth oracle pins for every name (only measure-and-score reads them here).
jq '.live_golden_rotation.pool = [.timed_prompt_pool[].r2_path | split("/")[-1] | sub("\\.golden\\.json$"; "")]
    | .live_golden_rotation.pending = []
    | (.timed_prompt_pool[0].r2_path | split("/")[:-1] | join("/")) as $prefix
    | .speculative_oracles = (reduce .live_golden_rotation.pool[] as $n ({};
        .[$n] = (reduce range(1; 7) as $d ({};
          .["mtp\($d)"] = {r2_path: "\($prefix)/\($n).mtp\($d).golden.json",
                           sha256: ("\($d)" * 64), bytes: (1000 + $d)}))))' \
  "${FIXTURE}" > "${WORK}/full.json"

# --- the draw -----------------------------------------------------------------
draws=800
counts="$(for _ in $(seq "${draws}"); do "${DRAW}" "${WORK}/full.json"; done | sort | uniq -c)"
distinct="$(wc -l <<<"${counts}" | tr -d ' ')"
[[ "${distinct}" -eq "${#NAMES[@]}" ]] \
  && ok "draw: ${draws} draws hit all ${#NAMES[@]} pool prompts" \
  || fail "draw: ${draws} draws hit ${distinct} of ${#NAMES[@]} prompts: ${counts}"
# Expected 100 per prompt, sd ~9.4; [50, 150] is a five-sigma band.
while read -r c name; do
  (( c >= 50 && c <= 150 )) || fail "draw: '${name}' drawn ${c} times of ${draws}; not uniform"
done <<<"${counts}"
ok "draw: every prompt drawn within five sigma of uniform"

jq 'del(.live_golden_rotation)' "${FIXTURE}" > "${WORK}/norot.json"
[[ "$("${DRAW}" "${WORK}/norot.json")" == "$(jq -r .live_golden "${FIXTURE}")" ]] \
  && ok "draw: no rotation block falls back to live_golden" \
  || fail "draw: no rotation block did not fall back to live_golden"
drawn="$("${DRAW}")"
jq -e --arg d "${drawn}" '.live_golden_rotation.pool | index($d)' "${FIXTURE}" >/dev/null \
  && ok "draw: the shipped fixture draws a pool name (${drawn})" \
  || fail "draw: the shipped fixture drew '${drawn}', not a pool name"

jq '.live_golden_rotation.pool += ["nosuch"]' "${FIXTURE}" > "${WORK}/bad.json"
out="$("${DRAW}" "${WORK}/bad.json" 2>&1)" && fail "draw: a pool name outside timed_prompt_pool was drawn from" \
  || { grep -q "'nosuch' has no timed_prompt_pool entry" <<<"${out}" && ok "draw: a pool name outside timed_prompt_pool is refused by name" || fail "draw: wrong refusal: ${out}"; }
jq '.live_golden_rotation.pool = []' "${FIXTURE}" > "${WORK}/empty.json"
"${DRAW}" "${WORK}/empty.json" >/dev/null 2>&1 && fail "draw: an empty pool drew something" || ok "draw: an empty pool is refused"
jq '.live_golden_rotation.mode = "round_robin"' "${FIXTURE}" > "${WORK}/mode.json"
"${DRAW}" "${WORK}/mode.json" >/dev/null 2>&1 && fail "draw: an unknown mode drew something" || ok "draw: an unknown mode is refused"

# --- measure-and-score uses the choice -------------------------------------------
ROOT="${WORK}/root"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures"
for entry in "${REPO_ROOT}"/tools/*; do ln -s "${entry}" "${ROOT}/tools/$(basename "${entry}")"; done
ln -s "${REPO_ROOT}/benchmark.json" "${ROOT}/benchmark.json"
cp "${REPO_ROOT}/mtp-head.manifest.json" "${ROOT}/mtp-head.manifest.json"
jq '.spec = {enabled: true, num_speculative_tokens: 3}' "${REPO_ROOT}/mtp-head.manifest.json" > "${WORK}/mtp3.manifest.json"
CONTRACT="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"

GOLDENS="${WORK}/goldens"
mkdir -p "${GOLDENS}" "${WORK}/ws" "${WORK}/weights"
for n in "${NAMES[@]}"; do
  : > "${GOLDENS}/${n}.golden.json"
  : > "${GOLDENS}/${n}.mtp3.golden.json"
done
: > "${WORK}/cal.json"
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/engine"
chmod +x "${WORK}/engine"

# benchd stub: `iterate --help` advertises the flags (STUB_FLAGS), a real run
# records its argv. With STUB_PLAN (a file of outcomes, one per run: pass,
# control, prefill, spawn, floor) it also seals score.json like benchd: the
# JSON on stdout, the marker on stderr, exit 1 unless it passed. Run n appends
# its outcome to STUB_RUNS and keeps its argv as STUB_RUNS.argv.n. With
# STUB_JOBDIR it logs how many files the per-job sandbox dirs hold
# (STUB_RUNS.seen), then plants one in each, as a memoizing candidate would.
cat > "${WORK}/benchd" <<'EOF'
#!/usr/bin/env bash
if [[ "$2" == --help ]]; then echo "${STUB_FLAGS:---control-golden --mtp-depth --live-prompt}"; exit 0; fi
printf '%s\n' "$@" > "${STUB_ARGV}"
[[ -n "${STUB_PLAN:-}" ]] || exit 0
n=$(( $(wc -l < "${STUB_RUNS}") + 1 ))
what="$(sed -n "${n}p" "${STUB_PLAN}")"
echo "${what}" >> "${STUB_RUNS}"
cp "${STUB_ARGV}" "${STUB_RUNS}.argv.${n}"
if [[ -n "${STUB_JOBDIR:-}" ]]; then
  find "${STUB_JOBDIR}" -mindepth 2 | wc -l >> "${STUB_RUNS}.seen"
  for d in cargo-home target kernels-out sock adapter-home; do echo memo > "${STUB_JOBDIR}/${d}/memo-${n}"; done
fi
score="$(grep -A1 -xF -- --score-path <<<"$(printf '%s\n' "$@")" | tail -1)"
case "${what}" in
  pass) json="{\"passed\": true, \"score\": 1.${n}, \"metrics\": {\"error\": \"\"}}" ;;
  control) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 47: expected oracle token 1278, engine returned 1799\"}}" ;;
  prefill) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: reference warmup leg failed: benchmark prefill token mismatch at step 0: expected oracle token 1, engine returned 2\"}}" ;;
  spawn) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: worker spawn failed\"}}" ;;
  floor) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"performance floor failed: decode_speedup=0.9\"}}" ;;
  boot) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the candidate leg's resident did not boot (1): abcd1234\", \"paired_legs\": []}}" ;;
  boot2) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the candidate leg's resident did not boot (1): abcd1234\", \"paired_legs\": [{\"pair\": 1, \"prompt_sha256\": \"deadbeef\", \"control_prefill_seconds_per_token\": 1.0, \"control_decode_seconds_per_token\": 2.0, \"candidate_prefill_seconds_per_token\": 1.1, \"candidate_decode_seconds_per_token\": 2.1}]}}" ;;
esac
echo "stub benchd run ${n}: ${what}" >&2
printf '%s\n' "${json}" | tee "${score}"
: > "${score}.sha256"
echo "benchd iterate: wrote ${score} (passed=$(jq -r .passed "${score}"), score=$(jq -r .score "${score}"))" >&2
[[ "${what}" == pass ]]
EOF
chmod +x "${WORK}/benchd"

measure() { # measure FIXTURE [env...]  -> stdout+stderr; argv in ${WORK}/argv
  cp "$1" "${CONTRACT}"; shift
  rm -f "${WORK}/argv"
  env -u MLXFAST_LIVE_GOLDEN_CHOICE -u SANDBOX_USER -u MLXFAST_QWEN_MTP_TRACK_ID \
    BENCHD="${WORK}/benchd" STUB_ARGV="${WORK}/argv" \
    MLXFAST_BASELINE_WORKSPACE="${WORK}/ws" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
    MLXFAST_NEMOTRON35_GOLDEN_DIR="${GOLDENS}" MLXFAST_ENGINE_BIN="${WORK}/engine" \
    MLXFAST_WEIGHTS_PATH="${WORK}/weights" SPEC_DECLARATION_MANIFEST="${ROOT}/mtp-head.manifest.json" \
    "$@" "${ROOT}/tools/nemotron35-measure-and-score.sh" ${MS_ARGS[@]+"${MS_ARGS[@]}"} 2>&1
}
MS_ARGS=()
argv_has() { grep -qxF -- "$1" "${WORK}/argv" 2>/dev/null; }
argv_after() { grep -A1 -xF -- "$1" "${WORK}/argv" | tail -1; }

out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle)"
if argv_has --live-prompt && [[ "$(argv_after --live-prompt)" == beagle ]] \
  && [[ "$(argv_after --golden)" == "${GOLDENS}/beagle.golden.json" ]] \
  && [[ "$(argv_after --control-golden)" == "${GOLDENS}/beagle.golden.json" ]]; then
  ok "measure: the drawn prompt (beagle) is the serial golden, the control golden and --live-prompt"
else
  fail "measure: serial run did not score the drawn prompt; out: $(tail -2 <<<"${out}"); argv: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi
want_sha="$(jq -r '.timed_prompt_pool[] | select(.r2_path | endswith("/beagle.golden.json")) | .sha256' "${FIXTURE}")"
[[ "$(argv_after --golden-sha256)" == "${want_sha}" ]] \
  && ok "measure: the golden pin is the drawn prompt's timed_prompt_pool pin" \
  || fail "measure: the golden pin is not beagle's"

out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=travel SPEC_DECLARATION_MANIFEST="${WORK}/mtp3.manifest.json")"
if [[ "$(argv_after --golden)" == "${GOLDENS}/travel.mtp3.golden.json" ]] \
  && [[ "$(argv_after --golden-sha256)" == "$(printf '3%.0s' $(seq 64))" ]] \
  && [[ "$(argv_after --control-golden)" == "${GOLDENS}/travel.golden.json" ]] \
  && [[ "$(argv_after --mtp-depth)" == 3 ]] && [[ "$(argv_after --live-prompt)" == travel ]]; then
  ok "measure: at mtp3 the candidate oracle is the drawn prompt's speculative_oracles.travel.mtp3"
else
  fail "measure: mtp3 run did not use travel's oracle; out: $(tail -2 <<<"${out}"); argv: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi

out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=nosuch)"
[[ ! -f "${WORK}/argv" ]] && grep -q "'nosuch' is not in live_golden_rotation.pool" <<<"${out}" \
  && ok "measure: a choice outside the rotation pool is refused by name" \
  || fail "measure: a choice outside the pool was not refused: $(tail -2 <<<"${out}")"

jq '.live_golden_rotation.pool = ["botany"]' "${FIXTURE}" > "${WORK}/botany-only.json"
out="$(measure "${WORK}/botany-only.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle)"
[[ ! -f "${WORK}/argv" ]] && grep -q "'beagle' is not in live_golden_rotation.pool" <<<"${out}" \
  && ok "measure: a pool of [botany] refuses beagle" \
  || fail "measure: a pool of [botany] accepted beagle: $(tail -2 <<<"${out}")"

out="$(measure "${WORK}/full.json")"
[[ ! -f "${WORK}/argv" ]] && grep -q "MLXFAST_LIVE_GOLDEN_CHOICE is unset" <<<"${out}" \
  && ok "measure: a real run with rotation and no drawn prompt is refused" \
  || fail "measure: an undrawn real run was not refused: $(tail -2 <<<"${out}")"

out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle STUB_FLAGS="--control-golden --mtp-depth")"
[[ ! -f "${WORK}/argv" ]] && grep -q "has no --live-prompt" <<<"${out}" \
  && ok "measure: a benchd without --live-prompt cannot score a non-default drawn prompt" \
  || fail "measure: an old benchd scored beagle: $(tail -2 <<<"${out}")"
out="$(measure "${FIXTURE}" MLXFAST_LIVE_GOLDEN_CHOICE=botany STUB_FLAGS="--control-golden --mtp-depth")"
[[ -f "${WORK}/argv" ]] && ! argv_has --live-prompt \
  && ok "measure: a benchd without --live-prompt still scores the default live_golden" \
  || fail "measure: an old benchd refused the default: $(tail -2 <<<"${out}")"

out="$(measure "${WORK}/norot.json")"
[[ "$(argv_after --golden)" == "${GOLDENS}/botany.golden.json" ]] && ! argv_has --live-prompt \
  && ok "measure: no rotation block scores live_golden, with no --live-prompt (as before rotation)" \
  || fail "measure: the no-rotation fallback changed: $(tail -2 <<<"${out}")"
out="$(measure "${WORK}/norot.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle)"
[[ ! -f "${WORK}/argv" ]] && grep -q "declares no live_golden_rotation" <<<"${out}" \
  && ok "measure: a choice with no rotation block is refused unless it is live_golden" \
  || fail "measure: a choice without rotation was honoured: $(tail -2 <<<"${out}")"

# --- the one control-leg retry ---------------------------------------------------
# plan OUTCOME... -> runs measure with that benchd plan; sets rc, out, runs.
printf 'MemTotal: %d kB\nMemAvailable: %d kB\n' $((100 * 1024 * 1024)) $((100 * 1024 * 1024)) > "${WORK}/meminfo"
PLAN_ENV=()
plan() {
  printf '%s\n' "$@" > "${WORK}/plan"
  : > "${WORK}/runs"
  rm -f "${ROOT}/score.json" "${ROOT}/score.json.sha256" "${WORK}"/runs.*
  out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle STUB_PLAN="${WORK}/plan" \
    STUB_RUNS="${WORK}/runs" SERVE_UP_MEMINFO="${WORK}/meminfo" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    ${PLAN_ENV[@]+"${PLAN_ENV[@]}"})"
  rc=$?
  runs="$(tr '\n' ' ' < "${WORK}/runs")"
}
markers() { grep -c "benchd iterate: wrote" <<<"${out}"; }

plan pass
[[ "${rc}" -eq 0 && "${runs}" == "pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" && jq -e '.score == 1.1' "${ROOT}/score.json" >/dev/null \
  && ok "retry: a passing run runs once, attempts=1, one marker" \
  || fail "retry: pass: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan control pass
[[ "${rc}" -eq 0 && "${runs}" == "control pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" && grep -q "(passed=true, score=1.2)" <<<"${out}" \
  && ! grep -q '"score": null' <<<"${out}" && jq -e '.score == 1.2' "${ROOT}/score.json" >/dev/null \
  && ok "retry: control failure then pass is scored from attempt 2 alone, attempts=2, one marker" \
  || fail "retry: control+pass: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan control control pass
[[ "${rc}" -eq 75 && "${runs}" == "control control " && "$(markers)" -eq 1 ]] \
  && grep -q "Serial control leg diverged twice" <<<"${out}" && grep -q "control_leg_attempts=2" <<<"${out}" \
  && ok "retry: two control failures end as infrastructure (exit 75), no third run" \
  || fail "retry: control twice: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan floor pass
[[ "${rc}" -eq 1 && "${runs}" == "floor " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: a candidate failure is never retried" \
  || fail "retry: floor: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan control floor
[[ "${rc}" -eq 1 && "${runs}" == "control floor " ]] && jq -e '.metrics.error | startswith("performance floor")' "${ROOT}/score.json" >/dev/null \
  && ok "retry: a candidate failure on attempt 2 is the result (no third run)" \
  || fail "retry: control+floor: rc=${rc} runs=${runs}"
cmp -s "${WORK}/runs.argv.1" "${WORK}/runs.argv.2" && grep -qxF beagle "${WORK}/runs.argv.2" \
  && ok "retry: attempt 2 runs benchd with the identical argv (same --live-prompt)" \
  || fail "retry: attempt 2's argv differs from attempt 1's"

plan prefill pass
[[ "${rc}" -eq 0 && "${runs}" == "prefill pass " ]] \
  && ok "retry: a reference warmup prefill-token divergence is retried" \
  || fail "retry: prefill divergence: rc=${rc} runs=${runs}"

plan spawn pass
[[ "${rc}" -eq 1 && "${runs}" == "spawn " ]] && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: a control-leg failure that is not a token divergence (worker spawn) is final" \
  || fail "retry: spawn: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan boot pass
[[ "${rc}" -eq 0 && "${runs}" == "boot pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" && grep -q "pair-1 candidate resident did not boot" <<<"${out}" \
  && jq -e '.score == 1.2' "${ROOT}/score.json" >/dev/null \
  && ok "retry: pair-1 candidate boot failure then pass is scored from attempt 2" \
  || fail "retry: boot+pass: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan boot2 pass
[[ "${rc}" -eq 1 && "${runs}" == "boot2 " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: pair-2 candidate boot failure (paired_legs non-empty) is never retried" \
  || fail "retry: boot2: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan control boot
[[ "${rc}" -eq 1 && "${runs}" == "control boot " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" \
  && ok "retry: control divergence then pair-1 boot failure uses the one retry budget (no third run)" \
  || fail "retry: control+boot: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

cat > "${WORK}/benchd-noseal" <<'EOF'
#!/usr/bin/env bash
if [[ "$2" == --help ]]; then echo "--control-golden --mtp-depth --live-prompt"; exit 0; fi
echo run >> "${STUB_RUNS}"; exit 1
EOF
chmod +x "${WORK}/benchd-noseal"
echo '{"passed": false, "score": null, "metrics": {"error": "SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 1"}}' > "${ROOT}/score.json"
: > "${WORK}/runs"
out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE=beagle STUB_RUNS="${WORK}/runs" \
  SERVE_UP_MEMINFO="${WORK}/meminfo" WAIT_FOR_MEMORY_TIMEOUT_S=0 BENCHD="${WORK}/benchd-noseal")"
rc=$?
[[ "${rc}" -eq 1 && "$(wc -l < "${WORK}/runs")" -eq 1 && ! -e "${ROOT}/score.json" ]] \
  && ok "retry: a stale control-divergence seal from before the run never triggers a retry" \
  || fail "retry: stale seal: rc=${rc} runs=$(wc -l < "${WORK}/runs")"

# The sandboxed teardown, with sudo and ipcs stubbed (the sandbox user is the
# current user; the stub ipcs lists nothing, so no real IPC object is touched).
SB="${WORK}/sb"
JOBDIR="${SB}/root/42-1"
mkdir -p "${SB}/bin" "${SB}/tmp" "${SB}/home" "${SB}/runner-temp"
for d in cargo-home target kernels-out sock adapter-home; do mkdir -p "${JOBDIR}/${d}"; done
printf '%s\n' "${SB}/tmp" > "${SB}/wipe-dirs"
cat > "${SB}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
shift 4
[[ "$1" == kill ]] && exit 0
exec "$@"
EOF
cat > "${SB}/bin/pgrep" <<EOF
#!/usr/bin/env bash
[[ "\$1 \$2" == "-u $(id -un)" ]] && exit 1
exec $(command -v pgrep) "\$@"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "${SB}/bin/ipcs"
chmod +x "${SB}/bin/"*
SB_ENV=(PATH="${SB}/bin:${PATH}" SANDBOX_USER="$(id -un)" SANDBOX_ROOT="${SB}/root" SANDBOX_JOB=42-1
        RUNNER_TEMP="${SB}/runner-temp" SANDBOX_WIPE_LIST="${SB}/wipe-dirs" SANDBOX_WIPE_HOME="${SB}/home")
env "${SB_ENV[@]}" "${ROOT}/tools/stage-cuda-engine.sh" --print-shim > "${SB}/engine"
cp "${WORK}/engine" "${SB}/engine.sandboxed"
chmod +x "${SB}/engine" "${SB}/engine.sandboxed"
echo setup-leftover > "${SB}/tmp/setup-leftover"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine")
plan pass
[[ "${rc}" -eq 0 && "${runs}" == "pass " && ! -e "${SB}/tmp/setup-leftover" ]] \
  && ok "measure: a file planted in shared tmp before the run is gone when attempt 1 starts" \
  || fail "measure: pre-measurement wipe: rc=${rc} runs=${runs} leftover=$(test -e "${SB}/tmp/setup-leftover" && echo yes || echo no)"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}")
plan control pass
[[ "${rc}" -eq 0 && "${runs}" == "control pass " && "$(tr '\n' ' ' < "${WORK}/runs.seen")" == "0 0 " ]] \
  && ok "retry: every file attempt 1 planted in the five per-job sandbox dirs is gone when attempt 2 starts" \
  || fail "retry: sandboxed teardown: rc=${rc} runs=${runs} seen=$(tr '\n' ' ' < "${WORK}/runs.seen" 2>/dev/null) out: $(tail -4 <<<"${out}")"

for d in cargo-home target kernels-out sock adapter-home; do rm -f "${JOBDIR}/${d}"/*; done
PLAN_ENV=("${SB_ENV[@]}" SANDBOX_WIPE_LIST="${SB}/no-such-list" MLXFAST_ENGINE_BIN="${SB}/engine")
plan pass
[[ "${rc}" -eq 75 && "${runs}" == "" ]] && grep -q "Pre-measurement sandbox reset failed" <<<"${out}" \
  && ok "measure: a pre-measurement wipe failure exits 75 before attempt 1" \
  || fail "measure: pre-measurement wipe failure: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
WIPE_REAL="${REPO_ROOT}/tools/sandbox-wipe.sh"
WIPE_COUNT="${SB}/wipe-count"
rm -f "${ROOT}/tools/sandbox-wipe.sh"
cat > "${ROOT}/tools/sandbox-wipe.sh" <<EOF
#!/usr/bin/env bash
n=0
[[ -f "${WIPE_COUNT}" ]] && n=\$(<"${WIPE_COUNT}")
echo \$((n + 1)) > "${WIPE_COUNT}"
[[ "\${SANDBOX_WIPE_FAIL_ON:-0}" -eq \$((n + 1)) ]] \
  && { echo "sandbox-wipe.sh: deliberate test failure" >&2; exit 1; }
exec bash "${WIPE_REAL}" "\$@"
EOF
chmod +x "${ROOT}/tools/sandbox-wipe.sh"
rm -f "${WIPE_COUNT}"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}" SANDBOX_WIPE_FAIL_ON=2)
plan control pass
ln -sf "${WIPE_REAL}" "${ROOT}/tools/sandbox-wipe.sh"
[[ "${rc}" -eq 75 && "${runs}" == "control " ]] && grep -q "retry teardown failed" <<<"${out}" \
  && ok "retry: a teardown that cannot reset the box exits 75 without a second run" \
  || fail "retry: teardown failure: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
PLAN_ENV=()

MS_ARGS=(--preflight-only)
out="$(measure "${WORK}/full.json")"
MS_ARGS=()
! grep -q "MLXFAST_LIVE_GOLDEN_CHOICE is unset" <<<"${out}" && grep -q "official_scoring_enabled" <<<"${out}" \
  && ok "measure: --preflight-only with no drawn prompt resolves the default" \
  || fail "measure: --preflight-only did not resolve the default: $(tail -2 <<<"${out}")"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-live-golden-rotation.sh: all cases passed"
  exit 0
fi
echo "test-live-golden-rotation.sh: ${failures} case(s) failed" >&2
exit 1
