#!/usr/bin/env bash
# Exercise the real local entry point, root proxy, and facade with a stub benchd.
# No model, GPU, toolchain, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Canonical (pwd -P): on macOS mktemp answers under /var, a symlink to
# /private/var, and the helper records its cwd resolved.
WORK="$(cd "$(mktemp -d)" && pwd -P)"
SOCKDIR="$(mktemp -d /tmp/localbase.XXXXXX)"
trap 'rm -rf "${WORK}" "${SOCKDIR}"' EXIT
# shellcheck source=tools/testlib/synthetic-window.sh
. "${REPO_ROOT}/tools/testlib/synthetic-window.sh"
synthetic_window_env
synthetic_window_meminfo "${WORK}/meminfo"
export SERVE_UP_MEMINFO="${WORK}/meminfo" SERVE_UP_SOCKET_DIR="${SOCKDIR}"
TEST_ROOT="${WORK}/checkout with spaces"
mkdir -p "${TEST_ROOT}/tools/testlib" "${TEST_ROOT}/fixtures" "${TEST_ROOT}/correctness_prompts" \
  "${TEST_ROOT}/.build/release" "${TEST_ROOT}/harness"
synthetic_window_weights "${TEST_ROOT}/reference_weights/Muse-Glimmer-30B-NVFP4"
cp -R "${REPO_ROOT}/harness/vllm_resident" "${TEST_ROOT}/harness/"
cp "${REPO_ROOT}/benchmark.sh" "${TEST_ROOT}/"
cp "${REPO_ROOT}/tools/"{local-baseline,benchmark,spec-declaration,serve-up}.sh "${TEST_ROOT}/tools/"
cp "${REPO_ROOT}/tools/testlib/serial-declaration.jq" "${TEST_ROOT}/tools/testlib/"
cp "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/mtp-head.manifest.json" "${TEST_ROOT}/"
cp "${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json" "${TEST_ROOT}/fixtures/"
# The real committed public golden: local-baseline.sh pins its bytes.
PUBLIC_GOLDEN=correctness_prompts/public-longcopy-gate-english-512.golden.json
cp "${REPO_ROOT}/${PUBLIC_GOLDEN}" "${TEST_ROOT}/${PUBLIC_GOLDEN}"
# The cases below run serial unless they say otherwise.
# The contract's pin, drafter off (not the shipped file: the base may switch it on).
SERIAL_DECL="$(jq -c -f "${REPO_ROOT}/tools/testlib/serial-declaration.jq" "${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json")"
printf '%s\n' "${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"

cat > "${WORK}/benchd" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${2:-}" == "--help" ]]; then
  echo '--mtp-depth'
  exit 0
fi
printf '%s\n' "$PWD" > "${CAPTURE}.cwd"
printf '%s\n' "$@" > "${CAPTURE}.argv"
printf '%s\n' "${SPEC_DECLARATION_MANIFEST:-}" > "${CAPTURE}.spec-manifest-path"
if [[ -n "${SPEC_DECLARATION_MANIFEST:-}" ]]; then
  cp "${SPEC_DECLARATION_MANIFEST}" "${CAPTURE}.spec-manifest"
fi
printf '%s\n' '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}'
exit "${STUB_EXIT:-0}"
STUB
chmod +x "${WORK}/benchd"
cp "${WORK}/benchd" "${TEST_ROOT}/.build/release/mlxfast-runtime-worker"

# Every case boots a REAL resident through serve-up.sh on the fake backend
# (host launcher, synthetic snapshot), so the dispatch is the one a real local
# run takes; the stub benchd stands in for the scorer.
run_case() {
  local name="$1"
  shift
  rm -f "${WORK}/${name}.argv" "${WORK}/${name}.spec-manifest" \
    "${WORK}/${name}.spec-manifest-path"
  rc=0
  (
    cd "${WORK}"
    env -u MLXFAST_ENGINE_BIN -u MLXFAST_CORRECTNESS_GOLDEN_PATH \
      -u MLXFAST_WEIGHTS_PATH -u MLXFAST_SCORE_PATH -u MLXFAST_TARGET_SNAPSHOT_DIR \
      -u MLXFAST_CORRECTNESS_GOLDEN_SHA256 -u MLXFAST_CORRECTNESS_GOLDEN_BYTES \
      -u MLXFAST_QWEN_MTP_TRACK_ID \
      -u SPEC_DECLARATION_MANIFEST -u SPEC_DECLARATION_CONTRACT \
      BENCHD="${WORK}/benchd" CAPTURE="${WORK}/${name}" "$@"
  ) > "${WORK}/${name}.stdout" 2> "${WORK}/${name}.stderr" || rc=$?
}

assert_arg() {
  local name="$1" flag="$2" value="$3"
  awk -v flag="${flag}" -v value="${value}" '
    previous == flag && $0 == value { found=1 }
    { previous=$0 }
    END { exit !found }
  ' "${WORK}/${name}.argv"
}

run_case defaults "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
[[ "$(cat "${WORK}/defaults.cwd")" == "${TEST_ROOT}" ]]
assert_arg defaults --mode local-iterate
assert_arg defaults --engine .build/release/mlxfast-runtime-worker
assert_arg defaults --weights reference_weights/Muse-Glimmer-30B-NVFP4
assert_arg defaults --golden "${PUBLIC_GOLDEN}"
assert_arg defaults --golden-sha256 "$(shasum -a 256 "${REPO_ROOT}/${PUBLIC_GOLDEN}" | awk '{print $1}')"
assert_arg defaults --golden-bytes "$(wc -c < "${REPO_ROOT}/${PUBLIC_GOLDEN}" | tr -d '[:space:]')"
assert_arg defaults --score-path score.local-iterate.json
grep -Fxq -- '--cool-gate' "${WORK}/defaults.argv"
grep -q 'public local baseline (unranked)' "${WORK}/defaults.stderr"
grep -q 'score=null is expected' "${WORK}/defaults.stderr"
[[ "$(cat "${WORK}/defaults.stdout")" == '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}' ]]

# With no explicit golden, an armed shipped declaration runs the public serial
# golden through a temporary serial declaration. The shipped file stays exact.
jq -c '.spec.enabled = true | .spec.num_speculative_tokens = 3' <<<"${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"
cp "${TEST_ROOT}/mtp-head.manifest.json" "${WORK}/armed-depth.manifest"
run_case armed_depth "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
if grep -Fxq -- '--mtp-depth' "${WORK}/armed_depth.argv"; then echo "test-local-baseline: unexpected match (line $LINENO)" >&2; exit 1; fi
jq -e '.spec.enabled == false and .spec.num_speculative_tokens == 0' \
  "${WORK}/armed_depth.spec-manifest" >/dev/null
grep -Fqx 'local-baseline.sh: the shipped declaration derives mtp3; this local run measures SERIAL against the public serial golden (no public mtp3 golden). Set MLXFAST_CORRECTNESS_GOLDEN_PATH to a mtp3 golden to run the drafter.' \
  "${WORK}/armed_depth.stderr"
cmp -s "${WORK}/armed-depth.manifest" "${TEST_ROOT}/mtp-head.manifest.json"
[[ ! -e "$(cat "${WORK}/armed_depth.spec-manifest-path")" ]]

# The serial fallback preserves benchmark.sh's failure and removes its manifest.
run_case fallback_failed STUB_EXIT=1 "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 ]]
[[ ! -e "$(cat "${WORK}/fallback_failed.spec-manifest-path")" ]]

# Invalid shipped declarations still fail before benchd starts.
jq -c '.spec.enabled = true | .spec.num_speculative_tokens = 7' <<<"${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"
run_case unpermitted_depth "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" != 0 ]]
[[ ! -f "${WORK}/unpermitted_depth.argv" ]]
grep -q 'not a contract-permitted draft depth' "${WORK}/unpermitted_depth.stderr"
printf '%s\n' "${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"

synthetic_window_weights "${TEST_ROOT}/box snapshot"
run_case snapshot_env MLXFAST_TARGET_SNAPSHOT_DIR='box snapshot' "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg snapshot_env --weights 'box snapshot'

cp "${WORK}/benchd" "${TEST_ROOT}/custom engine"
cp "${TEST_ROOT}/${PUBLIC_GOLDEN}" "${TEST_ROOT}/custom golden.json"
synthetic_window_weights "${TEST_ROOT}/custom weights"

# An explicit golden keeps the shipped speculative declaration. The fake
# drafter must fail the pinned-drafter check, so this case does not boot it.
DRAFTER="${WORK}/drafter"
mkdir -p "${DRAFTER}"
while IFS= read -r file; do
  printf '{}' > "${DRAFTER}/${file}"
done < <(jq -r '.mtp_head.files[].path' "${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json")
jq -c '.spec.enabled = true | .spec.num_speculative_tokens = 3' <<<"${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"
cp "${TEST_ROOT}/mtp-head.manifest.json" "${WORK}/mtp-golden.manifest"
run_case mtp_golden MLXFAST_CORRECTNESS_GOLDEN_PATH='custom golden.json' \
  SERVE_UP_DRAFTER_DIR="${DRAFTER}" "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" != 0 ]]
grep -Fq 'declared spec mtp3 (mtp-head.manifest.json' "${WORK}/mtp_golden.stderr"
if grep -Fq 'this local run measures SERIAL' "${WORK}/mtp_golden.stderr"; then echo "test-local-baseline: unexpected match (line $LINENO)" >&2; exit 1; fi
cmp -s "${WORK}/mtp-golden.manifest" "${TEST_ROOT}/mtp-head.manifest.json"

# A caller's declaration is never replaced by the local serial fallback.
printf '%s\n' "${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"
CALLER_MANIFEST="${WORK}/caller-mtp3.manifest.json"
jq -c '.spec.enabled = true | .spec.num_speculative_tokens = 3' <<<"${SERIAL_DECL}" > "${CALLER_MANIFEST}"
run_case caller_manifest SPEC_DECLARATION_MANIFEST="${CALLER_MANIFEST}" \
  "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" != 0 ]]
[[ ! -f "${WORK}/caller_manifest.argv" ]]
grep -q "the caller's SPEC_DECLARATION_MANIFEST derives mtp3" \
  "${WORK}/caller_manifest.stderr"

run_case overrides MLXFAST_ENGINE_BIN='custom engine' \
  MLXFAST_CORRECTNESS_GOLDEN_PATH='custom golden.json' \
  MLXFAST_TARGET_SNAPSHOT_DIR='box snapshot' \
  MLXFAST_WEIGHTS_PATH='custom weights' MLXFAST_SCORE_PATH='custom result.json' \
  "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg overrides --engine 'custom engine'
assert_arg overrides --golden 'custom golden.json'
assert_arg overrides --weights 'custom weights'
assert_arg overrides --score-path 'custom result.json'
# A caller's own golden carries no default pin.
if grep -Fxq -- '--golden-sha256' "${WORK}/overrides.argv"; then echo "test-local-baseline: unexpected match (line $LINENO)" >&2; exit 1; fi

# The default golden is pinned: an altered copy refuses before any run work.
cp "${TEST_ROOT}/${PUBLIC_GOLDEN}" "${WORK}/public-golden.keep"
printf ' ' >> "${TEST_ROOT}/${PUBLIC_GOLDEN}"
run_case altered_golden "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/altered_golden.argv" ]]
grep -q 'sha256 mismatch' "${WORK}/altered_golden.stderr"
cp "${WORK}/public-golden.keep" "${TEST_ROOT}/${PUBLIC_GOLDEN}"

run_case failed STUB_EXIT=1 "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 ]]
run_case help "${TEST_ROOT}/tools/local-baseline.sh" --help
[[ "${rc}" == 0 && ! -f "${WORK}/help.argv" ]]
run_case wrong_mode "${TEST_ROOT}/tools/local-baseline.sh" --official
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_mode.argv" ]]
run_case missing_golden MLXFAST_CORRECTNESS_GOLDEN_PATH=absent.json "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_golden.argv" ]]
grep -q 'correctness golden not found' "${WORK}/missing_golden.stderr"
run_case wrong_pin MLXFAST_CORRECTNESS_GOLDEN_SHA256=bad "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_pin.argv" ]]
grep -q 'sha256 mismatch' "${WORK}/wrong_pin.stderr"
run_case missing_engine MLXFAST_ENGINE_BIN=absent-engine "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_engine.argv" ]]
grep -q './setup.sh' "${WORK}/missing_engine.stderr"
run_case missing_snapshot MLXFAST_WEIGHTS_PATH=absent-snapshot "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_snapshot.argv" ]]
grep -q 'target snapshot directory not found' "${WORK}/missing_snapshot.stderr"

echo 'test-local-baseline.sh: all 14 cases passed'
