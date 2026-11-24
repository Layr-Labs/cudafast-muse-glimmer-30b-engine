#!/usr/bin/env bash
# setup.sh step 2 against a tiny pinned snapshot served from a file:// base.
# No GPU, toolchain, checkpoint, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1; else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}

# The checkout under test: the real setup.sh, a fixture whose target pins name
# five tiny files, and a staged engine and overlay so step 1 is skipped.
ROOT="${WORK}/checkout with spaces"
mkdir -p "${ROOT}/fixtures" "${ROOT}/tools" "${ROOT}/.build/release" "${ROOT}/.build/vllm-overlay/vllm"
cp "${REPO_ROOT}/setup.sh" "${ROOT}/"
# The trusted helpers step 2 calls for a contract that pins a drafter (case 11+).
cp "${REPO_ROOT}/tools/spec-declaration.sh" "${REPO_ROOT}/tools/drafter-dir-check.sh" "${ROOT}/tools/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/.build/release/mlxfast-runtime-worker"
touch "${ROOT}/.build/vllm-overlay/vllm/__init__.py"
chmod +x "${ROOT}/.build/release/mlxfast-runtime-worker"

# The source: the repository layout the pinned revision has, every file flat
# at the root.
SOURCE="${WORK}/source"
mkdir -p "${SOURCE}"
FILES=(
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  model.safetensors.index.json
  config.json
)
source_path() { echo "$1"; }
pins='[]'
for f in "${FILES[@]}"; do
  head -c $((RANDOM + 1000)) /dev/urandom > "${SOURCE}/$(source_path "${f}")"
  pins="$(jq -c --arg p "${f}" --argjson b "$(wc -c < "${SOURCE}/$(source_path "${f}")" | tr -d ' ')" \
    --arg s "$(sha256_of "${SOURCE}/$(source_path "${f}")")" '. + [{path:$p, bytes:$b, sha256:$s}]' <<<"${pins}")"
done
write_fixture() {
  jq -n --argjson files "$1" '{target:{upstream_model_id:"example/model", upstream_revision:"0000000", files:$files}}' \
    > "${ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
}
write_fixture "${pins}"

# run_setup NAME [ENV=VALUE...]: runs setup.sh from the checkout with the
# engine build skipped; captures stdout+stderr; sets rc.
run_setup() {
  local name="$1"; shift
  rc=0
  (
    cd "${ROOT}"
    env -u MLXFAST_TARGET_SNAPSHOT_DIR -u MLXFAST_TARGET_BASE_URL \
      MLXFAST_SKIP_ENGINE_BUILD=1 MLXFAST_TARGET_BASE_URL="file://${SOURCE}" "$@" ./setup.sh
  ) > "${WORK}/${name}.log" 2>&1 || rc=$?
}
downloads() { grep -c 'downloading ' "${WORK}/$1.log" || true; }
DEFAULT_DIR="${ROOT}/reference_weights/Muse-Glimmer-30B-NVFP4"

# 1. Fresh machine, no MLXFAST_TARGET_SNAPSHOT_DIR: the default directory is
#    filled from the source, verified, and marked.
run_setup fresh
[[ "${rc}" == 0 ]] || { cat "${WORK}/fresh.log"; exit 1; }
grep -q 'MLXFAST_TARGET_SNAPSHOT_DIR is unset; using' "${WORK}/fresh.log"
[[ "$(downloads fresh)" == 5 ]]
for f in "${FILES[@]}"; do cmp -s "${DEFAULT_DIR}/${f}" "${SOURCE}/$(source_path "${f}")"; done
grep -q 'by byte count and sha256' "${WORK}/fresh.log"
grep -q 'wrote the verification marker' "${WORK}/fresh.log"
marker="$(ls "${DEFAULT_DIR}"/.verified-*)"
[[ "$(wc -l < "${marker}" | tr -d ' ')" == 5 ]]
[[ -z "$(ls "${DEFAULT_DIR}"/*.partial 2>/dev/null)" ]]

# 2. Rerun on the same machine: nothing is fetched (the base is unreachable),
#    no shard is hashed, the marker answers.
run_setup rerun MLXFAST_TARGET_BASE_URL="file://${WORK}/nowhere"
[[ "${rc}" == 0 ]] || { cat "${WORK}/rerun.log"; exit 1; }
[[ "$(downloads rerun)" == 0 ]]
grep -q 'verified by marker: 5 pinned files unchanged' "${WORK}/rerun.log"
grep -q 'by byte count and the marker' "${WORK}/rerun.log"

# 3. One truncated file: only that file is fetched again; the set verifies in
#    full and a new marker is written.
truncate_target="${DEFAULT_DIR}/${FILES[2]}"
head -c 10 "${truncate_target}" > "${truncate_target}.tmp" && mv "${truncate_target}.tmp" "${truncate_target}"
run_setup truncated
[[ "${rc}" == 0 ]] || { cat "${WORK}/truncated.log"; exit 1; }
[[ "$(downloads truncated)" == 1 ]]
grep -q "${FILES[2]} is 10 bytes, pinned" "${WORK}/truncated.log"
cmp -s "${truncate_target}" "${SOURCE}/$(source_path "${FILES[2]}")"
grep -q 'by byte count and sha256' "${WORK}/truncated.log"

# 4. A file rewritten with the same size: its mtime and inode changed, so the
#    marker no longer matches, the full pass runs and refuses by name.
tamper_target="${DEFAULT_DIR}/${FILES[0]}"
size="$(wc -c < "${tamper_target}" | tr -d ' ')"
head -c "${size}" /dev/urandom > "${tamper_target}.tmp" && mv "${tamper_target}.tmp" "${tamper_target}"
run_setup tampered
[[ "${rc}" == 1 ]]
grep -q "sha256 mismatch for ${FILES[0]}" "${WORK}/tampered.log"
rm -f "${tamper_target}"

# 5. A corrupt source for a missing file: the download is discarded, nothing is
#    left in the snapshot, and setup fails closed.
good_source="${SOURCE}/$(source_path "${FILES[0]}")"
cp "${good_source}" "${WORK}/good.bak"
head -c "${size}" /dev/urandom > "${good_source}"
run_setup corrupt_source
[[ "${rc}" == 1 ]]
grep -q "downloaded ${FILES[0]} has sha256 .*; discarded" "${WORK}/corrupt_source.log"
[[ ! -e "${tamper_target}" && ! -e "${tamper_target}.partial" ]]
cp "${WORK}/good.bak" "${good_source}"

# 6. The pins change: a marker written under the old pins is not trusted.
run_setup restore
[[ "${rc}" == 0 ]] || { cat "${WORK}/restore.log"; exit 1; }
new_pins="$(jq -c '.[0].sha256 = ("0" * 64)' <<<"${pins}")"
write_fixture "${new_pins}"
run_setup repinned
[[ "${rc}" == 1 ]]
if grep -q 'verified by marker' "${WORK}/repinned.log"; then echo 'FAIL: a marker under old pins was trusted' >&2; exit 1; fi
grep -q "sha256 mismatch for ${FILES[0]}" "${WORK}/repinned.log"
write_fixture "${pins}"

# 7. An explicit MLXFAST_TARGET_SNAPSHOT_DIR (a box) holding the set: verified
#    in full on every run, nothing fetched, no marker written there.
BOX="${WORK}/box snapshot"
mkdir -p "${BOX}"
for f in "${FILES[@]}"; do cp "${SOURCE}/$(source_path "${f}")" "${BOX}/${f}"; done
for pass in box box_again; do
  run_setup "${pass}" MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}"
  [[ "${rc}" == 0 ]] || { cat "${WORK}/${pass}.log"; exit 1; }
  [[ "$(downloads "${pass}")" == 0 ]]
  grep -q 'by byte count and sha256' "${WORK}/${pass}.log"
  if grep -q 'verified by marker' "${WORK}/${pass}.log"; then echo 'FAIL: a staged snapshot used a marker' >&2; exit 1; fi
  [[ -z "$(ls "${BOX}"/.verified-* 2>/dev/null)" ]]
done

# 8. An explicit directory that is missing, or missing a file, refuses; nothing
#    is created and nothing is fetched.
run_setup box_missing_dir MLXFAST_TARGET_SNAPSHOT_DIR="${WORK}/no such box"
[[ "${rc}" == 1 && ! -d "${WORK}/no such box" ]]
grep -q 'target snapshot directory does not exist' "${WORK}/box_missing_dir.log"
rm -f "${BOX}/${FILES[4]}"
run_setup box_missing_file MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}"
[[ "${rc}" == 1 && "$(downloads box_missing_file)" == 0 && ! -e "${BOX}/${FILES[4]}" ]]
grep -q "pinned checkpoint file is missing: ${FILES[4]}" "${WORK}/box_missing_file.log"
head -c 10 "${SOURCE}/$(source_path "${FILES[4]}")" > "${BOX}/${FILES[4]}"
run_setup box_short_file MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}"
[[ "${rc}" == 1 && "$(downloads box_short_file)" == 0 && "$(wc -c < "${BOX}/${FILES[4]}" | tr -d ' ')" == 10 ]]
grep -q "byte-count mismatch for ${FILES[4]}" "${WORK}/box_short_file.log"

# 9. A stale lock on the default directory refuses rather than racing.
mkdir -p "${DEFAULT_DIR}/.setup-lock"
run_setup locked
[[ "${rc}" == 1 ]]
grep -q 'another setup is filling' "${WORK}/locked.log"
rmdir "${DEFAULT_DIR}/.setup-lock"

# 10. The download skip keeps its meaning: nothing fetched, nothing verified.
rm -rf "${DEFAULT_DIR}"
run_setup skipped MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
[[ "${rc}" == 0 ]]
[[ "$(downloads skipped)" == 0 && ! -d "${DEFAULT_DIR}" ]]
grep -q 'not verifying the checkpoint' "${WORK}/skipped.log"

# 11. A contract that pins a drafter (mtp_head.source "pinned"): step 2 also
#     fills and verifies it from its own pinned repository, under the same
#     rules; a staged MLXFAST_DRAFTER_DIR with a wrong digest refuses.
DSOURCE="${WORK}/drafter-source"
mkdir -p "${DSOURCE}"
dpins='[]'
for f in config.json model.safetensors; do
  head -c $((RANDOM + 500)) /dev/urandom > "${DSOURCE}/${f}"
  dpins="$(jq -c --arg p "${f}" --argjson b "$(wc -c < "${DSOURCE}/${f}" | tr -d ' ')" \
    --arg s "$(sha256_of "${DSOURCE}/${f}")" '. + [{path:$p, bytes:$b, sha256:$s}]' <<<"${dpins}")"
done
jq -n --argjson files "${pins}" --argjson dfiles "${dpins}" \
  '{target:{upstream_model_id:"example/model", upstream_revision:"0000000", files:$files},
    allowed_modes:["serial","mtp"],
    mtp_head:{source:"pinned", decoder:"dflash", upstream_model_id:"example/drafter", upstream_revision:"1111111",
              files:$dfiles, permitted_draft_depths:[1,2,3]}}' \
  > "${ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
DDEFAULT="${ROOT}/reference_weights/Muse-Glimmer-30B-assistant"
# 12. A SERIAL declaration (none, or spec disabled) needs no drafter: nothing is
#     required, hashed or fetched for it, even with no MLXFAST_DRAFTER_DIR and
#     an unreachable drafter source.
for decl in none disabled; do
  if [[ "${decl}" == disabled ]]; then
    printf '{"version":1,"spec":{"decoder":"dflash","enabled":false,"num_speculative_tokens":0}}\n' > "${ROOT}/mtp-head.manifest.json"
  fi
  run_setup "serial_${decl}" MLXFAST_DRAFTER_BASE_URL="file://${WORK}/nowhere"
  [[ "${rc}" == 0 ]] || { cat "${WORK}/serial_${decl}.log"; exit 1; }
  grep -q 'the declaration is serial: the pinned drafter is not needed, verified or fetched' "${WORK}/serial_${decl}.log"
  if grep -qE 'pinned drafter verified|verifying the pinned pinned drafter|MLXFAST_DRAFTER_DIR is unset' "${WORK}/serial_${decl}.log"; then
    cat "${WORK}/serial_${decl}.log"; exit 1
  fi
  [[ ! -e "${DDEFAULT}" ]]
done
# ...and a staged box that exports no drafter directory at all passes too.
cp "${SOURCE}/$(source_path "${FILES[4]}")" "${BOX}/${FILES[4]}"   # case 8 shortened it
run_setup serial_box MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}" MLXFAST_DRAFTER_DIR="${WORK}/no-such-drafter"
[[ "${rc}" == 0 && "$(downloads serial_box)" == 0 ]] || { cat "${WORK}/serial_box.log"; exit 1; }
# An invalid declaration is refused, never read as serial.
printf '{"version":1,"spec":{"enabled":true,"num_speculative_tokens":7}}\n' > "${ROOT}/mtp-head.manifest.json"
run_setup bad_decl
[[ "${rc}" == 1 ]]
grep -q 'not a contract-permitted draft depth' "${WORK}/bad_decl.log"
# 11. The declaration switches speculation on: the drafter is required.
printf '{"version":1,"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":2}}\n' > "${ROOT}/mtp-head.manifest.json"
run_setup drafter MLXFAST_DRAFTER_BASE_URL="file://${DSOURCE}"
[[ "${rc}" == 0 ]] || { cat "${WORK}/drafter.log"; exit 1; }
cmp -s "${DSOURCE}/model.safetensors" "${DDEFAULT}/model.safetensors"
grep -q 'MLXFAST_DRAFTER_DIR is unset; using' "${WORK}/drafter.log"
grep -q 'pinned drafter verified: 2 files by byte count and sha256' "${WORK}/drafter.log"
DBOX="${WORK}/box drafter"
mkdir -p "${DBOX}"
cp "${DSOURCE}/config.json" "${DBOX}/"
head -c "$(wc -c < "${DSOURCE}/model.safetensors" | tr -d ' ')" /dev/zero > "${DBOX}/model.safetensors"
cp "${SOURCE}/$(source_path "${FILES[4]}")" "${BOX}/${FILES[4]}"   # case 8 shortened it
run_setup drafter_box MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}" MLXFAST_DRAFTER_DIR="${DBOX}"
[[ "${rc}" == 1 && "$(downloads drafter_box)" == 0 ]]
grep -q 'sha256 mismatch for model.safetensors' "${WORK}/drafter_box.log"
# 13. A staged drafter directory holds EXACTLY the pinned files: the right
#     bytes pass, and the same directory with one unlisted file is refused.
DBOX2="${WORK}/box drafter exact"
mkdir -p "${DBOX2}"
cp "${DSOURCE}/config.json" "${DSOURCE}/model.safetensors" "${DBOX2}/"
run_setup drafter_exact MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}" MLXFAST_DRAFTER_DIR="${DBOX2}"
[[ "${rc}" == 0 && "$(downloads drafter_exact)" == 0 ]] || { cat "${WORK}/drafter_exact.log"; exit 1; }
grep -q 'holds exactly the 2 pinned drafter file(s)' "${WORK}/drafter_exact.log"
printf 'x' > "${DBOX2}/extra.safetensors"
run_setup drafter_extra MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}" MLXFAST_DRAFTER_DIR="${DBOX2}"
[[ "${rc}" == 1 ]] || { cat "${WORK}/drafter_extra.log"; exit 1; }
grep -q 'the contract does not pin: extra.safetensors' "${WORK}/drafter_extra.log"
write_fixture "${pins}"
printf '{"version":1}\n' > "${ROOT}/mtp-head.manifest.json"

echo 'test-setup-onboarding.sh: all 13 cases passed'
