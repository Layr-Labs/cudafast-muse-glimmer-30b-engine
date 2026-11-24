#!/usr/bin/env bash
# setup.sh -- build the engine and verify the pinned target snapshot and drafter.
#
# benchmark.json's setupCommand runs `./tools/fetch-benchd.sh && ./setup.sh`;
# benchmark.yml calls it on the ranked box before the GPU-locked measurement.
#
# Steps:
#   1. tools/engine/build.sh -- the protocol adapter (staged for benchd by
#      tools/stage-cuda-engine.sh), the pinned engine container by digest, the
#      vLLM overlay (the container's compiled vLLM with vllm/vllm/ on top) and
#      the optional kernels/. Needs cargo, docker and jq.
#   2. verify the pinned target snapshot against the pins in
#      fixtures/muse_glimmer_30b_cuda_v1_track.json: every safetensors
#      shard, the index, the configs and the tokenizer, byte count first, then
#      sha256. With MLXFAST_TARGET_SNAPSHOT_DIR set (a ranked box,
#      organizer-staged) the directory and every file must already be there:
#      nothing is fetched, nothing is deleted, and every run hashes the set.
#      With it UNSET (a participant's own machine) the snapshot lives under
#      this checkout, setup downloads each missing file from the pinned public
#      model repository at the pinned revision (resumable, kept only when both
#      pins match), and a verified set leaves a marker keyed on the pins and
#      each file's size, mtime and inode, so a rerun there reads no shard.
#      The same step, under the same rules, then verifies the pinned DFlash
#      drafter (fixture mtp_head.files: config.json and model.safetensors of
#      mtp_head.upstream_model_id at mtp_head.upstream_revision) when the
#      contract's mtp_head.source is "pinned" AND the declaration switches
#      speculation on (tools/spec-declaration.sh speculative = 1); a serial
#      entry skips it. On a box it is MLXFAST_DRAFTER_DIR; on a participant's
#      machine reference_weights/Muse-Glimmer-30B-assistant. Either directory
#      must hold exactly the pinned files (tools/drafter-dir-check.sh).
#
# Environment:
#   MLXFAST_TARGET_SNAPSHOT_DIR   the target snapshot. Set: verify only. Unset:
#                                 reference_weights/Muse-Glimmer-30B-NVFP4
#                                 under this checkout, which setup fills.
#   MLXFAST_TARGET_BASE_URL       where a missing file is fetched from on the
#                                 default path. Default: the fixture's
#                                 upstream_model_id at upstream_revision on
#                                 huggingface.co. A file:// base works.
#   MLXFAST_DRAFTER_DIR           the pinned drafter. Set: verify only. Unset:
#                                 reference_weights/Muse-Glimmer-30B-assistant
#                                 under this checkout, which setup fills.
#   MLXFAST_DRAFTER_BASE_URL      the drafter's MLXFAST_TARGET_BASE_URL.
#   MLXFAST_SKIP_ENGINE_BUILD=1   skip step 1 (a pre-built engine is staged)
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 / SKIP_MODEL_DOWNLOAD=1
#                                 skip step 2 (no scored run is possible)
#   MLXFAST_SKIP_WEIGHTS_SHA256=1 verify byte counts only
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'setup.sh: %s\n' "$*"; }
die() { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }

CONTRACT="${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json"
command -v jq >/dev/null 2>&1 || die "jq is required"

# The public model repository the target was pinned from, at that revision
# (the fixture's upstream_model_id / upstream_revision). Its files are flat at
# the repository root and the snapshot keeps them flat.
DEFAULT_SNAPSHOT_DIR="${ROOT_DIR}/reference_weights/Muse-Glimmer-30B-NVFP4"

# The pinned DFlash drafter (fixture mtp_head), from its own public repository
# at its pinned revision, kept flat the same way.
DEFAULT_DRAFTER_DIR="${ROOT_DIR}/reference_weights/Muse-Glimmer-30B-assistant"

resolve_pinned_source() {
  # resolve_pinned_source BASE_URL_OVERRIDE -- the base a missing file of the
  # current set (SET_SECTION) is fetched from. Read only when the default path
  # may fetch.
  local repo rev
  repo="$(jq -r "${SET_SECTION}.upstream_model_id // empty" "${CONTRACT}")"
  rev="$(jq -r "${SET_SECTION}.upstream_revision // empty" "${CONTRACT}")"
  [[ -n "${repo}" && -n "${rev}" ]] \
    || die "the contract names no ${SET_SECTION}.upstream_model_id / upstream_revision; cannot fetch a missing file"
  SET_BASE_URL="${1:-https://huggingface.co/${repo}/resolve/${rev}}"
}

source_path_for() {
  # The repository path of one pinned file: flat at the root.
  printf '%s\n' "$1"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

file_identity() {
  # size, mtime and inode: what the verified marker records per file.
  stat -c '%s %Y %i' "$1" 2>/dev/null || stat -f '%z %m %i' "$1"
}

download_pinned_file() {
  # download_pinned_file REL WANT_BYTES WANT_SHA -- fetch one pinned file into
  # the snapshot, resumable, and keep it only when both pins match.
  local rel="$1" want_bytes="$2" want_sha="$3"
  local dest="${SNAPSHOT_DIR}/${rel}" partial="${SNAPSHOT_DIR}/${rel}.partial"
  local url got_bytes got_sha
  url="${SET_BASE_URL}/$(source_path_for "${rel}")"
  command -v curl >/dev/null 2>&1 || die "curl is required to download ${rel}"
  log "downloading ${rel} (${want_bytes} bytes) from ${url}"
  curl -fL --retry 5 --retry-delay 5 -C - -o "${partial}" "${url}" \
    || die "download failed for ${rel}; rerun ./setup.sh to resume from ${partial}"
  got_bytes="$(wc -c < "${partial}" | tr -d '[:space:]')"
  if [[ "${got_bytes}" != "${want_bytes}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} is ${got_bytes} bytes, pinned ${want_bytes}; discarded"
  fi
  got_sha="$(sha256_of "${partial}")"
  if [[ "${got_sha}" != "${want_sha}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} has sha256 ${got_sha}, pinned ${want_sha}; discarded"
  fi
  mv -f "${partial}" "${dest}"
  log "downloaded and verified ${rel}"
}

snapshot_marker_path() {
  # Keyed on the pins themselves, so a fixture change invalidates every marker.
  local key
  key="$(jq -c "${SET_SECTION}.files" "${CONTRACT}" | sha256_of /dev/stdin | cut -c 1-16)"
  printf '%s/.verified-%s\n' "${SNAPSHOT_DIR}" "${key}"
}

snapshot_identity() {
  # One line per pinned file: path, size, mtime, inode.
  local rel
  while IFS=$'\t' read -r rel _ _; do
    [[ -n "${rel}" ]] || continue
    [[ -f "${SNAPSHOT_DIR}/${rel}" ]] || return 1
    printf '%s %s\n' "${rel}" "$(file_identity "${SNAPSHOT_DIR}/${rel}")"
  done <<< "${pins}"
}

# --- 1. engine build --------------------------------------------------------
if [[ "${MLXFAST_SKIP_ENGINE_BUILD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_ENGINE_BUILD=1: not building the engine; a pre-built engine must be staged"
  [[ -x "${ROOT_DIR}/.build/release/mlxfast-runtime-worker" ]] \
    || die "no staged engine at .build/release/mlxfast-runtime-worker"
  [[ -d "${ROOT_DIR}/.build/vllm-overlay/vllm" ]] \
    || die "no staged vLLM overlay at .build/vllm-overlay"
else
  log "building the engine (tools/engine/build.sh)"
  "${ROOT_DIR}/tools/engine/build.sh"
fi

# --- 2. pinned weight sets: the target snapshot and the drafter ---------------
# stage_pinned_set LABEL SECTION STAGED_DIR DEFAULT_DIR BASE_URL_OVERRIDE ENV_NAME
#   One pinned set of files, the fixture's SECTION.files ({path, bytes, sha256}).
#   STAGED_DIR set (a ranked box, organizer-staged): verify only. Empty (a
#   participant's own machine): DEFAULT_DIR, filled from the SECTION's pinned
#   repository at its pinned revision. The same rules hold for both sets.
stage_pinned_set() {
  local label="$1" staged="$3" default_dir="$4" base_override="$5" env_name="$6"
  local local_snapshot marker identity verified_by_marker count verify_dir jobs_max running verified
  local rel want_bytes want_sha f got_bytes r status got want setup_lock=""
  SET_SECTION="$2"
  SNAPSHOT_DIR="${staged}"
  pins="$(jq -r "${SET_SECTION}.files[] | [.path, (.bytes|tostring), .sha256] | @tsv" "${CONTRACT}")"
  [[ -n "${pins}" ]] || die "the contract pins no ${label} files (${SET_SECTION}.files is empty)"
  if [[ -n "${SNAPSHOT_DIR}" ]]; then
    # A staged set: present, complete, and hashed on every run. The ranked
    # path never fetches, deletes or short-cuts organizer material.
    local_snapshot=0
    [[ -d "${SNAPSHOT_DIR}" ]] || die "the ${label} directory does not exist: ${SNAPSHOT_DIR}"
  else
    local_snapshot=1
    SNAPSHOT_DIR="${default_dir}"
    log "${env_name} is unset; using ${SNAPSHOT_DIR}"
    mkdir -p "${SNAPSHOT_DIR}" || die "cannot create the ${label} directory: ${SNAPSHOT_DIR}"
    # One setup at a time fills the directory: two would append into the same
    # partial file.
    setup_lock="${SNAPSHOT_DIR}/.setup-lock"
    mkdir "${setup_lock}" 2>/dev/null \
      || die "another setup is filling ${SNAPSHOT_DIR}, or a previous one left ${setup_lock}; remove it when no setup is running"
    # shellcheck disable=SC2064 # the lock of THIS set, expanded now
    trap "rmdir '${setup_lock}' 2>/dev/null || true" EXIT
    resolve_pinned_source "${base_override}"
    # A file that is absent, or the wrong size, is fetched. A present file of
    # the right size is left for the digest pass below to judge.
    while IFS=$'\t' read -r rel want_bytes want_sha; do
      [[ -n "${rel}" ]] || continue
      f="${SNAPSHOT_DIR}/${rel}"
      if [[ -f "${f}" ]]; then
        got_bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
        [[ "${got_bytes}" == "${want_bytes}" ]] && continue
        log "${rel} is ${got_bytes} bytes, pinned ${want_bytes}; fetching it again"
        rm -f "${f}"
      fi
      download_pinned_file "${rel}" "${want_bytes}" "${want_sha}"
    done <<< "${pins}"
  fi

  # On the default path, a rerun on a machine that already holds the verified
  # set reads no shard: the marker written after a full pass records each
  # file's size, mtime and inode, and a set that still matches it is the set
  # that was verified. A staged set takes no marker and hashes every run.
  marker="$(snapshot_marker_path)"
  if [[ "${local_snapshot}" == "1" && "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && -f "${marker}" ]] \
      && identity="$(snapshot_identity)" && [[ "${identity}" == "$(cat "${marker}")" ]]; then
    log "${label} at ${SNAPSHOT_DIR} verified by marker: $(wc -l < "${marker}" | tr -d '[:space:]') pinned files unchanged since the last full verification"
    verified_by_marker=1
  else
    verified_by_marker=0
  fi

  log "verifying the pinned ${label} at ${SNAPSHOT_DIR} against fixtures/muse_glimmer_30b_cuda_v1_track.json ${SET_SECTION}.files"
  # Byte counts first (cheap, names a truncated file), then sha256 of every
  # file. The files are hashed CONCURRENTLY, one process per file, bounded by
  # MLXFAST_VERIFY_JOBS (default 8): the same digests, compared against the
  # same pins, in a fraction of the wall-clock. Nothing is skipped.
  count=0
  while IFS=$'\t' read -r rel want_bytes want_sha; do
    [[ -n "${rel}" ]] || continue
    f="${SNAPSHOT_DIR}/${rel}"
    [[ -f "${f}" ]] || die "pinned checkpoint file is missing: ${rel} (${label})"
    got_bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
    [[ "${got_bytes}" == "${want_bytes}" ]] \
      || die "byte-count mismatch for ${rel}: staged ${got_bytes}, pinned ${want_bytes}"
    count=$((count + 1))
  done <<< "${pins}"

  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && "${verified_by_marker}" != "1" ]]; then
    verify_dir="$(mktemp -d)"
    jobs_max="${MLXFAST_VERIFY_JOBS:-8}"
    running=0
    while IFS=$'\t' read -r rel want_bytes want_sha; do
      [[ -n "${rel}" ]] || continue
      (
        got_sha="$(sha256_of "${SNAPSHOT_DIR}/${rel}")"
        if [[ "${got_sha}" == "${want_sha}" ]]; then
          printf 'ok\t%s\n' "${rel}"
        else
          printf 'mismatch\t%s\t%s\t%s\n' "${rel}" "${got_sha}" "${want_sha}"
        fi
      ) > "${verify_dir}/$(printf '%s' "${rel}" | tr '/' '_').result" &
      running=$((running + 1))
      if [[ "${running}" -ge "${jobs_max}" ]]; then
        wait -n
        running=$((running - 1))
      fi
    done <<< "${pins}"
    wait
    verified=0
    for r in "${verify_dir}"/*.result; do
      IFS=$'\t' read -r status rel got want < "${r}"
      if [[ "${status}" != "ok" ]]; then
        rm -rf "${verify_dir}"
        die "sha256 mismatch for ${rel}: staged ${got}, pinned ${want}"
      fi
      verified=$((verified + 1))
    done
    rm -rf "${verify_dir}"
    [[ "${verified}" == "${count}" ]] || die "sha256 verified ${verified} of ${count} pinned files"
    if [[ "${local_snapshot}" == "1" ]]; then
      # The marker is a rerun convenience on the default path, never a gate.
      identity="$(snapshot_identity)"
      printf '%s\n' "${identity}" > "${marker}.tmp.$$"
      mv -f "${marker}.tmp.$$" "${marker}"
      log "wrote the verification marker $(basename "${marker}")"
    fi
  fi

  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" == "1" ]]; then
    log "${label} verified: ${count} files by byte count (sha256 skipped by MLXFAST_SKIP_WEIGHTS_SHA256=1)"
  elif [[ "${verified_by_marker}" == "1" ]]; then
    log "${label} verified: ${count} files by byte count and the marker of the last full sha256 pass"
  else
    log "${label} verified: ${count} files by byte count and sha256"
  fi
  if [[ -n "${setup_lock}" ]]; then
    rmdir "${setup_lock}" 2>/dev/null || true
    trap - EXIT
  fi
  log "${label} ready at ${SNAPSHOT_DIR}"
}

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_WEIGHTS_DOWNLOAD/SKIP_MODEL_DOWNLOAD set: not verifying the checkpoint or the drafter (no scored run is possible)"
else
  stage_pinned_set "target snapshot" .target "${MLXFAST_TARGET_SNAPSHOT_DIR:-}" \
    "${DEFAULT_SNAPSHOT_DIR}" "${MLXFAST_TARGET_BASE_URL:-}" MLXFAST_TARGET_SNAPSHOT_DIR
  log "local runs: tools/local-baseline.sh"
  # The pinned DFlash drafter a speculative leg mounts (tools/serve-up.sh). The
  # contract names it in mtp_head; a track with no pinned drafter skips this.
  # It is needed ONLY when this tree's declaration switches speculation on
  # (tools/spec-declaration.sh speculative = 1, which refuses an invalid
  # declaration): a serial entry neither requires, nor hashes, nor downloads it.
  # The directory must hold exactly the pinned files (tools/drafter-dir-check.sh).
  if [[ "$(jq -r '.mtp_head.source // empty' "${CONTRACT}")" == "pinned" ]]; then
    drafter_spec="$("${ROOT_DIR}/tools/spec-declaration.sh" speculative)" \
      || die "the speculative-decode declaration (mtp-head.manifest.json) is invalid; tools/spec-declaration.sh names why above"
    if [[ "${drafter_spec}" == "1" ]]; then
      stage_pinned_set "pinned drafter" .mtp_head "${MLXFAST_DRAFTER_DIR:-}" \
        "${DEFAULT_DRAFTER_DIR}" "${MLXFAST_DRAFTER_BASE_URL:-}" MLXFAST_DRAFTER_DIR
      if [[ -n "${MLXFAST_DRAFTER_DIR:-}" ]]; then
        "${ROOT_DIR}/tools/drafter-dir-check.sh" "${MLXFAST_DRAFTER_DIR}" \
          || die "the staged drafter directory holds more than the pinned files (above)"
      else
        "${ROOT_DIR}/tools/drafter-dir-check.sh" "${DEFAULT_DRAFTER_DIR}" --setup-default \
          || die "the drafter directory holds more than the pinned files (above)"
      fi
    else
      log "the declaration is serial: the pinned drafter is not needed, verified or fetched"
    fi
  fi
fi

log "setup complete: engine staged for benchd and the pinned weight sets verified per the steps above"
