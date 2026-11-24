#!/usr/bin/env bash
#
# drafter-dir-check.sh -- the pinned drafter directory holds EXACTLY the files
# the contract pins (fixture mtp_head.files), and nothing else.
#
# WHY. setup.sh verifies the bytes and the sha256 of every PINNED file, but a
# directory can carry more than the pinned files: an extra *.safetensors, a
# second config, a nested directory. tools/serve-up.sh mounts only the pinned
# files into a speculative leg, one read-only bind each, and this check refuses
# a directory with any unlisted entry before that, so a drafter directory can
# never be a place to stage weights the contract does not pin.
#
# A REFUSAL (exit 1, the entries named on stderr) for:
#   * a pinned file that is absent, a symlink, or not a regular file;
#   * any entry under DIR that is not a pinned file or a directory on the way
#     to one (a symlink, a socket, an unlisted file or directory, at any depth).
#
# --setup-default also admits what setup.sh itself writes into the drafter
# directory it fills on a participant's own machine: its verification marker,
# a top-level `.verified-<16 hex>` file. A ranked box stages the directory
# out of band and is checked without it.
#
# Usage: tools/drafter-dir-check.sh DIR [--setup-default]
#   DRAFTER_DIR_CHECK_CONTRACT   the contract (default: this tree's fixture)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${DRAFTER_DIR_CHECK_CONTRACT:-${REPO_ROOT}/fixtures/muse_glimmer_30b_cuda_v1_track.json}"

fail() {
  echo "drafter-dir-check.sh: REFUSING -- $*" >&2
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: drafter-dir-check.sh DIR [--setup-default]" >&2; exit 2; }
DIR="$1"
SETUP_DEFAULT=0
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--setup-default" ]] || { echo "usage: drafter-dir-check.sh DIR [--setup-default]" >&2; exit 2; }
  SETUP_DEFAULT=1
fi
command -v jq >/dev/null 2>&1 || fail "jq is required to read the contract"
[[ -f "${CONTRACT}" ]] || fail "the track contract is missing: ${CONTRACT}"
[[ -d "${DIR}" ]] || fail "the drafter directory does not exist: ${DIR}"
DIR="$(cd -- "${DIR}" && pwd -P)"

mapfile -t PINNED < <(jq -r '.mtp_head.files[]?.path' "${CONTRACT}")
[[ ${#PINNED[@]} -gt 0 ]] || fail "the contract pins no drafter files (mtp_head.files is empty)"

# The admitted names: every pinned file, and every directory on the way to one.
declare -A ADMITTED_FILE=() ADMITTED_DIR=()
for rel in "${PINNED[@]}"; do
  [[ -n "${rel}" && "${rel}" != /* ]] || fail "the contract pins a drafter file with an empty or absolute path: '${rel}'"
  case "/${rel}/" in *"/../"*|*"/./"*|*"//"*) fail "the contract pins a drafter file with a non-canonical path: '${rel}'" ;; esac
  ADMITTED_FILE["${rel}"]=1
  parent="${rel}"
  while [[ "${parent}" == */* ]]; do
    parent="${parent%/*}"
    ADMITTED_DIR["${parent}"]=1
  done
done

for rel in "${PINNED[@]}"; do
  f="${DIR}/${rel}"
  [[ ! -L "${f}" ]] || fail "the pinned drafter file ${rel} is a symlink in ${DIR}; only the pinned bytes themselves are mounted"
  [[ -f "${f}" ]] || fail "the drafter directory has no ${rel}: ${DIR} (run ./setup.sh)"
done

unlisted=()
while IFS= read -r -d '' entry; do
  rel="${entry#"${DIR}/"}"
  if [[ -L "${entry}" ]]; then
    unlisted+=("${rel} (symlink)")
  elif [[ -f "${entry}" && -n "${ADMITTED_FILE["${rel}"]+set}" ]]; then
    continue
  elif [[ -d "${entry}" && -n "${ADMITTED_DIR["${rel}"]+set}" ]]; then
    continue
  elif [[ "${SETUP_DEFAULT}" == 1 && -f "${entry}" && "${rel}" =~ ^\.verified-[0-9a-f]{16}$ ]]; then
    continue
  else
    unlisted+=("${rel}")
  fi
done < <(find "${DIR}" -mindepth 1 -print0)

if [[ ${#unlisted[@]} -gt 0 ]]; then
  shown="$(printf '%s, ' "${unlisted[@]:0:10}")"
  fail "the drafter directory ${DIR} holds ${#unlisted[@]} entr(ies) the contract does not pin: ${shown%, }$([[ ${#unlisted[@]} -gt 10 ]] && echo ", ..."); it must hold exactly mtp_head.files ($(printf '%s ' "${PINNED[@]}" | sed 's/ $//')) and nothing else"
fi
echo "drafter-dir-check.sh: ${DIR} holds exactly the ${#PINNED[@]} pinned drafter file(s)"
