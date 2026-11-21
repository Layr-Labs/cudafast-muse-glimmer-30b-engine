#!/usr/bin/env bash
# Rejects diffs that touch files outside the benchmark contract's editablePaths.
# The allowlist is read from the BASE commit so a PR cannot grant itself access.
# Usage: BASE_SHA=<sha> HEAD_SHA=<sha> [CONTRACT_PATH=<contract>] enforce-modifiable-surface.sh
#
# Re-implementation of Layr-Labs/qwen-3.8-mtp-challenge@bfab0de
# .github/scripts/enforce-modifiable-surface.sh:1-77, unchanged in behaviour,
# taken from the Muse Glimmer 30B MLX engine's copy of the same gate. This
# copy changes five things for this track: the comments name this
# repository's two other readers of the editable surface (the manifest linter
# and submission-static-review-checks.sh), the identity probe tries GNU
# `stat -c` before BSD `stat -f` (on Linux `stat -f` is a FILESYSTEM query: it
# prints filesystem text before it fails, so the fallback's identity carries
# that text and never matches), the case-fold note names the Linux box, and
# the diff runs with --no-renames -z (a rename cannot hide a protected path),
# and the committed mode of each changed path is checked (COMMITTED MODES).
set -euo pipefail

: "${BASE_SHA:?BASE_SHA is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"

# Which contract's editablePaths are in force. Deliberately the SAME variable
# name submission-static-review-checks.sh already reads, so the gates that consume an editable surface cannot disagree about
# which contract a submission is judged against.
CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

# This runs with the UNTRUSTED submission checkout as the working directory,
# where repo-local .git/config settings (core.fsmonitor, core.hooksPath,
# core.pager, filters) are attacker-influenced on submission branches. Every
# git read goes through hardened-git.sh -- resolved next to THIS script, i.e.
# the trusted checkout's copy, never one inside the submission worktree.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HARDENED_GIT="${SCRIPT_DIR}/hardened-git.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read editablePaths from ${CONTRACT_PATH}"
  exit 1
fi

# Read the contract from the BASE commit, never from the submission work tree:
# a candidate that could point this gate at its own copy of a contract could
# widen its own surface just by editing one.
#
# Fail closed on a contract that cannot be read or carries no editablePaths.
# Both are configuration errors, and neither may degrade into a pass or into a
# rejection that names the wrong contract. There is deliberately no fallback to
# benchmark.json: silently judging a submission against another track's surface
# would reject it with a message pointing at a contract it was never submitted
# under.
if ! contract_source="$("${HARDENED_GIT}" show "${BASE_SHA}:${CONTRACT_PATH}")"; then
  echo "::error::cannot read ${CONTRACT_PATH} from base commit ${BASE_SHA}"
  exit 1
fi
# A contract with no editablePaths key (or a non-list value) makes jq fail; a
# contract with an empty list makes jq succeed with no output. Both mean the
# same thing here -- this gate has no allowlist -- and both must be loud. An
# empty allowlist left to run would reject every changed file while naming the
# wrong reason, and would pass silently whenever the diff is also empty.
if ! allowed="$(jq -r '.editablePaths[]' <<<"${contract_source}")" \
  || [[ -z "${allowed//[[:space:]]/}" ]]; then
  echo "::error::${CONTRACT_PATH} at base commit ${BASE_SHA} lists no usable editablePaths"
  exit 1
fi
# --no-renames: with rename detection (git's default, and diff.renames in the
# submission's own repo-local config) --name-only reports a renamed file by its
# NEW name only, so a protected file renamed into an editable path would hide
# its own removal. Every changed path is listed on both sides of the diff.
# -z: paths are NUL-delimited and RAW, never C-quoted (`".gitmodule\305\277"`)
# and never split at a newline. The device:inode arm below stats each changed
# path; a quoted literal names no file, so a quoted form would leave the
# identity arm inert against exactly the non-ASCII spellings it exists to catch
# -- e.g. `.gitmoduleſ` (U+017F LONG S), which APFS case-folds to `.gitmodules`
# (same inode) but ASCII `tr` does not. The flags are set at THIS call site,
# not in hardened-git.sh (kept byte-faithful to the original). The list goes
# through a file, so a git failure stops the gate (set -e) instead of
# vanishing inside a process substitution. --ignore-submodules=none: a
# repo-local diff.ignoreSubmodules would otherwise hide a gitlink.
changed_file="$(mktemp)"; raw_file="$(mktemp)"
trap 'rm -f -- "${changed_file}" "${raw_file}"' EXIT
"${HARDENED_GIT}" diff --no-renames --ignore-submodules=none --name-only -z "${BASE_SHA}" "${HEAD_SHA}" > "${changed_file}"

# ADDITION (pinned measurement harness), the same rule the manifest linter
# asserts (FORBIDDEN_EDITABLE in tools/lint-benchmark-manifest.py): a diff reaching what decides which
# scorer runs is refused HERE too -- independently of what the base contract's
# editablePaths happen to say. Without this the gate is the one layer of the
# three that would admit the write if the trusted contract ever drifted.
#
# WHAT THE PROTECTED THING IS NOW. benchd used to be a SHA-pinned source
# submodule at `benchd`, pointed at by `.gitmodules`. It is now a pinned
# PREBUILT binary resolved from the dist channel (the `benchd.pin` sha pin is
# RETIRED: tools/fetch-benchd.sh carries the pin itself -- but the spelling stays FORBIDDEN below so
# a submission can never plant a pin file and confuse tooling): historically it named {branch, commit, sha256, bytes} and
# tools/fetch-benchd.sh resolves it into `benchd-bin/`. So the pin file and the
# resolved-binary directory are the live entries -- writing the pin repoints the
# scorer, writing benchd-bin swaps the bytes after verification. The two
# submodule spellings are KEPT rather than dropped: they cost nothing and a
# reintroduced gitlink must be covered on arrival, not after someone remembers.
#
# Two-armed, mirroring lint-benchmark-manifest.py, so the layers that read this
# editable surface cannot diverge about which spellings reach the pinned
# scorer.
#
#   (1) CASE FOLD. The ranked box is Linux (case-sensitive), but a checkout on a
#       case-INSENSITIVE filesystem (macOS APFS by default) makes `BENCHD.PIN`
#       and `benchd.pin` the same file, and a byte-comparison guard would stop
#       only one spelling of the same write.
#   (2) FILESYSTEM IDENTITY (device:inode). A changed path whose own prefix
#       RESOLVES to the protected inode is refused even when ASCII case folding
#       does not normalise its spelling (Unicode case folding, HFS+/APFS
#       normalisation). The linter refuses the same spellings by
#       str.casefold(), so under trusted-contract drift a spelling the
#       filesystem resolves to the protected path cannot pass HERE while the
#       linter refuses it.
FORBIDDEN_SURFACE_PATHS=("benchd" ".gitmodules" "benchd.pin" "benchd-bin")
fold_case() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Filesystem identity of an existing path, as device:inode. Empty (status 1)
# when the path does not exist. GNU `stat -c` first, BSD `stat -f` as the
# fallback: GNU `stat -f` is a FILESYSTEM query that prints filesystem text to
# stdout before it fails, which spoils the identity, and BSD stat refuses `-c`. Not a git read -- `stat` ignores the
# attacker-influenced repo-local git config the header warns about.
path_identity() {
  local path="$1"
  [[ -e "${path}" ]] || return 1
  stat -c '%d:%i' "${path}" 2>/dev/null || stat -f '%d:%i' "${path}" 2>/dev/null
}

reaches_forbidden_path() {
  local path="$1" forbidden folded_path folded_forbidden forbidden_identity prefix identity
  folded_path="$(fold_case "${path}")"
  for forbidden in "${FORBIDDEN_SURFACE_PATHS[@]}"; do
    folded_forbidden="$(fold_case "${forbidden}")"
    if [[ "${folded_path}" == "${folded_forbidden}" \
       || "${folded_path}" == "${folded_forbidden}/"* \
       || "${folded_forbidden}" == "${folded_path}/"* ]]; then
      return 0
    fi
    forbidden_identity="$(path_identity "${forbidden}")" || continue
    # Walk the changed path's own prefixes: `BENCHD-BIN/benchd` reaches the
    # scorer because `BENCHD-BIN` resolves to the pinned-binary directory even
    # though the full path may not exist on disk.
    prefix="${path}"
    while [[ -n "${prefix}" && "${prefix}" != "." && "${prefix}" != "/" ]]; do
      if identity="$(path_identity "${prefix}")" && [[ "${identity}" == "${forbidden_identity}" ]]; then
        return 0
      fi
      prefix="$(dirname -- "${prefix}")"
    done
  done
  return 1
}

bad=0
while IFS= read -r -d '' f; do
  [[ -z "${f}" ]] && continue
  if reaches_forbidden_path "${f}"; then
    echo "::error file=${f}::${f} reaches the measurement-harness surface (benchd-bin, or the retired benchd.pin spelling) or a retired submodule spelling; a submission must never be able to change what scores it"
    bad=1
    continue
  fi
  ok=0
  while IFS= read -r allowed_path; do
    [[ -z "${allowed_path}" ]] && continue
    # Exact match OR file is inside an allowed directory prefix.
    if [[ "${f}" == "${allowed_path}" || "${f}" == "${allowed_path}/"* ]]; then
      ok=1
      break
    fi
  done <<<"${allowed}"
  if [[ "${ok}" == "0" ]]; then
    echo "::error file=${f}::${f} is outside the modifiable surface (see editablePaths in ${CONTRACT_PATH})"
    bad=1
  fi
done < "${changed_file}"

# COMMITTED MODES. The checks above read names only, and the ranked review reads
# the checkout, where an uninitialized gitlink looks like an empty directory.
# So the mode each changed path has in HEAD's tree is checked too: a gitlink
# (160000) is refused anywhere, and an optional editable path
# (optionalEditablePaths) that is present must be a regular file, never a
# directory (a path under it).
optional="$(jq -r '(.optionalEditablePaths // [])[]' <<<"${contract_source}")"
"${HARDENED_GIT}" diff --no-renames --ignore-submodules=none --raw -z "${BASE_SHA}" "${HEAD_SHA}" > "${raw_file}"
# --raw -z: ":<old mode> <new mode> <old sha> <new sha> <status>" NUL <path> NUL.
while IFS= read -r -d '' meta && IFS= read -r -d '' f; do
  mode="$(cut -d' ' -f2 <<<"${meta}")"
  [[ "${mode}" == 000000 ]] && continue
  if [[ "${mode}" == 160000 ]]; then
    echo "::error file=${f}::${f} is a gitlink (submodule); the editable surface takes files only"
    bad=1
  elif grep -qxF -- "${f}" <<<"${optional}" && [[ "${mode}" != 100644 && "${mode}" != 100755 ]]; then
    echo "::error file=${f}::${f} is an optional editable path and must be a regular file (mode ${mode})"
    bad=1
  fi
  while IFS= read -r o; do
    [[ -n "${o}" && "${f}" == "${o}/"* ]] || continue
    echo "::error file=${f}::${f} is under the optional editable path ${o}, which must be a regular file"
    bad=1
  done <<<"${optional}"
done < "${raw_file}"
exit "${bad}"
