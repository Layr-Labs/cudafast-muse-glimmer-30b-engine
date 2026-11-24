#!/usr/bin/env bash
# test-enforce-modifiable-surface.sh -- .github/scripts/enforce-modifiable-surface.sh
# refuses a dispatched commit that changes a file outside the protected base's
# editablePaths, and the ranked workflow runs it from that base.
#
# Asserts, on scratch git repositories (the repository is read only):
#   1. a change inside editablePaths passes, also with a benchd-bin/ directory
#      on disk (the identity arm must not match every path);
#   2. a change outside editablePaths fails;
#   3. a HEAD commit that widens its own benchmark.json is still judged by the
#      base's editablePaths;
#   4. a write to benchd-bin/ fails even when the base contract lists it;
#   5. a case variant of benchd-bin fails the same way (case-fold arm);
#   6. a hard link inside editablePaths to the box's benchd.pin file fails
#      (device:inode arm: same inode, other spelling);
#   7. a base contract with no editablePaths fails closed;
#   8. an empty diff (a commit on the protected branch) passes;
#   9. benchmark.yml's surface-check job fetches full history and runs the
#      gate and its git wrapper as the merge base holds them, before the lint;
#  10. a protected file RENAMED into editablePaths fails and names the removed
#      protected path (rename detection must not hide the deletion), also when
#      the submission's own .git/config asks for rename detection;
#  11. a gitlink committed inside editablePaths fails (also when the
#      submission's .git/config sets diff.ignoreSubmodules), and so does an
#      optional editable path committed as a symlink or as a directory; as a
#      regular file it passes.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${ROOT_DIR}/.github/scripts/enforce-modifiable-surface.sh"
WORKFLOW="${ROOT_DIR}/.github/workflows/benchmark.yml"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

g() { git -C "$1" -c user.name=t -c user.email=t@e -c commit.gpgsign=false "${@:2}"; }

# new_repo NAME EDITABLE_JSON -> a repo with one base commit; prints its path.
new_repo() {
  local dir="${WORK}/$1"
  mkdir -p "${dir}/harness" "${dir}/tools"
  printf '{"editablePaths": %s}\n' "$2" > "${dir}/benchmark.json"
  printf 'engine\n' > "${dir}/harness/engine.py"
  printf 'trusted\n' > "${dir}/tools/trusted.sh"
  g "${dir}" init -q -b main
  g "${dir}" add -A
  g "${dir}" commit -q -m base
  printf '%s\n' "${dir}"
}

# gate DIR BASE HEAD -> runs the gate in DIR; sets rc and out.
gate() {
  out="$(cd "$1" && BASE_SHA="$2" HEAD_SHA="$3" CONTRACT_PATH=benchmark.json "${GATE}" 2>&1)"
  rc=$?
}

commit_all() { g "$1" add -A && g "$1" commit -q -m "$2" && g "$1" rev-parse HEAD; }

# 1
d="$(new_repo inside '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
mkdir -p "${d}/benchd-bin"   # untracked, as on a box that staged the pair
printf 'faster\n' > "${d}/harness/engine.py"
head="$(commit_all "${d}" inside)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -eq 0 ]] && ok "case 1: a change inside editablePaths passes" \
  || fail "case 1: rc=${rc}: ${out}"

# 2
d="$(new_repo outside '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
printf 'attacker\n' > "${d}/tools/trusted.sh"
head="$(commit_all "${d}" outside)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "tools/trusted.sh is outside the modifiable surface" <<<"${out}" \
  && ok "case 2: a change outside editablePaths fails" \
  || fail "case 2: rc=${rc}: ${out}"

# 3
d="$(new_repo widen '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
printf '{"editablePaths": ["harness", "tools"]}\n' > "${d}/benchmark.json"
printf 'attacker\n' > "${d}/tools/trusted.sh"
head="$(commit_all "${d}" widen)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "benchmark.json is outside the modifiable surface" <<<"${out}" \
  && grep -q "tools/trusted.sh is outside the modifiable surface" <<<"${out}" \
  && ok "case 3: a self-widened benchmark.json does not grant access" \
  || fail "case 3: rc=${rc}: ${out}"

# 4
d="$(new_repo pin '["harness", "benchd-bin"]')"; base="$(g "${d}" rev-parse HEAD)"
mkdir -p "${d}/benchd-bin"; printf 'attacker scorer\n' > "${d}/benchd-bin/benchd"
head="$(commit_all "${d}" pin)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "benchd-bin/benchd reaches the measurement-harness surface" <<<"${out}" \
  && ok "case 4: a benchd-bin write fails even when the base lists it" \
  || fail "case 4: rc=${rc}: ${out}"

# 5 -- staged as a blob, so it exists on a case-insensitive checkout too.
d="$(new_repo fold '["harness", "BENCHD-BIN"]')"; base="$(g "${d}" rev-parse HEAD)"
blob="$(printf 'attacker scorer\n' | g "${d}" hash-object -w --stdin)"
g "${d}" update-index --add --cacheinfo "100644,${blob},BENCHD-BIN/benchd"
g "${d}" commit -q -m fold; head="$(g "${d}" rev-parse HEAD)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "BENCHD-BIN/benchd reaches the measurement-harness surface" <<<"${out}" \
  && ok "case 5: a case variant of benchd-bin fails" \
  || fail "case 5: rc=${rc}: ${out}"

# 6
d="$(new_repo identity '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
printf 'pin\n' > "${d}/benchd.pin"   # untracked: the protected file on disk
ln "${d}/benchd.pin" "${d}/harness/pin"
g "${d}" add harness/pin && g "${d}" commit -q -m identity; head="$(g "${d}" rev-parse HEAD)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "harness/pin reaches the measurement-harness surface" <<<"${out}" \
  && ! grep -q "file=benchd.pin::" <<<"${out}" \
  && ok "case 6: a hard link to benchd.pin fails" \
  || fail "case 6: rc=${rc}: ${out}"

# 7
d="$(new_repo empty '[]')"; base="$(g "${d}" rev-parse HEAD)"
printf 'faster\n' > "${d}/harness/engine.py"
head="$(commit_all "${d}" change)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "lists no usable editablePaths" <<<"${out}" \
  && ok "case 7: a base contract with no editablePaths fails closed" \
  || fail "case 7: rc=${rc}: ${out}"

# 8
d="$(new_repo same '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
gate "${d}" "${base}" "${base}"
[[ "${rc}" -eq 0 ]] && ok "case 8: an empty diff passes" || fail "case 8: rc=${rc}: ${out}"

# 10 -- git mv keeps the content, so a diff with rename detection reports only
# the new, editable name. The repo-local config is the submission's.
d="$(new_repo rename '["harness"]')"; base="$(g "${d}" rev-parse HEAD)"
g "${d}" mv tools/trusted.sh harness/trusted.sh
g "${d}" commit -q -m rename; head="$(g "${d}" rev-parse HEAD)"
g "${d}" config diff.renames copies
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "tools/trusted.sh is outside the modifiable surface" <<<"${out}" \
  && ok "case 10: a protected file renamed into editablePaths fails and names the removed path" \
  || fail "case 10: rc=${rc}: ${out}"

# 11 -- committed modes: the gate reads HEAD's tree, not only names.
d="$(new_repo modes '["harness", "decl.json"], "optionalEditablePaths": ["decl.json"]')"; base="$(g "${d}" rev-parse HEAD)"
g "${d}" update-index --add --cacheinfo "160000,${base},harness/sub"
g "${d}" commit -q -m gitlink; head="$(g "${d}" rev-parse HEAD)"
g "${d}" config diff.ignoreSubmodules all
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "harness/sub is a gitlink" <<<"${out}" \
  && ok "case 11: a gitlink inside editablePaths fails" || fail "case 11: gitlink: rc=${rc}: ${out}"
g "${d}" reset -q --hard "${base}"; ln -s harness/engine.py "${d}/decl.json"; head="$(commit_all "${d}" symlink)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "decl.json is an optional editable path and must be a regular file" <<<"${out}" \
  && ok "case 11: an optional editable path committed as a symlink fails" || fail "case 11: symlink: rc=${rc}: ${out}"
g "${d}" reset -q --hard "${base}"; mkdir "${d}/decl.json"; printf '{}\n' > "${d}/decl.json/x"; head="$(commit_all "${d}" dir)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -ne 0 ]] && grep -q "decl.json/x is under the optional editable path decl.json" <<<"${out}" \
  && ok "case 11: an optional editable path committed as a directory fails" || fail "case 11: directory: rc=${rc}: ${out}"
g "${d}" reset -q --hard "${base}"; rm -rf "${d}/decl.json"; printf '{}\n' > "${d}/decl.json"; head="$(commit_all "${d}" regular)"
gate "${d}" "${base}" "${head}"
[[ "${rc}" -eq 0 ]] && ok "case 11: the optional editable path as a regular file passes" || fail "case 11: regular: rc=${rc}: ${out}"

# 9 -- the wiring, read from the surface-check job only.
job="$(awk '/^  surface-check:/{f=1; print; next} f && /^  [a-z][a-z-]*:/{exit} f' "${WORKFLOW}")"
gate_line="$(grep -n 'enforce-modifiable-surface.sh"$' <<<"${job}" | head -1 | cut -d: -f1)"
lint_line="$(grep -n 'name: Benchmark manifest lint' <<<"${job}" | cut -d: -f1)"
if grep -q 'fetch-depth: 0' <<<"${job}" \
  && grep -qF 'base_sha="$(git merge-base "${base_ref}" "${GITHUB_SHA}")"' <<<"${job}" \
  && grep -qF 'git show "${base_sha}:.github/scripts/${script}"' <<<"${job}" \
  && grep -qF 'for script in enforce-modifiable-surface.sh hardened-git.sh; do' <<<"${job}" \
  && [[ -n "${gate_line}" && -n "${lint_line}" && "${gate_line}" -lt "${lint_line}" ]]; then
  ok "case 9: surface-check runs the base's gate on full history, before the lint"
else
  fail "case 9: the surface-check job does not run the gate from the merge base (gate line ${gate_line:-none}, lint line ${lint_line:-none})"
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "test-enforce-modifiable-surface: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-enforce-modifiable-surface: all cases passed"
