#!/usr/bin/env bash
# test-sandbox-names.sh -- tools/sandbox-names.sh, the per-runner sandbox names
# (docs/sandbox-design.md 6b). No root, no user is created: scratch runner homes.
#
# Asserts:
#   1. a single runner (actions-runner) resolves the section 6a names with no
#      setting, and check-unique passes;
#   2. a second runner dir derives its own names (actions-runner-dev ->
#      nemosub-dev, nemoshare-dev, ...-dev), and check-unique passes from
#      either runner's job (separate users);
#   3. the environment wins over the runner's .env, and the .env wins over the
#      dir-name default; the dir is found from RUNNER_WORKSPACE inside a job;
#   4. two runners that resolve the same sandbox user are refused, from
#      either side and across sibling homes, and the refusal names the name;
#   5. another runner whose .env cannot be read is refused (skipped as root);
#   6. an odd runner dir name still gives a valid user name.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="${ROOT_DIR}/tools/sandbox-names.sh"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'chmod -R u+rwX "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

H="${WORK}/home"
PROD="${H}/runbench/actions-runner"
DEV="${H}/runbench/actions-runner-dev"
mkdir -p "${PROD}/_work/cudafast/cudafast"
: > "${PROD}/.runner"

# names DIR [VAR=value...] -- run the tool as a job of the runner in DIR.
names() {
  local dir="$1"; shift
  env -u SANDBOX_USER -u SHARE_GROUP -u SANDBOX_ROOT -u SANDBOX_ETC -u SUDOERS_NAME -u EGRESS_UNIT -u ACL_UNIT \
    -u RUNNER_WORKSPACE -u GITHUB_WORKSPACE -u SANDBOX_RUNNER_HOME RUNNER_DIR="${dir}" "$@" "${TOOL}" "${CMD}" 2>&1
}
value() { grep "^$1=" <<<"$2" | cut -d= -f2-; }

# 1. One runner.
CMD="env"; out="$(names "${PROD}")"
want="SANDBOX_USER=nemosub
SHARE_GROUP=nemoshare
SANDBOX_ROOT=/var/lib/muse-glimmer/sandbox
SANDBOX_ETC=/etc/muse-glimmer-sandbox
SUDOERS_NAME=muse-glimmer-sandbox
EGRESS_UNIT=muse-glimmer-sandbox-egress.service
ACL_UNIT=muse-glimmer-sandbox-acl.service"
[[ "${out}" == "${want}" ]] && ok "case 1: a single runner resolves the section 6a names with no setting" \
  || fail "case 1: got ${out}"
CMD=check-unique; out="$(names "${PROD}")"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "no other runner on this box (0 found)" <<<"${out}" \
  && ok "case 1: check-unique passes for a single runner" || fail "case 1: rc=${rc}: ${out}"

# 2. A second runner of the same account.
mkdir -p "${DEV}/_work/cudafast/cudafast"
: > "${DEV}/.runner"
CMD="env"; out="$(names "${DEV}")"
t=0
for pair in SANDBOX_USER=nemosub-dev SHARE_GROUP=nemoshare-dev SANDBOX_ROOT=/var/lib/muse-glimmer/sandbox-dev \
    SANDBOX_ETC=/etc/muse-glimmer-sandbox-dev SUDOERS_NAME=muse-glimmer-sandbox-dev \
    EGRESS_UNIT=muse-glimmer-sandbox-dev-egress.service ACL_UNIT=muse-glimmer-sandbox-dev-acl.service; do
  grep -qxF "${pair}" <<<"${out}" || { fail "case 2: no ${pair} in: ${out}"; t=1; }
done
[[ "${t}" -eq 0 ]] && ok "case 2: actions-runner-dev derives its own seven names"
CMD=check-unique
for dir in "${PROD}" "${DEV}"; do
  out="$(names "${dir}")"; rc=$?
  [[ "${rc}" -eq 0 ]] && grep -q "(1 found)" <<<"${out}" \
    && ok "case 2: separate users pass from $(basename "${dir}")'s job" || fail "case 2 ($(basename "${dir}")): rc=${rc}: ${out}"
done

# 3. Precedence and the job's runner dir.
printf 'LANG=C.UTF-8\nSANDBOX_USER="devsub"\n' > "${DEV}/.env"
CMD="env"
[[ "$(value SANDBOX_USER "$(names "${DEV}")")" == devsub ]] \
  && ok "case 3: the runner's .env wins over the dir-name default" || fail "case 3: .env not read"
[[ "$(value SANDBOX_USER "$(names "${DEV}" SANDBOX_USER=envsub)")" == envsub ]] \
  && ok "case 3: the environment wins over the .env" || fail "case 3: environment not first"
out="$(env -u RUNNER_DIR -u SANDBOX_USER -u GITHUB_WORKSPACE RUNNER_WORKSPACE="${DEV}/_work/cudafast" "${TOOL}" env)"
[[ "$(value SANDBOX_USER "${out}")" == devsub ]] \
  && ok "case 3: inside a job the runner dir comes from RUNNER_WORKSPACE" || fail "case 3: got ${out}"
rm -f "${DEV}/.env"

# 4. A shared sandbox user is refused.
CMD=check-unique
printf 'SANDBOX_USER=nemosub\n' > "${DEV}/.env"
out="$(names "${PROD}")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "REFUSING -- runner ${DEV} also uses SANDBOX_USER=nemosub" <<<"${out}" \
  && ok "case 4: the prod job refuses a dev .env that names nemosub" || fail "case 4 (prod): rc=${rc}: ${out}"
out="$(names "${DEV}")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "REFUSING -- runner ${PROD} also uses SANDBOX_USER=nemosub" <<<"${out}" \
  && ok "case 4: the dev job refuses it too" || fail "case 4 (dev): rc=${rc}: ${out}"
out="$(names "${DEV}" SANDBOX_USER=nemosub)"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "also uses SANDBOX_USER=nemosub" <<<"${out}" \
  && ok "case 4: a job environment that names the other runner's user is refused" || fail "case 4 (env): rc=${rc}: ${out}"
rm -f "${DEV}/.env"
mkdir -p "${H}/otherbench/actions-runner"
: > "${H}/otherbench/actions-runner/.runner"
out="$(names "${PROD}")"; rc=$?
[[ "${rc}" -eq 1 ]] && grep -q "runner ${H}/otherbench/actions-runner also uses SANDBOX_USER=nemosub" <<<"${out}" \
  && ok "case 4: a runner in a sibling home with the same default names is refused" || fail "case 4 (sibling): rc=${rc}: ${out}"
rm -rf "${H}/otherbench"

# 5. An unreadable .env of another runner.
if [[ "$(id -u)" -ne 0 ]]; then
  : > "${DEV}/.env"; chmod 000 "${DEV}/.env"
  out="$(names "${PROD}")"; rc=$?
  [[ "${rc}" -eq 1 ]] && grep -q "runner ${DEV} has a .env this account cannot read" <<<"${out}" \
    && ok "case 5: another runner's unreadable .env is refused" || fail "case 5: rc=${rc}: ${out}"
  chmod 600 "${DEV}/.env"; rm -f "${DEV}/.env"
else
  echo "skip: case 5 (root reads every file)"
fi

# 6. An odd dir name.
CMD="env"
odd="${H}/runbench/My_Runner.2"
mkdir -p "${odd}"
u="$(value SANDBOX_USER "$(names "${odd}")")"
[[ "${u}" == nemosub-my-runner-2 && "${#u}" -le 32 ]] \
  && ok "case 6: an odd runner dir name gives a valid user name (${u})" || fail "case 6: got ${u}"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-sandbox-names: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-sandbox-names: all cases passed"
