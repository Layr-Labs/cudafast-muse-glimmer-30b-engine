#!/usr/bin/env bash
# test-sandbox-grant-checkout.sh -- tools/sandbox-grant-checkout.sh, the ACL
# walk of the ranked workflow's "Sandbox start" step (docs/sandbox-design.md 6a).
#
# Asserts, with setfacl stubbed (it logs; no ACL is set) on a scratch runner
# home that holds two runners of one account, as a dev and a prod runner do:
#   1. the checkout gets rX (and the default ACL), each parent up to the home
#      gets --x, and the path itself is never denied;
#   2. every entry beside the path is denied (.env, _diag, ~/.cache, a stray
#      directory), and so are RUNNER_TEMP and _work's _actions and _tool;
#   3. the OTHER runner's install (a directory with a .runner file) is denied
#      to this job's sandbox user like any other entry, from either runner's
#      job, and no ACL names any other user: each runner has its own sandbox
#      user (tools/sandbox-names.sh), so the other runner's jobs keep theirs;
#   4. a checkout outside the runner home is refused before any ACL is set.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Physical path: the walk compares realpath results (macOS /var is a symlink).
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

H="${WORK}/home/runbench"
mkdir -p "${WORK}/bin" "${H}/.cache" "${H}/stray"
for r in actions-runner actions-runner-dev; do
  mkdir -p "${H}/${r}/_diag" "${H}/${r}/_work/_actions" "${H}/${r}/_work/_tool" "${H}/${r}/_work/_temp" \
    "${H}/${r}/_work/cudafast/cudafast/tools"
  : > "${H}/${r}/.runner"
  : > "${H}/${r}/.env"
done
cat > "${WORK}/bin/setfacl" <<STUB
#!/usr/bin/env bash
echo "setfacl \$*" >> "${WORK}/acl.log"
STUB
chmod +x "${WORK}/bin/setfacl"

grant() { # grant RUNNER [HOME] -- run the walk as that runner's job
  local r="$1" home="${2:-${H}}"
  : > "${WORK}/acl.log"
  env PATH="${WORK}/bin:${PATH}" HOME="${home}" SANDBOX_USER=sbx \
    GITHUB_WORKSPACE="${H}/${r}/_work/cudafast/cudafast" RUNNER_WORKSPACE="${H}/${r}/_work/cudafast" \
    RUNNER_TEMP="${H}/${r}/_work/_temp" "${ROOT_DIR}/tools/sandbox-grant-checkout.sh" > "${WORK}/out" 2>&1
}
has()  { grep -qxF "setfacl $1" "${WORK}/acl.log"; }
deny() { has "-m u:sbx:--- $1"; }

for pair in "actions-runner-dev actions-runner" "actions-runner actions-runner-dev"; do
  read -r me other <<<"${pair}"
  grant "${me}"; rc=$?
  [[ "${rc}" -eq 0 ]] || fail "${me}: the walk exited ${rc}: $(cat "${WORK}/out")"
  ws="${H}/${me}/_work/cudafast/cudafast"
  has "-R -P -m u:sbx:rX ${ws}" && has "-R -P -d -m u:sbx:rX ${ws}" \
    && ok "case 1 (${me}): the checkout gets rX and the default rX" || fail "case 1 (${me}): no rX on ${ws}"
  t=0; for d in "${H}" "${H}/${me}" "${H}/${me}/_work" "${H}/${me}/_work/cudafast"; do
    has "-m u:sbx:--x ${d}" || { fail "case 1 (${me}): no traverse on ${d}"; t=1; }
    deny "${d}" && { fail "case 1 (${me}): the path ${d} was denied"; t=1; }
  done
  [[ "${t}" -eq 0 ]] && ok "case 1 (${me}): traverse on each parent, no deny on the path"
  t=0; for d in "${H}/.cache" "${H}/stray" "${H}/${me}/.env" "${H}/${me}/.runner" "${H}/${me}/_diag" \
      "${H}/${me}/_work/_temp" "${H}/${me}/_work/_actions" "${H}/${me}/_work/_tool"; do
    deny "${d}" || { fail "case 2 (${me}): ${d} was not denied"; t=1; }
  done
  [[ "${t}" -eq 0 ]] && ok "case 2 (${me}): every entry beside the path, RUNNER_TEMP, _actions and _tool are denied"
  deny "${H}/${other}" && ! grep -v ' u:sbx:' "${WORK}/acl.log" | grep -q . \
    && ok "case 3 (${me}): the other runner's install ${other} is denied to sbx, and only sbx is named" \
    || fail "case 3 (${me}): the other runner's install ${other}: $(grep -F "${H}/${other}" "${WORK}/acl.log")"
done

mkdir -p "${WORK}/elsewhere"
grant actions-runner-dev "${WORK}/elsewhere"; rc=$?
[[ "${rc}" -ne 0 && ! -s "${WORK}/acl.log" ]] && grep -q "is not under runner's home" "${WORK}/out" \
  && ok "case 4: a checkout outside the runner home is refused before any ACL" \
  || fail "case 4: rc=${rc}, acl: $(cat "${WORK}/acl.log")"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-sandbox-grant-checkout: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-sandbox-grant-checkout: all cases passed"
