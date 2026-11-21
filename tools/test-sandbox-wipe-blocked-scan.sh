#!/usr/bin/env bash
# test-sandbox-wipe-blocked-scan.sh -- tools/sandbox-wipe.sh fails closed when
# entry code makes a path it must look at unreadable (docs/sandbox-design.md
# 6a, "The wipe fails closed").
#
# The sandbox user is the current user (sudo, ipcs stubbed), as in
# test-sandbox-wipe.sh. A stub entry (leg 1) does only what code running as
# the sandbox user can do on a box:
#   - plants a file in a mode-000 directory in the account home, then makes the
#     account home itself mode 000 (the sandbox user owns that home:
#     tools/box-runner/setup-sandbox-user.sh);
#   - plants a file in a mode-000 directory in a temp directory;
#   - plants a file by name in a listed directory it can write and enter but
#     not list (mode 0333, like a 1733 session directory).
# Then the wipe runs, and the next leg (the same account) looks for those
# files. A case passes when the file is gone, or when the wipe fails (the run is
# refused). A wipe that exits 0 while a file is still there is the fault.
#
# The account home is no longer emptied but must be SEALED (root-owned, out of
# the sandbox user's reach; tools/sandbox-access.sh sealed-home): a home the
# sandbox user owns fails the wipe. An absent home is sealed, so every case but
# case 1 names an absent one (tools/test-sandbox-home-sealed.sh checks a
# present sealed home as a real second user).
#
# Cases:
#   1. the three plants above, with the sandbox user's own home: refused;
#   1b. the temp-dir and listed-dir plants, with a sealed home: gone or refused;
#   2. a listed directory ANOTHER user owns that the sandbox user can enter but
#      not list: the wipe fails (it cannot prove the directory is clean);
#   3. a directory another user owns that the sandbox user cannot enter at all
#      is skipped: nothing in it is reachable, and the wipe passes;
#   4. a sandbox-owned directory the wipe cannot reopen (chmod fails) fails the
#      wipe, and no planted name is printed.
#
# Hermetic: no root, no GPU, no real IPC objects touched.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIPE="${ROOT_DIR}/tools/sandbox-wipe.sh"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwx "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
ME="$(id -un)"
NAME=planted-prompt-name

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin"
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
shift 4
[[ "$1" == kill ]] && exit 0
exec "$@"
EOF
cat > "${WORK}/bin/ipcs" <<'EOF'
#!/usr/bin/env bash
echo "------ header --------"
echo "key        id         owner      perms"
EOF
chmod +x "${WORK}/bin/"*
# A chmod that always fails, on PATH only for case 4.
mkdir -p "${WORK}/badbin"
printf '#!/usr/bin/env bash\nexit 1\n' > "${WORK}/badbin/chmod"; chmod +x "${WORK}/badbin/chmod"

# fresh -- empty box state: a temp dir, a listed write-only dir, the account home.
fresh() {
  chmod -R u+rwx "${WORK}/box" 2>/dev/null; rm -rf "${WORK}/box"
  mkdir -p "${WORK}/box/tmp" "${WORK}/box/wo" "${WORK}/box/home"
  chmod 0333 "${WORK}/box/wo"; chmod 0700 "${WORK}/box/home"
  printf '%s\n' "${WORK}/box/tmp" "${WORK}/box/wo" > "${WORK}/box/wipe-dirs"
}
run_wipe() {
  env PATH="${WORK}/bin:${PATH}" SANDBOX_USER="${ME}" \
    SANDBOX_WIPE_LIST="${WORK}/box/wipe-dirs" SANDBOX_WIPE_HOME="${WIPE_HOME:-${WORK}/box/no-home}" "$@" "${WIPE}"
}
# leg1_entry -- what entry code can leave behind as the sandbox user.
leg1_entry() {
  local h="${WORK}/box/home" t="${WORK}/box/tmp" w="${WORK}/box/wo"
  mkdir "${h}/hide" && echo leg1 > "${h}/hide/${NAME}" && chmod 000 "${h}/hide" && chmod 000 "${h}"
  mkdir "${t}/hide" && echo leg1 > "${t}/hide/${NAME}" && chmod 000 "${t}/hide"
  echo leg1 > "${w}/${NAME}"
}
# leg2_finds -- the next leg, as the same account: reopen what it owns, then
# look for leg 1's files. Prints each one it can still read.
leg2_finds() {
  local h="${WORK}/box/home" t="${WORK}/box/tmp" w="${WORK}/box/wo" f
  chmod u+rwx "${h}" "${h}/hide" "${t}/hide" 2>/dev/null
  for f in "${h}/hide/${NAME}" "${t}/hide/${NAME}" "${w}/${NAME}"; do
    [[ -r "${f}" ]] && echo "${f#"${WORK}/box/"}"
  done
}

# --- case 1: unreadable home (the sandbox user's own), temp dir, listed dir ---
fresh; leg1_entry
out="$(WIPE_HOME="${WORK}/box/home" run_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 1: a home the sandbox user owns fails the wipe (exit ${status}), so the run is refused" \
  || fail "case 1: the wipe exited 0 with the sandbox user's own home: ${out}"
grep -q "${NAME}" <<<"${out}" && fail "case 1: a planted name was printed" || ok "case 1: no planted name is printed"

# --- case 1b: unreadable temp dir and listed dir, sealed home ----------------
fresh; leg1_entry
out="$(run_wipe 2>&1)"; status=$?
left="$(leg2_finds | grep -v '^home/' | tr '\n' ' ')"
if [[ "${status}" -eq 0 && -n "${left}" ]]; then
  fail "case 1b: the wipe exited 0 but the next leg can still read leg 1's file(s): ${left}"
elif [[ "${status}" -eq 0 ]]; then
  ok "case 1b: the wipe passed and no leg-1 file is left"
else
  ok "case 1b: the wipe failed (exit ${status}), so the run is refused"
fi
grep -q "${NAME}" <<<"${out}" && fail "case 1b: a planted name was printed" || ok "case 1b: no planted name is printed"

# --- case 2: a listed dir another user owns, enterable but not listable ------
# SANDBOX_USER=root: every directory here belongs to "another user".
fresh; echo x > "${WORK}/box/wo/${NAME}"
out="$(run_wipe SANDBOX_USER=root 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 2: a directory the sandbox user can enter but not list fails the wipe (exit ${status})" \
  || fail "case 2: the wipe exited 0 although it could not list $(basename "${WORK}/box/wo"): ${out}"
grep -q "${NAME}" <<<"${out}" && fail "case 2: a planted name was printed" || ok "case 2: no planted name is printed"

# --- case 3: another user's directory the sandbox user cannot enter ----------
fresh; chmod 0700 "${WORK}/box/wo"
mkdir -p "${WORK}/box/tmp/private"; echo x > "${WORK}/box/tmp/private/f"; chmod 0600 "${WORK}/box/tmp/private"
out="$(run_wipe SANDBOX_USER=root 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 3: another user's directory the sandbox user cannot enter is skipped" \
  || fail "case 3: exit ${status}: ${out}"

# --- case 4: a sandbox-owned directory that cannot be reopened ----------------
fresh; mkdir "${WORK}/box/tmp/hide"; echo x > "${WORK}/box/tmp/hide/${NAME}"; chmod 000 "${WORK}/box/tmp/hide"
out="$(env PATH="${WORK}/badbin:${WORK}/bin:${PATH}" SANDBOX_USER="${ME}" \
  SANDBOX_WIPE_LIST="${WORK}/box/wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/box/no-home" "${WIPE}" 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 4: a directory the wipe cannot reopen fails the wipe (exit ${status})" \
  || fail "case 4: the wipe exited 0 with an unreadable directory left: ${out}"
grep -q "${NAME}" <<<"${out}" && fail "case 4: a planted name was printed" || ok "case 4: no planted name is printed"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-wipe-blocked-scan.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
