#!/usr/bin/env bash
# test-kernels-staged-sandbox.sh -- the participant kernels tools/engine/build.sh
# stages in .build/kernels are not writable by the sandbox user
# (docs/sandbox-design.md 6a, "Staged kernels").
#
# harness/kernels/build.sh runs as the sandbox user and writes kernels-out;
# build.sh then copies that output into the checkout's .build/kernels, which
# every candidate leg's resident loads. A copy that keeps the output's modes and
# ACLs (a 0666 file, a 0777 directory, an ACL entry granting the sandbox user
# write) leaves a runner-owned file the sandbox user can rewrite between legs,
# and the per-leg clear never looks there.
#
# The REAL tools/engine/build.sh runs with SANDBOX_USER=nobody and the real sudo
# for every probe (tools/testlib/real-sandbox-user.sh; SKIP without it). Only
# cargo, docker, pgrep, getent and the sandbox kill are stubs. The stub kernels
# build writes, AS the sandbox user, a 0666 file, a 0777 directory and a file
# with the ACL entry u:nobody:rw. After the build, a stub entry (the sandbox
# user) tries to write each staged file and to make a file in each staged
# directory. Pass = the build refused (and nothing is staged), or every write
# fails.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=tools/testlib/real-sandbox-user.sh
. "${ROOT_DIR}/tools/testlib/real-sandbox-user.sh"
real_sandbox_user_or_skip test-kernels-staged-sandbox.sh
trap sbx_cleanup EXIT
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required" >&2; exit 1; }

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

REPO="${SBX_WORK}/repo"; STUBS="${SBX_WORK}/bin"
mkdir -p "${REPO}/tools/engine" "${REPO}/fixtures" "${REPO}/harness/protocol-adapter" "${REPO}/harness/kernels" "${STUBS}"
cp "${ROOT_DIR}/tools/engine/build.sh" "${REPO}/tools/engine/"
cp "${ROOT_DIR}/tools/stage-cuda-engine.sh" "${ROOT_DIR}/tools/run-engine-sandboxed.sh" "${REPO}/tools/"
for f in sandbox-access.sh sandbox-clear-dirs.sh; do
  [[ -f "${ROOT_DIR}/tools/${f}" ]] && cp "${ROOT_DIR}/tools/${f}" "${REPO}/tools/"
done
cp "${ROOT_DIR}/fixtures/muse_glimmer_30b_cuda_v1_track.json" "${REPO}/fixtures/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${REPO}/tools/stage-vllm-overlay.sh"
touch "${REPO}/harness/kernels/build.sh"
printf '[package]\nname = "adapter"\nversion = "0.1.0"\n' > "${REPO}/harness/protocol-adapter/Cargo.toml"
printf 'version = 3\n\n[[package]]\nname = "adapter"\nversion = "0.1.0"\n' > "${REPO}/harness/protocol-adapter/Cargo.lock"
chmod +x "${REPO}/tools/"*.sh "${REPO}/tools/engine/build.sh"

# cargo: vendor makes a dir; build (run as the sandbox user) makes the adapter.
cat > "${STUBS}/cargo" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  vendor) mkdir -p "${@: -1}/some-crate"; printf '[source.crates-io]\nreplace-with = "vendored-sources"\n' ;;
  build) mkdir -p "${CARGO_TARGET_DIR}/release" && printf '#!/bin/sh\n' > "${CARGO_TARGET_DIR}/release/cuda-engine" \
           && chmod 0755 "${CARGO_TARGET_DIR}/release/cuda-engine" ;;
esac
EOF
# sudo: the sandbox kill is skipped (nothing of nobody's may be killed here);
# the cargo build runs as the sandbox user against the cargo stub; every other
# command (true, find, sh, chmod: the probes) is the REAL sudo.
cat > "${STUBS}/sudo" <<EOF
#!/usr/bin/env bash
args=("\$@")
[[ "\$1 \$2" == "-n -u" ]] || exec /usr/bin/sudo "\$@"
shift 3; [[ "\$1" == -- ]] && shift
[[ "\$1" == kill ]] && exit 0
if [[ "\$1" == prlimit ]]; then
  shift; while [[ "\$1" == --* ]]; do shift; done
  shift; vars=(); while [[ "\$1" == *=* ]]; do vars+=("\$1"); shift; done
  exec /usr/bin/sudo -n -u "${SBX}" -- env "\${vars[@]}" PATH="${STUBS}:/usr/bin:/bin" "${STUBS}/\$1" "\${@:2}"
fi
exec /usr/bin/sudo "\${args[@]}"
EOF
# docker: image present; the kernels "container" writes kernels-out as the sandbox user.
cat > "${STUBS}/docker" <<EOF
#!/usr/bin/env bash
[[ "\$1" == image ]] && exit 0
[[ "\$1" == run ]] || exit 1
for a in "\$@"; do
  [[ "\${a}" == KERNELS_OUT=* ]] || continue
  /usr/bin/sudo -n -u "${SBX}" -- sh -c 'cd "\$1" && umask 0000 && echo k > open.so && chmod 0666 open.so \
    && mkdir open-dir && chmod 0777 open-dir && echo k > open-dir/inner.so \
    && echo k > acl.so && chmod 0644 acl.so && setfacl -m u:${SBX}:rw acl.so' sh "\${a#KERNELS_OUT=}"
done
exit 0
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "${STUBS}/pgrep"
printf '#!/usr/bin/env bash\necho "nemoshare:x:%s:"\n' "$(id -g)" > "${STUBS}/getent"
chmod 0755 "${STUBS}"/*

# The sandbox root: the sandbox user and this account get, through inherited
# ACLs, the access the share group gives both on a box (the sandbox user is not
# in this account's group here).
SROOT="${SBX_WORK}/sandbox"
mkdir -m 0750 "${SROOT}"
setfacl -m "u:${SBX}:rx" -m "d:u:${SBX}:rwx" -m "d:u:$(id -un):rwx" "${SROOT}"

out="$(cd "${REPO}" && env PATH="${STUBS}:${PATH}" RUNNER_TEMP="${SBX_WORK}" SANDBOX_USER="${SBX}" \
  SANDBOX_ROOT="${SROOT}" SANDBOX_JOB=42-1 ./tools/engine/build.sh 2>&1)"; rc=$?
STAGE="${REPO}/.build/kernels"
if [[ "${rc}" -ne 0 ]]; then
  if grep -q 'kernels refused' <<<"${out}" && [[ ! -e "${STAGE}/open.so" ]]; then
    ok "the build refused a kernels output the sandbox user could write once staged, and staged nothing"
  else
    fail "the build failed for another reason (rc=${rc}): $(tail -5 <<<"${out}")"
  fi
else
  [[ -f "${STAGE}/open.so" && -f "${STAGE}/acl.so" && -f "${STAGE}/open-dir/inner.so" ]] \
    && ok "the build staged the kernels output" || fail "the staged kernels are missing: $(tail -5 <<<"${out}")"
  for f in open.so acl.so open-dir/inner.so; do
    if sbx sh -c 'printf leg1-marker >> "$1"' sh "${STAGE}/${f}" 2>/dev/null; then
      fail "the sandbox user can write the staged kernel ${f}"
    else
      ok "the sandbox user cannot write the staged kernel ${f}"
    fi
    sbx cat -- "${STAGE}/${f}" >/dev/null 2>&1 && ok "the resident (the sandbox user) can still read ${f}" \
      || fail "the sandbox user cannot read the staged kernel ${f}"
  done
  for d in . open-dir; do
    if sbx sh -c ': > "$1/planted"' sh "${STAGE}/${d}" 2>/dev/null; then
      fail "the sandbox user can make a file in the staged directory ${d}"
    else
      ok "the sandbox user cannot make a file in the staged directory ${d}"
    fi
  done
fi

if [[ "${failures}" -gt 0 ]]; then echo "FAIL: ${failures} failure(s)" >&2; exit 1; fi
echo "PASS: test-kernels-staged-sandbox.sh"
