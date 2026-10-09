#!/usr/bin/env bash
# test-converge.sh -- tools/box/converge.sh in a disposable ubuntu:24.04
# container, as root, with a small test fixture.
#
#   tools/box/test/test-converge.sh       starts the container (needs docker)
#
# Proves, on a fresh container:
#   1. a box file that is not values only is refused (exit 2)
#   2. --check on the fresh box exits 1, prints WOULD-CHANGE and no CONVERGED,
#      and changes nothing (the account, the files and the dirs are as before)
#   3. apply converges every unit under test and exits 0
#   4. a second apply prints no CONVERGED line
#   5. --check after apply exits 0
#   6. a changed weight file and a loose .env mode are found by --check and
#      fixed by apply
#   7. a value given on --secrets-stdin is in no file afterwards
#   8. apply refuses while a job of the runner account runs
# The units that need a GPU, the network or GitHub (rust apply, reference,
# runner-install, sandbox, preflight) are not in this test: the engine tools
# they run have their own tests, and the box apply proves them.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
REPO="$(cd -- "${HERE}/../../.." >/dev/null && pwd -P)"

if [[ "${CONVERGE_TEST_INSIDE:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "test-converge.sh: docker is required" >&2; exit 1; }
  exec docker run --rm -e CONVERGE_TEST_INSIDE=1 -v "${REPO}:/src:ro" ubuntu:24.04 \
    bash /src/tools/box/test/test-converge.sh
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends jq git curl ca-certificates procps passwd util-linux >/dev/null

PASS=0; FAILN=0
pass() { PASS=$((PASS + 1)); echo "PASS $*"; }
fail() { FAILN=$((FAILN + 1)); echo "FAIL $*"; }

RU=tbench; RH=/home/tbench; ST="${RH}/track"
ENGINE="${ST}/engine"; MIRROR="${ST}/mirror.git"
SRC=/srv-src; mkdir -p "${SRC}"
STUB=/stub; mkdir -p "${STUB}/bin" "${STUB}/state"

# stubs: systemctl, docker
cat > "${STUB}/bin/systemctl" <<'EOF'
#!/bin/bash
s=/stub/state; cmd="$1"; shift; [[ "$1" == "-q" ]] && shift; u="$1"
case "${cmd}" in
  is-active) [[ -f "${s}/${u}.active" ]] && { echo active; exit 0; }; echo inactive; exit 3 ;;
  is-enabled) [[ -f "${s}/${u}.enabled" ]] && { echo enabled; exit 0; }; echo disabled; exit 1 ;;
  enable) touch "${s}/${u}.enabled" ;;
  start|restart) touch "${s}/${u}.active"; echo "${cmd}" >> "${s}/${u}.log" ;;
esac
EOF
cat > "${STUB}/bin/docker" <<'EOF'
#!/bin/bash
case "$1 $2" in
  "image inspect") [[ -f /stub/state/image ]] ;;
  "pull -q") touch /stub/state/image ;;
  *) exit 1 ;;
esac
EOF
chmod +x "${STUB}/bin/"*
export PATH="${STUB}/bin:${PATH}"

# The operator's first checkout: a git repo of this working tree, with the
# test box file and fixture committed. The box mirror is cloned from it.
git config --system --add safe.directory '*'
OPREPO="${SRC}/repo"
mkdir -p "${OPREPO}"
tar -C /src --exclude=.git -cf - . | tar -C "${OPREPO}" -xf -

# The test track: two target files, one golden, a fake benchd pair.
printf 'weights-one\n' > "${SRC}/a.bin"; printf 'weights-two\n' > "${SRC}/b.json"
printf '{"golden":1}\n' > "${SRC}/g1.json"
sha() { sha256sum "$1" | cut -d' ' -f1; }
by() { stat -c %s "$1"; }
PIN_SRC="$(sed -n 's/^PINNED_SOURCE_COMMIT="\([0-9a-f]\{40\}\)"$/\1/p' /src/tools/fetch-benchd.sh)"
printf 'fake-benchd\n' > "${SRC}/benchd"
cat > "${SRC}/fixture.json" <<EOF
{"target": {"upstream_model_id": "x/y", "upstream_revision": "r",
  "files": [{"path": "a.bin", "bytes": $(by "${SRC}/a.bin"), "sha256": "$(sha "${SRC}/a.bin")"},
            {"path": "b.json", "bytes": $(by "${SRC}/b.json"), "sha256": "$(sha "${SRC}/b.json")"}]},
 "serve_configuration": {"container_image": "docker.io/x/engine:v1", "container_digest": "sha256:$(printf '%064d' 0)"},
 "baseline_reference_commit": "$(printf '%040d' 0)",
 "timed_prompt_pool": [{"r2_path": "correctness_prompts/t/g1.json", "sha256": "$(sha "${SRC}/g1.json")", "bytes": $(by "${SRC}/g1.json")}]}
EOF

HOSTN="$(hostname -s 2>/dev/null || hostname)"
BOXDIR="${OPREPO}/tools/box"
write_box() {
  cat > "${BOXDIR}/testbox.env" <<EOF
BOX_HOST="${HOSTN}"
BOX_TRACK_ID="t-cuda-v1"
BOX_TRACK_FIXTURE="tools/box/test/fixture.json"
BOX_GOLDEN_ENV="MLXFAST_T_GOLDEN_DIR"
BOX_RUNNER_USER="${RU}"
BOX_RUNNER_HOME="${RH}"
BOX_RUNNER_GROUPS="tdocker tbench-lock"
BOX_RUST_TOOLCHAIN="1.98.1"
BOX_STAGING_ROOT="${ST}"
BOX_MIRROR_DIR="${MIRROR}"
BOX_MIRROR_URL="${OPREPO}"
BOX_ENGINE_DIR="${ENGINE}"
BOX_BENCHD_BIN_DIR="${ST}/benchd-bin"
BOX_BENCHD_SHA256="$(sha "${SRC}/benchd")"
BOX_BENCHD_BYTES="$(by "${SRC}/benchd")"
BOX_SNAPSHOT_DIR="/srv/t/track/snapshot"
BOX_WEIGHTS_BASE_URL="file://${SRC}"
BOX_GOLDEN_DIR="${ST}/goldens"
BOX_BASELINE_WORKSPACE="${ST}/baseline/workspace"
BOX_BASELINE_CALIBRATION="${ST}/baseline/baseline-calibration.json"
BOX_ACTIONS_RUNNER_DIR="${RH}/actions-runner"
BOX_RUNNER_REPO_URL="https://github.com/o/r"
BOX_RUNNER_NAME="tbox"
BOX_RUNNER_LABELS="t-cuda-v1"
BOX_GPU_LOCK="/tmp/t-gpu.lock"
BOX_LOCK_GROUP="tbench-lock"
BOX_SANDBOX_USER="tsub"
BOX_SHARE_GROUP="tshare"
BOX_SANDBOX_ROOT="/var/lib/t/sandbox"
BOX_SANDBOX_ETC="/etc/t-sandbox"
BOX_SUDOERS_NAME="t-sandbox"
BOX_EGRESS_UNIT="t-egress.service"
BOX_ACL_UNIT="t-acl.service"
BOX_RUST_DIR="/opt/t-rust"
BOX_SERVE_UP_LOG_DIR="/var/lib/t/serve-log"
BOX_SERVE_UP_SOCKET_DIR="/var/lib/t/sock"
BOX_SERVE_UP_CONTAINER_PREFIX="t-serve-up-"
BOX_UNTOUCHED_UNITS="other-tenant.service"
EOF
}
cp "${SRC}/fixture.json" "${BOXDIR}/test/fixture.json"
write_box
git -C "${OPREPO}" init -q -b main
git -C "${OPREPO}" -c user.name=t -c user.email=t@t add -A
git -C "${OPREPO}" -c user.name=t -c user.email=t@t commit -q -m test
groupadd tdocker
export CONVERGE_SYSTEMD_DIR=/stub/systemd CONVERGE_TMPFILES_DIR=/stub/tmpfiles.d
mkdir -p "${CONVERGE_SYSTEMD_DIR}" "${CONVERGE_TMPFILES_DIR}"
UNITS=(--unit host --unit account --unit gpu-lock --unit dirs --unit engine --unit benchd --unit weights
       --unit image --unit goldens --unit runner-register --unit runner-env --unit calibration --unit service)
OPCONV="${OPREPO}/tools/box/converge.sh"
CONV="${ENGINE}/tools/box/converge.sh"

# The items that need GitHub or R2 on a real box are placed the way their tool
# places them, by a hook the test runs between the account and the units.
stage_offline_inputs() {
  install -d -o "${RU}" -g "${RU}" "${ST}/benchd-bin" "${ST}/goldens" "${ST}/baseline" "${RH}/actions-runner"
  install -o "${RU}" -m 0755 "${SRC}/benchd" "${ST}/benchd-bin/benchd"
  printf '{"source_commit":"%s","sha256":"%s","bytes":%s}\n' "${PIN_SRC}" "$(sha "${SRC}/benchd")" "$(by "${SRC}/benchd")" > "${ST}/benchd-bin/benchd.manifest.json"
  install -o "${RU}" -m 0640 "${SRC}/g1.json" "${ST}/goldens/g1.json"
  printf '{"box":"tbox","captured_at":"2026-10-09T00:00:00Z"}\n' > "${ST}/baseline/baseline-calibration.json"
  printf '#!/bin/sh\n' > "${RH}/actions-runner/config.sh"; cp "${RH}/actions-runner/config.sh" "${RH}/actions-runner/run.sh"
  chmod 0755 "${RH}/actions-runner/"*.sh
  printf '{"agentName":"tbox","gitHubUrl":"https://github.com/o/r"}\n' > "${RH}/actions-runner/.runner"
  printf '[Service]\nUser=%s\n' "${RU}" > "${CONVERGE_SYSTEMD_DIR}/actions.runner.o-r.tbox.service"
  chown -R "${RU}:${RU}" "${RH}/actions-runner" "${ST}"
}

fs_state() {
  { getent passwd "${RU}" || true; getent group tbench-lock || true
    find /srv /var/lib/t /stub "${RH}" /tmp/t-gpu.lock -printf '%p %m %u %s\n' 2>/dev/null || true; } | sort | sha256sum
}

# 1. values only
cp "${BOXDIR}/testbox.env" /tmp/bad.env
printf 'BOX_HOST="$(id)"\n' >> /tmp/bad.env
set +e; out="$(BOX_FILE=/tmp/bad.env "${OPCONV}" testbox --check 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 2 && "${out}" == *"is not a comment"* ]] && pass "a box file with a command is refused" || fail "values-only refusal: rc ${rc}: ${out}"

# 2. check on a fresh box changes nothing
before="$(fs_state)"
set +e; out="$("${OPCONV}" testbox --check "${UNITS[@]}" --unit rust 2>&1)"; rc=$?; set -e
after="$(fs_state)"
[[ "${rc}" -eq 1 ]] && pass "--check on a fresh box exits 1" || fail "--check fresh rc ${rc}"
grep -q 'WOULD-CHANGE' <<<"${out}" && pass "--check prints WOULD-CHANGE" || fail "no WOULD-CHANGE: ${out}"
grep -q 'CONVERGED' <<<"${out}" && fail "--check printed CONVERGED: ${out}" || pass "--check prints no CONVERGED"
[[ "${before}" == "${after}" ]] && pass "--check changed nothing on the box" || fail "--check changed the box"

# 3. apply: the first run from the operator checkout makes the account, the
# mirror and the box checkout and stops; then the offline inputs; then all
# units from the box checkout.
set +e; out="$("${OPCONV}" testbox "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 0 && -d "${ENGINE}/.git" && "${out}" == *"run every later apply"* ]] \
  && pass "first apply from the operator checkout makes the box checkout and stops" || fail "bootstrap apply rc ${rc}: ${out}"
[[ "$(git -C "${ENGINE}" rev-parse HEAD)" == "$(git -C "${OPREPO}" rev-parse HEAD)" ]] && pass "box checkout at the mirror main" || fail "box checkout commit"
stage_offline_inputs
set +e; out="$("${CONV}" testbox "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 0 ]] && pass "apply exits 0" || fail "apply rc ${rc}: ${out}"
[[ "$(stat -c '%a %U' /var/lib/t/sock)" == "700 ${RU}" ]] && pass "socket dir 0700 ${RU}" || fail "socket dir mode"
for want in "target snapshot: fetched" "engine image: docker pull" \
            "goldens: files 0440" "runner .env: write" "enable" "start"; do
  grep -q -- "${want}" <<<"${out}" && pass "apply: ${want}" || fail "apply did not report '${want}': ${out}"
done
[[ "$(stat -c '%a %U:%G' /tmp/t-gpu.lock)" == "660 root:tbench-lock" ]] && pass "GPU lock 0660 root:tbench-lock" || fail "lock mode"
[[ "$(stat -c '%a %U' "${RH}/actions-runner/.env")" == "600 ${RU}" ]] && pass ".env 0600 ${RU}" || fail ".env mode"
grep -qx "MLXFAST_T_GOLDEN_DIR=${ST}/goldens" "${RH}/actions-runner/.env" && pass ".env names the golden dir by BOX_GOLDEN_ENV" || fail ".env golden key"
[[ "$(stat -c '%a %U' /srv/t/track/snapshot/a.bin)" == "644 root" ]] && pass "weights root-owned 0644" || fail "weights mode"
id -nG "${RU}" | tr ' ' '\n' | grep -qx tdocker && pass "${RU} joined tdocker" || fail "groups"

# 4. second apply: no change
set +e; out="$("${CONV}" testbox "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 0 ]] && ! grep -q CONVERGED <<<"${out}" && pass "second apply converges nothing" || fail "second apply rc ${rc}: ${out}"

# 5. check after apply is clean
set +e; out="$("${CONV}" testbox --check "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 0 && "${out}" == *"the box matches"* ]] && pass "--check after apply exits 0" || fail "--check after apply rc ${rc}: ${out}"

# 6. drift is found and fixed
chmod u+w /srv/t/track/snapshot/a.bin; printf 'tampered\n' > /srv/t/track/snapshot/a.bin
truncate -s "$(by "${SRC}/a.bin")" /srv/t/track/snapshot/a.bin
chmod 0644 "${RH}/actions-runner/.env"
set +e; out="$("${CONV}" testbox --check "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 1 ]] && grep -q 'target snapshot: fetch 0 missing, 1 wrong' <<<"${out}" && grep -q 'runner .env: 0600' <<<"${out}" \
  && pass "--check finds a changed weight (same size) and a loose .env" || fail "drift check rc ${rc}: ${out}"
set +e; out="$("${CONV}" testbox "${UNITS[@]}" 2>&1)"; rc=$?; set -e
[[ "${rc}" -eq 0 && "$(sha /srv/t/track/snapshot/a.bin)" == "$(sha "${SRC}/a.bin")" && "$(stat -c %a "${RH}/actions-runner/.env")" == 600 ]] \
  && pass "apply fixes the drift" || fail "drift apply rc ${rc}: ${out}"

# 7. a secret is in no file
secret="tok-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
set +e; printf 'GITHUB_TOKEN=%s\nR2_ACCESS_KEY_ID=%s\n' "${secret}" "${secret}" | "${CONV}" testbox --secrets-stdin "${UNITS[@]}" >/tmp/secret-run.log 2>&1; rc=$?; set -e
if grep -rqsF -- "${secret}" /home /srv /var/lib /stub /etc /tmp /root 2>/dev/null; then fail "the secret was written to a file"
else pass "apply with --secrets-stdin (rc ${rc}) left the secret in no file"; fi

# 8. apply refuses while a job runs
runuser -u "${RU}" -- bash -c 'exec -a Runner.Worker sleep 30' &
busy=$!
sleep 1
set +e; out="$("${CONV}" testbox "${UNITS[@]}" 2>&1)"; rc=$?; set -e
kill "${busy}" 2>/dev/null || true; wait "${busy}" 2>/dev/null || true
[[ "${rc}" -ne 0 && "${out}" == *"apply at a quiet moment"* ]] && ! grep -q CONVERGED <<<"${out}" \
  && pass "apply refuses while a job of ${RU} runs" || fail "busy refusal rc ${rc}: ${out}"

echo "test-converge.sh: ${PASS} passed, ${FAILN} failed"
[[ "${FAILN}" -eq 0 ]]
