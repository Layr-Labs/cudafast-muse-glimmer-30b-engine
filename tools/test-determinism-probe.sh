#!/usr/bin/env bash
# test-determinism-probe.sh -- tools/determinism-probe.py resolves the engine
# from the checkout it runs in, never from a fixed home-directory path.
#
# Asserts, with a stub adapter (no GPU, no resident):
#   1. the probe runs <checkout>/.build/release/mlxfast-runtime-worker and
#      reports one stream for a deterministic engine;
#   2. MLXFAST_ENGINE_BIN overrides it, and a relative value is read from the
#      repository root;
#   3. with no staged engine the probe refuses by name before it reads a prompt.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/repo/tools" "${WORK}/repo/.build/release" "${WORK}/repo/alt"
cp "${ROOT_DIR}/tools/determinism-probe.py" "${WORK}/repo/tools/"
write_stub() { # write_stub PATH MARKER
  cat > "$1" <<STUB
#!/usr/bin/env python3
import json, sys
open("${WORK}/ran", "a").write("$2\n")
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["kind"] == "free_decode_begin":
        print(json.dumps({"id": req["id"], "seed_token": 7}), flush=True)
    else:
        print(json.dumps({"id": req["id"], "tokens": list(range(req["count"]))}), flush=True)
STUB
  chmod +x "$1"
}
write_stub "${WORK}/repo/.build/release/mlxfast-runtime-worker" staged
write_stub "${WORK}/repo/alt/engine" override
echo '[1, 2, 3]' > "${WORK}/prompt.json"
# The probe runs from an unrelated directory with an empty HOME, so neither the
# working directory nor a home-directory checkout can supply the engine.
probe() { # probe [VAR=VALUE ...]
  (cd "${WORK}" && env -u MLXFAST_ENGINE_BIN HOME="${WORK}/nohome" "$@" \
    python3 "${WORK}/repo/tools/determinism-probe.py" "${WORK}/sock" "${WORK}/prompt.json" 2 2>&1)
}

# --- case 1: the checkout's staged engine -----------------------------------
: > "${WORK}/ran"
out="$(probe)"; rc=$?
[[ "${rc}" -eq 0 ]] && grep -q "distinct streams: 1" <<<"${out}" && [[ "$(sort -u "${WORK}/ran")" == staged ]] \
  && ok "case 1: the probe runs this checkout's .build/release/mlxfast-runtime-worker" \
  || fail "case 1: rc=${rc}, ran='$(tr '\n' ' ' < "${WORK}/ran")', out='${out}'"

# --- case 2: MLXFAST_ENGINE_BIN, relative to the repository root ------------
: > "${WORK}/ran"
out="$(probe MLXFAST_ENGINE_BIN=alt/engine)"; rc=$?
[[ "${rc}" -eq 0 ]] && [[ "$(sort -u "${WORK}/ran")" == override ]] \
  && ok "case 2: MLXFAST_ENGINE_BIN overrides the engine, read from the repository root" \
  || fail "case 2: rc=${rc}, ran='$(tr '\n' ' ' < "${WORK}/ran")', out='${out}'"

# --- case 3: no staged engine refuses by name -------------------------------
rm "${WORK}/repo/.build/release/mlxfast-runtime-worker"
out="$(probe)"; rc=$?
[[ "${rc}" -ne 0 ]] && grep -q "no staged engine at ${WORK}/repo/.build/release/mlxfast-runtime-worker" <<<"${out}" \
  && ok "case 3: a checkout with no staged engine refuses by name" \
  || fail "case 3: rc=${rc}, out='${out}'"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-determinism-probe: ${failures} failure(s)" >&2
  exit 1
fi
echo "test-determinism-probe: all cases passed"
