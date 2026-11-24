#!/usr/bin/env bash
# .github/scripts/entry-weight-check.py against a throwaway git repository:
# a base commit with the editable surface, then one candidate commit per case.
# No GPU, network, credential or weights are used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECK="${REPO_ROOT}/.github/scripts/entry-weight-check.py"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT
fails=0
ok() { echo "ok    $*"; }
fail() { echo "FAIL  $*" >&2; fails=$((fails + 1)); }

G="${WORK}/repo"
mkdir -p "${G}/harness/vllm_resident" "${G}/vllm/vllm/model_executor" "${G}/tools"
g() { git -C "${G}" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
g init -q
jq -n '{editablePaths:["vllm","harness","mtp-head.manifest.json"], contractPath:"fixtures/track.json"}' > "${G}/benchmark.json"
# The base: a resident that loads the pinned drafter (the trusted call), and a
# model file that already names a loader (vendored code does).
cat > "${G}/harness/vllm_resident/backends.py" <<'EOF'
import os
def spec_config(depth):
    return {"method": "dflash", "model": os.environ["RESIDENT_DRAFT_MODEL_DIR"], "num_speculative_tokens": depth}
EOF
cat > "${G}/vllm/vllm/model_executor/model.py" <<'EOF'
from vllm.model_executor.model_loader import get_model_loader
def forward(x):
    return x
EOF
cp "${REPO_ROOT}/mtp-head.manifest.json" "${G}/mtp-head.manifest.json"
echo "trusted" > "${G}/tools/x.sh"
g add -A && g commit -qm base
BASE="$(g rev-parse HEAD)"
case_n=0

# candidate NAME SETUP -- a commit on a fresh branch from the base; SETUP runs in the repo.
candidate() {
  local name="$1" setup="$2"
  case_n=$((case_n + 1))
  g checkout -q -B "case-${case_n}" "${BASE}"
  ( cd "${G}" && eval "${setup}" )
  g add -A && g commit -qm "${name}" --allow-empty
}
run_check() { ( cd "${G}" && BASE_SHA="${BASE}" HEAD_SHA="$(g rev-parse HEAD)" CONTRACT_PATH=benchmark.json "${CHECK}" 2>&1 ); }
passes() { # passes LABEL SETUP
  local out rc=0
  candidate "$1" "$2"
  out="$(run_check)" || rc=$?
  [[ ${rc} -eq 0 ]] && ok "$1: passes" || fail "$1: refused (rc=${rc}): $(tail -2 <<<"${out}")"
}
refused() { # refused LABEL NEEDLE SETUP
  local out rc=0
  candidate "$1" "$3"
  out="$(run_check)" || rc=$?
  if [[ ${rc} -eq 1 ]] && grep -qF -- "$2" <<<"${out}"; then ok "$1: refused by name"
  else fail "$1: rc=${rc}, want 1 with '$2'; got: $(tail -2 <<<"${out}")"; fi
}

# --- clean entries pass ------------------------------------------------------
passes "a plain code edit" \
  "printf 'def forward(x):\n    return x * 1\n' > vllm/vllm/model_executor/fast.py"
passes "an edit beside an unchanged loader line" \
  "printf 'from vllm.model_executor.model_loader import get_model_loader\ndef forward(x):\n    return x + 0\n' > vllm/vllm/model_executor/model.py"
passes "the declaration switched on (it restates the pin: sha256 hex, model.safetensors)" \
  "jq '.spec.enabled = true | .spec.num_speculative_tokens = 3' mtp-head.manifest.json > m && mv m mtp-head.manifest.json"
passes "a small non-text file (1 KiB)" \
  "head -c 1024 /dev/urandom | tr -d '\000' > harness/icon.dat; printf '\377\000' >> harness/icon.dat"
passes "a deletion" "git rm -q vllm/vllm/model_executor/model.py"
passes "a tuning table of a few hundred numbers" \
  "seq 1 400 | jq -s '{sizes: .}' > vllm/vllm/model_executor/tune.json"

# --- weight-like payloads are refused ----------------------------------------
LCPP="gg""uf"   # the llama.cpp format; check-doc-consistency.py refuses the whole word
for ext in safetensors "${LCPP}" pt bin npy npz pkl; do
  refused "a .${ext} file" "has a weight-file extension" "printf 'x' > harness/w.${ext}"
done
refused "a safetensors container under another name" "safetensors container signature" \
  "python3 -c 'import struct,sys; h=b\"{\\\"a\\\":{}}\"; sys.stdout.buffer.write(struct.pack(\"<Q\", len(h)) + h + bytes(16))' > harness/data.txt"
refused "a llama.cpp container under another name" "${LCPP^^} container signature" "printf '${LCPP^^}xxxx' > harness/data.txt"
refused "a NumPy array under another name" "NumPy container signature" "printf '\223NUMPYxx' > harness/data.txt"
refused "a zip (torch checkpoint) under another name" "zip (torch checkpoint) container signature" "printf 'PK\003\004xx' > harness/blob.py"
refused "a non-text file over 1 MiB" "non-text data, over the 1048576-byte cap" \
  "head -c 1100000 /dev/urandom > harness/data.raw; printf '\000' >> harness/data.raw"
refused "non-text files over 1 MiB in total" "added non-text bytes to" \
  "for i in 1 2; do head -c 600000 /dev/urandom > harness/part\${i}.raw; printf '\000' >> harness/part\${i}.raw; done"
refused "a base64 blob in source" "base64/hex-like run" \
  "printf 'W = \"%s\"\n' \"\$(head -c 3000 /dev/urandom | base64 | tr -d '\n')\" > harness/w.py"
refused "a wrapped base64 blob in source (76 columns)" "characters of base64/hex-like runs" \
  "{ echo 'W = ('; head -c 3000 /dev/urandom | base64 -w 76 | sed 's/.*/    \"&\"/'; echo ')'; } > harness/w.py"
refused "a hex blob in source" "base64/hex-like run" \
  "printf 'W = bytes.hex(\"%s\")\n' \"\$(head -c 1000 /dev/urandom | od -An -tx1 | tr -d ' \n')\" > harness/w.py"
refused "an escaped byte string in source" "base64/hex-like run" \
  "python3 -c 'import os; print(\"W = b\\\"\" + \"\".join(\"\\\\x%02x\" % b for b in os.urandom(600)) + \"\\\"\")' > harness/w.py"
refused "a weight table of floats" "numeric literals, over the 16384 cap" \
  "python3 -c 'print(\"W = [\" + \", \".join(str(i / 7.0) for i in range(20000)) + \"]\")' > harness/w.py"
refused "an oversized declaration" "over 4096; the declaration is a handful of keys" \
  "jq --arg v \"\$(printf 'a %.0s' \$(seq 1 3000))\" '.version = \$v' mtp-head.manifest.json > m && mv m mtp-head.manifest.json"
refused "a base64 payload in the declaration" "base64/hex-like run" \
  "jq --arg v \"\$(head -c 1500 /dev/urandom | base64 | tr -d '\n')\" '.version = \$v' mtp-head.manifest.json > m && mv m mtp-head.manifest.json"
refused "a symlink" "is a symlink or a submodule" "ln -s /etc/hostname harness/link"

# --- editable code that names a weights dir or another loader is refused ------
refused "code naming the drafter dir variable" "names 'MLXFAST_DRAFTER_DIR'" \
  "printf 'import os\nD = os.environ[\"MLXFAST_DRAFTER_DIR\"]\n' > harness/x.py"
refused "code naming the target weights dir" "names 'RESIDENT_MODEL_DIR'" \
  "printf 'import os\nD = os.environ[\"RESIDENT_MODEL_DIR\"] + \"/extra\"\n' > harness/x.py"
refused "code naming the drafter repository" "names 'Muse-Glimmer-30B-assistant'" \
  "printf 'R = \"meta-models/Muse-Glimmer-30B-assistant\"\n' > harness/x.py"
refused "from_pretrained on an own path" "names 'from_pretrained'" \
  "printf 'from transformers import AutoModel\nm = AutoModel.from_pretrained(\"./own\")\n' > harness/x.py"
refused "a hub download" "names 'hf_hub'" "printf 'p = hf_hub_download(\"own/model\", \"w\")\n' > harness/x.py"
refused "safetensors load_file" "names 'safetensors'" "printf 'from safetensors.torch import load_file\n' > harness/x.py"
refused "torch.load" "names 'torch.load'" "printf 'import torch\nw = torch.load(p)\n' > vllm/vllm/model_executor/fast.py"
refused "np.load" "names 'np.load'" "printf 'import numpy as np\nw = np.load(p)\n' > harness/x.py"
refused "an unpacking path (base64)" "names 'base64'" "printf 'import base64\n' > harness/x.py"
refused "an unpacking path (zlib)" "names 'zlib'" "printf 'import zlib\n' > harness/x.py"
refused "a weight-file name in code" "names '.safetensors'" "printf 'P = \"own.safetensors\"\n' > harness/x.py"
refused "a changed trusted resident call" "names 'RESIDENT_DRAFT_MODEL_DIR'" \
  "python3 -c 'p = \"harness/vllm_resident/backends.py\"; s = open(p).read(); open(p, \"w\").write(s.replace(\"\\\"RESIDENT_DRAFT_MODEL_DIR\\\"]\", \"\\\"RESIDENT_DRAFT_MODEL_DIR\\\"] + \\\"/../own\\\"\"))'"
refused "a vLLM weights iterator" "names 'weights_iterator'" \
  "printf 'it = pt_weights_iterator(files)\n' > vllm/vllm/model_executor/fast.py"

# --- fail-closed on a missing base -------------------------------------------
out="$(cd "${G}" && env -u BASE_SHA CONTRACT_PATH=benchmark.json "${CHECK}" 2>&1)" && rc=0 || rc=$?
[[ ${rc} -eq 2 ]] && grep -q 'BASE_SHA is not set' <<<"${out}" \
  && ok "no BASE_SHA: refused (rc=2)" || fail "no BASE_SHA: rc=${rc} out=${out}"

if [[ ${fails} -eq 0 ]]; then echo "test-entry-weight-check.sh: all cases passed"; else echo "test-entry-weight-check.sh: ${fails} case(s) FAILED" >&2; exit 1; fi
