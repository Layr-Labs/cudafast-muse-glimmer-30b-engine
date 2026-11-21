#!/usr/bin/env bash
# test-check-doc-odd-names.sh -- tools/check-doc-consistency.py reads the
# tracked file list NUL-delimited, so an honest editable file whose name holds a
# space (or a newline) does not break CI.
#
# The checker used to split `git ls-files` output on whitespace, so
# "harness/fast adapter.py" became "harness/fast" and "adapter.py" and the
# check died opening a file that does not exist. This test copies the tracked
# tree, adds that file (and one with a newline in its name where the
# filesystem takes one), indexes the copy in a repository of its own and runs
# the checker there. Hermetic: no network, no GPU.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
T="${WORK}/tree"
mkdir -p "${T}"
(cd "${ROOT_DIR}" && git ls-files -z | tar --null -T - -cf -) | tar -xf - -C "${T}" \
  || { echo "FAIL: cannot copy the tree" >&2; exit 1; }
printf 'print("fast")\n' > "${T}/harness/fast adapter.py"
printf '# notes\n' > "${T}/harness/odd"$'\n'"name.md" 2>/dev/null || true

if out="$({ git -C "${T}" init -q && git -C "${T}" add -A && python3 -B "${T}/tools/check-doc-consistency.py"; } 2>&1)"; then
  echo "ok: check-doc-consistency.py passes a tree with a file named 'harness/fast adapter.py'"
  echo "PASS: test-check-doc-odd-names.sh"
  exit 0
fi
echo "FAIL: check-doc-consistency.py fails a tree with an editable file whose name holds a space: $(tail -3 <<<"${out}")" >&2
exit 1
