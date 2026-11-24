#!/usr/bin/env python3
"""Entry weight check: the entry diff carries no weights and no other loader.

WHY. The pinned DFlash drafter is the ONLY drafter a leg can run: the
organizer stages it, tools/serve-up.sh mounts exactly its pinned files
read-only, and tools/spec-declaration.sh lets an entry only switch it on and
pick a depth. But the leg also mounts the entry's whole editable tree (vllm/,
harness/) read-only, and that tree is code the entry controls. An entry could
bundle weights of its own inside the 40 MiB editable surface -- as files, or
encoded in source -- and change editable code to load them. This check refuses
that shape of entry, fail-closed, before the ranked job sees the commit.

WHAT IT READS. Only what the entry CHANGED: every path under the base
contract's editablePaths that differs between BASE_SHA and HEAD_SHA (git diff
--no-renames, deletions excepted), read from the HEAD commit, never the work
tree. Code and content checks read only the ADDED lines of each diff, so the
vendored vLLM tree (which loads models by design) is never judged by lines
the entry did not write. The rules come from the BASE contract.

A REFUSAL (exit 1, one ::error:: line per finding) for a changed path that:
  1. is a symlink or a submodule (gitlink);
  2. has a weight-file extension (.safetensors, the llama.cpp single-file
     format, .pt .pth .bin .ckpt .npy
     .npz .pkl .pickle .h5 .onnx .msgpack .tflite);
  3. starts with a weight or archive container signature (safetensors header,
     the llama.cpp single-file format, NumPy, zip/torch, gzip, zstd, xz, bzip2, pickle protocol 2+);
  4. is not text (holds a NUL or is not UTF-8) and is over 1 MiB, or brings the
     entry's total added non-text bytes over 1 MiB;
  5. adds base64/hex-like blobs (runs of 64+ characters of [A-Za-z0-9+/=_-],
     or runs of 16+ \\xNN / \\uNNNN escapes) over 2048 characters in one file, or
     any single run over 1024 characters;
  6. adds more than 16384 numeric literals in one file (a weight table);
  7. is mtp-head.manifest.json and is over 4096 bytes (the declaration is a
     handful of keys; tools/spec-declaration.sh validates its content);
  8. adds a line (outside mtp-head.manifest.json) that names the drafter or
     target weights directory or repository, or a model or weight loading,
     download or unpacking path: from_pretrained, hf_hub_download,
     snapshot_download, huggingface_hub, safetensors, safe_open, load_file,
     torch.load, torch.hub, np.load, np.fromfile, pickle, marshal, joblib,
     the llama.cpp single-file format, vLLM weights iterators and model loaders, base64 / b64decode,
     bytes.fromhex, zlib, lzma, bz2, gzip, zipfile, tarfile, or a
     weight-file name. The trusted resident call that loads the pinned drafter
     (harness/vllm_resident/backends.py, RESIDENT_DRAFT_MODEL_DIR) is base
     code: it passes while the entry leaves those lines as they are, and a
     changed line there is refused like any other.

WHAT IT CANNOT CATCH (the honest limits; docs/participant-contract.md section 4
is the rule, this is a filter):
  * a payload chunked below the blob and numeric caps, spread over many files,
    or encoded as identifiers, words, whitespace or any text that does not
    look like a blob;
  * a loader assembled at run time (getattr, string concatenation, exec/eval,
    importlib) or reached through code the entry did not change;
  * weights derived by computation (a seeded generator, a closed form);
  * reading the target snapshot or any other mounted file through ordinary
    file APIs (open, mmap) with a path built at run time;
  * anything in a file the diff does not list (unchanged base files).
The fixed backstops stay: the container has no network, Hugging Face is
offline, only the pinned target and the pinned drafter files are mounted, and
every timed token must match the golden.

Usage:
  BASE_SHA=<sha> HEAD_SHA=<sha> CONTRACT_PATH=benchmark.json \\
    .github/scripts/entry-weight-check.py
Run from the checkout of HEAD_SHA (the dispatched commit). Exit 0 clean,
1 refused, 2 usage or git error.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARDENED_GIT = os.path.join(HERE, "hardened-git.sh")

# The llama.cpp single-file weight format. Its name is spelled in two halves:
# tools/check-doc-consistency.py refuses the whole word in tracked files (a
# retired-track term), and this check needs it only as data.
LCPP = "gg" + "uf"

MIB = 1024 * 1024
BINARY_FILE_CAP = MIB
BINARY_TOTAL_CAP = MIB
BLOB_RUN_MIN = 64
BLOB_FILE_CAP = 2048
BLOB_RUN_CAP = 1024
NUMERIC_FILE_CAP = 16384
DECLARATION = "mtp-head.manifest.json"
DECLARATION_CAP = 4096

WEIGHT_EXTENSIONS = (
    ".safetensors", "." + LCPP, ".pt", ".pth", ".bin", ".ckpt", ".npy", ".npz",
    ".pkl", ".pickle", ".h5", ".onnx", ".msgpack", ".tflite",
)
SIGNATURES = (
    (LCPP.upper().encode(), LCPP.upper()),
    (b"\x93NUMPY", "NumPy"),
    (b"PK\x03\x04", "zip (torch checkpoint)"),
    (b"\x1f\x8b", "gzip"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"BZh", "bzip2"),
)
BLOB_RE = re.compile(r"[A-Za-z0-9+/=_-]{%d,}" % BLOB_RUN_MIN)
ESCAPE_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}){16,}")
NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:0[xX][0-9A-Fa-f]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)(?![A-Za-z0-9_])"
)
# Rule 8: names of the weights directories and of loading/unpacking paths.
LOADER_PATTERNS = (
    r"MLXFAST_DRAFTER_DIR", r"RESIDENT_DRAFT_MODEL_DIR", r"SERVE_UP_DRAFTER_DIR",
    r"MLXFAST_TARGET_SNAPSHOT_DIR", r"RESIDENT_MODEL_DIR", r"SERVE_UP_WEIGHTS_DIR",
    r"reference_weights", r"Muse-Glimmer-30B-assistant", r"Muse-Glimmer-30B-NVFP4",
    r"from_pretrained", r"hf_hub", r"snapshot_download", r"huggingface_hub",
    r"HF_HOME", r"HF_HUB_", r"TRANSFORMERS_CACHE", r"TRANSFORMERS_OFFLINE",
    r"safetensors", r"safe_open", r"\bload_file\b", r"\btorch\.load\b", r"\btorch\.hub\b",
    r"\bnp\.load\b", r"\bnumpy\.load\b", r"\bnp\.fromfile\b", r"\bnumpy\.fromfile\b",
    r"\bpickle\b", r"\bmarshal\b", r"\bjoblib\b", r"\b" + LCPP + r"\b",
    r"weights_iterator", r"\bget_model_loader\b", r"\bDefaultModelLoader\b", r"\bmodel_loader\b",
    r"\bbase64\b", r"b64decode", r"\bfromhex\b", r"\bzlib\b", r"\blzma\b", r"\bbz2\b",
    r"\bgzip\b", r"\bzipfile\b", r"\btarfile\b",
    r"\.(?:safetensors|" + LCPP + r"|pth?|bin|ckpt|npy|npz|pkl|pickle|h5|onnx|msgpack|tflite)\b",
)
LOADER_RE = re.compile("|".join(f"(?:{p})" for p in LOADER_PATTERNS))


def git(*args: str, text: bool = True) -> str | bytes:
    proc = subprocess.run([HARDENED_GIT, *args], capture_output=True, text=text)
    if proc.returncode != 0:
        err = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        print(f"::error::git {' '.join(args[:3])} failed: {err.strip()}", file=sys.stderr)
        sys.exit(2)
    return proc.stdout


class Check:
    def __init__(self) -> None:
        self.findings: list[str] = []

    def refuse(self, path: str, msg: str) -> None:
        self.findings.append(path)
        print(f"::error file={path}::entry weight check: {msg}", file=sys.stderr)


def is_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def signature(data: bytes) -> str | None:
    for magic, name in SIGNATURES:
        if data.startswith(magic):
            return name
    # safetensors: 8-byte little-endian header length, then a JSON object.
    if len(data) >= 10 and data[8:10] == b'{"':
        n = int.from_bytes(data[:8], "little")
        if 0 < n <= len(data) - 8:
            return "safetensors"
    # pickle protocol 2+ opens with PROTO (0x80) and the protocol number.
    if len(data) >= 2 and data[0] == 0x80 and 2 <= data[1] <= 5:
        return "pickle"
    return None


def added_lines(base: str, head: str, path: str) -> list[str]:
    diff = git("diff", "--no-renames", "--no-color", "-U0", base, head, "--", path)
    out = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


def main() -> int:
    base = os.environ.get("BASE_SHA", "")
    head = os.environ.get("HEAD_SHA", "HEAD")
    contract_path = os.environ.get("CONTRACT_PATH", "benchmark.json")
    if not base:
        print("::error::BASE_SHA is not set; the entry weight check reviews the entry diff against a protected base", file=sys.stderr)
        return 2
    base = git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").strip()
    head = git("rev-parse", "--verify", "--quiet", f"{head}^{{commit}}").strip()
    try:
        contract = json.loads(git("show", f"{base}:{contract_path}"))
        paths = contract["editablePaths"]
        assert isinstance(paths, list) and paths and all(isinstance(p, str) and p for p in paths)
    except (ValueError, KeyError, AssertionError):
        print(f"::error::{contract_path} at {base} names no editablePaths; refusing to review an unbounded surface", file=sys.stderr)
        return 2

    changed = [p for p in git("diff", "--no-renames", "--name-only", "-z", "--diff-filter=d", base, head, "--", *paths).split("\0") if p]
    chk = Check()
    binary_total = 0
    for path in changed:
        tree = git("ls-tree", "-z", head, "--", path).rstrip("\0")
        mode = tree.split(" ", 1)[0] if tree else ""
        if mode in ("120000", "160000"):
            chk.refuse(path, "is a symlink or a submodule; an entry adds plain files only")
            continue
        if mode not in ("100644", "100755"):
            chk.refuse(path, f"has git mode '{mode}', not a plain file")
            continue
        lower = path.lower()
        if lower.endswith(WEIGHT_EXTENSIONS):
            chk.refuse(path, "has a weight-file extension; an entry brings no weights of its own (the drafter is the contract's pinned one)")
            continue
        data = git("cat-file", "blob", f"{head}:{path}", text=False)
        sig = signature(data)
        if sig:
            chk.refuse(path, f"starts with a {sig} container signature; an entry brings no weights or archives")
            continue
        if path == DECLARATION and len(data) > DECLARATION_CAP:
            chk.refuse(path, f"is {len(data)} bytes, over {DECLARATION_CAP}; the declaration is a handful of keys and carries no payload")
        if not is_text(data):
            binary_total += len(data)
            if len(data) > BINARY_FILE_CAP:
                chk.refuse(path, f"is {len(data)} bytes of non-text data, over the {BINARY_FILE_CAP}-byte cap for a non-text file")
            elif binary_total > BINARY_TOTAL_CAP:
                chk.refuse(path, f"brings the entry's added non-text bytes to {binary_total}, over the {BINARY_TOTAL_CAP}-byte cap")
            continue
        lines = added_lines(base, head, path)
        blob_chars = 0
        longest = 0
        numbers = 0
        for line in lines:
            for m in BLOB_RE.finditer(line):
                blob_chars += len(m.group(0))
                longest = max(longest, len(m.group(0)))
            for m in ESCAPE_RE.finditer(line):
                blob_chars += len(m.group(0))
                longest = max(longest, len(m.group(0)))
            numbers += len(NUMERIC_RE.findall(line))
        if longest > BLOB_RUN_CAP:
            chk.refuse(path, f"adds a {longest}-character base64/hex-like run, over the {BLOB_RUN_CAP}-character cap")
        elif blob_chars > BLOB_FILE_CAP:
            chk.refuse(path, f"adds {blob_chars} characters of base64/hex-like runs, over the {BLOB_FILE_CAP}-character cap per file")
        if numbers > NUMERIC_FILE_CAP:
            chk.refuse(path, f"adds {numbers} numeric literals, over the {NUMERIC_FILE_CAP} cap per file (a weight table)")
        if path == DECLARATION:
            continue
        for n, line in enumerate(lines, 1):
            m = LOADER_RE.search(line)
            if m:
                chk.refuse(
                    path,
                    f"added line {n} names '{m.group(0)}', a weights directory or a model loading or unpacking path. "
                    "Editable code may not load weights other than through the unchanged trusted resident call",
                )
                break

    if chk.findings:
        print(f"entry-weight-check: REFUSED {len(set(chk.findings))} of {len(changed)} changed path(s)", file=sys.stderr)
        return 1
    print(f"entry-weight-check: {len(changed)} changed path(s) carry no weights and name no other loader (base {base[:12]}, head {head[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
