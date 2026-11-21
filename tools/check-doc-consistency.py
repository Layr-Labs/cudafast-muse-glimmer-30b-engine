#!/usr/bin/env python3
"""check-doc-consistency.py -- the docs must say what the manifest says.

The reference track shipped a TASK.md that listed three editable paths and
called the engine read-only while benchmark.json listed four, including the
engine; it disagreed with its own contract about re-quantizing the head, and
it pinned three draft depths while permitting six. This check is the tripwire
for that class of drift. It reads the values from benchmark.json and the track
fixture and requires every participant-facing document to state them.

This track is SERIAL ONLY, so the check pins the serial shape: one serve depth,
no speculative arm, one 512-token seed, every live golden measured. It also
refuses another track's vocabulary anywhere outside the vendor tree.
A calibration CV limit quoted next to its field name anywhere outside the
vendor tree must equal the fixture's calibration_max_cv_percent. CI runs every
test file tools/ci-tests.sh discovers: .github/workflows/ci.yml must call
`tools/ci-tests.sh run <suite>` for each suite, and every entry of the GPU
exception list tools/ci-test-exceptions.tsv must name a discovered test file
on disk and give a reason.

Exit 0 when consistent, 1 with every finding listed otherwise.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FINDINGS: list[str] = []


def need(cond: bool, msg: str) -> None:
    if not cond:
        FINDINGS.append(msg)


def main() -> int:
    manifest = json.loads((ROOT / "benchmark.json").read_text())
    fixture = json.loads((ROOT / manifest["contractPath"]).read_text())
    docs = {
        name: (ROOT / name).read_text()
        for name in ("TASK.md", "README.md", "AGENTS.md", "docs/participant-contract.md")
    }

    editable = manifest["editablePaths"]
    scoring = manifest["scoring"]
    track = manifest["trackId"]

    # 1. Every participant document names the track and every editable path.
    for name, text in docs.items():
        need(track in text, f"{name}: does not name the track id {track}")
    for name in ("TASK.md", "AGENTS.md", "docs/participant-contract.md"):
        for path in editable:
            need(f"`{path}`" in docs[name], f"{name}: does not name editable path `{path}`")
    # 1b. The count, where a document states one, is the manifest's.
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    for name, text in docs.items():
        for m in re.finditer(r"(one|two|three|four|five|\d) (?:editable paths|entries)", text):
            n = m.group(1)
            need(n in (str(len(editable)), words.get(len(editable), "")),
                 f"{name}: says '{m.group(0)}' but editablePaths has {len(editable)}")

    # 2. The serial envelope. One serve depth, no speculative arm anywhere.
    need(sorted(fixture["allowed_modes"]) == ["serial"],
         f"fixture: allowed_modes is {fixture['allowed_modes']}, expected ['serial']")
    serve_depths = sorted(fixture["serve_configuration"]["depths"])
    need(serve_depths == ["serial"],
         f"fixture: serve_configuration.depths {serve_depths} must be exactly ['serial']")
    need(fixture["mtp_head"].get("permitted_draft_depths") == [],
         "fixture: mtp_head.permitted_draft_depths must be empty (serial only)")
    need(not fixture.get("speculative_oracles"),
         "fixture: speculative_oracles must be empty (serial only)")
    for name in ("TASK.md", "docs/participant-contract.md"):
        need("serial only" in docs[name] or "SERIAL ONLY" in docs[name],
             f"{name}: does not say the track is serial only")

    # 2b. The declaration matches the contract's own source, disabled. The file
    # is an OPTIONAL editable path: the scoring reader (tools/spec-declaration.sh)
    # reads an absent file, an absent source and an absent spec block as the
    # defaults (serial), so this check does too.
    decl_path = ROOT / "mtp-head.manifest.json"
    decl = json.loads(decl_path.read_text()) if decl_path.is_file() else {}
    if "source" in decl:
        need(decl["source"] == fixture["mtp_head"].get("source"),
             f"mtp-head.manifest.json: source {decl['source']!r} disagrees with the fixture's "
             f"mtp_head.source {fixture['mtp_head'].get('source')!r}")
    spec = decl.get("spec") or {}
    need(spec.get("enabled", False) is False and int(spec.get("num_speculative_tokens", 0)) == 0,
         "mtp-head.manifest.json: the declaration must be disabled (serial only)")

    # 2c. Every job measures every live golden; the seed length is the fixture's.
    live = fixture.get("live_goldens") or []
    need(len(live) == 3, f"fixture: live_goldens must name the three pool prompts, got {live}")
    for name in ("TASK.md", "docs/participant-contract.md"):
        need("every live golden" in docs[name].lower() or "Every live golden" in docs[name],
             f"{name}: does not say every job measures every live golden (live_goldens)")
        need(f"{fixture['seed_tokens']}-token" in docs[name],
             f"{name}: does not state the {fixture['seed_tokens']}-token prompt (seed_tokens)")

    # 3. Scoring constants appear in TASK.md and the contract.
    for name in ("TASK.md", "docs/participant-contract.md"):
        text = docs[name]
        need(f"prefill_gain ^ {scoring['scoredExponents']['prefillGainExponent']}" in text
             and f"decode_gain ^ {scoring['scoredExponents']['decodeGainExponent']}" in text,
             f"{name}: the composite formula disagrees with scoredExponents")
        need(str(scoring["prefillSpeedupFloor"]) in text, f"{name}: missing the floor {scoring['prefillSpeedupFloor']}")
        need(str(scoring["decodeSpeedupCeiling"]) in text, f"{name}: missing the ceiling {scoring['decodeSpeedupCeiling']}")
        need(f"prefill ceiling is {scoring['prefillSpeedupCeiling']}" in text
             or f"`prefill_speedup_ceiling` = {scoring['prefillSpeedupCeiling']}" in text,
             f"{name}: missing the prefill ceiling {scoring['prefillSpeedupCeiling']}")
        need(f"`{scoring['kvBackend']}`" in text, f"{name}: missing the kv backend `{scoring['kvBackend']}`")
        need(f"{fixture['official_pairs']} pairs" in text, f"{name}: does not state {fixture['official_pairs']} pairs")
    need(fixture["kv_backend"] == scoring["kvBackend"], "fixture kv_backend disagrees with benchmark.json")
    need(fixture["token_tolerance_budget"] == 0.0,
         "fixture: token_tolerance_budget must be 0.0 on this track (exact match)")
    need("0.0" in docs["docs/participant-contract.md"],
         "docs/participant-contract.md: does not state the 0.0 token-tolerance budget")

    # 4. The fixture and the manifest agree on the scoring mode, and the mode is
    #    the one benchd emits.
    need(scoring.get("mode"), "benchmark.json: scoring.mode is missing")

    # 5. Another track's vocabulary stays out of every tracked file except the
    #    vendor tree, the provenance notices and this checker. The retired term
    #    is assembled from parts on purpose: this file must not itself trip the
    #    sweep it runs (the repository's own gate is a literal grep).
    retired = re.compile(
        r"\b(" + "nemo" + "tron" + r"[0-9A-Za-z_.-]*|ds4|Qwen|qwen[0-9.]*|GGUF|gguf|Gemma|macmon|libds4qwen)\b",
        re.IGNORECASE,
    )
    retired_numbers = re.compile(r"\b496-token\b|gate-english-496")
    # benchd's own vocabulary and the provenance of the trusted CI scripts are
    # not drift: they are names this repository reads from a pinned binary or
    # from the overlay it re-implements. They are scrubbed before the sweep.
    benchd_names = re.compile(
        r"qwen-native-mtp-paired-decode-only"
        r"|qwen-3\.8-mtp-challenge"
        r"|qwen-mtp-ranked-benchmark"
        r"|qwen signer"
        r"|MLXFAST_QWEN_MTP_TRACK_ID"
        r"|DS4_RESIDENT_SOCKET"
        r"|ds4-era"
    )
    allowed_files = {
        "tools/check-doc-consistency.py",
    }
    # NUL-delimited: a tracked name may hold a space or a newline.
    files = [f for f in subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                                       check=True).stdout.split("\0") if f]
    for rel in files:
        if rel.startswith("vllm/") or rel in allowed_files or not rel.endswith(
            (".md", ".sh", ".py", ".rs", ".yml", ".yaml", ".json", ".toml", ".service", ".template", ".example")
        ):
            continue
        for lineno, line in enumerate((ROOT / rel).read_text(errors="replace").splitlines(), 1):
            scrubbed = benchd_names.sub("", line)
            m = retired.search(scrubbed)
            if m:
                FINDINGS.append(f"{rel}:{lineno}: retired-track term '{m.group(0)}'")
            n = retired_numbers.search(scrubbed)
            if n:
                FINDINGS.append(f"{rel}:{lineno}: retired golden name '{n.group(0)}'")

    # 6. CI runs every test file on disk. tools/ci-tests.sh discovers the
    #    test files of each suite and runs each one; ci.yml must call it for
    #    every suite in a real `run:` command (a comment that names it does not
    #    count), and the GPU exception list must agree with the files on disk:
    #    each entry names a discovered test file, once, with a reason.
    ci_runs = set()
    for line in (ROOT / ".github/workflows/ci.yml").read_text().splitlines():
        m = re.match(r"\s*(?:-\s+)?run:\s*tools/ci-tests\.sh run ([a-z]+)\s*$", line)
        if m:
            ci_runs.add(m.group(1))
    for suite in ("tools", "router", "resident"):
        need(suite in ci_runs,
             f".github/workflows/ci.yml: no step runs `tools/ci-tests.sh run {suite}`, so the {suite} test files do not run in CI")
    discovered = subprocess.run(
        [str(ROOT / "tools/ci-tests.sh"), "discover", "all"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    exceptions_path = ROOT / "tools/ci-test-exceptions.tsv"
    if not exceptions_path.is_file():
        FINDINGS.append("tools/ci-test-exceptions.tsv: missing (it may hold no entry, but it must exist)")
    else:
        seen: set[str] = set()
        for lineno, line in enumerate(exceptions_path.read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            path, _, reason = line.partition("\t")
            need(bool(reason.strip()),
                 f"tools/ci-test-exceptions.tsv:{lineno}: {path} gives no reason (<file><TAB><reason>)")
            need(path in discovered,
                 f"tools/ci-test-exceptions.tsv:{lineno}: {path} is not a test file on disk; CI would skip nothing and the list would say it does")
            need(path not in seen, f"tools/ci-test-exceptions.tsv:{lineno}: {path} is listed twice")
            seen.add(path)

    # 7. A calibration CV limit quoted next to its field name is the fixture's
    #    value. benchd reads calibration_max_cv_percent from the fixture, so a
    #    stale copy in prose ("2 % on this track") misstates the gate.
    cv_limit = fixture.get("calibration_max_cv_percent")
    cv_quote = re.compile(r"calibration_max_cv_percent`?\s*(?:\(|=)\s*([0-9]+(?:\.[0-9]+)?)\s*%")
    for rel in files:
        if rel.startswith("vllm/") or not rel.endswith((".md", ".sh", ".py")):
            continue
        for lineno, line in enumerate((ROOT / rel).read_text(errors="replace").splitlines(), 1):
            for m in cv_quote.finditer(line):
                if cv_limit is None or float(m.group(1)) != float(cv_limit):
                    FINDINGS.append(
                        f"{rel}:{lineno}: quotes calibration_max_cv_percent as {m.group(1)} %, "
                        f"the fixture says {cv_limit!r}"
                    )

    if FINDINGS:
        print(f"check-doc-consistency: {len(FINDINGS)} finding(s)")
        for f in FINDINGS:
            print(f"  - {f}")
        return 1
    print("check-doc-consistency: the docs agree with benchmark.json and the fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
