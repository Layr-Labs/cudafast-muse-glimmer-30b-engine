# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-independent reduced vocabulary for greedy DFlash drafting.

The ranking is derived once at resident boot from public source text in the
submission tree.  Hidden/request tokens are never observed.  Tokens absent
from the corpus are tied by tokenizer id, which also makes the result stable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_TOKEN_IDS: tuple[int, ...] | None = None


def get_draft_vocab_token_ids() -> tuple[int, ...] | None:
    return _TOKEN_IDS


def _special_token_ids(model_dir: Path, vocab_size: int) -> set[int]:
    """Read scalar/list token-id fields from the pinned model config."""
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()

    found: set[int] = set()

    def visit(value: object, key: str = "") -> None:
        if key.endswith("_token_id"):
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, int) and 0 <= item < vocab_size:
                    found.add(item)
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)

    visit(config)
    return found


def _public_corpus_files(tree: Path) -> list[Path]:
    """A bounded multilingual code/prose corpus shipped in the public tree."""
    # Held out from the corpus because local evaluation uses these six files.
    evaluation_files = {
        tree / "README.md",
        tree / "docs/participant-contract.md",
        tree / "harness/kernels/decode6/gemv6_tma4.cu",
        tree / "harness/protocol-adapter/src/session_backend.rs",
        tree / "vllm/vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js",
        tree / "vllm/vllm/model_executor/models/qwen3_dflash.py",
    }
    roots = (tree / "vllm", tree / "harness", tree / "docs")
    suffix_budgets = {
        ".py": 8 << 20,
        ".rs": 2 << 20,
        ".cu": 2 << 20,
        ".cpp": 2 << 20,
        ".cc": 2 << 20,
        ".h": 2 << 20,
        ".js": 2 << 20,
        ".ts": 2 << 20,
        ".md": 2 << 20,
    }
    used = Counter[str]()
    selected: list[Path] = []
    candidates: list[Path] = [tree / "README.md", tree / "TASK.md"]
    for root in roots:
        if root.is_dir():
            candidates.extend(root.rglob("*"))
    for path in sorted(set(candidates)):
        budget = suffix_budgets.get(path.suffix.lower())
        if (
            path in evaluation_files
            or budget is None
            or not path.is_file()
            or used[path.suffix.lower()] >= budget
        ):
            continue
        try:
            size = min(path.stat().st_size, 512 << 10)
        except OSError:
            continue
        used[path.suffix.lower()] += size
        selected.append(path)
    return selected


def configure_draft_vocab(model_dir: str, tree: str, size: int) -> None:
    """Install a deterministic frequency ranking before target weight load."""
    global _TOKEN_IDS
    if size == 0:
        _TOKEN_IDS = None
        return
    if size not in (8192, 16384, 32768, 65536):
        raise ValueError(f"unsupported reduced draft vocabulary size: {size}")

    from tokenizers import Tokenizer

    model_path = Path(model_dir)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if size >= vocab_size:
        _TOKEN_IDS = None
        return

    counts: Counter[int] = Counter()
    corpus_bytes = 0
    corpus_files = _public_corpus_files(Path(tree))
    for path in corpus_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[: 512 << 10]
        except OSError:
            continue
        corpus_bytes += len(text.encode("utf-8"))
        counts.update(tokenizer.encode(text, add_special_tokens=False).ids)

    forced = _special_token_ids(model_path, vocab_size)
    ranked = sorted(range(vocab_size), key=lambda token: (-counts[token], token))
    chosen = list(sorted(forced))
    chosen_set = set(chosen)
    chosen.extend(token for token in ranked if token not in chosen_set)
    _TOKEN_IDS = tuple(chosen[:size])
    logger.info(
        "Reduced DFlash vocabulary ranked %d tokens from %d public files "
        "(%d bytes); selected %d rows",
        len(counts),
        len(corpus_files),
        corpus_bytes,
        size,
    )
