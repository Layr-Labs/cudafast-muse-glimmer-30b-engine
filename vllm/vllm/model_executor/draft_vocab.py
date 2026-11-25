# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language-universal BPE merge-rank reduced vocabulary for greedy DFlash drafting.

Derived deterministically once at resident boot strictly from the pinned checkpoint's
`tokenizer.json` and `config.json` (zero external weights, zero Python-only corpus bias,
< 0.05 s startup).

Modes (`MUSE_DRAFT_VOCAB_MODE`, default `ascii_bpe`):
  - `ascii_bpe`: forced special tokens -> all 256 raw byte tokens -> all pure-ASCII
    BPE tokens (covering Python, Rust, Go, SQL, C/C++, TS/JS, Shell, YAML, JSON, Markdown,
    and English prose proportionally to pretraining frequency) ordered by exact BPE merge
    rank -> remaining non-ASCII BPE tokens ordered by merge rank.
  - `bpe`: forced special tokens -> all non-placeholder BPE tokens ordered strictly by
    exact BPE merge rank (tokenizer ID order).

Combined with the dynamic <= 512-row exact-NVFP4 prompt-tail kernel (`MUSE_DRAFT_PROMPT_TAIL=1`),
every rare or non-ASCII identifier appearing in the prompt is also scored with zero quantization
error at ~15 us overhead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_TOKEN_IDS: tuple[int, ...] | None = None
_VOCAB_SIZE: int = 0
_CODE_SYNTAX_CHARS: frozenset[str] = frozenset("_.:/\\()[]{}=<>+*%&\t\n;`")


def get_draft_vocab_token_ids() -> tuple[int, ...] | None:
    return _TOKEN_IDS


def _bytes_to_unicode_inv() -> dict[str, int]:
    """Inverse of the standard GPT-2 / Qwen ByteLevel BPE char->byte map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


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


def _decode_pure_ascii(tok_str: str, b2u_inv: dict[str, int]) -> str | None:
    """Return decoded ASCII string if every byte of ByteLevel BPE token is < 128, else None."""
    if not tok_str:
        return None
    chars: list[str] = []
    for ch in tok_str:
        b = b2u_inv.get(ch)
        if b is not None:
            if b >= 128:
                return None
            chars.append(chr(b))
        else:
            o = ord(ch)
            if o >= 128:
                return None
            chars.append(ch)
    return "".join(chars)


def _is_pure_ascii_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    """True if every decoded byte of a ByteLevel BPE token is ASCII (< 128)."""
    return _decode_pure_ascii(tok_str, b2u_inv) is not None


def _is_single_byte_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    return len(tok_str) == 1 and (tok_str in b2u_inv or ord(tok_str) < 256)


def configure_draft_vocab(
    model_dir: str, size: int, mode: str = "ascii_bpe"
) -> None:
    """Install a deterministic BPE merge-rank vocabulary subset before target weight load."""
    global _TOKEN_IDS, _VOCAB_SIZE
    if size <= 0:
        _TOKEN_IDS = None
        return
    if size % 128 != 0 or size < 1024:
        raise ValueError(
            f"MUSE_DRAFT_VOCAB_SIZE must be 0 or a multiple of 128 >= 1024, got {size}"
        )

    model_path = Path(model_dir)
    tok_path = model_path / "tokenizer.json"
    if not tok_path.is_file():
        logger.warning("draft_vocab: %s not found; keeping full lm_head", tok_path)
        _TOKEN_IDS = None
        return

    data = json.loads(tok_path.read_text(encoding="utf-8"))
    model_obj = data.get("model", {})
    vocab_map: dict[str, int] = model_obj.get("vocab", {})
    added_list = data.get("added_tokens", [])
    added_ids: set[int] = set()
    non_reserved_special_ids: set[int] = set()
    id_to_str: dict[int, str] = {}
    for s, idx in vocab_map.items():
        if isinstance(idx, int) and idx >= 0:
            id_to_str[idx] = s
    for entry in added_list:
        idx = entry.get("id")
        if isinstance(idx, int) and idx >= 0:
            added_ids.add(idx)
            content = str(entry.get("content", ""))
            id_to_str.setdefault(idx, content)
            if entry.get("special") and "reserved_special_token" not in content:
                non_reserved_special_ids.add(idx)

    vocab_size = (max(id_to_str) + 1) if id_to_str else 0
    _VOCAB_SIZE = vocab_size
    if size >= vocab_size or vocab_size <= 0:
        _TOKEN_IDS = None
        return

    # Build exact BPE merge rank if `merges` is present; base tokens get rank -1,
    # unmerged / added tokens fall back to tokenizer ID.
    merge_rank: dict[int, int] = {}
    merges = model_obj.get("merges") or []
    for r, m in enumerate(merges):
        if isinstance(m, str):
            merged = m.replace(" ", "", 1)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merged = f"{m[0]}{m[1]}"
        else:
            continue
        tid = vocab_map.get(merged)
        if tid is not None and tid not in merge_rank:
            merge_rank[tid] = r

    forced = _special_token_ids(model_path, vocab_size) | {
        idx for idx in non_reserved_special_ids if 0 <= idx < vocab_size
    }
    b2u_inv = _bytes_to_unicode_inv()
    mode_norm = mode.strip().lower()
    use_ascii_tier = mode_norm != "bpe"
    code_boost = os.environ.get("MUSE_DRAFT_VOCAB_CODE_BOOST", "1").strip().lower() not in (
        "0",
        "",
        "false",
        "no",
    )

    tier0: list[int] = []  # forced special tokens (<|eot|>, <|eom|>, etc.) + all single-byte base tokens
    tier1: list[tuple[int, int]] = []  # ASCII BPE tokens (or all BPE tokens in `bpe` mode)
    tier1_code_ids: set[int] = set()
    tier2: list[tuple[int, int]] = []  # non-ASCII BPE tokens (in `ascii_bpe` mode)
    tier3: list[int] = []  # unforced added / placeholder tokens

    for tid in range(vocab_size):
        s = id_to_str.get(tid, "")
        if tid in forced:
            tier0.append(tid)
            continue
        if tid in added_ids or not s:
            tier3.append(tid)
            continue
        if _is_single_byte_token(s, b2u_inv):
            tier0.append(tid)
            continue
        rank = merge_rank.get(tid, len(merges) + tid)
        if not use_ascii_tier:
            tier1.append((rank, tid))
        else:
            da = _decode_pure_ascii(s, b2u_inv)
            if da is not None:
                tier1.append((rank, tid))
                if code_boost and any(ch in _CODE_SYNTAX_CHARS for ch in da):
                    tier1_code_ids.add(tid)
            else:
                tier2.append((rank, tid))

    tier0.sort()
    tier1.sort()
    tier2.sort()
    tier3.sort()

    chosen: list[int] = []
    chosen.extend(tier0)
    if use_ascii_tier and code_boost and size >= 65536 and tier1_code_ids:
        core_keep = max(0, size - 10304)
        chosen.extend(tid for _, tid in tier1[:core_keep])
        chosen.extend(tid for _, tid in tier1[core_keep:] if tid in tier1_code_ids)
        chosen.extend(tid for _, tid in tier1[core_keep:] if tid not in tier1_code_ids)
    else:
        chosen.extend(tid for _, tid in tier1)
    chosen.extend(tid for _, tid in tier2)
    chosen.extend(tier3)
    # Keeping sorted ID order preserves ascending original-ID locality in DRAM and makes ties deterministic.
    selected = tuple(sorted(chosen[:size]))
    _TOKEN_IDS = selected
    ascii_count = sum(
        1 for tid in selected if _is_pure_ascii_token(id_to_str.get(tid, ""), b2u_inv)
    )
    logger.info(
        "Reduced DFlash vocabulary (%s, code_boost=%d): selected %d / %d tokens "
        "(forced+bytes=%d, ascii=%d, code_syntax=%d, merges=%d)",
        mode_norm,
        int(code_boost),
        len(selected),
        vocab_size,
        len(tier0),
        ascii_count,
        sum(1 for tid in selected if tid in tier1_code_ids),
        len(merges),
    )
