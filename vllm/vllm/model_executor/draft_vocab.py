# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language-universal BPE merge-rank reduced vocabulary for greedy DFlash drafting.

Derived deterministically once at resident boot strictly from the pinned checkpoint's
`tokenizer.json` and `config.json` (zero external weights, zero Python-only corpus bias,
< 0.05 s startup).

Modes (`MUSE_DRAFT_VOCAB_MODE`, default `ascii_lm`):
  - `ascii_lm`: forced special tokens -> all 256 raw byte tokens -> top 50,000 pure-ASCII
    BPE tokens preserved verbatim in exact merge-rank order -> remaining pure-ASCII BPE
    tokens re-ranked by exact BPE merge rank plus a self-bootstrapped character 3/4-gram
    log-likelihood trained strictly on the tokenizer's own top-12,000 pure-ASCII BPE
    tokens (demotes rare foreign Latin-script tail entries in favor of productive English
    and code tokens at zero extra DRAM bytes) -> remaining non-ASCII BPE tokens.
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
import math
import os
from collections import Counter
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_TOKEN_IDS: tuple[int, ...] | None = None
_VOCAB_SIZE: int = 0


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


def _is_pure_ascii_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    """True if every decoded byte of a ByteLevel BPE token is ASCII (< 128)."""
    if not tok_str:
        return False
    for ch in tok_str:
        b = b2u_inv.get(ch)
        if b is not None:
            if b >= 128:
                return False
        else:
            if ord(ch) >= 128:
                return False
    return True


def _is_single_byte_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    return len(tok_str) == 1 and (tok_str in b2u_inv or ord(tok_str) < 256)


_ENG_CURLY_SUFFIXES = frozenset(("s", "t", "re", "ve", "ll", "d", "m"))
_GREEK_MATH_CPS = frozenset(range(0x0391, 0x03AA)) | frozenset(range(0x03B1, 0x03CA))


def _is_english_typo_math_utf8_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    """True for high-frequency English typography, math/logic operators, units, or single Greek symbols."""
    if not tok_str:
        return False
    raw = bytearray()
    for ch in tok_str:
        b = b2u_inv.get(ch)
        if b is None:
            o = ord(ch)
            if o >= 256:
                return False
            b = o
        raw.append(b)
    try:
        dec = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # Exclude NBSP+word tokens (e.g. '\xa0the') and long whitespace-only runs.
    if "\xa0" in dec and any(c.isalpha() for c in dec):
        return False
    if len(dec) > 8 and not dec.strip("\xa0 \u200b\u200c\u200d\u2009\u2002\u2003\u202f\n\t"):
        return False
    # If curly apostrophe ’ is followed by ASCII letters, allow only English contractions (’s, ’t, ’re, ’ve, ’ll, ’d, ’m).
    if "’" in dec:
        parts = dec.split("’")
        for tail in parts[1:]:
            alpha_prefix = []
            for ch in tail:
                if "a" <= ch <= "z" or "A" <= ch <= "Z":
                    alpha_prefix.append(ch.lower())
                else:
                    break
            if alpha_prefix and "".join(alpha_prefix) not in _ENG_CURLY_SUFFIXES:
                return False
    n_non_ascii = 0
    n_greek = 0
    for c in dec:
        cp = ord(c)
        if cp < 0x80:
            continue
        n_non_ascii += 1
        if cp in _GREEK_MATH_CPS:
            n_greek += 1
            if n_greek > 1:
                return False
        elif not (
            0x00A0 <= cp <= 0x00BF
            or cp in (0x00D7, 0x00F7)
            or 0x2000 <= cp <= 0x22FF
        ):
            return False
    return n_non_ascii > 0


def _is_valid_utf8_prose_token(tok_str: str, b2u_inv: dict[str, int]) -> bool:
    """True if decoded bytes form valid UTF-8 Latin/Greek/Punctuation/Math (no split bytes or CJK)."""
    if not tok_str:
        return False
    raw = bytearray()
    for ch in tok_str:
        b = b2u_inv.get(ch)
        if b is None:
            o = ord(ch)
            if o >= 256:
                return False
            b = o
        raw.append(b)
    try:
        dec = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    has_non_ascii = False
    for c in dec:
        cp = ord(c)
        if cp < 0x80:
            continue
        has_non_ascii = True
        if not (
            0x00A0 <= cp <= 0x024F  # Latin-1 Supplement + Latin Extended-A/B
            or 0x0370 <= cp <= 0x03FF  # Greek (alpha, beta, pi, sigma, ...)
            or 0x2000 <= cp <= 0x22FF  # General Punctuation (’, “, ”, —, –, …, •), Superscripts, Arrows, Math
        ):
            return False
    return has_non_ascii


def _decode_ascii_alpha(tok_str: str, b2u_inv: dict[str, int]) -> str:
    """Extract lowercase ASCII alphabetic characters from a pure-ASCII BPE token."""
    chars: list[str] = []
    for ch in tok_str:
        b = b2u_inv.get(ch)
        if b is None:
            b = ord(ch)
        if 65 <= b <= 90:
            chars.append(chr(b + 32))
        elif 97 <= b <= 122:
            chars.append(chr(b))
    return "".join(chars)


def _rerank_ascii_tier_self_lm(
    tier1: list[tuple[int, int]],
    id_to_str: dict[int, str],
    b2u_inv: dict[str, int],
    *,
    keep_top: int = 50000,
    lm_top_n: int = 12000,
    lm_weight: float = 150000.0,
    lm_center: float = 2.6,
    min_alpha_len: int = 3,
) -> list[tuple[int, int]]:
    """Re-rank pure-ASCII BPE tail tokens using a self-bootstrapped char 3/4-gram LM.

    Trained strictly from the tokenizer's own top-`lm_top_n` pure-ASCII BPE tokens
    (weighted by inverse log-rank). Preserves the top `keep_top` pure-ASCII tokens
    verbatim in merge-rank order, and adjusts only the lower tail (`tier1[keep_top:]`)
    so productive English/code tokens outrank rare foreign Latin-script tail entries.
    """
    if len(tier1) <= keep_top or lm_top_n <= 0:
        return tier1

    alphas = [_decode_ascii_alpha(id_to_str.get(tid, ""), b2u_inv) for _, tid in tier1]

    c3: Counter[str] = Counter()
    c2: Counter[str] = Counter()
    c4: Counter[str] = Counter()
    c3_ctx: Counter[str] = Counter()
    fit_n = min(lm_top_n, len(tier1))
    for idx in range(fit_n):
        alpha = alphas[idx]
        if len(alpha) < 2:
            continue
        w = 1.0 / math.log2(idx + 4.0)
        pad = f"^{alpha}$"
        n_pad = len(pad)
        for i in range(n_pad - 2):
            c3[pad[i : i + 3]] += w
            c2[pad[i : i + 2]] += w
        if n_pad >= 4:
            for i in range(n_pad - 3):
                c4[pad[i : i + 4]] += w
                c3_ctx[pad[i : i + 3]] += w

    if not c3 or not c4:
        return tier1

    vocab_alpha = 28.0
    smooth = 0.05
    log_smooth = math.log(smooth)
    default_denom = math.log(smooth * vocab_alpha)
    c2_denom = {k: math.log(v + smooth * vocab_alpha) for k, v in c2.items()}
    c3_denom = {k: math.log(v + smooth * vocab_alpha) for k, v in c3_ctx.items()}
    log_c3 = {k: math.log(v + smooth) - c2_denom[k[:2]] for k, v in c3.items()}
    log_c4 = {k: math.log(v + smooth) - c3_denom[k[:3]] for k, v in c4.items()}

    tail_scored: list[tuple[float, int, int]] = []
    for idx in range(keep_top, len(tier1)):
        rank, tid = tier1[idx]
        alpha = alphas[idx]
        if len(alpha) < min_alpha_len:
            eff = float(rank)
        else:
            pad = f"^{alpha}$"
            m3 = len(pad) - 2
            s3 = 0.0
            for i in range(m3):
                tri = pad[i : i + 3]
                lp = log_c3.get(tri)
                if lp is not None:
                    s3 += lp
                else:
                    s3 += log_smooth - c2_denom.get(tri[:2], default_denom)
            s3 /= m3

            m4 = len(pad) - 3
            s4 = 0.0
            for i in range(m4):
                quad = pad[i : i + 4]
                lp = log_c4.get(quad)
                if lp is not None:
                    s4 += lp
                else:
                    s4 += log_smooth - c3_denom.get(quad[:3], default_denom)
            s4 /= m4

            sc = 0.5 * (s3 + s4)
            eff = rank - lm_weight * (sc + lm_center)
        tail_scored.append((eff, rank, tid))

    tail_scored.sort(key=lambda item: (item[0], item[1], item[2]))
    head = tier1[:keep_top]
    return head + [(rank, tid) for _, rank, tid in tail_scored]


def configure_draft_vocab(
    model_dir: str, size: int, mode: str = "ascii_lm"
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
    id_to_str: dict[int, str] = {}
    for s, idx in vocab_map.items():
        if isinstance(idx, int) and idx >= 0:
            id_to_str[idx] = s
    for entry in added_list:
        idx = entry.get("id")
        if isinstance(idx, int) and idx >= 0:
            added_ids.add(idx)
            id_to_str.setdefault(idx, str(entry.get("content", "")))

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

    forced = _special_token_ids(model_path, vocab_size)
    for entry in added_list:
        idx = entry.get("id")
        content = str(entry.get("content", ""))
        if (
            isinstance(idx, int)
            and 0 <= idx < vocab_size
            and entry.get("special")
            and "reserved_special_token" not in content
        ):
            forced.add(idx)
    b2u_inv = _bytes_to_unicode_inv()
    mode_norm = mode.strip().lower()
    use_ascii_tier = mode_norm != "bpe"
    use_utf8_tier = (
        os.environ.get("MUSE_DRAFT_VOCAB_UTF8", "1" if size >= 120000 else "0").strip()
        not in ("0", "", "false", "no")
    )
    default_utf8_swap = "256" if (use_ascii_tier and 65536 <= size < 120000) else "0"
    try:
        utf8_swap = max(0, int(os.environ.get("MUSE_DRAFT_UTF8_SWAP", default_utf8_swap).strip() or "0"))
    except ValueError:
        utf8_swap = 0
    utf8_swap_max_rank = int(os.environ.get("MUSE_DRAFT_UTF8_MAX_RANK", "90000"))

    tier0: list[int] = []  # forced special tokens + all single-byte base tokens
    tier1: list[tuple[int, int]] = []  # ASCII BPE tokens (or all BPE tokens in `bpe` mode)
    tier2_typo: list[tuple[int, int]] = []  # English typography / math / 1-Greek UTF-8 tokens
    tier2_utf8: list[tuple[int, int]] = []  # valid UTF-8 Latin/Greek/Punctuation/Math BPE tokens
    tier2: list[tuple[int, int]] = []  # remaining non-ASCII BPE tokens
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
        if not use_ascii_tier or _is_pure_ascii_token(s, b2u_inv):
            tier1.append((rank, tid))
        elif utf8_swap > 0 and rank < utf8_swap_max_rank and _is_english_typo_math_utf8_token(s, b2u_inv):
            tier2_typo.append((rank, tid))
        elif use_utf8_tier and _is_valid_utf8_prose_token(s, b2u_inv):
            tier2_utf8.append((rank, tid))
        else:
            tier2.append((rank, tid))

    tier0.sort()
    tier1.sort()
    if mode_norm == "ascii_lm":
        tier1 = _rerank_ascii_tier_self_lm(tier1, id_to_str, b2u_inv)
    tier2_typo.sort()
    tier2_utf8.sort()
    tier2.sort()
    tier3.sort()

    chosen: list[int] = []
    chosen.extend(tier0)
    if utf8_swap > 0 and tier2_typo and len(tier0) + len(tier1) > size:
        swap_ids = [tid for _, tid in tier2_typo[:utf8_swap]]
        ascii_keep = max(0, size - len(tier0) - len(swap_ids))
        chosen.extend(tid for _, tid in tier1[:ascii_keep])
        chosen.extend(swap_ids)
        chosen.extend(tid for _, tid in tier1[ascii_keep:])
    else:
        chosen.extend(tid for _, tid in tier1)
        chosen.extend(tid for _, tid in tier2_typo)
    chosen.extend(tid for _, tid in tier2_utf8)
    chosen.extend(tid for _, tid in tier2)
    chosen.extend(tier3)
    # Sort final selected IDs? Keeping sorted ID order preserves ascending original-ID locality
    # in DRAM and makes ties deterministic.
    selected = tuple(sorted(chosen[:size]))
    _TOKEN_IDS = selected
    ascii_count = sum(
        1 for tid in selected if _is_pure_ascii_token(id_to_str.get(tid, ""), b2u_inv)
    )
    logger.info(
        "Reduced DFlash vocabulary (%s): selected %d / %d tokens "
        "(forced+bytes=%d, ascii=%d, merges=%d)",
        mode_norm,
        len(selected),
        vocab_size,
        len(tier0),
        ascii_count,
        len(merges),
    )
