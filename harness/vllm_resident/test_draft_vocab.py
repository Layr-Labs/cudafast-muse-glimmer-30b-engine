"""GPU-free unit tests for deterministic reduced draft vocabulary selection."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _load_draft_vocab():
    path = Path(__file__).resolve().parents[2] / "vllm" / "vllm" / "model_executor" / "draft_vocab.py"
    added_stubs: list[str] = []
    if "vllm.logger" not in sys.modules:
        if "vllm" not in sys.modules:
            sys.modules["vllm"] = types.ModuleType("vllm")
            added_stubs.append("vllm")
        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
        sys.modules["vllm.logger"] = vllm_logger
        added_stubs.append("vllm.logger")
    try:
        spec = importlib.util.spec_from_file_location("draft_vocab_under_test", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in reversed(added_stubs):
            sys.modules.pop(name, None)


draft_vocab = _load_draft_vocab()


def _write_synthetic_model(root: Path) -> None:
    vocab: dict[str, int] = {}
    merges: list[str] = []
    next_id = 0

    # Reserve 256 single-byte base tokens (printable ASCII subset + synthetic placeholders).
    b2u = {v: k for k, v in draft_vocab._bytes_to_unicode_inv().items()}
    for b in range(256):
        vocab[b2u[b]] = next_id
        next_id += 1

    # Add 600 English-like pure-ASCII tokens at early ranks so the self-LM learns
    # English character 3/4-grams (e.g. "ing", "tion", "str", "rea", "dat").
    stems = [
        "string", "stream", "action", "option", "buffer", "router", "parser",
        "server", "client", "render", "filter", "sender", "reader", "writer",
        "vector", "matrix", "tensor", "kernel", "thread", "memory", "static",
        "module", "import", "export", "return", "struct", "object", "config",
        "status", "result", "report", "metric", "sample", "target", "source",
        "verify", "format", "encode", "decode", "packet", "socket", "header",
        "footer", "layout", "window", "cursor", "scroll", "button", "dialog",
        "schema", "record", "column", "select", "insert", "update", "delete",
        "create", "branch", "commit", "merged", "staged", "cached", "hashed",
        "sorted", "hashed", "packed", "mapped", "linked", "queued", "locked",
    ]
    suffixes = ["", "s", "ed", "er", "ing", "al", "ic", "ion", "ly", "or"]
    for suf in suffixes:
        for st in stems:
            tok = f"Ġ{st}{suf}"
            if tok not in vocab:
                vocab[tok] = next_id
                merges.append(f"Ġ{st[:2]} {st[2:]}{suf}")
                next_id += 1

    # Add 500 synthetic unusual-cluster Latin tokens at medium ranks.
    for i in range(500):
        tok = f"Ġqx{chr(97 + (i % 26))}z{chr(97 + ((i // 26) % 26))}vj"
        if tok not in vocab:
            vocab[tok] = next_id
            merges.append(f"Ġqx {tok[3:]}")
            next_id += 1

    # Add 100 productive English-like tokens at the very tail of merges.
    tail_english = [
        f"Ġ{a}{b}"
        for a in ("re", "pre", "sub", "un", "de", "con", "trans", "inter", "over", "under")
        for b in ("streamed", "action", "buffer", "parser", "router", "module", "format", "verify", "kernel", "target")
    ]
    for tok in tail_english:
        if tok not in vocab:
            vocab[tok] = next_id
            merges.append(f"{tok[:3]} {tok[3:]}")
            next_id += 1

    # Add a forced special token at the end.
    eos_id = next_id
    added_tokens = [{"id": eos_id, "content": "<|eos|>", "special": True}]

    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": vocab, "merges": merges}, "added_tokens": added_tokens}),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"eos_token_id": eos_id, "vocab_size": eos_id + 1}),
        encoding="utf-8",
    )


class DraftVocabTest(unittest.TestCase):
    def tearDown(self) -> None:
        draft_vocab.configure_draft_vocab("", 0)

    def test_modes_deterministic_and_preserve_tier0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_model(root)

            for mode in ("ascii_lm", "ascii_bpe", "bpe"):
                draft_vocab.configure_draft_vocab(str(root), 1024, mode=mode)
                ids_a = draft_vocab.get_draft_vocab_token_ids()
                self.assertIsNotNone(ids_a)
                assert ids_a is not None
                self.assertEqual(len(ids_a), 1024)
                self.assertEqual(ids_a, tuple(sorted(set(ids_a))))

                # All 256 single-byte tokens + forced EOS must be present.
                for b_id in range(256):
                    self.assertIn(b_id, ids_a)
                cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
                self.assertIn(cfg["eos_token_id"], ids_a)

                # Deterministic on repeated calls.
                draft_vocab.configure_draft_vocab(str(root), 1024, mode=mode)
                self.assertEqual(draft_vocab.get_draft_vocab_token_ids(), ids_a)

    def test_self_lm_promotes_english_tail_over_rare_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_model(root)
            tok_data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
            vocab_map = tok_data["model"]["vocab"]
            b2u_inv = draft_vocab._bytes_to_unicode_inv()
            id_to_str = {tid: s for s, tid in vocab_map.items()}

            tier1 = [(tid - 256, tid) for tid in range(256, len(vocab_map))]
            reranked = draft_vocab._rerank_ascii_tier_self_lm(
                tier1,
                id_to_str,
                b2u_inv,
                keep_top=400,
                lm_top_n=400,
                lm_weight=500.0,
            )
            top_ids = {tid for _, tid in reranked[:750]}
            # Tail English-like tokens should be promoted into the top 750.
            promoted_en = [
                tid
                for s, tid in vocab_map.items()
                if s.startswith("Ġre") and s.endswith(("streamed", "buffer", "parser", "router"))
            ]
            self.assertTrue(promoted_en)
            for tid in promoted_en:
                self.assertIn(tid, top_ids)

    def test_missing_tokenizer_falls_back_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            draft_vocab.configure_draft_vocab(tmp, 1024, mode="ascii_lm")
            self.assertIsNone(draft_vocab.get_draft_vocab_token_ids())

    def test_non_reserved_special_tokens_preserved_and_reserved_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_model(root)
            tok_path = root / "tokenizer.json"
            data = json.loads(tok_path.read_text(encoding="utf-8"))
            base_id = max(e["id"] for e in data["added_tokens"]) + 1
            eot_id = base_id
            res_id = base_id + 1
            data["added_tokens"].append({"id": eot_id, "content": "<|eot|>", "special": True})
            data["added_tokens"].append(
                {"id": res_id, "content": "<|reserved_special_token_42|>", "special": True}
            )
            tok_path.write_text(json.dumps(data), encoding="utf-8")

            draft_vocab.configure_draft_vocab(str(root), 1024, mode="ascii_lm")
            ids = draft_vocab.get_draft_vocab_token_ids()
            self.assertIsNotNone(ids)
            assert ids is not None
            self.assertIn(eot_id, ids)
            self.assertNotIn(res_id, ids)


if __name__ == "__main__":
    unittest.main()
