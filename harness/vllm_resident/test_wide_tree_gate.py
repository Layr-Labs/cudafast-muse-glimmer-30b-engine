"""GPU-free unit tests for M<=32 5-bit KV remap packing and adaptive M=16/M=32 host gate."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path


def _load_native_leaf_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "vllm"
        / "v1"
        / "worker"
        / "gpu"
        / "spec_decode"
        / "dflash"
        / "native_leaf.py"
    )
    added: list[str] = []
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: (lambda fn: fn)  # type: ignore[attr-defined]
        fake_torch.Tensor = object  # type: ignore[attr-defined]
        fake_torch.device = object  # type: ignore[attr-defined]
        sys.modules["torch"] = fake_torch
        added.append("torch")
    if "vllm.logger" not in sys.modules:
        if "vllm" not in sys.modules:
            sys.modules["vllm"] = types.ModuleType("vllm")
            added.append("vllm")
        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
        sys.modules["vllm.logger"] = vllm_logger
        added.append("vllm.logger")
    try:
        spec = importlib.util.spec_from_file_location("native_leaf_under_test", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)


class WideTreeGateTests(unittest.TestCase):
    def test_5bit_remap_roundtrip_all_32_rows_depth6(self) -> None:
        """6 levels of 5-bit physical row IDs (0..31) fit strictly in positive int32 (< 2^30)."""
        chains = [
            [7, 8, 9, 10, 11, 12],
            [15, 16, 20, 24, 29, 31],
            [31, 30, 29, 28, 27, 26],
        ]
        for chain in chains:
            packed = 0
            for dp_idx, r in enumerate(chain):
                packed |= (r & 0x1F) << (5 * dp_idx)
            # Fits inside signed int32 without sign-bit wrap.
            self.assertLess(packed, 1 << 30)
            unpacked = [(packed >> (5 * j)) & 0x1F for j in range(len(chain))]
            self.assertEqual(unpacked, chain)

    def test_adaptive_cpc_gate_switches_cleanly_zero_sync(self) -> None:
        nl = _load_native_leaf_module()
        nl._S.ready = True
        nl._S.tree_m_ready = 16
        nl._S.tree_wide_m_ready = 32
        nl._S._host_req_id = None
        nl._S._host_rounds = 0
        nl.TREE_WIDE_GATE = "cpc"
        nl.TREE_WIDE_CPC = 4.0

        # Step 0 of a new request starts wide (M=32) with decoded_tokens=0 (num_computed == prefill_len).
        self.assertEqual(nl.select_verify_m("req-hard", 0), 32)
        nl._S._host_rounds += 1
        # Hard prompt: 1 committed token on round 0 -> decoded_tokens=1 must NOT reset _host_rounds!
        self.assertEqual(nl.select_verify_m("req-hard", 1), 32)
        self.assertEqual(nl._S._host_rounds, 1)
        nl._S._host_rounds += 1
        # Hard prompt: 6 committed tokens after 2 rounds (cpc = 3.0 < 4.0) -> stays M=32.
        self.assertEqual(nl.select_verify_m("req-hard", 6), 32)
        nl._S._host_rounds += 1
        # Hard prompt: 11 committed tokens after 3 rounds (cpc = 3.67 < 4.0) -> stays M=32.
        self.assertEqual(nl.select_verify_m("req-hard", 11), 32)

        # Easy prompt: switches to M=16 as soon as running cpc >= 4.0!
        self.assertEqual(nl.select_verify_m("req-easy", 0), 32)
        nl._S._host_rounds += 1
        # Committed 5 tokens on round 0 -> cpc = 5.0 >= 4.0 -> drops to M=16!
        self.assertEqual(nl.select_verify_m("req-easy", 5), 16)
        nl._S._host_rounds += 1
        # Committed 10 tokens in 2 rounds -> stays M=16 (0 extra cost on easy prompts!).
        self.assertEqual(nl.select_verify_m("req-easy", 10), 16)
        nl._S._host_rounds += 1
        # If easy prompt suddenly gets hard (11 tokens in 3 rounds, cpc=3.67 < 4.0) -> re-widens to M=32!
        self.assertEqual(nl.select_verify_m("req-easy", 11), 32)

    def test_desired_verify_ms_default_and_adaptive(self) -> None:
        nl = _load_native_leaf_module()
        nl.ENABLED = True
        nl.TREE_M = 16
        nl.TREE_WIDE_M = 0
        self.assertEqual(nl.desired_verify_ms(), (16,))
        self.assertEqual(nl.desired_verify_m(), 16)

        nl.TREE_WIDE_M = 32
        self.assertEqual(nl.desired_verify_ms(), (16, 32))
        self.assertEqual(nl.desired_verify_m(), 32)

    def test_default_wide_cpc_and_step0_reset(self) -> None:
        nl = _load_native_leaf_module()
        self.assertGreaterEqual(nl.TREE_WIDE_CPC, 4.2)
        self.assertLessEqual(nl.TREE_WIDE_CPC, 5.0)

        class _Buf:
            def __init__(self) -> None:
                self.ops: list[tuple[str, object]] = []

            def __getitem__(self, _idx):
                return self

            def zero_(self):
                self.ops.append(("zero_", None))
                return self

            def fill_(self, val):
                self.ops.append(("fill_", val))
                return self

        nl._S.ready = True
        nl._S.leaf_flag = _Buf()
        nl._S.prev_ns = _Buf()
        nl._S.alt_ids = _Buf()
        nl._S.alt_margins = _Buf()
        nl._S.tree_top = _Buf()
        nl._S.tree_vals = _Buf()
        nl._S.req_acc_state = _Buf()
        nl._S._last_req_id = "req-prev"
        nl._S._host_req_id = "req-prev"
        nl._S._host_rounds = 12

        # Zero-prep prefill flush calls on_non_leaf_step(req_id, 0) BEFORE speculator.propose():
        # resets request tracking so Step 0 setup_step sees matching _last_req_id and keeps leaves!
        nl.on_non_leaf_step("req-new", 0)
        self.assertEqual(nl._S._last_req_id, "req-new")
        self.assertEqual(nl._S._host_req_id, "req-new")
        self.assertEqual(nl._S._host_rounds, 0)
        self.assertIn(("fill_", -1), nl._S.tree_top.ops)
        self.assertIn(("fill_", -1e30), nl._S.tree_vals.ops)

        # Transient ceiling clamp (0 < num_draft_tokens < 6) must NOT hard-latch req_acc_state[2]=2.
        nl._S.leaf_flag.ops.clear()
        nl._S.prev_ns.ops.clear()
        nl._S.req_acc_state.ops.clear()
        nl.on_non_leaf_step("req-new", 4)
        self.assertIn(("zero_", None), nl._S.leaf_flag.ops)
        self.assertIn(("fill_", 7), nl._S.prev_ns.ops)
        self.assertEqual(nl._S.req_acc_state.ops, [])


if __name__ == "__main__":
    unittest.main()
