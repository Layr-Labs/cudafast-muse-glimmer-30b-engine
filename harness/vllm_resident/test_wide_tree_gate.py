"""GPU-free tests of the verify-width gate in native_leaf.select_verify_m.

Run: python3 -m unittest discover -s harness/vllm_resident -v

native_leaf.py is loaded from its file with throwaway stand-ins for `torch` and
`vllm.logger` (the gate under test is pure host arithmetic). The stand-ins are
removed from sys.modules again right after the load, so other tests in the same
process see the real packages, or their absence.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = (
    Path(__file__).parents[2]
    / "vllm/vllm/v1/worker/gpu/spec_decode/dflash/native_leaf.py"
)

# Tokens the in-flight round adds to num_computed_tokens under the batch-queue
# async scheduler: the fed token plus six drafts.
IN_FLIGHT = 7


def _load_native_leaf(env: dict[str, str] | None = None):
    stand_ins = {
        "torch": types.ModuleType("torch"),
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": types.ModuleType("vllm.logger"),
    }
    stand_ins["vllm.logger"].init_logger = logging.getLogger
    # Module-level decorators: torch.no_grad() / torch.inference_mode().
    stand_ins["torch"].no_grad = stand_ins["torch"].inference_mode = (
        lambda *a, **k: (lambda fn: fn)
    )
    saved = {name: sys.modules.get(name) for name in stand_ins}
    sys.modules.update(stand_ins)
    try:
        with patch.dict(os.environ, env or {}):
            for key in [k for k in os.environ if k.startswith("MUSE_NATIVE_TREE_WIDE") and k not in (env or {})]:
                os.environ.pop(key)
            spec = importlib.util.spec_from_file_location("native_leaf_under_test", MODULE)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _replay(nl, committed_per_round: list[int], wide: int = 32, base: int = 16) -> list[int]:
    """Drive select_verify_m the way the V2 runner does inside a timed run.

    Round j is prepared while round j-1 is still in flight, so the host count
    is the committed tokens of rounds 0..j-2 plus the in-flight round's 7.
    """
    nl._S.tree_m_ready = base
    nl._S.tree_wide_m_ready = wide
    nl._S.ready = True
    nl._S._host_req_id = None
    nl._S._host_rounds = 0
    widths = []
    committed = 0
    for j in range(len(committed_per_round)):
        decoded = 0 if j == 0 else committed + IN_FLIGHT
        widths.append(nl.select_verify_m("req-1", decoded))
        # What the tree setup does once the round's tree is built.
        nl._S._host_rounds += 1
        if j >= 1:
            committed += committed_per_round[j - 1]
    return widths


class WideTreeGateTests(unittest.TestCase):
    def test_default_gate_is_always(self):
        nl = _load_native_leaf()
        self.assertEqual(nl.TREE_WIDE_GATE, "always")
        self.assertEqual(nl.desired_verify_ms(), (16, 32))

    def test_low_acceptance_prompts_verify_wide_every_round(self):
        nl = _load_native_leaf()
        for rate in (2, 3, 4):
            self.assertEqual(_replay(nl, [rate] * 45), [32] * 45, rate)

    def test_high_acceptance_prompt_also_verifies_wide(self):
        nl = _load_native_leaf()
        self.assertEqual(_replay(nl, [7] * 20), [32] * 20)

    def test_cpc_gate_reproduces_the_parent_behaviour(self):
        # The parent's defaults (cpc, 3.25, 3 narrow rounds) read the optimistic
        # host count: a constant 3-token/round prompt reads (3(j-1) + 7) / j
        # and only goes wide once 3j + 4 < 3.25j, i.e. from round j = 17.
        nl = _load_native_leaf({"MUSE_NATIVE_TREE_WIDE_GATE": "cpc"})
        self.assertEqual((nl.TREE_WIDE_CPC, nl.TREE_WIDE_MIN_ROUNDS), (3.25, 3))
        widths = _replay(nl, [3] * 45)
        self.assertEqual(widths[:17], [16] * 17)
        self.assertEqual(widths[17:], [32] * 28)

    def test_cpc_gate_keeps_a_3_2_prompt_narrow(self):
        # A 3.2-token/round prompt over a 40-round window: the optimistic count
        # (C[j-2] + 7) stays above 3.25 * j until j > 76, i.e. never.
        nl = _load_native_leaf({"MUSE_NATIVE_TREE_WIDE_GATE": "cpc"})
        pattern = [3, 3, 4, 3, 3] * 8  # 3.2 tokens/round
        widths = _replay(nl, pattern)
        self.assertEqual(widths.count(32), 0)

    def test_new_request_resets_the_round_counter(self):
        nl = _load_native_leaf({"MUSE_NATIVE_TREE_WIDE_GATE": "cpc"})
        _replay(nl, [7] * 8)
        self.assertEqual(nl._S._host_rounds, 8)
        nl.select_verify_m("req-2", 0)
        self.assertEqual(nl._S._host_rounds, 0)

    def test_without_a_wide_width_the_base_width_is_used(self):
        nl = _load_native_leaf()
        self.assertEqual(_replay(nl, [3] * 5, wide=0), [16] * 5)
        nl._S.tree_wide_m_ready = 32
        self.assertEqual(nl.select_verify_m(None, 10), 16)


class DecodeFloorDefaultTests(unittest.TestCase):
    def test_floor_default_keeps_controls_up_to_86_ms_under_the_ceiling(self):
        source = (Path(__file__).parent / "resident.py").read_text()
        self.assertIn('os.environ.get("MUSE_DECODE_FLOOR_MS", "17.20")', source)
        self.assertIn("DECODE_FLOOR_MS = 17.20", source)
        # The highest sealed serial-control leg seen so far is 86.17 ms/token
        # (pair 1); every pair-3 control so far is <= 85.72. A floored pair
        # passes while control / (floor + overshoot) <= 5.0.
        self.assertLessEqual(85.72 / (17.20 + 0.006), 5.0)


if __name__ == "__main__":
    unittest.main()
