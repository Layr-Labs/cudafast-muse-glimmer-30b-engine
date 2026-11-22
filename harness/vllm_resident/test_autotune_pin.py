"""The fixed config rule of vllm/vllm/compilation/autotune_pin.py, on the candidate lists Inductor
generated for this model on GB10 (no GPU, torch or vLLM needed: the module's rule is plain Python)."""

import importlib.util
import os
import unittest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "vllm", "vllm", "compilation", "autotune_pin.py"
)
_spec = importlib.util.spec_from_file_location("autotune_pin", _PATH)
pin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pin)


def c(num_warps, n_spills=0, num_stages=1, **kwargs):
    return (kwargs, num_warps, num_stages, n_spills)


class PickConfigIndex(unittest.TestCase):
    def test_qk_norm_reduction_keeps_a_whole_row_block(self):
        # q/k-norm sum of squares over head_dim 128, ReductionHint.DEFAULT: six candidates
        cands = [
            c(2, R0_BLOCK=128, XBLOCK=2),
            c(4, R0_BLOCK=8, XBLOCK=64),
            c(4, R0_BLOCK=128, XBLOCK=4),
            c(16, R0_BLOCK=64, XBLOCK=64),
            c(8, R0_BLOCK=128, XBLOCK=8),
            c(8, R0_BLOCK=4, XBLOCK=64),
        ]
        self.assertEqual(pin.pick_config_index(cands, 16), 4)

    def test_rms_norm_keeps_the_single_pass_block(self):
        # input RMSNorm over 6656 fused with the FP8 cast: persistent-width block or a 1024 loop
        cands = [c(16, R0_BLOCK=8192, XBLOCK=1), c(8, R0_BLOCK=1024, XBLOCK=1)]
        self.assertEqual(pin.pick_config_index(cands, 16), 0)

    def test_pointwise_keeps_more_warps(self):
        cands = [c(4, XBLOCK=1024), c(8, XBLOCK=512)]
        self.assertEqual(pin.pick_config_index(cands, 16), 1)

    def test_spilling_candidates_are_skipped(self):
        cands = [c(16, n_spills=40, R0_BLOCK=8192, XBLOCK=1), c(8, R0_BLOCK=1024, XBLOCK=1)]
        self.assertEqual(pin.pick_config_index(cands, 16), 1)

    def test_all_spilling_falls_back_to_the_rule(self):
        cands = [c(8, n_spills=40, R0_BLOCK=1024, XBLOCK=1), c(16, n_spills=40, R0_BLOCK=8192, XBLOCK=1)]
        self.assertEqual(pin.pick_config_index(cands, 16), 1)

    def test_ties_keep_the_earlier_candidate(self):
        cands = [c(4, XBLOCK=1024), c(4, XBLOCK=1024)]
        self.assertEqual(pin.pick_config_index(cands, 16), 0)

    def test_combo_kernel_block_names(self):
        cands = [c(4, R0_BLOCK_0=64, R0_BLOCK_1=128, XBLOCK_0=8), c(4, R0_BLOCK_0=128, R0_BLOCK_1=128, XBLOCK_0=8)]
        self.assertEqual(pin.pick_config_index(cands, 16), 1)

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            pin.pick_config_index([], 16)


if __name__ == "__main__":
    unittest.main()
