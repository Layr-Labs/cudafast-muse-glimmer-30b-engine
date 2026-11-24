"""GPU-free tests of the leg's speculative decoder choice (backends.py).

The resident builds vLLM's `speculative_config` from what tools/serve-up.sh
exports: RESIDENT_SPEC_DEPTH, RESIDENT_SPEC_METHOD and, for a dflash leg,
RESIDENT_DRAFT_MODEL_DIR (the contract's pinned drafter, mounted read-only).

Run: python3 -m unittest discover -s harness/vllm_resident -p test_spec_method.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import FakeBackend, VllmConfig, check_spec_method, speculative_config  # noqa: E402
from resident import Session  # noqa: E402


def cfg(depth: int, method: str = "mtp", draft: str = "") -> VllmConfig:
    return VllmConfig(
        model_dir="/w/target",
        depth=depth,
        ctx_size=8192,
        gpu_memory_utilization=0.45,
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=4 << 30,
        ident="test",
        spec_method=method,
        draft_model_dir=draft,
    )


class SpeculativeConfig(unittest.TestCase):
    def test_serial_builds_no_spec(self):
        self.assertIsNone(speculative_config(cfg(0)))
        self.assertIsNone(speculative_config(cfg(0, "dflash", "")))

    def test_mtp_path_is_unchanged(self):
        self.assertEqual(speculative_config(cfg(3)), {"method": "mtp", "num_speculative_tokens": 3})

    def test_dflash_passes_the_pinned_drafter_and_the_depth(self):
        self.assertEqual(
            speculative_config(cfg(4, "dflash", "/srv/drafter")),
            {"method": "dflash", "model": "/srv/drafter", "num_speculative_tokens": 4},
        )


class CheckSpecMethod(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.drafter = self.tmp.name
        with open(os.path.join(self.drafter, "config.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_dflash_with_a_drafter_dir_is_accepted(self):
        check_spec_method("dflash", 2, self.drafter)

    def test_unknown_method_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not one of mtp, dflash"):
            check_spec_method("eagle", 2, "")

    def test_dflash_without_a_drafter_dir_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "needs RESIDENT_DRAFT_MODEL_DIR"):
            check_spec_method("dflash", 2, "")

    def test_dflash_dir_without_config_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "holds no config.json"):
            check_spec_method("dflash", 2, os.path.join(self.drafter, "missing"))

    def test_a_drafter_dir_on_a_serial_or_mtp_leg_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "set for a serial leg"):
            check_spec_method("dflash", 0, self.drafter)
        with self.assertRaisesRegex(RuntimeError, "set for a mtp leg"):
            check_spec_method("mtp", 2, self.drafter)

    def test_from_env_reads_method_and_drafter(self):
        env = {
            "RESIDENT_MODEL_DIR": "/w/target",
            "RESIDENT_SPEC_DEPTH": "3",
            "RESIDENT_CTX_SIZE": "8192",
            "RESIDENT_GPU_MEMORY_UTILIZATION": "0.45",
            "RESIDENT_MAX_NUM_BATCHED_TOKENS": "8192",
            "RESIDENT_KV_CACHE_MEMORY_BYTES": str(4 << 30),
            "RESIDENT_SPEC_METHOD": "dflash",
            "RESIDENT_DRAFT_MODEL_DIR": self.drafter,
        }
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            c = VllmConfig.from_env()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((c.spec_method, c.draft_model_dir, c.depth), ("dflash", self.drafter, 3))


class Hello(unittest.TestCase):
    def test_hello_names_the_drafter_dir_on_a_dflash_leg(self):
        hello = Session(FakeBackend(depth=2, draft_model_path="/srv/drafter")).hello()
        self.assertEqual(hello["mtp_head_path"], "/srv/drafter")
        self.assertTrue(hello["mtp_armed"])

    def test_hello_keeps_the_mtp_form_without_a_drafter(self):
        self.assertEqual(Session(FakeBackend(depth=2)).hello()["mtp_head_path"], "/fake/model#mtp.*")
        self.assertEqual(Session(FakeBackend(depth=0)).hello()["mtp_head_path"], "")


if __name__ == "__main__":
    unittest.main()
