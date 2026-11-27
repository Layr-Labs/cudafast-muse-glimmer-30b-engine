"""CPU unit tests for native_leaf verify-width selection (TREE_M / TREE_WIDE_M, wide 48 default, 32 unchanged)."""

import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NL_PATH = _REPO_ROOT / "vllm/vllm/v1/worker/gpu/spec_decode/dflash/native_leaf.py"
_ENV_KEYS = ("MUSE_NATIVE_TREE_M", "MUSE_NATIVE_TREE_WIDE_M", "MUSE_NATIVE_LEAF", "MUSE_NATIVE_TREE_WIDE_GATE")


def _torch_stub():
    t = types.ModuleType("torch")

    class _Any:
        def __init__(self, *a, **k):
            pass

    t.Tensor = _Any
    t.device = _Any
    t.no_grad = lambda: (lambda f: f)
    t.inference_mode = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *e: False)
    t.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_current_stream_capturing=lambda: False,
        CUDAGraph=_Any,
        Stream=_Any,
    )
    for name in ("int32", "int64", "float32", "bool", "bfloat16"):
        setattr(t, name, name)
    return t


def _load(env: dict[str, str] | None = None):
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    added = []
    stubs = {"torch": _torch_stub(), "vllm": types.ModuleType("vllm")}
    lg = types.ModuleType("vllm.logger")
    lg.init_logger = lambda name: logging.getLogger(name)
    stubs["vllm.logger"] = lg
    for name, mod in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = mod
            added.append(name)
    try:
        spec = importlib.util.spec_from_file_location("_native_leaf_under_test", _NL_PATH)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in added:
            sys.modules.pop(name, None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class NativeLeafWidthTest(unittest.TestCase):
    def test_default_is_16_48(self):
        nl = _load()
        self.assertEqual((nl.TREE_M, nl.TREE_WIDE_M, nl.TREE_MAX_M), (16, 48, 48))
        self.assertEqual(nl.desired_verify_ms(), (16, 48))
        self.assertEqual(nl.desired_verify_m(), 48)

    def test_wide_32_override_unchanged(self):
        nl = _load({"MUSE_NATIVE_TREE_WIDE_M": "32"})
        self.assertEqual((nl.TREE_M, nl.TREE_WIDE_M, nl.TREE_MAX_M), (16, 32, 32))
        self.assertEqual(nl.desired_verify_ms(), (16, 32))
        self.assertEqual(nl.desired_verify_m(), 32)

    def test_invalid_wide_disables_wide(self):
        for bad in ("40", "64", "16", "abc", "0"):
            nl = _load({"MUSE_NATIVE_TREE_WIDE_M": bad})
            self.assertEqual(nl.TREE_WIDE_M, 0, bad)
            self.assertEqual(nl.desired_verify_ms(), (16,), bad)
            self.assertEqual(nl.TREE_MAX_M, 32, bad)

    def test_48_is_never_base(self):
        nl = _load({"MUSE_NATIVE_TREE_M": "48"})
        self.assertEqual(nl.TREE_M, 0)
        self.assertEqual(nl.TREE_WIDE_M, 0)

    def test_wide_must_exceed_base(self):
        nl = _load({"MUSE_NATIVE_TREE_M": "32", "MUSE_NATIVE_TREE_WIDE_M": "48"})
        self.assertEqual(nl.desired_verify_ms(), (32, 48))
        nl = _load({"MUSE_NATIVE_TREE_M": "32", "MUSE_NATIVE_TREE_WIDE_M": "32"})
        self.assertEqual(nl.desired_verify_ms(), (32,))

    def test_select_verify_m_cpc_gate(self):
        nl = _load()
        nl._S.ready = True
        nl._S.tree_m_ready = 16
        nl._S.tree_wide_m_ready = 48
        self.assertEqual(nl.active_verify_m(), 16)
        self.assertEqual(nl.select_verify_m("r", 0), 48)  # first round: wide
        nl._S._host_rounds = 10
        self.assertEqual(nl.select_verify_m("r", 30), 48)  # cpc 3.0 < 4.8 -> wide
        self.assertEqual(nl.select_verify_m("r", 60), 16)  # cpc 6.0 -> base
        nl._S.tree_wide_m_ready = 0
        self.assertEqual(nl.select_verify_m("r", 30), 16)

    def test_prepare_tree_width_cap(self):
        calls = []
        nl = _load()
        nl._S.ready = True
        nl._get_bfx = lambda: types.SimpleNamespace(native_leaf_prepare_tree=lambda p, s, m: calls.append(m))
        self.assertTrue(nl.prepare_tree(None, None, 48))
        self.assertFalse(nl.prepare_tree(None, None, 49))
        nl32 = _load({"MUSE_NATIVE_TREE_WIDE_M": "32"})
        nl32._S.ready = True
        nl32._get_bfx = nl._get_bfx
        self.assertTrue(nl32.prepare_tree(None, None, 32))
        self.assertFalse(nl32.prepare_tree(None, None, 48))
        self.assertEqual(calls, [48, 32])


if __name__ == "__main__":
    unittest.main()
