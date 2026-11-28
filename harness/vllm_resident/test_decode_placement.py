"""Unit tests for 2-stage M=32 decode & prefill placement logic in decode_placement.py."""

import importlib.util
import logging
import os
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DP_PATH = _REPO_ROOT / "vllm/vllm/model_executor/kernels/linear/nvfp4/decode_placement.py"


def _load_dp_module(env_overrides: dict[str, str] | None = None):
    """Load decode_placement.py on CPU (with lightweight stubs if torch/vllm are unavailable)."""
    saved_env = {}
    if env_overrides:
        for k, v in env_overrides.items():
            saved_env[k] = os.environ.get(k)
            os.environ[k] = v

    added_stubs = []
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        torch_stub.nn = types.SimpleNamespace(Module=object, Parameter=object)
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch_stub
        added_stubs.append("torch")
    if "vllm" not in sys.modules:
        vllm_pkg = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm_pkg
        added_stubs.append("vllm")
    if "vllm.logger" not in sys.modules:
        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = lambda name: logging.getLogger(name)
        sys.modules["vllm.logger"] = vllm_logger
        added_stubs.append("vllm.logger")

    try:
        spec = importlib.util.spec_from_file_location("_decode_placement_under_test", _DP_PATH)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in added_stubs:
            sys.modules.pop(name, None)
        if env_overrides:
            for k, old in saved_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old


class DecodePlacementLogicTest(unittest.TestCase):
    def test_defaults_and_legacy_fallback(self):
        mod_on = _load_dp_module({"MUSE_PLACE_M32": "1"})
        self.assertTrue(mod_on.PLACE_M32)
        self.assertEqual(mod_on.PLACE_M, 32)
        self.assertEqual(mod_on.S1_ROUNDS, 3)
        self.assertEqual(mod_on.S2_ROUNDS, 7)
        self.assertGreater(mod_on.MIN_SWAP_GAIN, 0.0)

        mod_off = _load_dp_module({"MUSE_PLACE_M32": "0"})
        self.assertFalse(mod_off.PLACE_M32)
        self.assertEqual(mod_off.PLACE_M, 1)
        self.assertEqual(mod_off.S1_ROUNDS, mod_off.ROUNDS)
        self.assertEqual(mod_off.S2_ROUNDS, 0)
        self.assertEqual(mod_off.MIN_SWAP_GAIN, 0.0)

    def test_round_order_avoids_boundary_repeats(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        rng = random.Random(12345)
        prev = 0
        for _ in range(200):
            order = mod._make_round_order(6, rng, prev)
            self.assertEqual(sorted(order), [0, 1, 2, 3, 4, 5])
            self.assertNotEqual(order[0], prev)
            prev = order[-1]

    def test_two_stage_eliminates_winners_curse(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        # 2 homes + 20 spares. Spare #5 got lucky in Stage 1 (0.50ms) but is truly slow in Stage 2 (1.30ms).
        # Spare #12 was 2nd in Stage 1 (0.75ms) and confirmed fast in Stage 2 (0.74ms).
        count = 2
        cands = list(range(count + 20))
        s1_times = [1.00, 1.10] + [1.05 + 0.01 * i for i in range(20)]
        s1_times[count + 5] = 0.50   # lucky Stage-1 fluke
        s1_times[count + 12] = 0.75  # genuinely fast
        s2_true = [1.00, 1.10] + [1.05 + 0.01 * i for i in range(20)]
        s2_true[count + 5] = 1.30    # exposed as slow in Stage 2!
        s2_true[count + 12] = 0.74   # confirmed fast!

        calls = []

        def fake_time_fn(sub_cands, rounds, flush_pc, seed):
            calls.append((len(sub_cands), rounds, flush_pc, seed))
            if len(calls) == 1:
                return [s1_times[c] for c in sub_cands]
            return [s2_true[c] for c in sub_cands]

        t1, fin_ids, score_map = mod._rank_candidates_2stage(
            fake_time_fn, cands, count, s1_rounds=3, s2_rounds=7, flush_s1=True, flush_s2=True
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], 3)
        self.assertEqual(calls[1][1], 7)
        self.assertNotEqual(calls[0][3], calls[1][3])  # independent RNG seeds
        self.assertIn(0, fin_ids)
        self.assertIn(1, fin_ids)
        self.assertIn(count + 5, fin_ids)
        self.assertIn(count + 12, fin_ids)

        kept, pairs = mod._plan_swaps(count, fin_ids, score_map, min_gain=0.003)
        # Only home 1 (1.10ms) should move to spare #12 (0.74ms); lucky fluke #5 must NOT be chosen!
        self.assertEqual(pairs, [(1, count + 12)])
        self.assertEqual(kept, {0})

    def test_plan_swaps_enforces_min_gain_margin(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        # Home 0 = 1.000ms, Spare 2 = 0.9985ms (0.15% noise gain < 0.3% min_gain -> skip!)
        # Home 1 = 1.050ms, Spare 3 = 0.980ms  (6.7% real gain >= 0.3% -> swap!)
        score_map = {0: 1.000, 1: 1.050, 2: 0.9985, 3: 0.980}
        kept, pairs = mod._plan_swaps(2, [0, 1, 2, 3], score_map, min_gain=0.003)
        self.assertEqual(pairs, [(1, 3)])
        self.assertEqual(kept, {0})

    def test_merge_s1_s2_scores_corrects_thermal_drift(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        # Stage 2 ran 10% warmer globally; finalist 0 went 1.0 -> 1.1, finalist 1 went 0.9 -> 0.99.
        # Non-finalist 2 had s1=1.05 -> projected to 1.155 (correctly stays slower than finalist 0's 1.10!).
        s1 = [1.00, 0.90, 1.05]
        s2 = {0: 1.10, 1: 0.99}
        merged = mod._merge_s1_s2_scores(s1, s2)
        self.assertAlmostEqual(merged[0], 1.10, places=5)
        self.assertAlmostEqual(merged[1], 0.99, places=5)
        self.assertAlmostEqual(merged[2], 1.155, places=5)
        self.assertLess(merged[0], merged[2])

    def test_can_append_slow_hold_fixes_100_slab_bug(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        mod._SLOW_HOLD.clear()
        mod._SLOW_HOLD.extend([None] * 175)  # post full_ram_scan state!
        with mock.patch.object(mod, "_mem_available", return_value=40 << 30):
            self.assertTrue(mod._can_append_slow_hold(64 << 20))
        with mock.patch.object(mod, "_mem_available", return_value=30 << 30):
            # Below HOST_HEADROOM_BYTES (24 GiB) + 8 GiB margin -> refuse!
            self.assertFalse(mod._can_append_slow_hold(64 << 20))
        mod._SLOW_HOLD.clear()

    def test_time_candidates_does_not_misclassify_gate_up_as_lm_pad128(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})

        class FakeDev:
            index = 0

        class FakeTensor:
            def __init__(self, shape=(1,), ptr=0x1000, dtype="bf16"):
                self.shape = shape
                self._ptr = ptr
                self.device = FakeDev()
                self.dtype = dtype

            def data_ptr(self):
                return self._ptr

            def __mod__(self, _):
                return self

            def __sub__(self, _):
                return self

            def float(self):
                return self

            def to(self, _):
                return self

            def reshape(self, *shape):
                return FakeTensor(shape, self._ptr + 16, self.dtype)

            def contiguous(self):
                return self

            def new_empty(self, shape, dtype=None):
                return FakeTensor(shape, self._ptr + 32, dtype or self.dtype)

        class FakeCtx:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        fake_torch = types.SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            int64="i64",
            arange=lambda n, device=None: FakeTensor((n,), 0x2000),
            zeros=lambda shape, dtype=None, device=None: FakeTensor(shape if isinstance(shape, tuple) else (shape,), 0x3000, dtype),
            cuda=types.SimpleNamespace(
                device=lambda d: FakeCtx(),
                current_device=lambda: 0,
                current_stream=lambda d=None: types.SimpleNamespace(cuda_stream=0x999),
            ),
        )

        ilv_m_calls = []
        lmpost_m_calls = []
        pad128_queries = []

        fake_m6 = types.ModuleType("muse_marlin6_gemv")
        fake_m6.decode_linear6 = lambda *a, **kw: None
        fake_m6.is_tma4_ilv = lambda s: True
        fake_m6.is_lm_pad128 = lambda s: (pad128_queries.append(s) or True)
        fake_m6.is_lm_split2 = lambda s: False
        fake_m6._get_lmpost_m_bufs = lambda d: (FakeTensor((4112,), 0x4000), FakeTensor((96, 32, 128), 0x4100))

        def _fake_ilv_m(*args):
            ilv_m_calls.append(args)
            return 0

        def _fake_lmpost_m(*args):
            lmpost_m_calls.append(args)
            return 0

        fake_m6._sk_tma4_ilv_m = _fake_ilv_m
        fake_m6._sk_tma4_ilv_lmpost_m = _fake_lmpost_m
        fake_m6._sk_tma4_ilv_lmpost = lambda *a: 0

        layer = types.SimpleNamespace(
            weight_global_scale=FakeTensor((1,), 0x5000),
            workspace=FakeTensor((48,), 0x5100),
            _marlin_decode_palette=FakeTensor((64,), 0x5200),
            _marlin_decode_scales6=FakeTensor((100, 32), 0x5300),
            marlin_decode6_fixed_grid=False,
            org_vocab_size=202048,
        )

        def fake_time_calls(run_fn, cands, rounds, evict_buf, flush_per_cand=False, seed=0):
            for c in cands:
                run_fn(c)
            return [1.0 for _ in cands]

        with mock.patch.object(mod, "torch", fake_torch), \
             mock.patch.object(mod, "_time_calls", side_effect=fake_time_calls), \
             mock.patch.dict(sys.modules, {"muse_marlin6_gemv": fake_m6}):
            # 1) gate_up (39936x6656): MUST NOT call is_lm_pad128 and MUST pass eff_n=39936 (NOT 202112!)
            mod._time_candidates(None, layer, [FakeTensor((1,), 0x6000)], 39936, 6656, 3, FakeTensor(), probe_m=32)
            self.assertEqual(len(pad128_queries), 0)
            self.assertEqual(len(ilv_m_calls), 1)
            # args: 8 ptrs, then (eff_m, eff_n, k, blocks, dev_idx, stream)
            self.assertEqual(ilv_m_calls[0][8:13], (32, 39936, 6656, 48, 0))

            # 2) lm_head (202048x6656) with pad128=True: MUST call _sk_tma4_ilv_lmpost_m with eff_m=32, eff_n=202112
            mod._time_candidates(None, layer, [FakeTensor((1,), 0x6100)], 202048, 6656, 3, FakeTensor(), probe_m=32)
            self.assertEqual(len(pad128_queries), 1)
            self.assertEqual(len(lmpost_m_calls), 1)
            # args: 10 ptrs, then (org_vocab, eff_m, eff_n, k, lm_blocks, dev_idx, stream)
            self.assertEqual(lmpost_m_calls[0][10:16], (202048, 32, 202112, 6656, 48, 0))

            # 3) lm_head (202048x6656) with pad128=False, split2=True: MUST clamp eff_m=16, eff_n=202048, blocks=96
            fake_m6.is_lm_pad128 = lambda s: False
            fake_m6.is_lm_split2 = lambda s: True
            mod._time_candidates(None, layer, [FakeTensor((1,), 0x6200)], 202048, 6656, 3, FakeTensor(), probe_m=32)
            self.assertEqual(len(lmpost_m_calls), 2)
            self.assertEqual(lmpost_m_calls[1][10:16], (202048, 16, 202048, 6656, 96, 0))

    def test_buddyinfo_sponge_exports_preserved(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1", "MUSE_BUDDY_SPONGE": "0"})
        self.assertEqual(mod.take_buddyinfo_sponge(), 0)
        mod.release_buddyinfo_sponge()

    def test_drop_model_page_cache_runs_before_buddyinfo_sponge(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1", "MUSE_BUDDY_SPONGE": "1"})
        call_order = []

        def fake_drop():
            call_order.append("drop")

        def fake_usable(_min_chunk=2 << 20):
            call_order.append("usable")
            return 10 << 30  # below reserve -> exits cleanly without libcuda

        with mock.patch.object(mod, "_drop_model_page_cache", side_effect=fake_drop), \
             mock.patch.object(mod, "_usable_buddyinfo_bytes", side_effect=fake_usable):
            self.assertEqual(mod.take_buddyinfo_sponge(), 0)
        self.assertEqual(call_order, ["drop", "usable"])

    def test_partition_fast_pool_bandwidth_gates_slow_slabs(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        # Clean boot: 200 unused slabs all within 1.08x of best (0.530 ms -> cut 0.5724 ms)
        clean_unused = [(0.535 + 0.0001 * i, f"b{i}") for i in range(200)]
        fast_c, rest_c = mod._partition_fast_pool(clean_unused, best_time=0.530, max_fast_keep=180, ratio=1.08)
        self.assertEqual(len(fast_c), 180)
        self.assertEqual(len(rest_c), 20)

        # Fragmented boot (like w68b_full_37): best=0.555 ms (cut=0.5994 ms), only 12 unused slabs <= 0.5994,
        # remaining 188 are slow 0.66-0.79 ms -> MUST NOT poison fast_pool with slow slabs!
        frag_unused = [(0.560 + 0.003 * i, f"f{i}") for i in range(12)] + [
            (0.660 + 0.001 * i, f"s{i}") for i in range(188)
        ]
        fast_f, rest_f = mod._partition_fast_pool(frag_unused, best_time=0.555, max_fast_keep=180, ratio=1.08)
        self.assertEqual(len(fast_f), 12)
        self.assertEqual(len(rest_f), 188)

    def test_adaptive_retry_trigger_clean_vs_fragmented_boot(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        # Clean boot (w68b_det12x2: 6656x19968 min=0.287, med=0.289, max=0.290) -> NO retry!
        clean_times = [0.287 + 0.0001 * i for i in range(40)]
        self.assertEqual(mod._select_retry_indices(clean_times, 1.08), [])

        # Fragmented boot (w68b_full_37: 6656x19968 min=0.299, 10 fast + 30 slow @ 0.362-0.383) -> retries ONLY the 30 slow!
        frag_times = [0.299 + 0.001 * i for i in range(10)] + [0.350 + 0.001 * i for i in range(30)]
        slow_ids = mod._select_retry_indices(frag_times, 1.08)
        self.assertEqual(slow_ids, list(range(10, 40)))

        # Legacy disabled (ratio=0.0) -> NO retry
        self.assertEqual(mod._select_retry_indices(frag_times, 0.0), [])

    def test_wave83_fast_pool_tight_ratio_with_fallback_and_cross_group_floor(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        self.assertAlmostEqual(mod.FAST_POOL_RATIO, 1.035)
        self.assertAlmostEqual(mod.ADAPTIVE_RETRY_RATIO, 1.04)
        self.assertAlmostEqual(mod.DOWN_GU_GBPS_RATIO, 0.922)

        # Clean boot with 140 slabs <= 1.035x and 60 slabs at 1.05..1.07x -> keeps ONLY the 140 tight slabs!
        mixed = [(0.530 + 0.0001 * i, f"t{i}") for i in range(140)] + [
            (0.558 + 0.0002 * i, f"m{i}") for i in range(60)
        ]
        fast_t, rest_t = mod._partition_fast_pool(
            mixed, best_time=0.530, max_fast_keep=180, ratio=1.035, fallback_ratio=1.08, min_slabs=110
        )
        self.assertEqual(len(fast_t), 140)
        self.assertEqual(len(rest_t), 60)

        # Mildly fragmented boot: only 50 slabs <= 1.035x (<110) -> auto-relaxes to 1.08x (110 more slabs <= 0.5724)
        sparse = [(0.530 + 0.0002 * i, f"t{i}") for i in range(50)] + [
            (0.555 + 0.0001 * i, f"m{i}") for i in range(100)
        ] + [(0.650 + 0.001 * i, f"s{i}") for i in range(50)]
        fast_s, rest_s = mod._partition_fast_pool(
            sparse, best_time=0.530, max_fast_keep=180, ratio=1.035, fallback_ratio=1.08, min_slabs=110
        )
        self.assertEqual(len(fast_s), 150)
        self.assertEqual(len(rest_s), 50)

        # Cross-group floor check for 6656x19968 (w82c_clean_det12x2: min=0.2929, med=0.2977 vs floor 0.2935)
        down_degraded = [0.2929] * 5 + [0.2977] * 35
        self.assertEqual(mod._select_retry_indices(down_degraded, 1.04), [])
        self.assertEqual(mod._select_retry_indices(down_degraded, 1.04, floor_max_ms=0.2935), list(range(5, 40)))
        # Clean 6656x19968 (med=0.2898 <= 0.2935) -> NO retry!
        down_clean = [0.2867 + 0.00015 * i for i in range(40)]
        self.assertEqual(mod._select_retry_indices(down_clean, 1.04, floor_max_ms=0.2935), [])

    def test_ensure_slow_hold_headroom_flushes_cache_before_trimming(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        mod._SLOW_HOLD.clear()
        mod._SLOW_HOLD.extend([object() for _ in range(16)])
        state = {"avail": 22 << 30, "flushes": 0}

        def fake_empty_cache():
            state["flushes"] += 1
            # Flushing cleared rejected_hold recovers 14 GiB -> 36 GiB >= 32 GiB without losing any _SLOW_HOLD!
            state["avail"] = 36 << 30

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=fake_empty_cache)
        )
        with mock.patch.object(mod, "torch", fake_torch), \
             mock.patch.object(mod, "_mem_available", side_effect=lambda: state["avail"]):
            mod.ensure_slow_hold_headroom(32 << 30)
        self.assertEqual(state["flushes"], 1)
        self.assertEqual(len(mod._SLOW_HOLD), 16)
        mod._SLOW_HOLD.clear()

    def test_spares_room_enforces_hard_os_mem_floor(self):
        mod = _load_dp_module({"MUSE_PLACE_M32": "1"})
        mod._SLOW_HOLD.clear()
        mod._SLOW_HOLD.extend([object() for _ in range(8)])
        hold_list = [object() for _ in range(8)]
        state = {"avail": 14 << 30, "cached": 24 << 30, "flushes": 0}

        def fake_empty_cache():
            state["flushes"] += 1
            if state["cached"] > 0:
                state["avail"] += 12 << 30
                state["cached"] = 0

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                empty_cache=fake_empty_cache,
                memory_reserved=lambda: state["cached"],
                memory_allocated=lambda: 0,
            )
        )
        with mock.patch.object(mod, "torch", fake_torch), \
             mock.patch.object(mod, "_mem_available", side_effect=lambda: state["avail"]):
            # 1) Hard 16 GiB OS floor: cached_free (24 GiB) MUST NOT mask avail=14 GiB!
            self.assertEqual(mod._spares_allowed(10, 64 << 20), 0)
            # 2) Even with allow_slow_trim=False (is_small attn), low OS MemAvailable (<18 GiB)
            #    must flush hold_list + empty_cache() to recover OS headroom before returning!
            allowed = mod._ensure_spares_room(10, 32 << 20, hold_list, allow_slow_trim=False)
            self.assertGreaterEqual(state["flushes"], 1)
            self.assertGreaterEqual(state["avail"], 20 << 30)
            self.assertGreater(allowed, 0)
            # 3) >128 MiB per-iteration guard stops when avail - payload < 16 GiB
            state["avail"] = 16 << 30
            self.assertFalse(mod._os_headroom_ok(256 << 20))
            self.assertTrue(mod._os_headroom_ok(64 << 20))
        mod._SLOW_HOLD.clear()


if __name__ == "__main__":
    unittest.main()
