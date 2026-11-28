"""GPU-free tests of the resident's bookkeeping and wire (FakeBackend).

Run: python3 -m unittest discover -s harness/vllm_resident -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import DEFAULT, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import FakeBackend, VllmBackend  # noqa: E402
from resident import IMPLEMENTED_DEPTH, RefusedError, Session, serve_connection  # noqa: E402

PROMPT = list(range(100, 164))


def serial_stream(n: int) -> list[int]:
    s = Session(FakeBackend(depth=0))
    out, t = [], s.sync(PROMPT)
    for _ in range(n):
        t = s.eval(t)
        out.append(t)
    return out


class SessionTests(unittest.TestCase):
    def test_serial_eval_advances_one_token(self):
        s = Session(FakeBackend(depth=0))
        first = s.sync(PROMPT)
        self.assertEqual(s.eval(first), s.frontier.token)
        self.assertEqual(s.ctx, PROMPT + [first])

    def test_split_serial_run_preserves_steps_and_top_k_mode(self):
        for argmax_only in (False, True):
            with self.subTest(argmax_only=argmax_only):
                backend = FakeBackend()
                session = Session(backend)
                reference = Session(FakeBackend())
                pending = session.sync(PROMPT, argmax_only=argmax_only)
                expected_pending = reference.sync(PROMPT)
                self.assertEqual(pending, expected_pending)

                steps = 0
                with patch.object(backend, "step", wraps=backend.step) as step:
                    for count in (1, 31, 96):
                        expected = []
                        for _ in range(count):
                            expected_pending = reference.eval(expected_pending)
                            expected.append(expected_pending)
                        tokens = session.serial_run(pending, count)
                        pending = tokens[-1]
                        steps += count
                        self.assertEqual(tokens, expected)
                        self.assertEqual(step.call_count, steps)
                        self.assertEqual(session.ctx, reference.ctx)
                        self.assertEqual(session.argmax_only, argmax_only)
                        if argmax_only:
                            self.assertEqual(session.frontier.top, ())
                            with self.assertRaisesRegex(RefusedError, "argmax_only"):
                                session.top_logits(8)
                        else:
                            self.assertEqual(session.top_logits(8), reference.top_logits(8))
                self.assertEqual(backend.start_modes, [not argmax_only])

    def test_a_failed_serial_run_drops_the_partially_advanced_sequence(self):
        backend = FakeBackend()
        session = Session(backend)
        seed = session.sync(PROMPT)
        failure = RuntimeError("the request reached the context limit")
        with patch.object(backend, "step", wraps=backend.step, side_effect=[DEFAULT, failure]):
            with self.assertRaisesRegex(RuntimeError, "context limit"):
                session.serial_run(seed, 2)

        self.assertFalse(session.ctx)
        with self.assertRaisesRegex(RefusedError, "no live sequence"):
            session.top_logits(8)
        self.assertEqual(session.sync(PROMPT), seed)
        self.assertEqual(session.serial_run(seed, 2), serial_stream(2))

    def test_spec_run_equals_the_serial_stream_at_every_depth(self):
        want = serial_stream(128)
        for depth in range(1, IMPLEMENTED_DEPTH + 1):
            s = Session(FakeBackend(depth=depth))
            seed = s.sync(PROMPT)
            rounds, frontiers = s.spec_run(seed, 128)
            got = []
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            self.assertEqual(got, want, f"depth {depth}")
            self.assertEqual(sum(len(r) for r in rounds), 128)
            self.assertTrue(any(len(r) > 1 for r in rounds), f"depth {depth} never accepted")
            self.assertGreater(s.drafts, 0)

    def test_spec_run_never_overshoots_the_count(self):
        for count in (1, 2, 3, 7, 129):
            s = Session(FakeBackend(depth=6))
            seed = s.sync(PROMPT)
            rounds, _ = s.spec_run(seed, count)
            self.assertEqual(sum(len(r) for r in rounds), count)

    def test_overhang_is_consumed_by_the_next_run(self):
        want = serial_stream(20)
        s = Session(FakeBackend(depth=6))
        seed = s.sync(PROMPT)
        got = []
        pending = seed
        for count in (3, 5, 12):
            rounds, frontiers = s.spec_run(pending, count)
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            pending = frontiers[-1]
        self.assertEqual(got, want)

    def test_teacher_forcing_reprefills(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        first = s.sync(PROMPT)
        forced = (first + 1) % 1000 + 20
        s.eval(forced)
        self.assertEqual(s.ctx, PROMPT + [forced])
        self.assertEqual(backend.starts, 2)

    def test_a_bounded_sync_leaves_the_engine_idle(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        s.sync(PROMPT, max_tokens=1)
        self.assertFalse(backend.abort(), "a 1-token request must have finished on its own")

    def test_an_unbounded_sync_is_stopped_by_invalidate(self):
        s = Session(FakeBackend(depth=3))
        seed = s.sync(PROMPT)
        s.spec_run(seed, 16)
        s.invalidate()
        self.assertEqual(s.stopped, 1, "the run-on request is stopped and counted")
        s.sync(PROMPT, max_tokens=1)
        s.invalidate()
        self.assertEqual(s.stopped, 1, "a finished request needs no stop")

    def test_a_bounded_sequence_that_goes_on_is_still_the_serial_stream(self):
        want = serial_stream(40)
        s = Session(FakeBackend(depth=0))
        t, got = s.sync(PROMPT, max_tokens=1), []
        for _ in range(40):
            t = s.eval(t)
            got.append(t)
        self.assertEqual(got, want)
        for depth in range(1, IMPLEMENTED_DEPTH + 1):
            s = Session(FakeBackend(depth=depth))
            seed = s.sync(PROMPT, max_tokens=1)
            rounds, frontiers = s.spec_run(seed, 40)
            got = []
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            self.assertEqual(got, want, f"depth {depth}")

    def test_a_non_positive_bound_is_refused(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.sync(PROMPT, max_tokens=0)

    def test_top_logits_are_the_frontiers(self):
        s = Session(FakeBackend(depth=0))
        first = s.sync(PROMPT)
        ids, logits = s.top_logits(8)
        self.assertEqual(ids[0], first)
        self.assertEqual(len(ids), 8)
        self.assertEqual(logits, sorted(logits, reverse=True))

    def test_an_argmax_only_sync_computes_no_top_logits(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        first = s.sync(PROMPT, argmax_only=True)
        self.assertEqual(backend.start_modes, [False], "the engine is asked for no top-k")
        self.assertEqual(s.frontier.top, ())
        with self.assertRaises(RefusedError):
            s.top_logits(8)
        # The tokens do not depend on it.
        self.assertEqual(first, Session(FakeBackend(depth=0)).sync(PROMPT))

    def test_argmax_only_runs_are_the_serial_stream_at_every_depth(self):
        want = serial_stream(64)
        for depth in range(0, IMPLEMENTED_DEPTH + 1):
            backend = FakeBackend(depth=depth)
            s = Session(backend)
            seed = s.sync(PROMPT, argmax_only=True)
            rounds, frontiers = s.spec_run(seed, 64)
            got = []
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            self.assertEqual(got, want, f"depth {depth}")
            self.assertEqual(backend.start_modes, [False], f"depth {depth}")

    def test_a_sequence_keeps_its_top_k_mode_through_teacher_forcing(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        first = s.sync(PROMPT, argmax_only=True)
        s.eval((first + 1) % 1000 + 20)  # forced: re-prefills
        self.assertEqual(backend.start_modes, [False, False])
        with self.assertRaises(RefusedError):
            s.top_logits(8)
        # A gate sequence keeps its top-k through the same re-prefill.
        first = s.sync(PROMPT)
        s.eval((first + 1) % 1000 + 20)
        self.assertEqual(backend.start_modes, [False, False, True, True])
        self.assertEqual(len(s.top_logits(8)[0]), 8)

    def test_invalidate_and_a_plain_sync_restore_the_top_k(self):
        s = Session(FakeBackend(depth=2))
        s.sync(PROMPT, max_tokens=1, argmax_only=True)
        s.invalidate()
        self.assertFalse(s.argmax_only)
        first = s.sync(PROMPT)
        ids, _ = s.top_logits(8)
        self.assertEqual(ids[0], first)

    def test_sync_counts_split_by_top_k_mode(self):
        s = Session(FakeBackend(depth=0))
        s.sync(PROMPT, argmax_only=True)
        s.sync(PROMPT)
        s.sync(PROMPT, max_tokens=1, argmax_only=True)
        self.assertEqual(s.take_sync_counts(), (2, 1))
        self.assertEqual(s.take_sync_counts(), (0, 0))

    def test_verbs_refuse_without_a_sequence(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.eval(5)
        with self.assertRaises(RefusedError):
            s.top_logits(8)

    def test_out_of_vocab_is_refused(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.sync([1, 2, 131072])

    def test_depth_ext_emission_keeps_drafts_ge_hits(self):
        from backends import Emit

        backend = FakeBackend(depth=6)
        s = Session(backend)
        seed = s.sync(PROMPT)
        # Simulate a depth-extension verify step returning 11 tokens (> depth + 1 = 7).
        fake_emitted = [Emit(token=1000 + i, top=()) for i in range(11)]
        with patch.object(backend, "step", return_value=fake_emitted):
            rounds, frontiers = s.spec_run(seed, 11)
        self.assertEqual(sum(len(r) for r in rounds), 11)
        self.assertEqual(s.hits, 10)
        self.assertGreaterEqual(s.drafts, s.hits)
        self.assertEqual(frontiers[-1], fake_emitted[-1].token)

    def test_hello_reports_depth_in_the_reference_convention(self):
        self.assertEqual(Session(FakeBackend(depth=0)).hello()["draft_tokens"], 1)
        h = Session(FakeBackend(depth=3)).hello()
        self.assertEqual(h["draft_tokens"], 4)
        self.assertTrue(h["mtp_armed"])

    def test_implemented_depth_agrees_with_the_adapter(self):
        src = os.path.join(
            os.path.dirname(__file__), "..", "protocol-adapter", "src", "session_backend.rs"
        )
        with open(src) as fp:
            line = next(l for l in fp if l.startswith("pub const ENGINE_IMPLEMENTED_DEPTH"))
        self.assertEqual(int(line.split("=")[1].strip().rstrip(";")), IMPLEMENTED_DEPTH)


class CheckIdsTests(unittest.TestCase):
    """The fast sweep accepts and refuses exactly what the per-token check does."""

    def test_the_sweep_and_the_per_token_check_agree(self):
        class Sub(int):
            pass

        s = Session(FakeBackend(depth=0))
        vocab = s.b.vocab_size
        accepted = [[0], [vocab - 1], list(range(512)), [Sub(7), 3], []]
        refused = [[vocab], [-1], [3, True], [False], [1.0], ["1"], [None], [Sub(vocab)], [5, vocab, 6]]
        for tokens in accepted:
            s._check_ids(tokens)
        for tokens in refused:
            with self.assertRaises(RefusedError, msg=repr(tokens)):
                s._check_ids(tokens)


class CountingConn:
    """A server-side connection that counts the replies written to it."""

    def __init__(self, conn):
        self._conn = conn
        self.sent = 0

    def sendall(self, data):
        self._conn.sendall(data)
        self.sent += 1

    def __getattr__(self, name):
        return getattr(self._conn, name)


class SettleTests(unittest.TestCase):
    """REPLY FIRST, THEN CLEAN UP: one settle per reply, after it is written."""

    def serve(self, backend):
        client, server = socket.socketpair()
        conn = CountingConn(server)
        session = Session(backend)
        thread = threading.Thread(target=serve_connection, args=(session, conn, 5.0))
        thread.start()
        reader = client.makefile("rb")

        def call(obj):
            client.sendall((json.dumps(obj) + "\n").encode())
            return json.loads(reader.readline())

        def close():
            reader.close()
            client.close()
            thread.join(timeout=10)
            server.close()

        self.addCleanup(close)
        # The reply reaches the client BEFORE the server settles, so a test
        # that counts settles first waits for the server to finish the phase.
        self.server_done = lambda: thread.join(timeout=10)
        return session, conn, call

    def test_the_resident_settles_once_after_every_reply(self):
        seen = []

        class Recording(FakeBackend):
            def settle(inner):
                seen.append(conn.sent)
                super().settle()

        backend = Recording(depth=0)
        session, conn, call = self.serve(backend)
        seed = call({"op": "sync", "tokens": PROMPT, "max_tokens": 1, "argmax_only": True})["token"]
        self.assertFalse(call({"op": "nope"})["ok"])
        self.assertEqual(call({"op": "sync", "tokens": PROMPT, "argmax_only": True})["token"], seed)
        self.assertEqual(call({"op": "serial_run", "first_token": seed, "count": 8})["tokens"], serial_stream(8))
        self.assertTrue(call({"op": "invalidate"})["ok"])
        self.assertTrue(call({"op": "bye"})["ok"])
        self.server_done()
        # Settle number i ran when exactly i replies had been written.
        self.assertEqual(seen, [1, 2, 3, 4, 5, 6])
        self.assertEqual(backend.settles, 6)

    def test_a_failed_settle_drops_the_sequence_and_keeps_the_phase(self):
        class Failing(FakeBackend):
            def settle(inner):
                super().settle()
                if inner.settles == 1:
                    raise RuntimeError("cleanup failed")

        session, conn, call = self.serve(Failing(depth=0))
        seed = call({"op": "sync", "tokens": PROMPT})["token"]
        # The sync's own reply was already written; the sequence is gone.
        self.assertFalse(call({"op": "argmax"})["ok"])
        self.assertEqual(call({"op": "sync", "tokens": PROMPT})["token"], seed)
        self.assertEqual(call({"op": "argmax"})["token"], seed)


class FakeEngine:
    """The slice of LLMEngine that VllmBackend.abort and .settle touch."""

    def __init__(self):
        self.unfinished = False
        self.aborted: list[list[str]] = []
        self.steps = 0

    def has_unfinished_requests(self):
        return self.unfinished

    def abort_request(self, request_ids):
        self.aborted.append(list(request_ids))
        self.unfinished = False

    def step(self):
        self.steps += 1
        return []


class BackendSettleTests(unittest.TestCase):
    def backend(self, deferred=True):
        b = VllmBackend.__new__(VllmBackend)
        b._engine = FakeEngine()
        b._live = None
        self.core_settles = 0

        def settle_core():
            self.core_settles += 1

        b._settle_core = settle_core if deferred else None
        return b

    def test_settle_finishes_a_request_that_completed_on_its_own(self):
        b = self.backend()
        b._live = "r1"
        b.settle()
        self.assertIsNone(b._live)
        self.assertEqual(b._engine.aborted, [["r1"]])
        self.assertGreaterEqual(self.core_settles, 1)
        # The next start's abort finds nothing left to do in the engine.
        self.assertFalse(b.abort())
        self.assertEqual(b._engine.aborted, [["r1"]])

    def test_settle_never_touches_a_request_that_is_still_generating(self):
        b = self.backend()
        b._live = "r1"
        b._engine.unfinished = True
        b.settle()
        self.assertEqual(b._live, "r1")
        self.assertEqual(b._engine.aborted, [])
        self.assertTrue(b.abort())
        self.assertEqual(b._engine.aborted, [["r1"]])

    def test_abort_settles_the_core_even_with_nothing_live(self):
        b = self.backend()
        self.assertFalse(b.abort())
        self.assertEqual(self.core_settles, 1)

    def test_without_the_deferral_settle_is_a_no_op(self):
        b = self.backend(deferred=False)
        b._live = "r1"
        b.settle()
        self.assertEqual(b._live, "r1")
        self.assertEqual(b._engine.aborted, [])


class WireTests(unittest.TestCase):
    """Boots the real resident process on the fake backend and talks NDJSON."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sock = os.path.join(self.dir, "r.sock")
        self.ready = os.path.join(self.dir, "ready")
        env = dict(
            os.environ,
            RESIDENT_BACKEND="fake",
            RESIDENT_SPEC_DEPTH="2",
            RESIDENT_SOCKET=self.sock,
            RESIDENT_READY_FILE=self.ready,
        )
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "resident.py")],
            env=env,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 20
        while not os.path.exists(self.ready):
            if time.time() > deadline or self.proc.poll() is not None:
                self.fail("the resident never became ready")
            time.sleep(0.05)

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)

    def call(self, conn, reader, obj):
        conn.sendall((json.dumps(obj) + "\n").encode())
        return json.loads(reader.readline())

    def test_phase_round_trip_and_reset(self):
        for _phase in range(2):
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(self.sock)
            reader = conn.makefile("rb")
            hello = self.call(conn, reader, {"op": "hello"})
            self.assertTrue(hello["ok"])
            self.assertEqual(hello["draft_tokens"], 3)
            # A fresh phase has no live sequence.
            self.assertFalse(self.call(conn, reader, {"op": "argmax"})["ok"])
            seed = self.call(conn, reader, {"op": "sync", "tokens": PROMPT})["token"]
            run = self.call(conn, reader, {"op": "spec_run", "first_token": seed, "count": 32})
            self.assertTrue(run["ok"])
            self.assertEqual(sum(len(r) for r in run["rounds"]), 32)
            self.assertEqual(run["frontiers"][-1], run["token"])
            top = self.call(conn, reader, {"op": "top_logits", "k": 8})
            self.assertEqual(len(top["ids"]), 8)
            bounded = self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "max_tokens": 1})
            self.assertEqual(bounded["token"], seed)
            self.assertFalse(
                self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "max_tokens": 0})["ok"]
            )
            # The timed phases' argmax-only sync: same tokens, no top-k.
            lean = self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "argmax_only": True})
            self.assertEqual(lean["token"], seed)
            run = self.call(conn, reader, {"op": "spec_run", "first_token": seed, "count": 32})
            self.assertEqual(run["frontiers"][-1], top["ids"][0])
            self.assertFalse(self.call(conn, reader, {"op": "top_logits", "k": 8})["ok"])
            self.assertFalse(
                self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "argmax_only": 1})["ok"]
            )
            plain = self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "argmax_only": False})
            self.assertEqual(plain["token"], seed)
            self.assertEqual(len(self.call(conn, reader, {"op": "top_logits", "k": 8})["ids"]), 8)
            bad = self.call(conn, reader, {"op": "nope"})
            self.assertFalse(bad["ok"])
            self.assertTrue(self.call(conn, reader, {"op": "bye"})["ok"])
            reader.close()
            conn.close()


if __name__ == "__main__":
    unittest.main()
