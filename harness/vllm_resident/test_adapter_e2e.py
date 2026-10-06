"""End to end with no GPU: the real `cuda-engine` binary attached to the real
resident process (fake backend), driven over Engine Protocol v1 the way benchd
drives a leg: hello, free_decode_begin, free_decode_run, then a correctness
phase. It proves the Rust client and the Python resident agree on the wire and
that every depth commits exactly the serial stream.

Needs `cargo build --release` in harness/protocol-adapter first; skips if the
binary is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_resident import PROMPT, serial_stream  # noqa: E402

ENGINE = os.path.join(HERE, "..", "protocol-adapter", "target", "release", "cuda-engine")


@unittest.skipUnless(os.path.exists(ENGINE), "build harness/protocol-adapter --release first")
class AdapterEndToEnd(unittest.TestCase):
    def boot(self, depth: int) -> tuple[subprocess.Popen, str]:
        d = tempfile.mkdtemp()
        sock, ready = os.path.join(d, "r.sock"), os.path.join(d, "ready")
        env = dict(
            os.environ,
            RESIDENT_BACKEND="fake",
            RESIDENT_SPEC_DEPTH=str(depth),
            RESIDENT_SOCKET=sock,
            RESIDENT_READY_FILE=ready,
        )
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "resident.py")], env=env, stderr=subprocess.DEVNULL
        )
        deadline = time.time() + 20
        while not os.path.exists(ready):
            if time.time() > deadline or proc.poll() is not None:
                self.fail("the resident never became ready")
            time.sleep(0.05)
        self.addCleanup(lambda: (proc.terminate(), proc.wait(timeout=10)))
        return proc, sock

    def phase(self, sock: str, requests: list[dict]) -> list[dict]:
        env = dict(os.environ, BENCH_WORKER_RESIDENT_SOCKET=sock)
        env.pop("DS4_RESIDENT_SOCKET", None)
        stdin = "".join(json.dumps(r) + "\n" for r in requests)
        out = subprocess.run(
            [ENGINE], input=stdin, capture_output=True, text=True, env=env, timeout=60
        )
        lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
        self.assertTrue(lines, out.stderr)
        return lines

    def test_every_depth_commits_the_serial_stream(self):
        want = serial_stream(128)
        for depth in range(0, 7):
            with self.subTest(depth=depth):
                _, sock = self.boot(depth)
                spec = {"mode": "serial"} if depth == 0 else {"mode": "mtp", "mtp": {"depth": depth}}
                replies = self.phase(
                    sock,
                    [
                        {"id": 1, "kind": "free_decode_begin", "seed_tokens": PROMPT, "spec": spec},
                        {"id": 2, "kind": "free_decode_run", "count": 128},
                    ],
                )
                hello, begin, run = replies
                self.assertTrue(hello["backend"].startswith("vllm-resident load_epoch="))
                self.assertTrue(begin["ok"], begin)
                self.assertTrue(run["ok"], run)
                self.assertEqual(run["tokens"], want)
                self.assertEqual(run["committed_total"], 128)
                self.assertEqual(sum(run["acceptance_lengths"]), 128)
                if depth:
                    self.assertEqual(begin["effective_spec"]["mtp"]["depth"], depth)
                    self.assertGreater(run["drafted_total"], 0)

    def test_correctness_phase_reports_top8(self):
        _, sock = self.boot(0)
        want = serial_stream(3)
        replies = self.phase(
            sock,
            [
                {"id": 1, "kind": "correctness_begin", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "correctness_step", "token": None},
            ],
        )
        begin = replies[1]
        self.assertTrue(begin["ok"], begin)
        self.assertEqual(len(begin["top_logits"]), 8)
        step = self.phase(
            sock,
            [
                {"id": 1, "kind": "correctness_begin", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "correctness_step", "token": begin["token"]},
            ],
        )[2]
        self.assertTrue(step["ok"], step)
        self.assertEqual(step["token"], want[0])


if __name__ == "__main__":
    unittest.main()
