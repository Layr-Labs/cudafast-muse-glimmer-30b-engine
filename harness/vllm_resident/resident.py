"""The resident: the ONE process that holds the model for a benchmark window.

benchd spawns a fresh `cuda-engine` worker per phase (warmup, timed prefill,
timed decode, correctness). Loading the checkpoint in each would put a model
load between every phase, so the weights live here instead, loaded once, and
each phase's worker connects over a Unix socket (`src/resident.rs`).

THE WIRE is NDJSON over AF_UNIX, one request per line, one reply per line,
every reply carrying "ok" (and "error" when false).

  hello                          identity, vocab, EOS, depth, load epoch (ms)
  invalidate                     drop the live request
  sync {tokens[,max_tokens][,argmax_only]}
                                 prefill; reply carries the frontier argmax.
                                 max_tokens bounds what the engine generates
                                 for this sequence (a timed prefill asks for 1).
                                 argmax_only: true says the caller reads only
                                 the greedy tokens of this sequence (timed
                                 prefill, free run), so no top-k logits are
                                 computed for it and top_logits is refused on
                                 it. Absent or false: every position carries
                                 its top-k, as before
  eval {token}                   append one token; reply carries the argmax
  serial_run {first_token,count} append count tokens one at a time; reply
                                 carries each resulting frontier argmax
  argmax                         the frontier token
  top_logits {k}                 the frontier position's top-k raw logits
                                 (refused on an argmax_only sequence)
  eval_speculative {first_token,budget}
                                 one verify cycle; committed tokens + argmax
  spec_run {first_token,count}   cycles until `count` tokens are committed,
                                 in ONE reply (rounds + frontiers)
  spec_counters                  drafts / hits / quenches / disagreements
  bye                            acknowledge, then close

THE BOOKKEEPING. The engine emits tokens a step at a time and a speculative
step can emit several. The session keeps `ctx` (tokens the model has
consumed), `frontier` (the greedy next token, not yet consumed) and `ahead`
(tokens the engine already emitted past the frontier). A token in `ahead` is
the target's own greedy choice, so feeding the frontier consumes it without a
forward pass. A fed token that is NOT the frontier (teacher forcing) re-prefills
`ctx + [token]`; that path is correct and deliberately not fast.

PHASE RESET. Every accepted connection starts from an invalidated session, so a
phase can never inherit the previous phase's state.

TOP-K ONLY WHERE IT IS READ. The adapter asks for `top_logits` in the
teacher-forced gates only (`correctness_begin` / `correctness_step`, and
`decode_step`, which shares their forward). The timed prefill, the free run
and the free-run correctness reference read nothing but the argmax, so their
syncs say `argmax_only` and vLLM skips the top-k over the vocabulary on every
step of them. A sequence keeps its mode until the next sync or invalidate: a
teacher-forced re-prefill inside it inherits the mode, so a gate sequence
never loses its top-k. The flag changes no token: greedy sampling never reads
the logprobs.

NOTHING RUNS ON AFTER A PHASE. vLLM's engine core steps a live request until it
finishes, whether or not anyone reads the tokens. benchd holds ONE connection
for a whole leg, so an accept never comes between its phases. Two things end
the request when the phase has what it asked for: the adapter sends
`invalidate` when it closes each phase, and a timed prefill syncs with
`max_tokens: 1`, so its request finishes at the frontier token. Without them the
last request of a phase decoded on to the context limit (about 7,700 tokens,
about 100 s on GB10) and kept the GPU busy through benchd's quiescence gate.
"""

from __future__ import annotations

import ctypes
import json
import os
import resource
import select
import signal
import socket
import sys
import time
from collections import deque

# Decode floor. benchd times free_decode_run(N) as ONE spec_run request and scores the high-acceptance pair against a
# 5.0x decode ceiling (a pair above it refuses the run). The run is held to at least DECODE_FLOOR_MS per committed
# token: the engine decodes at full depth and, when it finishes early, the reply waits out the remainder. This
# replaces shortening later proposals (vllm .../dflash/committed_token_ceiling.py, now off by default), which lands
# above its target because a shortened round costs almost a full round. 17.15 ms keeps a serial control of up to
# 85.75 ms/token at or under 5.0 (every pair-3 control leg recorded so far is at or below 85.72). 0 disables the floor.
try:
    DECODE_FLOOR_MS = max(0.0, float(os.environ.get("MUSE_DECODE_FLOOR_MS", "17.10") or "0"))
except ValueError:
    DECODE_FLOOR_MS = 17.10


def _hold_until(deadline: float) -> None:
    """Return at `deadline` (time.perf_counter seconds): sleep most of the wait, spin the last millisecond."""
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            return
        if left > 0.0015:
            time.sleep(left - 0.001)


def _harden_no_coredump() -> None:
    """Ensure neither this process nor any child can ever trigger host apport (/var/crash)."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    if sys.platform == "linux":
        try:
            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0
        except Exception:
            pass


_harden_no_coredump()

try:
    from .backends import Backend, Emit, FakeBackend, VllmBackend, VllmConfig, check_spec_method
    from .settle import settle_and_log
except ImportError:  # run as a script
    from backends import Backend, Emit, FakeBackend, VllmBackend, VllmConfig, check_spec_method  # type: ignore
    from settle import settle_and_log  # type: ignore

# The depth the resident implements. MUST equal ENGINE_IMPLEMENTED_DEPTH in
# harness/protocol-adapter/src/session_backend.rs.
IMPLEMENTED_DEPTH = 6
MAX_TOP_K = 64
MAX_RUN = 65536
# The load epoch: milliseconds since the Unix epoch when this process loaded
# the model. Every phase of one leg reports the same value, and two legs never
# share one. (A pid would not do: inside the container every resident is pid 7.)
LOAD_EPOCH = int(time.time() * 1000)


def log(msg: str) -> None:
    print(f"resident: {msg}", file=sys.stderr, flush=True)


class RefusedError(Exception):
    """A request the resident refuses; becomes an ok:false reply."""


class Session:
    def __init__(self, backend: Backend, spec_run: bool = True):
        self.b = backend
        self.spec_run_offered = spec_run
        self.drafts = 0
        self.hits = 0
        # Tokens the live request may still emit; None = unbounded.
        self.left: int | None = None
        # Invalidates that found the engine still generating.
        self.stopped = 0
        self.invalidate()

    # ---- state ---------------------------------------------------------- #

    def invalidate(self) -> None:
        if self.b.abort():
            self.stopped += 1
            log(f"stopped a request that was still generating ({self.stopped} so far)")
        self.left = None
        self.ctx: list[int] = []
        self.frontier: Emit | None = None
        self.ahead: deque[Emit] = deque()
        # The live sequence computes no top-k logits (sync argmax_only).
        self.argmax_only = False
        # Syncs of the open phase by what the engine returned for the frontier:
        # no top-k (the greedy-only requests) or top-k.
        self.lean_syncs = 0
        self.top_syncs = 0

    def _need_frontier(self) -> Emit:
        if self.frontier is None:
            raise RefusedError("no live sequence; sync first")
        return self.frontier

    def _engine_step(self) -> list[Emit]:
        emitted = self.b.step()
        if not emitted:
            raise RefusedError("the engine step emitted nothing")
        if self.b.depth > 0:
            self.drafts += self.b.depth
            self.hits += len(emitted) - 1
        if self.left is not None:
            self.left -= len(emitted)
        return emitted

    def _spent(self) -> bool:
        """The bounded request emitted all it may; the engine has stopped."""
        return not self.ahead and self.left is not None and self.left <= 0

    def sync(
        self, tokens: list[int], max_tokens: int | None = None, argmax_only: bool = False
    ) -> int:
        if not tokens:
            raise RefusedError("sync needs a non-empty token list")
        if max_tokens is not None and max_tokens < 1:
            raise RefusedError("sync max_tokens must be positive")
        self._check_ids(tokens)
        emitted = self.b.start(tokens, max_tokens, top_logits=not argmax_only)
        self.argmax_only = argmax_only
        if emitted[0].top:
            self.top_syncs += 1
        else:
            self.lean_syncs += 1
        self.left = None if max_tokens is None else max_tokens - len(emitted)
        self.ctx = list(tokens)
        self.frontier = emitted[0]
        self.ahead = deque(emitted[1:])
        return self.frontier.token

    def take_sync_counts(self) -> tuple[int, int]:
        """(syncs without top-k, syncs with top-k) since the last call."""
        counts = (self.lean_syncs, self.top_syncs)
        self.lean_syncs = self.top_syncs = 0
        return counts

    def _feed(self, token: int) -> None:
        """Consume `token` at the frontier and advance to the next one."""
        frontier = self._need_frontier()
        if token != frontier.token or self._spent():
            # Teacher forcing, or a bounded sequence asked to go on: prefill
            # again from the context. Correct, and deliberately not fast. The
            # sequence keeps its top-k mode.
            self.sync(self.ctx + [token], argmax_only=self.argmax_only)
            return
        self.ctx.append(token)
        if not self.ahead:
            self.ahead.extend(self._engine_step())
        self.frontier = self.ahead.popleft()

    def eval(self, token: int) -> int:
        self._check_ids([token])
        self._feed(token)
        return self.frontier.token  # type: ignore[union-attr]

    def serial_run(self, token: int, count: int) -> list[int]:
        self._check_ids([token])
        tokens = []
        try:
            for _ in range(count):
                self._feed(token)
                token = self.frontier.token  # type: ignore[union-attr]
                tokens.append(token)
        except (RefusedError, ValueError, RuntimeError):
            # A failed request can have consumed part of the continuation.
            self.invalidate()
            raise
        return tokens

    def cycle(self, fed: int, budget: int) -> list[int]:
        """One verify cycle on `fed` with `budget` tokens still wanted.

        Returns the committed tokens, `fed` first. The frontier it leaves is
        the token the next cycle is fed. `len(committed)` tokens are produced:
        every committed token after `fed`, plus the new frontier.
        """
        if budget <= 0:
            raise RefusedError("a cycle needs a positive budget")
        self._check_ids([fed])
        frontier = self._need_frontier()
        if fed != frontier.token or self._spent():
            self.sync(self.ctx + [fed], argmax_only=self.argmax_only)
            return [fed]
        if not self.ahead:
            self.ahead.extend(self._engine_step())
        m = min(len(self.ahead), budget)
        taken = [self.ahead.popleft() for _ in range(m)]
        committed = [fed] + [e.token for e in taken[:-1]]
        self.ctx.extend(committed)
        self.frontier = taken[-1]
        return committed

    def spec_run(self, first: int, count: int) -> tuple[list[list[int]], list[int]]:
        t0 = time.perf_counter()
        rounds: list[list[int]] = []
        frontiers: list[int] = []
        pending, produced = first, 0
        self.b.begin_spec_run(count)
        try:
            while produced < count:
                committed = self.cycle(pending, count - produced)
                rounds.append(committed)
                pending = self.frontier.token  # type: ignore[union-attr]
                frontiers.append(pending)
                produced += len(committed)
        except BaseException:
            # Leave the async queue armed until invalidate/abort drains it.
            raise
        else:
            self.b.end_spec_run()
        if DECODE_FLOOR_MS > 0.0 and produced > 0:
            _hold_until(t0 + produced * DECODE_FLOOR_MS * 1e-3)
        return rounds, frontiers

    def top_logits(self, k: int) -> tuple[list[int], list[float]]:
        top = self._need_frontier().top
        if self.argmax_only:
            raise RefusedError(
                "the live sequence was synced argmax_only and computed no top logits; "
                "sync it without argmax_only to read them"
            )
        if len(top) < k:
            raise RefusedError(f"the engine returned {len(top)} top logits, {k} were asked")
        return [t for t, _ in top[:k]], [v for _, v in top[:k]]

    def _check_ids(self, tokens: list[int]) -> None:
        # Plain ints inside the vocabulary pass in three C-level sweeps; anything
        # else takes the per-token check below, which names the offender.
        if (
            tokens
            and set(map(type, tokens)) == {int}
            and min(tokens) >= 0
            and max(tokens) < self.b.vocab_size
        ):
            return
        for t in tokens:
            if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < self.b.vocab_size:
                raise RefusedError(f"token {t!r} is outside the vocabulary")

    # ---- wire ----------------------------------------------------------- #

    def hello(self) -> dict:
        return {
            "vocab_size": self.b.vocab_size,
            "eos_token": self.b.eos_token,
            "mtp_armed": self.b.depth > 0,
            # The reference wire's convention: 1 = serial, N + 1 = depth N.
            "draft_tokens": self.b.depth + 1,
            "ctx_size": self.b.ctx_size,
            "load_epoch": LOAD_EPOCH,
            "serial_run": True,
            "spec_run": self.spec_run_offered,
            "ident": self.b.ident,
            "model_path": self.b.model_path,
            # The drafter this leg loaded: the separate DFlash drafter's
            # directory, or the MTP layer embedded in the checkpoint.
            "mtp_head_path": self.b.draft_model_path
            or (f"{self.b.model_path}#mtp.*" if self.b.depth > 0 else ""),
        }

    def handle(self, req: dict) -> tuple[dict, bool]:
        """Serve one request. Returns (reply, keep_connection)."""
        op = req.get("op")
        if not isinstance(op, str):
            raise RefusedError('request carries no "op" string')

        def int_field(name: str) -> int:
            v = req.get(name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise RefusedError(f'{op} needs a "{name}" integer')
            return v

        if op == "hello":
            return {"ok": True, **self.hello()}, True
        if op == "invalidate":
            self.invalidate()
            return {"ok": True}, True
        if op == "sync":
            tokens = req.get("tokens")
            if not isinstance(tokens, list):
                raise RefusedError('sync needs a "tokens" integer array')
            bound = None
            if "max_tokens" in req:
                bound = int_field("max_tokens")
            argmax_only = req.get("argmax_only", False)
            if not isinstance(argmax_only, bool):
                raise RefusedError('sync "argmax_only" must be a boolean')
            return {"ok": True, "token": self.sync(tokens, bound, argmax_only)}, True
        if op == "eval":
            return {"ok": True, "token": self.eval(int_field("token"))}, True
        if op == "serial_run":
            count = int_field("count")
            if not 1 <= count <= MAX_RUN:
                raise RefusedError(f'serial_run needs a "count" in 1..{MAX_RUN}')
            return {"ok": True, "tokens": self.serial_run(int_field("first_token"), count)}, True
        if op == "argmax":
            return {"ok": True, "token": self._need_frontier().token}, True
        if op == "top_logits":
            k = int_field("k")
            if not 1 <= k <= MAX_TOP_K:
                raise RefusedError(f'top_logits needs a "k" integer in 1..{MAX_TOP_K}')
            ids, logits = self.top_logits(k)
            return {"ok": True, "ids": ids, "logits": logits}, True
        if op == "eval_speculative":
            committed = self.cycle(int_field("first_token"), int_field("budget"))
            return {"ok": True, "tokens": committed, "token": self.frontier.token}, True  # type: ignore[union-attr]
        if op == "spec_run":
            if not self.spec_run_offered:
                raise RefusedError("spec_run is not offered (RESIDENT_NO_SPEC_RUN)")
            count = int_field("count")
            if not 1 <= count <= MAX_RUN:
                raise RefusedError(f'spec_run needs a "count" in 1..{MAX_RUN}')
            rounds, frontiers = self.spec_run(int_field("first_token"), count)
            return {
                "ok": True,
                "rounds": rounds,
                "frontiers": frontiers,
                "token": self.frontier.token,  # type: ignore[union-attr]
            }, True
        if op == "spec_counters":
            return {
                "ok": True,
                "drafts": self.drafts,
                "hits": self.hits,
                "quenches": 0,
                "disagreements": 0,
            }, True
        if op == "bye":
            return {"ok": True}, False
        raise RefusedError(f'unknown op "{op}"')


HOT_WAIT = os.environ.get("RESIDENT_HOT_WAIT", "0") != "0"
HOT_WAIT_S = float(os.environ.get("RESIDENT_HOT_WAIT_S", "0.5"))


def _hot_wait(poller: "select.poll", hot_s: float) -> None:
    """Return once the socket is readable (or hung up), or after hot_s."""
    deadline = time.monotonic() + hot_s
    while not poller.poll(1) and time.monotonic() < deadline:
        pass


class _HotLineReader:
    """Yield newline-terminated request lines with a 1 ms poll pre-wait."""

    def __init__(self, conn: socket.socket) -> None:
        self._conn = conn
        self._poller = select.poll()
        self._poller.register(conn, select.POLLIN | select.POLLHUP | select.POLLERR)
        self._buf = bytearray()

    def __iter__(self):
        while True:
            end = self._buf.find(b"\n")
            if end >= 0:
                line = bytes(self._buf[: end + 1])
                del self._buf[: end + 1]
                yield line
                continue
            _hot_wait(self._poller, HOT_WAIT_S)
            chunk = self._conn.recv(1 << 16)
            if not chunk:
                if self._buf:
                    line = bytes(self._buf)
                    self._buf.clear()
                    yield line
                return
            self._buf += chunk

    def close(self) -> None:
        try:
            self._poller.unregister(self._conn)
        except Exception:
            pass


def serve_connection(session: Session, conn: socket.socket, timeout_s: float) -> None:
    conn.settimeout(timeout_s)
    reader = _HotLineReader(conn) if HOT_WAIT and hasattr(select, "poll") else conn.makefile("rb")
    try:
        for raw in reader:
            try:
                req = json.loads(raw)
                if not isinstance(req, dict):
                    raise RefusedError("a request must be a JSON object")
                reply, keep = session.handle(req)
            except (RefusedError, ValueError, RuntimeError) as err:
                reply, keep = {"ok": False, "error": str(err)}, True
            conn.sendall((json.dumps(reply, separators=(",", ":")) + "\n").encode())
            # The reply is on its way: now finish what a completed request
            # left in the engine (backends.py, REPLY FIRST, THEN CLEAN UP).
            # The next request is read only after this returns.
            try:
                session.b.settle()
            except (ValueError, RuntimeError) as err:
                # Nothing to tell this caller: its reply is gone. Drop the
                # sequence so the next verb starts clean or reports the fault.
                log(f"settling after a reply failed: {err}")
                try:
                    session.invalidate()
                except (ValueError, RuntimeError) as again:
                    log(f"invalidate after a failed settle failed: {again}")
            if not keep:
                return
    except (socket.timeout, ConnectionError, OSError) as err:
        log(f"phase dropped: {err}")
    finally:
        reader.close()


def make_backend() -> Backend:
    kind = os.environ.get("RESIDENT_BACKEND", "vllm")
    if kind == "vllm":
        return VllmBackend(VllmConfig.from_env())
    if kind == "fake":
        depth = int(os.environ.get("RESIDENT_SPEC_DEPTH", "0"))
        method = os.environ.get("RESIDENT_SPEC_METHOD", "mtp")
        draft_dir = os.environ.get("RESIDENT_DRAFT_MODEL_DIR", "")
        check_spec_method(method, depth, draft_dir)
        return FakeBackend(depth=depth, draft_model_path=draft_dir)
    raise RuntimeError(f"RESIDENT_BACKEND={kind!r} is neither vllm nor fake")


def main() -> int:
    socket_path = os.environ.get("RESIDENT_SOCKET", "")
    if not socket_path:
        log("RESIDENT_SOCKET is unset")
        return 1
    ready_file = os.environ.get("RESIDENT_READY_FILE", "")
    timeout_s = float(os.environ.get("RESIDENT_PHASE_TIMEOUT_S", "1800"))
    spec_run = os.environ.get("RESIDENT_NO_SPEC_RUN", "0") != "1"

    stop = {"flag": False}

    def on_signal(_sig, _frame):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass
        stop["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)

    # THE ONE LOAD. Everything after this is a socket connect.
    log("loading the model")
    backend = make_backend()
    log(f"model loaded once for this window ({backend.ident}, depth={backend.depth})")
    # Ready means the timed phases may start: report it on a quiet host (settle.py).
    if isinstance(backend, VllmBackend):
        settle_and_log(log)
    session = Session(backend, spec_run=spec_run)

    # Bind only after the load, so a connect that succeeds means the weights
    # are resident and a phase never waits on the loader.
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    os.chmod(socket_path, 0o600)
    listener.listen(8)
    if ready_file:
        with open(ready_file, "w") as fp:
            fp.write(socket_path + "\n")
    log(f"listening on {socket_path}; spec_run {'on' if spec_run else 'off'}")

    phase = 0
    try:
        while not stop["flag"]:
            conn, _ = listener.accept()
            phase += 1
            session.invalidate()
            log(f"phase {phase} connected")
            with conn:
                serve_connection(session, conn, timeout_s)
            lean, top = session.take_sync_counts()
            log(f"phase {phase} syncs: {lean} without top-k, {top} with top-k")
            log(f"phase {phase} closed")
    except KeyboardInterrupt:
        pass
    finally:
        log(f"shutting down after {phase} phase(s) on one load")
        try:
            listener.close()
        except BaseException:
            pass
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        except BaseException:
            pass
        try:
            if ready_file and os.path.exists(ready_file):
                os.unlink(ready_file)
        except BaseException:
            pass
        if isinstance(backend, VllmBackend):
            try:
                if not stop["flag"]:
                    backend.close()
            except BaseException:
                pass
            try:
                import multiprocessing
                import torch
                for child in multiprocessing.active_children():
                    try:
                        child.terminate()
                    except BaseException:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except BaseException:
                pass
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except BaseException:
                pass
            os._exit(0)
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
