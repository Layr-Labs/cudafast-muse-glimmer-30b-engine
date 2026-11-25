# SPDX-License-Identifier: Apache-2.0
"""One-leaf draft tree in the padding row of the depth-6 DFlash verify (proposals only; committed tokens unchanged).

A depth-6 verify scores 7 rows [b, d1..d6] and the CUDA graph pads the batch to 8 rows, so row 8 is computed for
nothing. Here row 8 carries `alt`, the drafter's second choice for position p+1 (the position of d1), as a one-node
branch that sees only the context and the bonus token b:

  * its K/V is written by the fused norm/rope/KV kernel into a scratch KV block (the last block of each cache);
  * per attention layer, the scratch block first receives the real block's slots below p+1 (incl. b's fresh K/V);
  * its attention runs through the same multi-row verify kernel as a 2-row sequence [b, alt] over the real block
    table with the scratch block in place of the block that holds p+1 (row b of that call is discarded);
  * after rejection sampling: if d1 was rejected (one token sampled) and the target's token equals `alt`, the step
    commits [alt, u] with u the argmax of row 8's logits, i.e. exactly the token the next verify would produce for
    that position; the scratch K/V of `alt` is copied to slot p+1 at the next forward, and the drafter's context
    rows take row 8's hidden states.

MUSE_LEAF=0 disables everything (row 8 stays padding).
"""
import os

import torch

ENABLED = os.environ.get("MUSE_LEAF", "1") != "0"
try:
    import muse_leaf_ext as _ext
except Exception:  # noqa: BLE001
    _ext = None
if _ext is None:
    ENABLED = False
LEAF_ROW = 6
_NOHIT = os.environ.get("MUSE_LEAF_NOHIT", "0") == "1"
_SELFTEST = os.environ.get("MUSE_LEAF_SELFTEST", "0") == "1"
LOW_REGIME = int(os.environ.get("MUSE_LEAF_LOW", "3"))  # leaf mode when the previous verify committed <= this


class _Group:
    def __init__(self, kv_blocks: int, bs: int, max_blocks: int, device, gid: int = 0):
        # one scratch block per kv-cache group: groups may share a cache tensor (disjoint block ids)
        self.sb = kv_blocks - 1 - gid
        self.bs = bs
        i64 = dict(dtype=torch.int64, device=device)
        self.bt2 = torch.zeros((1, max_blocks), dtype=torch.int32, device=device)
        self.seq2 = torch.zeros((1,), dtype=torch.int32, device=device)
        self.cu2 = torch.zeros((2,), dtype=torch.int32, device=device)
        self.cu2[1:].fill_(2)
        # scratch build: rows (block, head, slot) for 2 heads x bs slots
        self.hh = torch.zeros((2 * bs,), **i64)
        self.ss = torch.zeros((2 * bs,), **i64)
        self.hh.copy_(torch.arange(2 * bs, **i64) // bs)
        self.ss.copy_(torch.arange(2 * bs, **i64) % bs)
        self.rb_rows = torch.zeros((2 * bs,), **i64)
        self.sb_rows = torch.zeros((2 * bs,), **i64)
        self.sb_rows.fill_(self.sb)
        self.mask = torch.zeros((2 * bs, 1), dtype=torch.bool, device=device)
        # deferred fix of the previous step's hit: scratch slot -> real slot p+1, 2 kv heads
        self.h2 = torch.zeros((2,), **i64)
        self.h2.copy_(torch.arange(2, **i64))
        self.fix_sb = torch.zeros((2,), **i64)
        self.fix_sb.fill_(self.sb)
        self.fix_rb = torch.zeros((2,), **i64)
        self.fix_n = torch.zeros((2,), **i64)
        self.rb1 = torch.zeros((1,), **i64)
        self.n1 = torch.zeros((1,), **i64)


class _State:
    def __init__(self):
        self.groups: dict[int, _Group] = {}  # keyed by kv-cache group index
        self.layer_gid: dict[str, int] = {}
        self.mode = None  # bool[1]: this step runs the leaf in row 6 (low-acceptance regime)
        self.prev_ns = None  # int[1]: tokens committed by the previous verify
        self.alt = None  # int64[4]: drafter top-4 for position p+1 (set inside the drafter graph)
        self.q2 = None
        self.o2 = None
        self.hit = None  # bool[1]: this step committed the leaf
        self.hit_prev = None  # bool[1]: previous step committed the leaf (KV fix pending)
        self.active = False  # set per step on the host
        self.stats = None
        self.u_prev = None
        self.ustat = None
        self.st2 = None


S = _State()
_DBG = {"attn_calls": 0, "attn_skipped": 0, "setup_true": 0, "setup_false": 0, "commits": 0, "capturing_seen": 0}


def _dump():
    try:
        st = S.stats.tolist() if S.stats is not None else None
        with open("/root/leafstats.txt", "a") as fh:
            fh.write(f"keys={list(S.groups.keys())} gw={_DBG.get('gw')} st2={S.st2.tolist() if S.st2 is not None else None} ustat={S.ustat.tolist() if S.ustat is not None else None} groups={len(S.groups)} alt={'set' if S.alt is not None else None} dbg={_DBG} stats={st}\n")
    except Exception:  # noqa: BLE001
        pass


if os.environ.get("MUSE_LEAF_DUMP", "0") == "1":
    import atexit as _atexit
    _atexit.register(_dump)


def ensure_common(device, num_heads: int, head_dim: int):
    if S.q2 is None:
        S.q2 = torch.zeros((2, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        S.o2 = torch.zeros((2, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        S.hit = torch.zeros((1,), dtype=torch.bool, device=device)
        S.hit_prev = torch.zeros((1,), dtype=torch.bool, device=device)
        S.stats = torch.zeros((4,), dtype=torch.int64, device=device)
        S.mode = torch.zeros((1,), dtype=torch.bool, device=device)
        S.prev_ns = torch.full((1,), 7, dtype=torch.int64, device=device)
        if S.alt is None:
            S.alt = torch.zeros((4,), dtype=torch.int64, device=device)


def _wkey(window) -> int:
    if window is None:
        return -1
    return int(window[0]) if int(window[0]) >= 0 else -1


def group_for(block_table: torch.Tensor, kv_cache: torch.Tensor, layer=None) -> _Group | None:
    if S.q2 is None:
        ensure_common(kv_cache.device, 32, 128)
    name = getattr(layer, "layer_name", None)
    if name is None or name not in S.layer_gid:
        _DBG["no_gid"] = _DBG.get("no_gid", 0) + 1
        return None
    key = S.layer_gid[name]
    g = S.groups.get(key)
    if g is None:
        g = _Group(int(kv_cache.shape[0]), int(kv_cache.shape[2]), int(block_table.shape[1]), kv_cache.device, key)
        S.groups[key] = g
    return g


_SIDE = os.environ.get("MUSE_LEAF_SIDE", "1") != "0"
_side = {}


def _side_stream(dev):
    st = _side.get(dev)
    if st is None:
        st = torch.cuda.Stream(device=dev)
        _side[dev] = st
    return st


@torch.no_grad()
def prep_hook(query, kv_cache, attn_metadata, window, scale, run_verify, layer=None):
    """Per attention layer before the 7-row verify: last step's K/V commit, scratch build, query rows -> q2;
    then the leaf row's own 2-row attention, on a side stream so it overlaps the main verify attention."""
    g = group_for(attn_metadata.block_table, kv_cache, layer)
    if g is None or S.q2 is None:
        return None
    _DBG["attn_calls"] += 1
    _ext.prep(kv_cache, g.sb, g.rb1, g.n1, g.fix_rb, g.fix_n, S.hit_prev, S.mode, query, LEAF_ROW, S.q2)
    if not _SIDE:
        run_verify(S.q2, kv_cache, S.o2, g.seq2, g.bt2, g.cu2, 2, scale, window)
        return "noside"
    main = torch.cuda.current_stream()
    side = _side_stream(kv_cache.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _ok = run_verify(S.q2, kv_cache, S.o2, g.seq2, g.bt2, g.cu2, 2, scale, window)
    _DBG["verify_ok" if _ok else "verify_fail"] = _DBG.get("verify_ok" if _ok else "verify_fail", 0) + 1
    return side


@torch.no_grad()
def leaf_hook(output, token) -> None:
    """After the 7-row verify: join the side stream and select the leaf row's attention in leaf mode."""
    if token is None:
        return
    if not isinstance(token, str):
        torch.cuda.current_stream().wait_stream(token)
    _ext.select(S.mode, S.o2, output[LEAF_ROW])


@torch.no_grad()
def setup_step(input_ids: torch.Tensor, positions: torch.Tensor, block_tables, slot_mappings: torch.Tensor,
               group_windows=None) -> bool:
    """Host-side per-step setup (GPU ops only, no sync) for a single-request 7-token verify padded to 8."""
    if S.alt is None or not S.groups:
        _DBG["setup_false"] += 1
        return False
    _DBG["setup_true"] += 1
    pos1 = positions[1:2].clone()  # p+1 (int64)
    S.mode.copy_((S.prev_ns <= LOW_REGIME) & (S.alt[1:2] != S.alt[0:1]))
    r = LEAF_ROW
    _leaf_tok = input_ids[1:2].clone() if _SELFTEST else S.alt[1:2].to(input_ids.dtype)
    input_ids[r:r + 1].copy_(torch.where(S.mode, _leaf_tok, input_ids[r:r + 1]))
    positions[r:r + 1].copy_(torch.where(S.mode, pos1, positions[r:r + 1]))
    _DBG["gw"] = (list(group_windows) if group_windows is not None else None, len(block_tables))
    for gi, bt in enumerate(block_tables):
        g = S.groups.get(gi)
        if g is None:
            continue
        bs = g.bs
        blk = torch.div(pos1, bs, rounding_mode="floor")
        n = pos1 - blk * bs
        rb = bt[0].to(torch.int64).index_select(0, blk)
        # previous fix indices are kept until this step's fix ran (kv_fix_hook uses them); move current -> fix later
        _w = min(g.bt2.shape[1], bt.shape[1])
        g.bt2[0, :_w].copy_(bt[0, :_w])
        g.bt2[0].index_fill_(0, blk, g.sb)
        g.seq2.copy_((pos1 + 1).to(torch.int32))
        s = torch.arange(bs, device=pos1.device, dtype=torch.int64)
        h = torch.arange(2, device=pos1.device, dtype=torch.int64)
        g.rb_rows.copy_(rb.expand(2 * bs))
        g.rb1.copy_(rb)
        g.n1.copy_(n)
        g.mask.copy_((s[None, :] < n).expand(2, bs).reshape(-1, 1))
        slot_mappings[gi, r:r + 1].copy_(torch.where(S.mode, g.sb * bs + n, slot_mappings[gi, r:r + 1]))
        if _DBG.get("btlog", 0) < 4 and gi in (0, 8):
            _DBG["btlog"] = _DBG.get("btlog", 0) + 1
            _b = int(blk.item())
            _DBG.setdefault("bt", []).append(dict(gi=gi, pos1=int(pos1.item()), bs=bs, blk=_b, rb=int(rb.item()), sb=g.sb,
                bt_row=bt[0, max(0, _b - 2):_b + 2].tolist(), bt2_row=g.bt2[0, max(0, _b - 2):_b + 2].tolist(),
                seq2=int(g.seq2.item()), bt_w=int(bt.shape[1]), bt2_w=int(g.bt2.shape[1]), n=int(n.item())))
        g._next_fix_rb = rb.expand(2).clone()
        g._next_fix_n = n.expand(2).clone()
    S.active = True
    return True


@torch.no_grad()
def selftest(all_logits: torch.Tensor) -> None:
    if S.st2 is None:
        S.st2 = torch.zeros((4,), dtype=torch.int64, device=all_logits.device)
    m = S.mode
    same_arg = all_logits[LEAF_ROW].argmax() == all_logits[1].argmax()
    same_all = torch.equal(all_logits[LEAF_ROW], all_logits[1])
    S.st2[0] += m.sum()
    S.st2[1] += (m & same_arg).sum()
    if bool(m.item()):
        S.st2[2] += int(same_all)
        d = (all_logits[LEAF_ROW].float() - all_logits[1].float()).abs().max()
        S.st2[3] = torch.maximum(S.st2[3], (d * 1000).long())


def commit(sampled: torch.Tensor, num_sampled: torch.Tensor, num_rejected: torch.Tensor, leaf_logits: torch.Tensor):
    """sampled [1, 7], num_sampled [1], num_rejected [1]; leaf_logits = logits of row 6 (the leaf row in leaf mode)."""
    _DBG["commits"] += 1
    m = S.mode
    # leaf mode: row 6 is not d6, so a run of 6 "accepted" drafts stops at row 5's output
    clamp = m & (num_sampled[:1] == 7)
    num_sampled.sub_(clamp.to(num_sampled.dtype))
    num_rejected.add_(clamp.to(num_rejected.dtype))
    u = leaf_logits.reshape(1)[0].to(sampled.dtype)  # argmax taken before rejection sampling
    alt1 = S.alt[1:2].to(sampled.dtype)
    hit = m & (num_sampled[:1] == 1) & (sampled[0, :1] == alt1)
    if _NOHIT:
        if S.u_prev is None:
            S.u_prev = torch.zeros((1,), dtype=sampled.dtype, device=sampled.device)
            S.would_prev = torch.zeros((1,), dtype=torch.bool, device=sampled.device)
            S.ustat = torch.zeros((2,), dtype=torch.int64, device=sampled.device)
        # previous step would have hit: its u must equal this step's first target token
        S.ustat[0] += S.would_prev.sum()
        S.ustat[1] += (S.would_prev & (sampled[0, :1] == S.u_prev)).sum()
        S.would_prev.copy_(hit)
        S.u_prev.copy_(u.view(1))
        hit = hit & False
    sampled[0, 1:2].copy_(torch.where(hit, u.view(1), sampled[0, 1:2]))
    num_sampled.add_(hit.to(num_sampled.dtype))
    num_rejected.sub_(hit.to(num_rejected.dtype))
    S.hit.copy_(hit)
    S.hit_prev.copy_(hit)
    S.prev_ns.copy_(num_sampled[:1].to(torch.int64))
    for g in S.groups.values():
        if hasattr(g, "_next_fix_rb"):
            g.fix_rb.copy_(g._next_fix_rb)
            g.fix_n.copy_(g._next_fix_n)
    S.stats[0] += 1
    S.stats[1] += hit.sum()
    S.stats[2] += m.sum()
    S.stats[3] += clamp.sum()
    S.active = False
    return hit


@torch.no_grad()
def fix_hidden(hidden: torch.Tensor | None, aux: list | None) -> None:
    """On a hit, the drafter's context row 1 (position p+1) takes row 8's hidden states."""
    if S.hit is None:
        return
    m = S.hit.view(1, 1)
    for t in ([hidden] if hidden is not None else []) + (list(aux) if aux else []):
        if t is not None and t.shape[0] >= 7:
            t[1:2].copy_(torch.where(m, t[LEAF_ROW:LEAF_ROW + 1], t[1:2]))
