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
LOW_REGIME = int(os.environ.get("MUSE_LEAF_LOW", "2"))  # leaf mode when the previous verify committed <= this


class _Group:
    """Views into the shared per-group state rows (S.gbt2 / S.gseq / S.gstate)."""

    def __init__(self, kv_blocks: int, bs: int, gid: int):
        self.gid = gid
        self.sb = kv_blocks - 1 - gid  # one scratch block per kv-cache group (groups may share a cache tensor)
        self.bs = bs
        self.bt2 = S.gbt2[gid:gid + 1]
        self.seq2 = S.gseq[gid:gid + 1]
        self.rb1 = S.gstate[gid, 0:1]
        self.n1 = S.gstate[gid, 1:2]
        self.fix_rb = S.gstate[gid, 2:3]
        self.fix_n = S.gstate[gid, 3:4]


class _State:
    def __init__(self):
        self.groups: dict[int, _Group] = {}  # keyed by kv-cache group index
        self.gbt2 = None
        self.gseq = None
        self.gstate = None
        self.kv_blocks = None
        self.bs = None
        self.bt_ptrs = None
        self.bt_ptrs_key = None
        self.G = 0
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
        if S.gbt2 is None:
            W = int(block_table.shape[1])
            S.gbt2 = torch.zeros((16, W), dtype=torch.int32, device=kv_cache.device)
            S.gseq = torch.zeros((16,), dtype=torch.int32, device=kv_cache.device)
            S.gstate = torch.zeros((16, 8), dtype=torch.int64, device=kv_cache.device)
            S.kv_blocks = int(kv_cache.shape[0])
            S.bs = int(kv_cache.shape[2])
        if key >= 16 or int(kv_cache.shape[0]) != S.kv_blocks:
            return None
        g = _Group(int(kv_cache.shape[0]), int(kv_cache.shape[2]), key)
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
def prep_hook(query, kv_cache, attn_metadata, layer=None):
    """Per attention layer before the 7-row verify: last step's leaf K/V commit + scratch build (one fused kernel).
    Returns the leaf arguments for the verify kernel (native leaf row), or None."""
    g = group_for(attn_metadata.block_table, kv_cache, layer)
    if g is None or S.q2 is None:
        return None
    _DBG["attn_calls"] += 1
    _ext.prep(kv_cache, g.sb, g.rb1, g.n1, g.fix_rb, g.fix_n, S.hit_prev, S.mode, query, LEAF_ROW, S.q2)
    return (g.bt2, g.seq2, S.mode, LEAF_ROW)


@torch.no_grad()
def setup_step(input_ids: torch.Tensor, positions: torch.Tensor, block_tables, slot_mappings: torch.Tensor,
               group_windows=None) -> bool:
    """One fused kernel: leaf mode, row-6 inputs, per-group alternate block tables / key counts / slots."""
    if S.alt is None or S.gbt2 is None or not S.groups:
        _DBG["setup_false"] += 1
        return False
    G = len(block_tables)
    if G > 16:
        _DBG["setup_false"] += 1
        return False
    key = tuple(bt.data_ptr() for bt in block_tables)
    if key != S.bt_ptrs_key:
        S.bt_ptrs = torch.tensor(list(key), dtype=torch.int64, device=positions.device)
        S.bt_ptrs_key = key
    _DBG["setup_true"] += 1
    _ext.setup(positions, input_ids, slot_mappings, S.bt_ptrs, S.gbt2, S.gseq, S.gstate, G, S.bs, S.kv_blocks - 1,
               S.alt, S.prev_ns, LOW_REGIME, S.mode, LEAF_ROW)
    S.G = G
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


@torch.no_grad()
def commit(sampled: torch.Tensor, num_sampled: torch.Tensor, num_rejected: torch.Tensor, leaf_u: torch.Tensor):
    """One fused kernel: clamp in leaf mode, commit [alt, u] on a hit, carry the K/V fix to the next forward."""
    _DBG["commits"] += 1
    _ext.commit(sampled, num_sampled, num_rejected, leaf_u, S.alt, S.mode, S.prev_ns, S.hit, S.hit_prev,
                S.gstate, S.G, S.stats, 1 if _NOHIT else 0)
    S.active = False


@torch.no_grad()
def fix_hidden(hidden: torch.Tensor | None, aux: list | None) -> None:
    """On a hit, the drafter's context row 1 (position p+1) takes row 8's hidden states."""
    if S.hit is None:
        return
    m = S.hit.view(1, 1)
    for t in ([hidden] if hidden is not None else []) + (list(aux) if aux else []):
        if t is not None and t.shape[0] >= 7:
            t[1:2].copy_(torch.where(m, t[LEAF_ROW:LEAF_ROW + 1], t[1:2]))
