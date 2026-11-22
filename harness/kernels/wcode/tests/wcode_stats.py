"""Phase-1 kill test for lossless weight codes of the decode step (statistics only, no timing).

For every linear weight the single-token decode step reads (language-model projections and lm_head), straight from
the pinned safetensors, one tensor at a time on the GPU:
  * NVFP4 block scales (one FP8 E4M3 byte per 16 weights): distinct bytes, order-0 entropy, coverage of 63/31/15
    entry tables plus an escape, exponent-window codes over blocks along N and along K;
  * NVFP4 values: entropy of the 4-bit codes;
  * FP8 E4M3 weights (merged and requantized exactly as vLLM's ModelOpt FP8 path does): byte entropy, conditional
    exponent entropy per output column, the fp8c code (per-column 8-exponent window, least-frequent code as escape),
    a per-128-K-chunk window variant, a per-tensor window and mixed-radix windows of 5/6/7 exponents;
  * BF16 projections: sign + 4-bit exponent offset + 7-bit mantissa (12-bit) per tensor / per column.
Writes stats.json to $GLIM_OUT and prints one line per tensor plus a summary table.

Run: bin/atom glim exec SHA harness/kernels/wcode/tests/wcode_stats.py [--layers 0,1,...]"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
import time

import torch
from safetensors import safe_open

DEV = "cuda"
BW = 240e9  # bytes/s DRAM streaming rate used for the ms estimate


def log(*a):
    print(*a, flush=True)


class Snap:
    def __init__(self, snap: str):
        self.snap = snap
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self._files = {}

    def f(self, name):
        fn = os.path.join(self.snap, self.wmap[name])
        if fn not in self._files:
            self._files[fn] = safe_open(fn, "pt", device="cpu")
        return self._files[fn]

    def get(self, name):
        return self.f(name).get_tensor(name)

    def meta(self, name):
        sl = self.f(name).get_slice(name)
        return sl.get_dtype(), tuple(sl.get_shape())


GROUPS = [("qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
          ("attn_gate", ["self_attn.gate_proj"]), ("o_proj", ["self_attn.o_proj"]),
          ("gate_up", ["mlp.gate_proj", "mlp.up_proj"]), ("down", ["mlp.down_proj"])]


def ent(h: torch.Tensor) -> float:
    """Order-0 entropy in bits of a histogram."""
    h = h.double()
    t = h.sum()
    p = h[h > 0] / t
    return float(-(p * p.log2()).sum())


def hist_u8(t: torch.Tensor, minlength: int) -> torch.Tensor:
    """bincount of an integer tensor in chunks (bounded int64 temporaries)."""
    h = torch.zeros(minlength, dtype=torch.long, device=DEV)
    flat = t.reshape(-1)
    step = 1 << 26
    for i in range(0, flat.numel(), step):
        h += torch.bincount(flat[i:i + step].long(), minlength=minlength)
    return h


def chunked_rows(n: int, step: int):
    for r0 in range(0, n, step):
        yield r0, min(n, r0 + step)


# ------------------------------------------------------------------------------------------------ NVFP4 scales


def scale_stats(s: torch.Tensor) -> dict:
    """s: [N, G] uint8 scale bytes of a merged NVFP4 linear (row = output column, G = K/16 groups)."""
    n, g = s.shape
    flat = s.reshape(-1).long()
    h = torch.bincount(flat, minlength=256)
    tot = int(h.sum())
    hs, _ = torch.sort(h, descending=True)
    cov = {f"esc_t{t}": round(1.0 - float(hs[:t].sum()) / tot, 6) for t in (255, 63, 31, 15)}
    out = {"n": n, "groups": g, "distinct": int((h > 0).sum()), "H0": round(ent(h), 4),
           "neg": int(h[128:].sum()), "zero": int(h[0]), **cov}
    # table codes: (2^b - 1) entries + escape symbol, escaped bytes raw (8 bits each)
    for b, t in ((6, 63), (5, 31), (4, 15)):
        out[f"bits_table{b}"] = round(b + 8 * out[f"esc_t{t}"], 4)
    # exponent-window codes: per block of B consecutive scales, choose the 2^w exponent window with most members;
    # code = (e - base) w bits + 3 mantissa bits; header 4 bits/block; out-of-window values escape (8 bits raw +
    # one reserved code, counted as 8 bits)
    e = ((s >> 3) & 15).long()
    for axis in ("N", "K"):
        ee = e.t().contiguous() if axis == "N" else e  # blocks run along the last dim
        for B in (16, 64):
            L = ee.shape[1] // B * B
            if L == 0:
                continue
            blk = ee[:, :L].reshape(-1, B)
            nb = blk.shape[0]
            hist = torch.zeros(nb * 16, dtype=torch.int32, device=DEV)
            idx = (torch.arange(nb, device=DEV).view(-1, 1) * 16 + blk).reshape(-1)
            hist.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
            hist = hist.view(nb, 16)
            cs = torch.cat([torch.zeros(nb, 1, dtype=torch.int32, device=DEV), hist.cumsum(1, dtype=torch.int32)], 1)
            for w in (0, 1, 2):
                W = 1 << w
                inw = (cs[:, W:] - cs[:, :-W]).max(1).values
                esc = 1.0 - float(inw.sum()) / (nb * B)
                out[f"win{W}_{axis}{B}_esc"] = round(esc, 6)
                out[f"win{W}_{axis}{B}_bits"] = round(w + 3 + 4.0 / B + 8 * esc, 4)
            del hist, idx, cs
    return out


# ------------------------------------------------------------------------------------------------ FP8 weights


def fp8_stats(w: torch.Tensor) -> dict:
    """w: [N, K] uint8 E4M3 bytes (the merged, requantized weight, one row per output column)."""
    n, k = w.shape
    h = hist_u8(w, 256)
    tot = n * k
    e_hist = torch.zeros(16, dtype=torch.long, device=DEV)
    for b in range(256):
        e_hist[(b >> 3) & 15] += h[b]
    out = {"n": n, "k": k, "H0": round(ent(h), 4), "He": round(ent(e_hist), 4),
           "exp_hist": e_hist.tolist(), "neg_frac": round(float(h[128:].sum()) / tot, 5),
           "zero": int(h[0]), "negzero": int(h[128])}
    # global 8-window
    cs = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), e_hist.cumsum(0)])
    wins = cs[8:] - cs[:-8]
    e0g = int(wins.argmax())
    out["glob_e0"] = e0g
    out["glob_esc"] = round(1.0 - float(wins.max()) / tot, 6)
    # per column: exponent histogram [N, 16], code histogram [N, 128]
    H_e_col = 0.0
    esc_fp8c = 0
    esc_chunk = 0
    inwin_r = {4: 0, 5: 0, 6: 0, 7: 0}
    e0_hist = torch.zeros(16, dtype=torch.long, device=DEV)
    CH = 128
    for r0, r1 in chunked_rows(n, max(1, (1 << 26) // k)):
        x = w[r0:r1]
        rn = r1 - r0
        e = ((x >> 3) & 15).long()
        rows = torch.arange(rn, device=DEV).view(-1, 1)
        hc = torch.bincount((rows * 16 + e).reshape(-1), minlength=rn * 16).view(rn, 16)
        # conditional exponent entropy per column
        p = hc.double() / k
        H_e_col += float(-(torch.where(p > 0, p * p.clamp_min(1e-300).log2(), torch.zeros_like(p))).sum())
        ccs = torch.cat([torch.zeros(rn, 1, dtype=torch.long, device=DEV), hc.cumsum(1)], 1)
        win8 = ccs[:, 8:] - ccs[:, :-8]  # [rn, 9]
        # fp8c tie-break: highest count, then highest e0
        e0 = (win8 * 16 + torch.arange(9, device=DEV)).argmax(1)
        e0_hist += torch.bincount(e0, minlength=16)
        inwin = (e >= e0.view(-1, 1)) & (e <= e0.view(-1, 1) + 7)
        code = ((x.long() & 0x80) >> 1) | (((e - e0.view(-1, 1)) & 7) << 3) | (x.long() & 7)
        cc = torch.bincount((rows * 128 + code)[inwin], minlength=rn * 128).view(rn, 128)
        esc_code = cc.min(1).values  # least frequent window code becomes the escape
        esc_fp8c += int((~inwin).sum()) + int(esc_code.sum())
        for W in (4, 5, 6, 7):
            inwin_r[W] += int((ccs[:, W:] - ccs[:, :-W]).max(1).values.sum())
        # per (column, 128-K chunk) 8-window
        kc = k // CH
        ek = e[:, : kc * CH].reshape(rn * kc, CH)
        rr = torch.arange(rn * kc, device=DEV).view(-1, 1)
        hk = torch.bincount((rr * 16 + ek).reshape(-1), minlength=rn * kc * 16).view(rn * kc, 16)
        kcs = torch.cat([torch.zeros(rn * kc, 1, dtype=torch.long, device=DEV), hk.cumsum(1)], 1)
        esc_chunk += rn * kc * CH - int((kcs[:, 8:] - kcs[:, :-8]).max(1).values.sum())
        del x, e, hc, ccs, win8, inwin, code, cc, ek, hk, kcs
    out["H_e_given_col"] = round(H_e_col / n, 4)
    out["bound_col_static"] = round(1 + H_e_col / n + 3, 4)  # sign 1 bit + H(e|col) + 3 mantissa bits
    out["fp8c_esc"] = round(esc_fp8c / tot, 6)
    # fp8c header: u16 (e0|esc) per column + u16 offset per (8 columns x 128 K) warp-chunk
    hdr = 16.0 / k + 16.0 / (8 * 128)
    out["fp8c_bits"] = round(7 + 8 * esc_fp8c / tot + hdr, 4)
    out["chunk8_esc"] = round(esc_chunk / tot, 6)
    out["chunk8_bits"] = round(7 + 8 * esc_chunk / tot + 4.0 / 128 + 16.0 / (8 * 128), 4)
    out["glob7_bits"] = round(7 + 8 * out["glob_esc"], 4)
    for W in (4, 5, 6, 7):
        esc = 1.0 - inwin_r[W] / tot
        out[f"radix{W}_esc"] = round(esc, 6)
        # 16*W codes (one is the escape) packed in mixed radix: log2(16W) bits per weight
        out[f"radix{W}_bits"] = round(math.log2(16 * W) + 8 * esc + hdr, 4)
    out["e0_hist"] = e0_hist[:9].tolist()
    return out


# ------------------------------------------------------------------------------------------------ BF16 weights


def bf16_stats(w: torch.Tensor) -> dict:
    n, k = w.shape
    u = w.view(torch.int16).long() & 0xFFFF
    e = (u >> 7) & 0xFF
    eh = torch.bincount(e.reshape(-1), minlength=256)
    tot = n * k
    cs = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), eh.cumsum(0)])
    inw = int((cs[16:] - cs[:-16]).max())
    out = {"n": n, "k": k, "He": round(ent(eh), 4), "H0_bytes_hi": None,
           "exp_min": int((eh > 0).nonzero().min()), "exp_max": int((eh > 0).nonzero().max()),
           "glob16_esc": round(1 - inw / tot, 6)}
    esc = 0
    for r0, r1 in chunked_rows(n, max(1, (1 << 26) // k)):
        ee = e[r0:r1]
        rn = r1 - r0
        rows = torch.arange(rn, device=DEV).view(-1, 1)
        hc = torch.bincount((rows * 256 + ee).reshape(-1), minlength=rn * 256).view(rn, 256)
        ccs = torch.cat([torch.zeros(rn, 1, dtype=torch.long, device=DEV), hc.cumsum(1)], 1)
        esc += rn * k - int((ccs[:, 16:] - ccs[:, :-16]).max(1).values.sum())
    out["col16_esc"] = round(esc / tot, 6)
    out["bits12_col"] = round(12 + 16 * esc / tot + 8.0 / k, 4)
    out["bits12_glob"] = round(12 + 16 * out["glob16_esc"], 4)
    hh = torch.bincount((u >> 8).reshape(-1), minlength=256)
    out["H0_bytes_hi"] = round(ent(hh), 4)
    return out


# ------------------------------------------------------------------------------------------------ main


def linears(snap: Snap):
    """[(layer or -1 for lm_head, kind, fmt, [prefixes])] of every linear the text decode step reads."""
    pat = re.compile(r"^(.*language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+))\.weight$")
    per = collections.defaultdict(dict)
    for name in sorted(snap.wmap):
        m = pat.match(name)
        if m:
            per[int(m.group(2))][m.group(3) + "." + m.group(4)] = m.group(1)
    out = []
    for li in sorted(per):
        for kind, ps in GROUPS:
            if not all(p in per[li] for p in ps):
                continue
            prefs = [per[li][p] for p in ps]
            dt, _ = snap.meta(prefs[0] + ".weight")
            fmt = {"U8": "nvfp4", "F8_E4M3": "fp8", "BF16": "bf16"}.get(dt, dt)
            if fmt == "nvfp4" and prefs[0] + ".weight_scale" not in snap.wmap:
                fmt = "u8?"
            out.append((li, kind, fmt, prefs))
    lm = [n for n in snap.wmap if re.match(r"^(.*\.)?lm_head\.weight$", n)]
    for name in lm:
        dt, _ = snap.meta(name)
        fmt = {"U8": "nvfp4", "F8_E4M3": "fp8", "BF16": "bf16"}.get(dt, dt)
        out.append((-1, "lm_head", fmt, [name[: -len(".weight")]]))
    return out


def main():
    snap = Snap(os.environ["GLIM_SNAPSHOT"])
    out_dir = os.environ.get("GLIM_OUT", "/tmp")
    only = None
    if "--layers" in sys.argv:
        only = {int(v) for v in sys.argv[sys.argv.index("--layers") + 1].split(",")}
    with open(os.path.join(snap.snap, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    ql = qc.get("quantized_layers", {})
    algos = collections.Counter(
        (v.get("quant_algo") if isinstance(v, dict) else str(v)) for v in (ql.values() if isinstance(ql, dict) else []))
    log("quantization_config keys:", sorted(qc), "quantized_layers algos:", dict(algos))
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings"))
    log("tie_word_embeddings:", tied)
    lin = linears(snap)
    log(f"{len(lin)} decode linears;", collections.Counter((k, f) for _, k, f, _ in lin))
    rec = []
    t0 = time.time()
    for li, kind, fmt, prefs in lin:
        if only is not None and li not in only and li != -1:
            continue
        r = {"layer": li, "kind": kind, "fmt": fmt}
        if fmt == "nvfp4":
            ws = [snap.get(p + ".weight") for p in prefs]
            ss = [snap.get(p + ".weight_scale") for p in prefs]
            w = torch.cat(ws, 0).to(DEV)
            s = torch.cat(ss, 0).to(DEV).view(torch.uint8)
            n, k2 = w.shape
            k = 2 * k2
            r["n"], r["k"] = n, k
            r["bytes_vals"] = n * k // 2
            r["bytes_scales"] = s.numel()
            nh = hist_u8(w & 15, 16) + hist_u8(w >> 4, 16)
            r["nib_H0"] = round(ent(nh), 4)
            r["nib_hist"] = nh.tolist()
            r["scale"] = scale_stats(s)
            gs = [float(snap.get(p + ".weight_scale_2").float().reshape(-1)[0]) for p in prefs]
            r["global_scales_equal"] = len(set(gs)) == 1
            del w, s
        elif fmt == "fp8":
            ws = [snap.get(p + ".weight") for p in prefs]
            wsc = [float(snap.get(p + ".weight_scale").float().reshape(-1)[0]) for p in prefs]
            w = torch.cat(ws, 0).to(DEV)
            if len(set(wsc)) > 1:
                # vLLM: requantize_with_max_scale (dequant each shard with its scale, quant with the max)
                from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale
                wt = torch.tensor(wsc, device=DEV, dtype=torch.float32)
                _, w = requantize_with_max_scale(w, wt, [x.shape[0] for x in ws])
                r["requantized"] = True
            u = w.view(torch.uint8)
            r["n"], r["k"] = u.shape
            r["bytes"] = u.numel()
            r["fp8"] = fp8_stats(u)
            del w, u
        elif fmt == "bf16":
            w = torch.cat([snap.get(p + ".weight") for p in prefs], 0).to(DEV)
            r["n"], r["k"] = w.shape
            r["bytes"] = w.numel() * 2
            r["bf16"] = bf16_stats(w)
            del w
        else:
            r["skip"] = True
        torch.cuda.empty_cache()
        rec.append(r)
        brief = {kk: vv for kk, vv in r.items() if kk not in ("scale", "fp8", "bf16", "nib_hist")}
        extra = ""
        if "scale" in r:
            sc = r["scale"]
            extra = (f" scales: distinct {sc['distinct']} H0 {sc['H0']} esc63 {sc['esc_t63']} esc31 {sc['esc_t31']} "
                     f"esc15 {sc['esc_t15']} t6 {sc['bits_table6']} t5 {sc['bits_table5']} t4 {sc['bits_table4']} "
                     f"w4N64 {sc.get('win4_N64_bits')} w4K64 {sc.get('win4_K64_bits')} | nib H0 {r['nib_H0']}")
        if "fp8" in r:
            fp = r["fp8"]
            extra = (f" fp8: H0 {fp['H0']} bound {fp['bound_col_static']} fp8c {fp['fp8c_bits']} (esc {fp['fp8c_esc']}) "
                     f"chunk8 {fp['chunk8_bits']} glob7 {fp['glob7_bits']} r5 {fp['radix5_bits']} "
                     f"r6 {fp['radix6_bits']} r7 {fp['radix7_bits']}")
        if "bf16" in r:
            b = r["bf16"]
            extra = f" bf16: He {b['He']} col16_esc {b['col16_esc']} bits12 {b['bits12_col']} glob {b['bits12_glob']}"
        log(f"[{time.time() - t0:6.1f}s] {brief}{extra}")
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(rec, f)
    summarize(rec)


def summarize(rec):
    log("\n==== summary (GB per decode token; ms at 240 GB/s) ====")
    cls = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rec:
        if r.get("skip"):
            continue
        if r["fmt"] == "nvfp4":
            key = "nvfp4_lm_head" if r["kind"] == "lm_head" else "nvfp4"
            c = cls[key]
            c["vals"] += r["bytes_vals"]
            c["scales"] += r["bytes_scales"]
            nsc = r["bytes_scales"]
            sc = r["scale"]
            for b in (6, 5, 4):
                c[f"scales_t{b}"] += nsc * sc[f"bits_table{b}"] / 8
            c["scales_H0"] += nsc * sc["H0"] / 8
            for key2 in ("win4_N64_bits", "win4_K64_bits", "win2_N64_bits", "win2_K64_bits", "win4_N16_bits",
                         "win4_K16_bits"):
                if key2 in sc:
                    c["scales_" + key2] += nsc * sc[key2] / 8
            c["vals_H0"] += r["bytes_vals"] * r["nib_H0"] / 4
            c["distinct_max"] = max(c["distinct_max"], sc["distinct"])
        elif r["fmt"] == "fp8":
            c = cls["fp8_" + r["kind"]]
            fp = r["fp8"]
            c["bytes"] += r["bytes"]
            for key2 in ("H0", "bound_col_static", "fp8c_bits", "chunk8_bits", "glob7_bits", "radix4_bits",
                         "radix5_bits", "radix6_bits", "radix7_bits"):
                c[key2] += r["bytes"] * fp[key2] / 8
            c["count"] += 1
        elif r["fmt"] == "bf16":
            c = cls["bf16_" + r["kind"]]
            c["bytes"] += r["bytes"]
            c["bits12_col"] += r["bytes"] * r["bf16"]["bits12_col"] / 16
            c["count"] += 1
    total = 0.0
    for key, c in sorted(cls.items()):
        if key.startswith("nvfp4"):
            tb = c["vals"] + c["scales"]
            total += tb
            log(f"{key}: values {c['vals'] / 1e9:.3f} GB + scales {c['scales'] / 1e9:.3f} GB; "
                f"max distinct scale bytes {int(c['distinct_max'])}")
            for b in (6, 5, 4):
                sv = c["scales"] - c[f"scales_t{b}"]
                log(f"   scales table{b}: -> {c[f'scales_t{b}'] / 1e9:.4f} GB, saves {sv / 1e9:.4f} GB = "
                    f"{sv / BW * 1e3:.2f} ms")
            for key2 in ("win4_N64_bits", "win4_K64_bits", "win2_N64_bits", "win2_K64_bits", "win4_N16_bits",
                         "win4_K16_bits"):
                v = c.get("scales_" + key2, 0)
                if v:
                    log(f"   scales {key2}: -> {v / 1e9:.4f} GB, saves {(c['scales'] - v) / 1e9:.4f} GB = "
                        f"{(c['scales'] - v) / BW * 1e3:.2f} ms")
            log(f"   scales H0 bound: saves {(c['scales'] - c['scales_H0']) / 1e9:.4f} GB; "
                f"values H0 bound saves {(c['vals'] - c['vals_H0']) / 1e9:.4f} GB = "
                f"{(c['vals'] - c['vals_H0']) / BW * 1e3:.2f} ms")
        else:
            total += c["bytes"]
            bpw = 16 if key.startswith("bf16") else 8
            parts = " ".join(f"{k2} {bpw * c[k2] / c['bytes']:.3f}b (saves {(c['bytes'] - c[k2]) / BW * 1e3:.2f} ms)"
                             for k2 in c if k2 not in ("bytes", "count"))
            log(f"{key}: {int(c['count'])} tensors {c['bytes'] / 1e9:.3f} GB; {parts}")
    log(f"total weight bytes per decode token: {total / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
