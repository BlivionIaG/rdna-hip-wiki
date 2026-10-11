#!/usr/bin/env python3
"""mma-probe harness: MMA-Sim-style closed-loop feature probing for RDNA DOT/WMMA.

Usage (on the GPU box, after ./build.sh <arch>):
    python3 harness.py --arch gfx1030 --build build/gfx1030 --out results/gfx1030
    python3 harness.py --arch gfx1100 --build build/gfx1100 --out results/gfx1100
Self-test of the inference logic without a GPU:
    python3 harness.py --selftest

Method (arXiv 2511.10909 §III-A, adapted):
  T0  determinism (Step 1) and, for WMMA, cross-position consistency
  T1  summation tree from (U, -U, v) inputs (Step 2, FPRev)
  T2  fused-sum precision with cancellation (FPT 1)
  T3  rounding mode and tie-break (FPT 2), plus a sticky/extra-bit probe
  T5  FP16/BF16/FP32 subnormal handling (input, accumulator, output)
  T6  Inf/NaN propagation
  T7  odd symmetry  d(-a,b,-c) == -d(a,b,c)
  T8  randomized differential test against candidate models (Step 4)
Each of T3/T5/T6 is repeated under every MODE register setting the kernels
were built for (s_setreg of FP_ROUND / FP_DENORM) and under the descriptor
mode of both build variants (default and -fgpu-flush-denormals-to-zero).
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpmodel as M  # noqa: E402

P2 = lambda e: Fr(2) ** e  # noqa: E731

OPS = {
    # op: (family, kind, K, input fmt, clamp)
    "dot2_f32_f16": ("dot2_f16", "dot", 2, "f16", False),
    "dot2c_f32_f16": ("dot2_f16", "dot", 2, "f16", False),
    "fdot2_builtin": ("dot2_f16", "dot", 2, "f16", False),
    "fdot2_builtin_clamp": ("dot2_f16", "dot", 2, "f16", True),
    "dot2_f32_bf16": ("dot2_bf16", "dot", 2, "bf16", False),
    "fdot2_f32_bf16_builtin": ("dot2_bf16", "dot", 2, "bf16", False),
    "wmma_f32_16x16x16_f16": ("wmma", "dot", 16, "f16", False),
    "wmma_f32_16x16x16_bf16": ("wmma", "dot", 16, "bf16", False),
    "fma_mix_f32": ("fma_baseline", "dot", 1, "f16", False),
    "fma_mix_c": ("fma_baseline", "dot", 1, "f16", False),
    "fma_f32": ("fma_baseline", "fma32", 1, "f32", False),
    "fma_c": ("fma_baseline", "fma32", 1, "f32", False),
    "mul_add_c": ("fma_baseline", "fma32", 1, "f32", False),
    "pk_fma_f16": ("fma_baseline", "pkfma", 1, "f16", False),
    "dot4_i32_i8": ("dot4_i8", "idot", 4, "i8", False),
    "dot4_i32_i8_clamp": ("dot4_i8", "idot", 4, "i8", True),
    "dot4c_i32_i8": ("dot4_i8", "idot", 4, "i8", False),
    "sdot4_builtin": ("dot4_i8", "idot", 4, "i8", False),
    "sudot4_ss_builtin": ("dot4_i8", "idot", 4, "i8", False),
    "sudot4_ss_builtin_clamp": ("dot4_i8", "idot", 4, "i8", True),
}
ARCH_FAMILIES = {
    "gfx1030": ["dot2_f16", "dot4_i8", "fma_baseline"],
    "gfx1100": ["dot2_f16", "dot2_bf16", "wmma", "dot4_i8", "fma_baseline"],
}
FP_MODES = ["desc", "m00", "m10", "m20", "m30", "m40", "m50", "m60", "m70", "m80",
            "m90", "ma0", "mb0", "mc0", "md0", "me0", "mf0", "mf1", "mf2", "mf3",
            "mf4", "mf8", "mfc"]
DENORM_TXT = {0: "flush in+out", 1: "allow in/flush out", 2: "flush in/allow out", 3: "allow both"}
ROUND_TXT = {0: "RNE", 1: "+inf", 2: "-inf", 3: "RZ"}


def mode_txt(m):
    return "round32=%s round16/64=%s den32=%s den16/64=%s" % (
        ROUND_TXT[m & 3], ROUND_TXT[(m >> 2) & 3], DENORM_TXT[(m >> 4) & 3], DENORM_TXT[(m >> 6) & 3])


# ---------------------------------------------------------------------------
# Test representation: a test is (alist, blist, c) of raw bit patterns.
# ---------------------------------------------------------------------------
def fmt_of(op):
    return M.FMTS.get(OPS[op][3])


def vbits(x, fmt):
    return M.enc_exact(x, fmt)


def prod_pair(p, fmt):
    """Split an exact power-of-two-times-mantissa product p into a*b (both exact in fmt)."""
    if p == 0:
        return 0, vbits(Fr(1), fmt)
    neg = p < 0
    ap = -p if neg else p
    e = M.ilog2(ap)
    mant = ap / P2(e)            # in [1, 2)
    lo = fmt.emin
    ea = max(lo, min(fmt.emax, e - lo)) if e < 2 * lo + 2 else e // 2
    eb = e - ea
    if not (lo <= eb <= fmt.emax and lo <= ea <= fmt.emax):
        raise ValueError("product %s not splittable in %s" % (p, fmt))
    a = mant * P2(ea)
    b = P2(eb)
    return vbits(-a if neg else a, fmt), vbits(b, fmt)


def make_dot(op, prods, c):
    """prods: list of K exact products (Fractions); c: f32 Fraction or bits (int with tag)."""
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    al, bl = [], []
    for p in prods + [0] * (K - len(prods)):
        a, b = prod_pair(Fr(p), fmt)
        al.append(a)
        bl.append(b)
    cb = c[1] if isinstance(c, tuple) else M.enc_exact(c, M.F32)
    return (al, bl, cb)


def raw(op, al, bl, c):
    K = OPS[op][2]
    return (list(al) + [0] * (K - len(al)), list(bl) + [0] * (K - len(bl)), c)


def to_record(op, t):
    fam, kind, K, f, _ = OPS[op]
    al, bl, c = t
    if kind == "dot" and K == 1:
        return [al[0], bl[0], c]
    if kind == "dot" and K == 2:
        return [al[0] | al[1] << 16, bl[0] | bl[1] << 16, c]
    if kind == "dot" and K == 16:
        A = [al[2 * i] | al[2 * i + 1] << 16 for i in range(8)]
        B = [bl[2 * i] | bl[2 * i + 1] << 16 for i in range(8)]
        return A + B + [c]
    if kind == "fma32":
        return [al[0], bl[0], c]
    if kind == "pkfma":   # same test in both halves
        return [al[0] | al[0] << 16, bl[0] | bl[0] << 16, c | c << 16]
    if kind == "idot":
        return [al[0], bl[0], c]
    raise ValueError(kind)


def expected(op, model, t):
    """Evaluate a model on a test (models take decoded values)."""
    fam, kind, K, f, _ = OPS[op]
    al, bl, c = t
    if kind == "dot":
        fmt = M.FMTS[f]
        return model([M.decode(x, fmt) for x in al], [M.decode(x, fmt) for x in bl], M.decode(c, M.F32))
    return model(t)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class HwBackend:
    def __init__(self, build, variant):
        self.dir = os.path.join(build, variant)
        self.calls = 0

    def run(self, op, mode, tests):
        fam = OPS[op][0]
        exe = os.path.join(self.dir, fam)
        recs = [to_record(op, t) for t in tests]
        with tempfile.TemporaryDirectory() as td:
            fi, fo = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
            with open(fi, "wb") as fh:
                for r in recs:
                    fh.write(struct.pack("<%dI" % len(r), *r))
            subprocess.run([exe, "run", op, mode, fi, fo], check=True)
            data = open(fo, "rb").read()
        self.calls += 1
        w = struct.unpack("<%dI" % (len(data) // 4), data)
        ow = 3 if OPS[op][1] == "dot" and OPS[op][2] == 16 else 2
        out = []
        for i in range(len(tests)):
            o = w[ow * i: ow * i + ow]
            d = o[0]
            if OPS[op][1] == "pkfma":
                lo, hi = d & 0xFFFF, d >> 16
                d = lo if lo == hi else ("split", lo, hi)
            out.append({"d": d, "mode": o[-1], "uniform": (o[1] == 1) if ow == 3 else True})
        return out


class FakeBackend:
    """Pretends to be hardware using a fixed model (for --selftest)."""

    def __init__(self, dot_model, fma_fused=True, ftz_in="none"):
        self.dot_model, self.fma_fused, self.ftz_in = dot_model, fma_fused, ftz_in
        self.calls = 0

    def run(self, op, mode, tests):
        self.calls += 1
        kind = OPS[op][1]
        m = 0xF0 if mode == "desc" else int(mode[1:], 16)
        out = []
        for t in tests:
            if kind == "dot":
                d = expected(op, self.dot_model, t)
            elif kind == "fma32":
                d = M.fma_model(M.decode(t[0][0], M.F32), M.decode(t[1][0], M.F32), M.decode(t[2], M.F32),
                                M.F32, fused=self.fma_fused)
            elif kind == "pkfma":
                d = M.fma_model(M.decode(t[0][0], M.F16), M.decode(t[1][0], M.F16), M.decode(t[2], M.F16), M.F16)
            else:
                d = idot_model(t, False)
            out.append({"d": d, "mode": m, "uniform": True})
        return out


# ---------------------------------------------------------------------------
# Integer DOT
# ---------------------------------------------------------------------------
def s8(x, i):
    v = (x >> (8 * i)) & 0xFF
    return v - 256 if v & 0x80 else v


def idot_model(t, clamp):
    a, b, c = t[0][0], t[1][0], t[2]
    cs = c - (1 << 32) if c & 0x80000000 else c
    s = cs + sum(s8(a, i) * s8(b, i) for i in range(4))
    if clamp:
        s = max(-(1 << 31), min((1 << 31) - 1, s))
    return s & 0xFFFFFFFF


def probe_idot(be, op):
    clamp = OPS[op][4]
    rep = {"op": op, "kind": "idot"}
    cases = {
        "max_pos_overflow": ([0x80808080], [0x80808080], 0x7FFFFFFF),
        "max_neg_overflow": ([0x80808080], [0x7F7F7F7F], 0x80000000),
        "no_overflow": ([0x01FF7F80], [0x80017FFF], 5),
    }
    res = be.run(op, "desc", [cases[k] for k in cases])
    rep["edge"] = {}
    for (k, t), r in zip(cases.items(), res):
        rep["edge"][k] = {"d": "0x%08x" % r["d"], "wrap": r["d"] == idot_model(t, False),
                          "saturate": r["d"] == idot_model(t, True)}
    rnd = random.Random(7)
    tests = [([rnd.getrandbits(32)], [rnd.getrandbits(32)], rnd.getrandbits(32)) for _ in range(4000)]
    res = be.run(op, "desc", tests)
    rep["random_exact_wrap"] = sum(r["d"] == idot_model(t, False) for t, r in zip(tests, res))
    rep["random_saturate"] = sum(r["d"] == idot_model(t, True) for t, r in zip(tests, res))
    rep["random_n"] = len(tests)
    n = len(tests)
    edge_wrap = all(e["wrap"] for e in rep["edge"].values())
    edge_sat = all(e["saturate"] for e in rep["edge"].values())
    # Overflow edges decide wrap vs saturate; random tests check exactness.
    if edge_sat and not edge_wrap and rep["random_saturate"] == n:
        rep["verdict"] = "exact, saturating on i32 overflow"
    elif edge_wrap and not edge_sat and rep["random_exact_wrap"] == n:
        rep["verdict"] = "exact, wraps mod 2^32"
    else:
        rep["verdict"] = "matches neither exact-wrap nor exact-saturate: see edge cases"
    return rep


# ---------------------------------------------------------------------------
# Floating-point DOT / WMMA / FMA-mix probes
# ---------------------------------------------------------------------------
def fval(bits, fmt=M.F32):
    v = M.decode(bits, fmt)
    if v.kind == M.NAN:
        return "NaN(0x%x)" % bits
    if v.kind == M.INF:
        return "-inf" if v.neg else "+inf"
    if v.x == 0:
        return "-0" if v.neg else "+0"
    e = M.ilog2(abs(v.x))
    return "%s%s*2^%d" % ("-" if v.x < 0 else "", str(abs(v.x) / P2(e)), e)


def realize_tree(n, L):
    """FPRev-style realization from L[(i,j)] = d/v counts. Returns (tree, consistent)."""
    size = {k: n - c for k, c in L.items()}
    parent = list(range(n))
    nodes = {i: str(i) for i in range(n)}
    members = {i: {i} for i in range(n)}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    ok = True
    for s in sorted(set(size.values())):
        groups = {}
        pairs = [k for k, v in size.items() if v == s]
        roots = set()
        for i, j in pairs:
            roots.add(find(i))
            roots.add(find(j))
        # union all clusters connected by pairs of this size
        adj = {r: set() for r in roots}
        for i, j in pairs:
            adj[find(i)].add(find(j))
            adj[find(j)].add(find(i))
        seen = set()
        for r in roots:
            if r in seen:
                continue
            comp, stack = [], [r]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.append(x)
                stack.extend(adj[x] - seen)
            new = min(comp)
            mem = set().union(*(members[c] for c in comp))
            if len(mem) != s:
                ok = False
            label = "(" + " + ".join(sorted((nodes[c] for c in comp), key=len)) + ")"
            for c in comp:
                parent[c] = new
            nodes[new], members[new] = label, mem
    root = find(0)
    return nodes[root], ok and len(members[root]) == n


def leaf_name(i, K):
    return "c" if i == K else "p%d" % i


def probe_tree(be, op, mode="desc"):
    fam, kind, K, f, _ = OPS[op]
    if K < 2:
        return {"skipped": "single product"}
    U, v = P2(30), P2(-24)
    n = K + 1
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tests = []
    for sgn in (1, -1):
        for i, j in pairs:
            leaves = [v] * n
            leaves[i], leaves[j] = sgn * U, -sgn * U
            tests.append(make_dot(op, leaves[:K], leaves[K]))
    res = be.run(op, mode, tests)
    L, raw_counts, clean = {}, {}, True
    for idx, (i, j) in enumerate(pairs):
        r1, r2 = res[idx], res[idx + len(pairs)]
        d1 = M.decode(r1["d"], M.F32)
        cnt = None if d1.kind != M.NUM else d1.x / v
        raw_counts["%s,%s" % (leaf_name(i, K), leaf_name(j, K))] = str(cnt)
        if cnt is None or cnt.denominator != 1 or not (0 <= cnt <= n - 2) or r2["d"] != r1["d"]:
            clean = False
        else:
            L[(i, j)] = int(cnt)
    out = {"d_over_v": raw_counts, "U": "2^30", "v": "2^-24"}
    if not clean:
        out["tree"] = None
        out["note"] = "non-integer or sign-dependent counts: not a pure swamping pattern; use T8"
        return out
    if all(c == n - 2 for c in L.values()):
        out["tree"] = "fused " + str(n) + "-term sum, no intermediate rounding visible (high precision)"
        return out
    tree, ok = realize_tree(n, L)
    names = {str(i): leaf_name(i, K) for i in range(n)}
    for k in sorted(names, key=lambda s: -len(s)):
        tree = tree.replace("(" + k + " ", "(" + names[k] + " ").replace(" " + k + ")", " " + names[k] + ")").replace(" " + k + " ", " " + names[k] + " ")
    out["tree"] = tree
    out["tree_consistent_binary_LP"] = ok
    if not ok:
        out["note"] = "counts do not realize a limited-precision tree; a high-precision fused node is likely (MMA-Sim Eq. 9)"
    return out


def probe_precision(be, op, mode="desc"):
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    out = {}
    if K >= 2:
        # (a) +U, -U as products, c = 2^-e : does c survive exactly?
        es = list(range(0, 150))
        tests = [make_dot(op, [P2(30), -P2(30)], P2(-e)) for e in es]
        res = be.run(op, mode, tests)
        keep = [e for e, t, r in zip(es, tests, res) if r["d"] == t[2]]
        out["c_after_cancel_smallest_exact"] = ("2^-%d" % max(keep)) if keep else "c lost (absorbed before cancellation)"
        # (b) c = +U, p0 = -U, p1 = 2^-e
        emin_prod = 2 * fmt.emin
        es = [e for e in range(0, -emin_prod + 1) if -e >= max(emin_prod, -126)]
        tests = [make_dot(op, [-P2(30), P2(-e)], P2(30)) for e in es]
        res = be.run(op, mode, tests)
        keep = [e for e, r in zip(es, res) if r["d"] == M.enc_exact(P2(-e), M.F32)]
        lim = max(es)
        if keep:
            m = max(keep)
            out["product_after_cancel_smallest_exact"] = "2^-%d" % m
            out["fused_bits_below_leading_2^30"] = (">= %d (probe limit)" % (30 + m)) if m == lim else (30 + m)
            out["note"] = "FPT1: F = e_U - log2(eps) - 1 fractional bits; e_U = 30"
        else:
            out["product_after_cancel_smallest_exact"] = "product lost (absorbed into U before cancellation)"
    return out


def classify_rounding(r):
    """r: dict of outcomes for +1.5e,+0.5e,-1.5e,-0.5e as 'up'/'down' (magnitude)."""
    key = (r["+1.5"], r["+0.5"], r["-1.5"], r["-0.5"])
    table = {("away", "toward", "away", "toward"): "RN*", ("toward", "toward", "toward", "toward"): "RZ",
             ("away", "away", "toward", "toward"): "RU", ("toward", "toward", "away", "away"): "RD",
             ("away", "away", "away", "away"): "RA"}
    return table.get(key, "unclassified %s" % (key,))


def classify_ties(r):
    key = (r["+1"], r["+3"], r["-1"], r["-3"])
    table = {(0, 4, 0, 4): "RNE", (2, 4, 2, 4): "RNA", (0, 2, 0, 2): "RNZ",
             (2, 4, 0, 2): "RNU (ties up)", (0, 2, 2, 4): "RND (ties down)", (2, 2, 2, 2): "RNO (ties to odd)"}
    return table.get(key, "unclassified %s" % (key,))


def probe_rounding(be, op, mode):
    """FPT 2 with U = 1.0 (f32 ulp 2^-23, eps = 2^-24); two placements."""
    fam, kind, K, f, _ = OPS[op]
    eps = P2(-24)
    ks = {"+1.5": Fr(3, 2), "+0.5": Fr(1, 2), "+1": Fr(1), "+3": Fr(3)}
    out = {}
    for place in ("c_is_U", "product_is_U"):
        tests, keys = [], []
        for sgn in (1, -1):
            for k, mult in ks.items():
                small, big = sgn * mult * eps, sgn * Fr(1)
                key = ("+" if sgn > 0 else "-") + k[1:]
                if kind == "fma32":
                    if place == "c_is_U":
                        t = ([M.enc_exact(small, M.F32)], [M.enc_exact(Fr(1), M.F32)], M.enc_exact(big, M.F32))
                    else:
                        t = ([M.enc_exact(big, M.F32)], [M.enc_exact(Fr(1), M.F32)], M.enc_exact(small, M.F32))
                elif kind == "pkfma":
                    e16 = P2(-11)
                    small16 = sgn * mult * e16
                    if place == "c_is_U":
                        a, b = prod_pair(small16, M.F16)
                        t = ([a], [b], M.enc_exact(big, M.F16))
                    else:
                        t = ([M.enc_exact(big, M.F16)], [M.enc_exact(Fr(1), M.F16)], M.enc_exact(small16, M.F16))
                else:
                    t = make_dot(op, [small], big) if place == "c_is_U" else make_dot(op, [big], small)
                tests.append(t)
                keys.append(key)
        res = be.run(op, mode, tests)
        fmt_out = M.F16 if kind == "pkfma" else M.F32
        ulp = P2(-10) if kind == "pkfma" else P2(-23)
        half = ulp / 2
        r = {}
        for key, rr in zip(keys, res):
            if isinstance(rr["d"], tuple):
                r[key] = "split"
                continue
            v = M.decode(rr["d"], fmt_out)
            if v.kind != M.NUM:
                r[key] = "special"
                continue
            delta = (abs(v.x) - 1) / half   # in units of eps
            if key[1:] in ("1.5", "0.5"):
                r[key] = "away" if delta > 0 else "toward"
            else:
                r[key] = int(delta) if delta.denominator == 1 else str(delta)
        out[place] = {"mode": classify_rounding(r), "ties": classify_ties(r) if all(isinstance(r[k], int) for k in ("+1", "+3", "-1", "-3")) else "n/a", "raw": r}
    return out


def probe_sticky(be, op, mode="desc"):
    """Does a term far below the rounding point still break a tie? (extra bits / sticky)"""
    fam, kind, K, f, _ = OPS[op]
    if kind != "dot" or K < 2:
        return {"skipped": "needs >= 2 products"}
    fmt = M.FMTS[f]
    kmax = min(60, -2 * fmt.emin - 4)
    ks = list(range(1, kmax + 1))
    up, down = [], []
    t1 = [make_dot(op, [P2(-4), P2(-4 - k)], P2(20)) for k in ks]
    t2 = [make_dot(op, [P2(-4), -P2(-4 - k)], P2(20) + P2(-3)) for k in ks]
    r1, r2 = be.run(op, mode, t1), be.run(op, mode, t2)
    for k, a, b in zip(ks, r1, r2):
        if a["d"] == M.enc_exact(P2(20) + P2(-3), M.F32):
            up.append(k)
        if b["d"] == M.enc_exact(P2(20) + P2(-3), M.F32):
            down.append(k)
    def run_len(lst):
        n = 0
        for k in ks:
            if k in lst:
                n = k
            else:
                break
        return n
    return {"tie_broken_up_by_positive_tiny_down_to_k": run_len(up),
            "tie_broken_down_by_negative_tiny_down_to_k": run_len(down), "k_max_probed": kmax,
            "note": "c=2^20 (ulp 2^-3), p0=2^-4 (exact tie), p1=+/-2^-(4+k). Correct RNE needs p1 seen for every k; "
                    "a value < k_max means terms more than k bits below the output half-ulp are dropped before rounding"}


def subnormal_cases(op):
    fam, kind, K, f, _ = OPS[op]
    C = []
    if kind == "dot":
        fmt = M.FMTS[f]
        one, sub = M.enc_exact(Fr(1), fmt), 1
        negz = 1 << (fmt.width - 1)
        big = M.enc_exact(P2(10), fmt)
        sub_big = 1 << (fmt.mbits - 1)            # 2^(emin-1)
        C.append(("in_sub_a", raw(op, [sub], [one], 0), {"keep": M.encode(M.decode(sub, fmt).x, M.F32), "flush": 0}))
        C.append(("in_sub_b", raw(op, [one], [sub], 0), {"keep": M.encode(M.decode(sub, fmt).x, M.F32), "flush": 0}))
        C.append(("in_sub_big", raw(op, [sub_big], [big], 0),
                  {"keep": M.encode(M.decode(sub_big, fmt).x * 1024, M.F32), "flush": 0}))
        if K >= 2:
            C.append(("in_sub_neg_sign", raw(op, [sub | negz, negz], [one, 0], 0x80000000),
                      {"keep": M.encode(-M.decode(sub, fmt).x, M.F32), "flush_to_+0": 0, "flush_keep_sign": 0x80000000}))
        C.append(("c32_sub", raw(op, [0], [0], 0x00000001), {"keep": 1, "flush": 0}))
        C.append(("c32_sub_big", raw(op, [0], [0], 0x00400000), {"keep": 0x00400000, "flush": 0}))
        if K >= 2:
            C.append(("c32_sub_neg", raw(op, [negz, negz], [0, 0], 0x80400000),
                      {"keep": 0x80400000, "flush_to_+0": 0, "flush_keep_sign": 0x80000000}))
        if f == "bf16":
            a = M.enc_exact(P2(-70), fmt)
            b = M.enc_exact(P2(-60), fmt)
            C.append(("out32_sub_from_normal_inputs", raw(op, [a], [b], 0), {"keep": 0x00080000, "flush": 0}))
    elif kind == "fma32":
        one = M.enc_exact(Fr(1), M.F32)
        C.append(("in32_sub_a", ([1], [M.enc_exact(P2(30), M.F32)], 0), {"keep": M.enc_exact(P2(-119), M.F32), "flush": 0}))
        C.append(("out32_sub", ([M.enc_exact(P2(-100), M.F32)], [M.enc_exact(P2(-30), M.F32)], 0), {"keep": 0x00080000, "flush": 0}))
        C.append(("c32_sub", ([0], [one], 0x00400000), {"keep": 0x00400000, "flush": 0}))
    elif kind == "pkfma":
        one = M.enc_exact(Fr(1), M.F16)
        C.append(("in16_sub", ([1], [M.enc_exact(P2(10), M.F16)], 0), {"keep": 0x0400, "flush": 0}))
        C.append(("out16_sub", ([M.enc_exact(P2(-7), M.F16)], [M.enc_exact(P2(-8), M.F16)], 0), {"keep": 0x0200, "flush": 0}))
        C.append(("c16_sub", ([0], [one], 0x0001), {"keep": 0x0001, "flush": 0}))
    return C


def special_cases(op):
    fam, kind, K, f, _ = OPS[op]
    if kind != "dot":
        return []
    fmt = M.FMTS[f]
    one = M.enc_exact(Fr(1), fmt)
    inf = ((1 << fmt.ebits) - 1) << fmt.mbits
    ninf = inf | 1 << (fmt.width - 1)
    qnan = inf | 1 << (fmt.mbits - 1)
    snan = inf | 1
    C = [("inf_x_1", raw(op, [inf], [one], 0)), ("inf_x_0", raw(op, [inf], [0], 0)),
         ("qnan_in_a", raw(op, [qnan], [one], 0)), ("snan_in_a", raw(op, [snan], [one], 0)),
         ("c_qnan", raw(op, [one], [one], 0x7FC00000)), ("c_snan", raw(op, [one], [one], 0x7F800001)),
         ("c_+inf", raw(op, [one], [one], 0x7F800000)), ("c_-inf_plus_inf_product", raw(op, [inf], [one], 0xFF800000))]
    if K >= 2:
        C.append(("inf_plus_neg_inf_products", raw(op, [inf, ninf], [one, one], 0)))
    return C


def run_named(be, op, mode, cases):
    if not cases:
        return {}
    res = be.run(op, mode, [c[1] for c in cases])
    out = {}
    fmt_out = M.F16 if OPS[op][1] == "pkfma" else M.F32
    for c, r in zip(cases, res):
        d = r["d"]
        ent = {"d": "0x%x" % d if isinstance(d, int) else str(d), "value": fval(d, fmt_out) if isinstance(d, int) else "split"}
        if len(c) > 2:
            hit = [k for k, v in c[2].items() if v == d]
            ent["matches"] = hit or ["none"]
        out[c[0]] = ent
    return out


def subnormal_verdict(sub):
    v = {}
    def m(k):
        return sub.get(k, {}).get("matches", [])
    ins = [m(k) for k in ("in_sub_a", "in_sub_b", "in_sub_big", "in16_sub", "in32_sub_a") if k in sub]
    if ins:
        v["input_subnormal"] = ("preserved" if all("keep" in x for x in ins) else
                                "flushed" if all(any(y.startswith("flush") for y in x) for x in ins) else "mixed: " + str(ins))
    if "in_sub_neg_sign" in sub:
        v["flushed_input_sign"] = m("in_sub_neg_sign")
    cs = [m(k) for k in ("c32_sub", "c32_sub_big", "c16_sub") if k in sub]
    if cs:
        v["accumulator_subnormal"] = ("preserved" if all("keep" in x for x in cs) else
                                      "flushed" if all(any(y.startswith("flush") for y in x) for x in cs) else "mixed: " + str(cs))
    os_ = [m(k) for k in ("out32_sub_from_normal_inputs", "out32_sub", "out16_sub") if k in sub]
    if os_:
        v["output_subnormal"] = "preserved" if all("keep" in x for x in os_) else "flushed" if all("flush" in x for x in os_) else "mixed"
    elif OPS_KIND_DOT_F16(sub):
        v["output_subnormal"] = "not separable for f16 dot (smallest normal f16 product is 2^-28); see accumulator row"
    return v


def OPS_KIND_DOT_F16(sub):
    return "in_sub_a" in sub and "out32_sub_from_normal_inputs" not in sub


def symmetry(be, op, rnd, n=2000):
    fam, kind, K, f, _ = OPS[op]
    if kind != "dot":
        return {}
    fmt = M.FMTS[f]
    tests, neg = [], []
    for _ in range(n):
        t = gen_normalish(op, rnd)
        tests.append(t)
        sb = 1 << (fmt.width - 1)
        neg.append(([x ^ sb for x in t[0]], t[1], t[2] ^ 0x80000000))
    r1, r2 = be.run(op, "desc", tests), be.run(op, "desc", neg)
    bad = sum(1 for a, b in zip(r1, r2) if M.decode(a["d"], M.F32).kind != M.NAN and b["d"] != a["d"] ^ 0x80000000)
    return {"n": n, "violations": bad}


# ---------------------------------------------------------------------------
# Random generators (MMA-Sim Step 4 input families)
# ---------------------------------------------------------------------------
def rand_val(rnd, fmt, lo, hi):
    e = rnd.randint(lo, hi)
    m = rnd.getrandbits(fmt.mbits)
    x = (1 + Fr(m, 1 << fmt.mbits)) * P2(e)
    return M.encode(-x if rnd.random() < 0.5 else x, fmt)


def gen_bits(op, rnd):
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    return ([rnd.getrandbits(fmt.width) for _ in range(K)], [rnd.getrandbits(fmt.width) for _ in range(K)],
            rnd.getrandbits(32))


def gen_normalish(op, rnd):
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    return ([rand_val(rnd, fmt, -3, 2) for _ in range(K)], [rand_val(rnd, fmt, -3, 2) for _ in range(K)],
            rand_val(rnd, M.F32, -4, 4))


def gen_wide(op, rnd):
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    lo, hi = fmt.emin - (2 if rnd.random() < 0.2 else 0), min(fmt.emax, 15)
    al = [rand_val(rnd, fmt, lo, hi) if rnd.random() > 0.05 else rnd.getrandbits(fmt.mbits) for _ in range(K)]
    bl = [rand_val(rnd, fmt, lo, hi) if rnd.random() > 0.05 else rnd.getrandbits(fmt.mbits) for _ in range(K)]
    al = [M.encode(M.decode(x, fmt).x, fmt) if M.decode(x, fmt).kind == M.NUM else 0 for x in al]
    return (al, bl, rand_val(rnd, M.F32, -40, 40))


def gen_cancel(op, rnd):
    """Large condition number: products and c nearly cancel."""
    fam, kind, K, f, _ = OPS[op]
    fmt = M.FMTS[f]
    t = gen_normalish(op, rnd)
    al, bl = list(t[0]), list(t[1])
    s = sum(M.decode(a, fmt).x * M.decode(b, fmt).x for a, b in zip(al, bl))
    c = M.encode(-s, M.F32, "rz")
    c ^= rnd.getrandbits(2)
    return (al, bl, c)


GENS = [("bitstream", gen_bits), ("normal", gen_normalish), ("wide_exponent", gen_wide), ("cancellation", gen_cancel)]


def differential(be, op, rnd, n, mode="desc"):
    """MMA-Sim Step 4: keep only candidates that match every test bit-for-bit (NaN payload ignored)."""
    fam, kind, K, f, clamp = OPS[op]
    tests, src = [], []
    for i in range(n):
        g = GENS[i % len(GENS)]
        tests.append(g[1](op, rnd))
        src.append(g[0])
    res = be.run(op, mode, tests)
    if kind == "dot":
        cands = M.candidates(K, clamp)
    elif kind in ("fma32", "pkfma"):
        cands = fma_candidates(op)
    else:
        return {}
    alive = []
    first_fail = {}
    for m in cands:
        ok = True
        for t, r, s in zip(tests, res, src):
            e = expected(op, m, t)
            d = r["d"]
            fo = M.F16 if kind == "pkfma" else M.F32
            if isinstance(d, tuple):
                ok = False
                break
            if e == d or (M.decode(e, fo).kind == M.NAN and M.decode(d, fo).kind == M.NAN):
                continue
            ok = False
            first_fail[m.name] = (s, t, d, e)
            break
        if ok:
            alive.append(m.name)
    nan_bits = sorted({"0x%x" % r["d"] for r in res if isinstance(r["d"], int) and
                       M.decode(r["d"], M.F16 if kind == "pkfma" else M.F32).kind == M.NAN})
    out = {"n": n, "generators": [g[0] for g in GENS], "matching_models": alive, "nan_encodings_seen": nan_bits}
    if not alive:
        out["note"] = ("no candidate matched: CLFP Step 5 needed. Closest failures listed; reduce with the "
                       "minimum-failing-subset method (paper §III-A5).")
        best = sorted(first_fail.items())[:5]
        out["sample_failures"] = [{"model": k, "gen": v[0], "test": [list(map(hex, v[1][0])), list(map(hex, v[1][1])), hex(v[1][2])],
                                   "hw": hex(v[2]) if isinstance(v[2], int) else str(v[2]), "model_out": hex(v[3])} for k, v in best]
    return out


class FmaCand:
    def __init__(self, fmt, fused, rho, fi, fc, fo):
        self.fmt, self.fused, self.rho, self.fi, self.fc, self.fo = fmt, fused, rho, fi, fc, fo
        self.name = "%s[rho=%s,in=%s,c=%s,out=%s]" % ("fma_fused" if fused else "mul_then_add", rho, fi, fc, fo)

    def __call__(self, t):
        f = self.fmt
        return M.fma_model(M.decode(t[0][0], f), M.decode(t[1][0], f), M.decode(t[2], f), f,
                           self.rho, self.fused, self.fi, self.fc, self.fo)


def fma_candidates(op):
    fmt = M.F16 if OPS[op][1] == "pkfma" else M.F32
    return [FmaCand(fmt, fu, r, fi, fc, fo) for fu in (True, False) for r in ("rne", "rz", "ru", "rd")
            for fi in ("none", "pos", "sign") for fc in ("none", "sign") for fo in ("none", "sign")]


def gen_fma(op, rnd):
    fmt = M.F16 if OPS[op][1] == "pkfma" else M.F32
    r = rnd.random()
    if r < 0.25:
        return ([rnd.getrandbits(fmt.width)], [rnd.getrandbits(fmt.width)], rnd.getrandbits(fmt.width))
    lo = fmt.emin - 2 if r < 0.5 else -4
    hi = 4
    return ([rand_val(rnd, fmt, lo, hi)], [rand_val(rnd, fmt, lo, hi)], rand_val(rnd, fmt, 2 * lo, hi))


def fused_check(be, op):
    kind = OPS[op][1]
    if kind == "fma32":
        a = M.enc_exact(1 + P2(-12), M.F32)
        t = ([a], [a], M.enc_exact(Fr(-1), M.F32))
        fused, unfused = P2(-11) + P2(-24), P2(-11)
        fo = M.F32
    elif kind == "pkfma":
        a = M.enc_exact(1 + P2(-6), M.F16)
        t = ([a], [a], M.enc_exact(Fr(-1), M.F16))
        fused, unfused = P2(-5) + P2(-12), P2(-5)
        fo = M.F16
    else:
        return {"note": "f16 x f16 products are exact in f32 (<= 22 significant bits), so fused vs unfused is not observable"}
    r = be.run(op, "desc", [t])[0]
    v = M.decode(r["d"], fo)
    return {"d": fval(r["d"], fo), "verdict": "fused (single rounding)" if v.x == fused else
            "unfused (product rounded)" if v.x == unfused else "neither"}


# ---------------------------------------------------------------------------
def analyze_op(bes, op, args):
    fam, kind, K, f, clamp = OPS[op]
    rnd = random.Random(args.seed)
    be = bes["default"]
    rep = {"op": op, "family": fam, "kind": kind, "K": K, "input_format": f}
    if kind == "idot":
        rep.update(probe_idot(be, op))
        return rep
    # T0 determinism (+ WMMA cross-position consistency)
    tests = [gen_bits(op, rnd) if kind == "dot" else gen_fma(op, rnd) for _ in range(512)]
    r1, r2 = be.run(op, "desc", tests), be.run(op, "desc", tests)
    rep["T0_deterministic"] = all(a["d"] == b["d"] for a, b in zip(r1, r2))
    rep["T0_all_outputs_identical_across_positions"] = all(a["uniform"] for a in r1)
    rep["T0_mode_readback_desc"] = {v: "0x%02x %s" % (bes[v].run(op, "desc", tests[:1])[0]["mode"],
                                                      mode_txt(bes[v].run(op, "desc", tests[:1])[0]["mode"]))
                                    for v in bes}
    if kind == "dot":
        rep["T1_tree"] = probe_tree(be, op)
        rep["T2_precision"] = probe_precision(be, op)
        rep["T3b_sticky"] = probe_sticky(be, op)
        rep["T7_symmetry"] = symmetry(be, op, rnd)
    else:
        rep["fused"] = fused_check(be, op)
    # Mode sweep for T3, T5, T6 (+ desc mode of every build variant)
    sweep = {}
    runs = [("default", m) for m in FP_MODES] + [(v, "desc") for v in bes if v != "default"]
    for variant, mode in runs:
        b = bes[variant]
        key = mode if variant == "default" else "%s:%s" % (variant, mode)
        sub = run_named(b, op, mode, subnormal_cases(op))
        mr = b.run(op, mode, [subnormal_cases(op)[0][1]])[0]["mode"]
        sweep[key] = {"mode_readback": "0x%02x" % mr, "mode": mode_txt(mr), "T3_rounding": probe_rounding(b, op, mode),
                      "T5_subnormal": sub, "T5_verdict": subnormal_verdict(sub),
                      "T6_specials": run_named(b, op, mode, special_cases(op))}
    rep["mode_sweep"] = sweep
    # T8 randomized differential in desc and in an explicit all-denorm RNE mode
    for mode in ("desc", "mf0", "m00"):
        if kind == "dot":
            rep["T8_differential_" + mode] = differential(be, op, rnd, args.random if K <= 2 else args.random_wmma, mode)
        else:
            rep["T8_differential_" + mode] = differential_fma(be, op, rnd, args.random, mode)
    return rep


def differential_fma(be, op, rnd, n, mode):
    tests = [gen_fma(op, rnd) for _ in range(n)]
    res = be.run(op, mode, tests)
    kind = OPS[op][1]
    fo = M.F16 if kind == "pkfma" else M.F32
    alive = []
    for m in fma_candidates(op):
        if all((not isinstance(r["d"], tuple)) and (m(t) == r["d"] or (M.decode(m(t), fo).kind == M.NAN and M.decode(r["d"], fo).kind == M.NAN))
               for t, r in zip(tests, res)):
            alive.append(m.name)
    return {"n": n, "matching_models": alive}


def summarize(rep):
    """One-line verdicts for the markdown table."""
    lines = []
    if rep["kind"] == "idot":
        return [("arith", rep["verdict"])]
    if "T1_tree" in rep:
        lines.append(("summation tree (desc)", str(rep["T1_tree"].get("tree"))))
        lines.append(("fused precision (desc)", json.dumps(rep["T2_precision"])))
        lines.append(("sticky / extra bits (desc)", "%s / %s of %s" % (
            rep["T3b_sticky"].get("tie_broken_up_by_positive_tiny_down_to_k"),
            rep["T3b_sticky"].get("tie_broken_down_by_negative_tiny_down_to_k"), rep["T3b_sticky"].get("k_max_probed"))))
    if "fused" in rep:
        lines.append(("fused vs unfused", json.dumps(rep["fused"])))
    for mode in ("desc", "mf0", "m00"):
        d = rep.get("T8_differential_" + mode, {})
        ms = d.get("matching_models", [])
        lines.append(("models matching all %s random tests (%s)" % (d.get("n"), mode),
                      ("; ".join(ms[:6]) + (" (+%d more)" % (len(ms) - 6) if len(ms) > 6 else "")) or "NONE"))
    return lines


def write_report(reports, meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump({"meta": meta, "ops": reports}, fh, indent=1, default=str)
    L = ["# mma-probe results: %s" % meta.get("arch"), ""]
    for k, v in meta.items():
        L.append("- **%s:** `%s`" % (k, v))
    L.append("")
    for rep in reports:
        L.append("## %s (%s, K=%s, %s)" % (rep["op"], rep["family"], rep["K"], rep["input_format"]))
        L.append("")
        if "T0_deterministic" in rep:
            L.append("- deterministic: %s; all positions identical: %s" % (rep["T0_deterministic"], rep["T0_all_outputs_identical_across_positions"]))
            for v, s in rep["T0_mode_readback_desc"].items():
                L.append("- MODE at launch, build `%s`: %s" % (v, s))
        if "T7_symmetry" in rep:
            L.append("- odd symmetry violations: %s / %s" % (rep["T7_symmetry"].get("violations"), rep["T7_symmetry"].get("n")))
        L.append("")
        L.append("| property | observed |")
        L.append("|---|---|")
        for a, b in summarize(rep):
            L.append("| %s | %s |" % (a, b.replace("|", "/")))
        if "mode_sweep" in rep:
            L.append("")
            L.append("| MODE | readback | rounding (c=U / p=U) | ties | input sub | acc sub | output sub | flushed sign |")
            L.append("|---|---|---|---|---|---|---|---|")
            for k, s in rep["mode_sweep"].items():
                t3, v = s["T3_rounding"], s["T5_verdict"]
                L.append("| %s | %s | %s / %s | %s | %s | %s | %s | %s |" % (
                    k, s["mode_readback"], t3["c_is_U"]["mode"], t3["product_is_U"]["mode"], t3["c_is_U"]["ties"],
                    v.get("input_subnormal", "-"), v.get("accumulator_subnormal", "-"),
                    str(v.get("output_subnormal", "-"))[:40], ",".join(v.get("flushed_input_sign", ["-"]))))
            sp = rep["mode_sweep"]["desc"]["T6_specials"]
            if sp:
                L.append("")
                L.append("Specials (desc): " + "; ".join("%s -> %s" % (k, v["value"]) for k, v in sp.items()))
        L.append("")
    with open(os.path.join(outdir, "results.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    return os.path.join(outdir, "results.md")


def read_meta(build, arch):
    meta = {"arch": arch}
    for name in ("toolchain.txt", "isa-check.txt"):
        p = os.path.join(build, name)
        if os.path.exists(p):
            meta[name] = open(p).read().strip().replace("\n", " | ")[:2000]
    return meta


def selftest():
    """Feed known models through a fake backend and check the harness recovers them."""
    rnd_n = 300
    ok = True
    class A: pass
    args = A()
    args.seed, args.random, args.random_wmma = 1, rnd_n, 60
    cases = [
        ("dot2_f32_f16", M.Model("exact", M.candidate_bases(2)[0][1]), "exact[rho=rne,in16=none"),
        ("dot2_f32_f16", M.Model("chain_fma", M.candidate_bases(2)[1][1], ftz_in="pos"), "chain_fma[rho=rne,in16=pos"),
        ("wmma_f32_16x16x16_f16", M.Model("tfdpa_L16_F24", dict(M.candidate_bases(16))["tfdpa_L16_F24"], rho="rz"), "tfdpa_L16_F24[rho=rz"),
        ("wmma_f32_16x16x16_f16", M.Model("ftz_muladd_P4", dict(M.candidate_bases(16))["ftz_muladd_P4"], ftz_in="pos", ftz_c="sign", ftz_out="sign"), "ftz_muladd_P4[rho=rne,in16=pos,c32=sign,out=sign"),
    ]
    for op, model, want in cases:
        be = FakeBackend(model)
        rnd = random.Random(3)
        K = OPS[op][2]
        d = differential(be, op, rnd, rnd_n if K == 2 else 60)
        tree = probe_tree(be, op)
        hit = any(m.startswith(want) for m in d["matching_models"])
        print("selftest %-24s %-40s -> tree=%s | matched=%s" % (op, model.name[:40], str(tree.get("tree"))[:60], hit))
        if not hit:
            ok = False
            print("   survivors:", d["matching_models"][:5])
    be = FakeBackend(M.Model("exact", M.candidate_bases(2)[0][1]))
    r = probe_rounding(be, "dot2_f32_f16", "desc")
    print("selftest rounding(exact RNE) ->", r["c_is_U"]["mode"], r["c_is_U"]["ties"])
    ok = ok and r["c_is_U"]["ties"] == "RNE"
    s = probe_sticky(be, "dot2_f32_f16")
    print("selftest sticky(exact) ->", s["tie_broken_up_by_positive_tiny_down_to_k"], s["k_max_probed"])
    be2 = FakeBackend(M.Model("chain_fma", M.candidate_bases(2)[1][1], ftz_in="pos"))
    sub = run_named(be2, "dot2_f32_f16", "desc", subnormal_cases("dot2_f32_f16"))
    print("selftest subnormal(chain, ftz_in=pos) ->", subnormal_verdict(sub))
    ok = ok and subnormal_verdict(sub)["input_subnormal"] == "flushed"
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCH_FAMILIES))
    ap.add_argument("--build", help="build/<arch> directory from build.sh")
    ap.add_argument("--out", help="output directory for results.md / results.json")
    ap.add_argument("--ops", help="comma-separated subset of ops")
    ap.add_argument("--random", type=int, default=4000, help="random tests per scalar op and mode")
    ap.add_argument("--random-wmma", type=int, default=600, help="random tests per WMMA op and mode")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.arch and args.build and args.out):
        ap.error("--arch, --build and --out are required")
    bes = {"default": HwBackend(args.build, "default"), "ftz": HwBackend(args.build, "ftz")}
    info = subprocess.run([os.path.join(args.build, "default", "dot2_f16"), "info"], capture_output=True, text=True)
    meta = read_meta(args.build, args.arch)
    isa = os.path.join(args.build, "isa-check.txt")
    if not os.path.exists(isa) or "ISA CHECK PASS" not in open(isa).read():
        print("WARNING: %s missing or not 'ISA CHECK PASS'; results may not describe the intended "
              "instruction (rerun ./build.sh and read isa-check.txt)" % isa, flush=True)
        meta["isa_check_warning"] = True
    try:
        j = json.loads(info.stdout)
        meta.update({"device": j["device"], "gcnArchName": j["gcnArchName"], "hip_runtime": j["hip_runtime"]})
    except Exception:
        meta["device_info_error"] = info.stderr.strip()[:300]
    fams = ARCH_FAMILIES[args.arch]
    present = set()
    for fam in fams:
        exe = os.path.join(args.build, "default", fam)
        if os.path.exists(exe):
            j = json.loads(subprocess.run([exe, "info"], capture_output=True, text=True, check=True).stdout)
            present |= {e["op"] for e in j["entries"]}
    ops = [o for o in OPS if o in present]
    if args.ops:
        ops = [o for o in ops if o in args.ops.split(",")]
    reports = []
    for op in ops:
        print("probing", op, flush=True)
        reports.append(analyze_op(bes, op, args))
    print("wrote", write_report(reports, meta, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
