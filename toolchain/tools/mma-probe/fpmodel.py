"""Exact reference arithmetic for mma-probe (stdlib only).

Values are decoded to fractions.Fraction so every candidate model is computed
exactly and rounded once per modelled rounding step. Candidate model families
follow MMA-Sim (arXiv 2511.10909 §IV, Table II): chained FMA, exact FDPA
(E-FDPA), FTZ-Mul/Add pairwise (CDNA2) and truncated FDPA (T-FDPA). Names are
ours; parameters (L, F, rho) are the paper's.
"""
from fractions import Fraction
import math
import struct

NAN, INF, NUM = "nan", "inf", "num"


class Fmt:
    def __init__(self, name, ebits, mbits):
        self.name, self.ebits, self.mbits = name, ebits, mbits
        self.bias = (1 << (ebits - 1)) - 1
        self.emin = 1 - self.bias
        self.emax = self.bias
        self.width = 1 + ebits + mbits

    def __repr__(self):
        return self.name


F16 = Fmt("f16", 5, 10)
BF16 = Fmt("bf16", 8, 7)
F32 = Fmt("f32", 8, 23)
FMTS = {"f16": F16, "bf16": BF16, "f32": F32}


class Val:
    """Decoded floating-point value: kind in {nan, inf, num}."""
    __slots__ = ("kind", "neg", "x", "sub", "bits", "fmt")

    def __init__(self, kind, neg, x, sub, bits, fmt):
        self.kind, self.neg, self.x, self.sub, self.bits, self.fmt = kind, neg, x, sub, bits, fmt

    def exp(self):
        """Unbiased exponent of a finite value (emin for subnormals/zero)."""
        if self.x == 0 or self.sub:
            return self.fmt.emin
        return ilog2(abs(self.x))


def ilog2(q):
    """floor(log2(q)) for a positive Fraction."""
    q = Fraction(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    elif Fraction(2) ** (e + 1) <= q:
        e += 1
    return e


def decode(bits, fmt):
    m = bits & ((1 << fmt.mbits) - 1)
    e = (bits >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    neg = bool(bits >> (fmt.width - 1) & 1)
    if e == (1 << fmt.ebits) - 1:
        return Val(NAN if m else INF, neg, None, False, bits, fmt)
    if e == 0:
        x = Fraction(m, 1 << fmt.mbits) * Fraction(2) ** fmt.emin
        sub = m != 0
    else:
        x = (1 + Fraction(m, 1 << fmt.mbits)) * Fraction(2) ** (e - fmt.bias)
        sub = False
    return Val(NUM, neg, -x if neg else x, sub, bits, fmt)


def flush_in(v, how):
    """Input flush: how in {none, pos, sign}."""
    if how == "none" or v.kind != NUM or not v.sub:
        return v
    neg = v.neg if how == "sign" else False
    return Val(NUM, neg, Fraction(0), False, v.bits, v.fmt)


def quiet_nan_bits(fmt):
    return ((1 << (fmt.ebits + 1)) - 1) << (fmt.mbits - 1)


def _round_int(n, rho, neg):
    """Round a non-negative Fraction n to an integer (magnitude), sign neg."""
    fl = n.numerator // n.denominator
    if fl == n:
        return fl
    if rho == "rz":
        return fl
    if rho == "ru":
        return fl if neg else fl + 1
    if rho == "rd":
        return fl + 1 if neg else fl
    # rne
    rem = n - fl
    if rem > Fraction(1, 2):
        return fl + 1
    if rem < Fraction(1, 2):
        return fl
    return fl if fl % 2 == 0 else fl + 1


def encode(x, fmt, rho="rne", ftz_out="none", neg_zero=False):
    """Encode exact Fraction x (or 'nan'/'+inf'/'-inf') to fmt bits.

    neg_zero: sign to use if the exact value is zero.
    ftz_out: none | sign | pos   (flush subnormal results).
    """
    sbit = 1 << (fmt.width - 1)
    if isinstance(x, str):
        if x == "nan":
            return quiet_nan_bits(fmt)
        inf = ((1 << fmt.ebits) - 1) << fmt.mbits
        return inf | (sbit if x == "-inf" else 0)
    x = Fraction(x)
    if x == 0:
        return sbit if neg_zero else 0
    neg = x < 0
    ax = -x if neg else x
    e = max(ilog2(ax), fmt.emin)
    q = Fraction(2) ** (e - fmt.mbits)
    n = _round_int(ax / q, rho, neg)
    v = n * q
    if v == 0:
        return sbit if neg else 0
    if v >= Fraction(2) ** (fmt.emax + 1):
        maxf = (((1 << fmt.ebits) - 2) << fmt.mbits) | ((1 << fmt.mbits) - 1)
        inf = ((1 << fmt.ebits) - 1) << fmt.mbits
        to_inf = rho == "rne" or (rho == "ru" and not neg) or (rho == "rd" and neg)
        return (inf if to_inf else maxf) | (sbit if neg else 0)
    if v < Fraction(2) ** fmt.emin:
        if ftz_out != "none":
            return sbit if (neg and ftz_out == "sign") else 0
        mant = int(v / Fraction(2) ** (fmt.emin - fmt.mbits))
        return mant | (sbit if neg else 0)
    ev = ilog2(v)
    mant = int(v / Fraction(2) ** (ev - fmt.mbits)) - (1 << fmt.mbits)
    return ((ev + fmt.bias) << fmt.mbits) | mant | (sbit if neg else 0)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_float(bits, fmt=F32):
    v = decode(bits, fmt)
    if v.kind == NAN:
        return float("nan")
    if v.kind == INF:
        return float("-inf") if v.neg else float("inf")
    return float(v.x) if v.x != 0 else (-0.0 if v.neg else 0.0)


def enc_exact(x, fmt):
    """Encode a value that must be exactly representable (test-vector builder)."""
    b = encode(Fraction(x), fmt, "rne")
    if decode(b, fmt).x != Fraction(x):
        raise ValueError("%r not exact in %s" % (x, fmt))
    return b


# --------------------------------------------------------------------------
# Products and specials
# --------------------------------------------------------------------------
class Term:
    """An addend: exact value plus the exponent MMA-Sim uses for alignment."""
    __slots__ = ("x", "neg", "e")

    def __init__(self, x, neg, e):
        self.x, self.neg, self.e = x, neg, e


def products(avals, bvals, ftz_in):
    """Return (special, terms). special is None, 'nan', '+inf' or '-inf'."""
    terms, infs = [], set()
    for a, b in zip(avals, bvals):
        a, b = flush_in(a, ftz_in), flush_in(b, ftz_in)
        if a.kind == NAN or b.kind == NAN:
            return "nan", []
        if a.kind == INF or b.kind == INF:
            other = b if a.kind == INF else a
            if other.kind == NUM and other.x == 0:
                return "nan", []
            infs.add(a.neg ^ b.neg)
            continue
        terms.append(Term(a.x * b.x, a.neg ^ b.neg, a.exp() + b.exp()))
    if infs:
        if len(infs) == 2:
            return "nan", []
        return ("-inf" if True in infs else "+inf"), []
    return None, terms


def c_term(c, ftz_c):
    c = flush_in(c, ftz_c)
    if c.kind == NAN:
        return "nan", None
    if c.kind == INF:
        return ("-inf" if c.neg else "+inf"), None
    return None, Term(c.x, c.neg, c.exp())


def merge_special(s1, s2):
    if s1 == "nan" or s2 == "nan":
        return "nan"
    if s1 and s2 and s1 != s2:
        return "nan"
    return s1 or s2


def zero_sign(terms, rho):
    """IEEE sign of an exactly-zero sum."""
    if terms and all(t.x == 0 for t in terms):
        if all(t.neg for t in terms):
            return True
        return rho == "rd" and any(t.neg for t in terms)
    return rho == "rd"


def round_sum(terms, rho, ftz_out, fmt=F32):
    s = sum((t.x for t in terms), Fraction(0))
    return s, encode(s, fmt, rho, ftz_out, neg_zero=zero_sign(terms, rho) if s == 0 else False)


def as_term(bits, fmt=F32):
    v = decode(bits, fmt)
    if v.kind != NUM:
        return None
    return Term(v.x, v.neg, v.exp())


def truncate(t, quantum):
    """Round a term toward zero to a multiple of quantum (RZ_QF)."""
    n = abs(t.x) / quantum
    n = n.numerator // n.denominator
    x = n * quantum
    return Term(-x if t.x < 0 else x, t.neg, t.e)


# --------------------------------------------------------------------------
# Candidate models. Each takes (avals, bvals, cval) and returns f32 bits.
# --------------------------------------------------------------------------
class Model:
    def __init__(self, name, fn, ftz_in="none", ftz_c="none", ftz_out="none", rho="rne", clamp=False):
        self.base, self.fn = name, fn
        self.ftz_in, self.ftz_c, self.ftz_out, self.rho, self.clamp = ftz_in, ftz_c, ftz_out, rho, clamp
        self.name = "%s[rho=%s,in16=%s,c32=%s,out=%s%s]" % (
            name, rho, ftz_in, ftz_c, ftz_out, ",clamp01" if clamp else "")

    def __call__(self, avals, bvals, cval):
        sp, terms = products(avals, bvals, self.ftz_in)
        sc, ct = c_term(cval, self.ftz_c)
        sp = merge_special(sp, sc)
        if sp:
            r = encode(sp, F32)
        else:
            r = self.fn(terms, ct, self)
        return clamp01(r) if self.clamp else r


def clamp01(bits):
    v = decode(bits, F32)
    if v.kind == NAN:
        return 0  # DX10_CLAMP=1 (descriptor default): NaN clamps to 0 (hypothesis)
    if v.kind == INF:
        return 0 if v.neg else f32_bits(1.0)
    if v.x <= 0:
        return 0
    if v.x >= 1:
        return f32_bits(1.0)
    return bits


def _special_of(bits_list):
    """IEEE special result of adding f32 intermediates, or None if all finite."""
    sp = None
    for b in bits_list:
        v = decode(b, F32)
        if v.kind == NAN:
            return "nan"
        if v.kind == INF:
            sp = merge_special(sp, "-inf" if v.neg else "+inf")
    return sp


def _add(parts, rho, ftz_out):
    """Round the exact sum of a mix of Terms and f32 bit patterns (intermediates)."""
    bits = [p for p in parts if isinstance(p, int)]
    sp = _special_of(bits)
    if sp:
        return encode(sp, F32)
    terms = [p if isinstance(p, Term) else as_term(p) for p in parts]
    return round_sum(terms, rho, ftz_out)[1]


def _exact(terms, ct, m):
    return round_sum(terms + [ct], m.rho, m.ftz_out)[1]


def _efdpa(L):
    def fn(terms, ct, m):
        d = _add([ct], m.rho, m.ftz_out)
        for i in range(0, len(terms), L):
            d = _add([d] + terms[i:i + L], m.rho, m.ftz_out)
        return d
    return fn


def _chain_rev(terms, ct, m):
    return _efdpa(1)(list(reversed(terms)), ct, m)


def _pair_then_c(terms, ct, m):
    s = _add(terms, m.rho, m.ftz_out)
    return _add([s, ct], m.rho, m.ftz_out)


def _ftz_muladd(P):
    """CDNA2 Phi_FTZ-MulAdd (MMA-Sim Alg. 2): products and adds round to f32 RNE."""
    def fn(terms, ct, m):
        prods = [_add([t], "rne", m.ftz_out) for t in terms]
        d = _add([ct], m.rho, m.ftz_out)
        for i in range(0, len(prods), P):
            s = _add(prods[i:i + 2], "rne", m.ftz_out)
            if P == 4:
                s2 = _add(prods[i + 2:i + 4], "rne", m.ftz_out)
                s = _add([s, s2], "rne", m.ftz_out)
            d = _add([d, s], m.rho, m.ftz_out)
        return d
    return fn


def _tfdpa(L, F):
    """T-FDPA (MMA-Sim Alg. 7): align at emax, truncate to F fractional bits, round once."""
    def fn(terms, ct, m):
        d = _add([ct], m.rho, m.ftz_out)
        for i in range(0, len(terms), L):
            if _special_of([d]):
                return d
            block = [as_term(d)] + terms[i:i + L]
            nz = [t for t in block if t.x != 0]
            if nz:
                q = Fraction(2) ** (max(t.e for t in nz) - F)
                block = [truncate(t, q) for t in block]
            d = round_sum(block, m.rho, m.ftz_out)[1]
        return d
    return fn


def candidate_bases(K):
    b = [("exact", _exact), ("chain_fma", _efdpa(1)), ("chain_fma_rev", _chain_rev)]
    if K == 2:
        b.append(("pair_then_c", _pair_then_c))
    for L in (2, 4, 8):
        if L < K:
            b.append(("efdpa_L%d" % L, _efdpa(L)))
    if K >= 2:
        b.append(("ftz_muladd_P2", _ftz_muladd(2)))
    if K >= 4:
        b.append(("ftz_muladd_P4", _ftz_muladd(4)))
    for L in sorted({min(K, x) for x in (2, 4, 8, 16)}):
        for F in (23, 24, 25, 26, 28, 30):
            b.append(("tfdpa_L%d_F%d" % (L, F), _tfdpa(L, F)))
    return b


def candidates(K, clamp=False, rhos=("rne", "rz", "ru", "rd")):
    out = []
    for name, fn in candidate_bases(K):
        for rho in rhos:
            for ftz_in in ("none", "pos", "sign"):
                for ftz_c in ("none", "sign"):
                    for ftz_out in ("none", "sign"):
                        out.append(Model(name, fn, ftz_in, ftz_c, ftz_out, rho, clamp))
    return out


# --------------------------------------------------------------------------
# Scalar FMA-like ops
# --------------------------------------------------------------------------
def fma_model(a, b, c, fmt_out, rho="rne", fused=True, ftz_in="none", ftz_c="none", ftz_out="none"):
    """d = a*b + c, fused (one rounding) or unfused (product rounded to fmt_out first)."""
    sp, terms = products([a], [b], ftz_in)
    sc, ct = c_term(c, ftz_c)
    if not sp and not fused:
        pb = encode(terms[0].x, fmt_out, rho, ftz_out, neg_zero=terms[0].neg)
        pv = decode(pb, fmt_out)
        if pv.kind == INF:
            sp = "-inf" if pv.neg else "+inf"
        else:
            terms = [Term(pv.x, pv.neg, pv.exp())]
    sp = merge_special(sp, sc)
    if sp:
        return encode(sp, fmt_out)
    return round_sum(terms + [ct], rho, ftz_out, fmt_out)[1]
