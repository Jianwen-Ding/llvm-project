"""Exact semantics of llvm.convert.from.arbitrary.fp (LangRef): the source
call is replaced by a lookup in a constant table of the result's bit
patterns, computed here with exact rational arithmetic.  Widening is exact;
infinities are preserved; a NaN becomes a NaN of the same sign whose
mantissa is the source mantissa placed at the top (so the quiet bit is
preserved); the LangRef allows other payloads, so differences in NaN bits
alone are reported as `nan-payload`.
"""
from fractions import Fraction
import re

# name: (exponent bits, mantissa bits, bias, specials)
#   ieee: all-ones exponent is Inf (m == 0) / NaN;  fn_e4m3: only S.1111.111 is NaN;
#   finite: no Inf/NaN
FORMATS = {
    "Float8E5M2": (5, 2, 15, "ieee"),
    "Float8E4M3": (4, 3, 7, "ieee"),
    "Float8E3M4": (3, 4, 3, "ieee"),
    "Float8E4M3FN": (4, 3, 7, "fn_e4m3"),
    "Float6E3M2FN": (3, 2, 3, "finite"),
    "Float6E2M3FN": (2, 3, 1, "finite"),
    "Float4E2M1FN": (2, 1, 1, "finite"),
}
TARGETS = {"half": (5, 10), "bfloat": (8, 7), "float": (8, 23), "double": (11, 52)}


def decode(fmt, code):
    E, M, B, sp = FORMATS[fmt]
    s = code >> (E + M) & 1
    e = code >> M & ((1 << E) - 1)
    m = code & ((1 << M) - 1)
    emax = (1 << E) - 1
    if sp == "ieee" and e == emax:
        return s, ("inf", None) if m == 0 else ("nan", (m, M))
    if sp == "fn_e4m3" and e == emax and m == (1 << M) - 1:
        return s, ("nan", (m, M))
    if e == 0:
        v = Fraction(m, 1 << M) * Fraction(2) ** (1 - B)
    else:
        v = (1 + Fraction(m, 1 << M)) * Fraction(2) ** (e - B)
    return s, ("num", v)


def encode(ty, s, kind):
    TE, TM = TARGETS[ty]
    TB = (1 << (TE - 1)) - 1
    W = 1 + TE + TM
    sign = s << (W - 1)
    k, v = kind
    if k == "inf":
        return sign | (((1 << TE) - 1) << TM)
    if k == "nan":
        m, M = v
        return sign | (((1 << TE) - 1) << TM) | (m << (TM - M))
    if v == 0:
        return sign
    e = v.numerator.bit_length() - v.denominator.bit_length()
    while Fraction(2) ** e > v:
        e -= 1
    while Fraction(2) ** (e + 1) <= v:
        e += 1
    if e + TB >= 1:
        frac = v / Fraction(2) ** e - 1
        mant = frac * (1 << TM)
        assert mant.denominator == 1 and e + TB < (1 << TE) - 1
        return sign | ((e + TB) << TM) | int(mant)
    mant = v / Fraction(2) ** (1 - TB) * (1 << TM)
    assert mant.denominator == 1
    return sign | int(mant)


def table(fmt, ty):
    E, M, _, _ = FORMATS[fmt]
    return [encode(ty, *decode(fmt, c)) for c in range(1 << (1 + E + M))]


CALL_RE = re.compile(r'^(\s*%[\w.$-]+\s*=\s*(?:tail\s+)?call\s+)(?:[\w ]+\s+)?(<[^>]*>|\S+)\s+'
                     r'@llvm\.convert\.from\.arbitrary\.fp\.[\w.]+\(\s*(<[^>]*>|\S+)\s+([^,]+),\s*'
                     r'metadata\s+!"(\w+)"\s*\)(.*)$')


def expand(text):
    out, defs = [], {}
    for line in text.split("\n"):
        m = CALL_RE.match(line) if "convert.from.arbitrary.fp" in line and "call" in line else None
        if not m:
            out.append(line)
            continue
        head, rt, it, val, fmt, tail = m.groups()
        vm = re.fullmatch(r"<(\d+) x (\w+)>", rt)
        n, ety = (int(vm.group(1)), vm.group(2)) if vm else (1, rt)
        ivm = re.fullmatch(r"<(\d+) x i(\d+)>", it) if vm else re.fullmatch(r"()i(\d+)", it)
        if fmt not in FORMATS or ety not in TARGETS or not ivm:
            out.append(line)
            continue
        N = int(ivm.group(2))
        E, M, _, _ = FORMATS[fmt]
        if N != 1 + E + M:
            out.append(line)
            continue
        key = (fmt, rt, it)
        if key not in defs:
            fname = "ptxtv_arbfp_%d" % len(defs)
            TE, TM = TARGETS[ety]
            W = 1 + TE + TM
            tab = table(fmt, ety)
            tv = "<%d x i%d> <%s>" % (len(tab), W, ", ".join("i%d %d" % (W, x) for x in tab))
            b = ["define internal %s @%s(%s %%x) alwaysinline {" % (rt, fname, it)]

            def one(src, k):
                b.append("  %%z%d = zext i%d %s to i32" % (k, N, src))
                b.append("  %%t%d = extractelement %s, i32 %%z%d" % (k, tv, k))
                b.append("  %%f%d = bitcast i%d %%t%d to %s" % (k, W, k, ety))
                return "%%f%d" % k
            if not vm:
                b.append("  ret %s %s" % (rt, one("%x", 0)))
            else:
                acc = "poison"
                for k in range(n):
                    b.append("  %%e%d = extractelement %s %%x, i32 %d" % (k, it, k))
                    f = one("%%e%d" % k, k)
                    b.append("  %%v%d = insertelement %s %s, %s %s, i32 %d" % (k, rt, acc, ety, f, k))
                    acc = "%%v%d" % k
                b.append("  ret %s %s" % (rt, acc))
            b.append("}")
            defs[key] = (fname, "\n".join(b))
        out.append("%s%s @%s(%s %s)%s" % (head, rt, defs[key][0], it, val.strip(), tail))
    if not defs:
        return text, False
    out += [d for _, d in defs.values()]
    return "\n".join(out), True


if __name__ == "__main__":
    # spot checks against known values
    assert encode("float", *decode("Float8E5M2", 0x3c)) == 0x3f800000      # 1.0
    assert encode("float", *decode("Float8E4M3FN", 0x7e)) == 0x43e00000    # 448
    assert encode("half", *decode("Float8E5M2", 0x01)) == 0x0100            # 2^-16 (f16 subnormal)
    assert encode("float", *decode("Float4E2M1FN", 0x7)) == 0x40c00000     # 6.0
    assert encode("float", *decode("Float6E2M3FN", 0x01)) == 0x3e000000    # 0.125
    assert encode("float", *decode("Float8E5M2", 0x7c)) == 0x7f800000      # inf
    print("ok")


# ---- llvm.convert.to.arbitrary.fp -------------------------------------------
import struct

ROUNDING = ("round.tonearest", "round.towardzero")
SRC_TYPES = ("half", "bfloat", "float", "double")


def dhex(v):
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def positive_codes(fmt):
    """[(code, value)] of finite non-negative values in increasing order,
    and the encoding of +Inf (or None)."""
    E, M, _, sp = FORMATS[fmt]
    out, inf = [], None
    for c in range(1 << (E + M)):
        _, (k, v) = decode(fmt, c)
        if k == "num":
            out.append((c, v))
        elif k == "inf":
            inf = c
    out.sort(key=lambda cv: cv[1])
    return out, inf


def ceil_bits(ty, v):
    """(bits, exact): the smallest non-negative value of type ty that is >= v
    (a Fraction), as its bit pattern."""
    TE, TM = TARGETS[ty]
    TB = (1 << (TE - 1)) - 1
    if v == 0:
        return 0, True
    e = v.numerator.bit_length() - v.denominator.bit_length()
    while Fraction(2) ** e > v:
        e -= 1
    while Fraction(2) ** (e + 1) <= v:
        e += 1
    if e + TB >= 1:
        q = (v / Fraction(2) ** e - 1) * (1 << TM)
        mant = -((-q.numerator) // q.denominator)
        bits = ((e + TB) << TM) + mant          # a carry into the exponent is fine
        return bits, q.denominator == 1
    q = v / Fraction(2) ** (1 - TB) * (1 << TM)
    mant = -((-q.numerator) // q.denominator)
    return mant, q.denominator == 1


def to_ir(fmt, rnd, sat, sty, x, op, N):
    """IR computing the iN encoding of x (type sty); returns the value.
    Works on the bit pattern of x: for non-NaN values, the order of
    magnitudes is the integer order of the magnitude bits."""
    E, M, _, sp = FORMATS[fmt]
    vals, inf = positive_codes(fmt)
    assert [c for c, _ in vals] == list(range(len(vals)))   # codes increase with value
    maxc, maxv = vals[-1]
    nextv = maxv + (vals[-1][1] - vals[-2][1])   # one step above the maximum
    TE, TM = TARGETS[sty]
    W = 1 + TE + TM
    infb = ((1 << TE) - 1) << TM
    xb = op("bitcast %s %s to i%d" % (sty, x, W))
    ab = op("and i%d %s, %d" % (W, xb, (1 << (W - 1)) - 1))
    sgn = op("lshr i%d %s, %d" % (W, xb, W - 1))
    sgn = op("trunc i%d %s to i%d" % (W, sgn, N)) if W > N else \
        (op("zext i%d %s to i%d" % (W, sgn, N)) if W < N else sgn)

    def ge(v, strict):
        """|x| > v (strict) or |x| >= v, for non-NaN x."""
        bits, exact = ceil_bits(sty, v)
        if bits >= infb:
            return op("icmp uge i%d %s, %d" % (W, ab, infb)) if bits == infb and not \
                (strict and exact) else "false"
        return op("icmp %s i%d %s, %d" % ("ugt" if strict and exact else "uge", W, ab, bits))
    mag = None
    seq = [v for _, v in vals] + [nextv]
    for i in range(1, len(seq)):
        if rnd == "round.towardzero":
            c = ge(seq[i], False)
        else:
            c = ge((seq[i - 1] + seq[i]) / 2, i % 2 == 1)   # ties to the even code
        if c == "false":
            continue
        z = op("zext i1 %s to i%d" % (c, N))
        mag = z if mag is None else op("add i%d %s, %s" % (N, mag, z))
    if rnd == "round.towardzero":
        ovf = op("icmp eq i%d %s, %d" % (W, ab, infb))     # only an infinite input
        mag = op("select i1 %s, i%d %d, i%d %s" % (ovf, N, maxc + 1, N, mag))
    else:
        ovf = op("icmp ugt i%d %s, %d" % (N, mag, maxc))
    if sat:
        omag = str(maxc)
    elif inf is not None:
        omag = str(inf)
    else:
        # LangRef: poison; `freeze poison` (any value) so that the source's
        # optimiser cannot fold the select away
        omag = op("freeze i%d poison" % N)
    mag = op("select i1 %s, i%d %s, i%d %s" % (ovf, N, omag, N, mag))
    code = op("or i%d %s, %s" % (N, op("shl i%d %s, %d" % (N, sgn, N - 1)), mag))
    isnan = op("icmp ugt i%d %s, %d" % (W, ab, infb))
    if sp == "ieee":
        # any NaN encoding: all-ones exponent, non-zero mantissa, any sign
        f = op("freeze i%d poison" % N)
        mm = op("and i%d %s, %d" % (N, f, (1 << M) - 1))
        mz = op("icmp eq i%d %s, 0" % (N, mm))
        mm = op("select i1 %s, i%d 1, i%d %s" % (mz, N, N, mm))
        sb = op("and i%d %s, %d" % (N, f, 1 << (N - 1)))
        nan = op("or i%d %s, %d" % (N, op("or i%d %s, %s" % (N, sb, mm)), ((1 << E) - 1) << M))
    elif sp == "fn_e4m3":
        f = op("freeze i%d poison" % N)
        sb = op("and i%d %s, %d" % (N, f, 1 << (N - 1)))
        nan = op("or i%d %s, %d" % (N, sb, (1 << (N - 1)) - 1))
    else:
        nan = op("freeze i%d poison" % N)
    return op("select i1 %s, i%d %s, i%d %s" % (isnan, N, nan, N, code))


TO_RE = re.compile(r'^(\s*%[\w.$-]+\s*=\s*(?:tail\s+)?call\s+)(?:[\w ]+\s+)?(<[^>]*>|\S+)\s+'
                   r'@llvm\.convert\.to\.arbitrary\.fp\.[\w.]+\(\s*(<[^>]*>|\S+)\s+([^,]+),\s*'
                   r'metadata\s+!"(\w+)",\s*metadata\s+!"([\w.]+)",\s*i1\s+(true|false|0|1)\s*\)(.*)$')


def expand_to(text):
    out, defs = [], {}
    for line in text.split("\n"):
        m = TO_RE.match(line) if "convert.to.arbitrary.fp" in line and "call" in line else None
        if not m:
            out.append(line)
            continue
        head, rt, st, val, fmt, rnd, sat, tail = m.groups()
        sat = sat in ("true", "1")
        vm = re.fullmatch(r"<(\d+) x i(\d+)>", rt)
        n, N = (int(vm.group(1)), int(vm.group(2))) if vm else (1, int(rt[1:]) if re.fullmatch(r"i\d+", rt) else 0)
        svm = re.fullmatch(r"<(\d+) x (\w+)>", st)
        ety = svm.group(2) if svm else st
        if fmt not in FORMATS or rnd not in ROUNDING or ety not in SRC_TYPES or \
                N != 1 + FORMATS[fmt][0] + FORMATS[fmt][1] or bool(vm) != bool(svm):
            out.append(line)
            continue
        key = (fmt, rnd, sat, rt, st)
        if key not in defs:
            fname = "ptxtv_arbfp_to_%d" % len(defs)
            b = ["define internal %s @%s(%s %%x) alwaysinline {" % (rt, fname, st)]
            cnt = [0]

            def op(text):
                cnt[0] += 1
                t = "%%q%d" % cnt[0]
                b.append("  %s = %s" % (t, text))
                return t
            if not vm:
                r = to_ir(fmt, rnd, sat, ety, "%x", op, N)
            else:
                r = "poison"
                for k in range(n):
                    e = op("extractelement %s %%x, i32 %d" % (st, k))
                    c = to_ir(fmt, rnd, sat, ety, e, op, N)
                    r = op("insertelement %s %s, i%d %s, i32 %d" % (rt, r, N, c, k))
            b.append("  ret %s %s\n}" % (rt, r))
            defs[key] = (fname, "\n".join(b))
        out.append("%s%s @%s(%s %s)%s" % (head, rt, defs[key][0], st, val.strip(), tail))
    if not defs:
        return text, False
    out += [d for _, d in defs.values()]
    return "\n".join(out), True
