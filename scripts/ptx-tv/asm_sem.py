"""Semantics for inline PTX assembly in the source.

`call T asm [sideeffect] "template", "constraints"(args)` is replaced by an
always-inline function whose body is the template, with its operands bound
to parameters, lifted by ptx_lift.  llc copies the template into its output
with the operands replaced by registers, and the lifter lifts it there as
ordinary PTX, so both sides get the same instruction semantics; the check
covers how the backend binds operands and uses results.  Comments and empty
templates are no-ops on both sides.

Not handled (left as inline asm, reported `src-inline-asm`): memory
operands, outputs of class `b`, operand modifiers other than `c`/`n`, and
templates the lifter cannot lift.
"""
import re

from llvmsig import parse_type, split_top, TypeParseError

CLASS_BITS = {"h": 16, "r": 32, "l": 64, "f": 32, "d": 64, "q": 128, "b": 1, "c": 8}
FP_BITS = {"half": 16, "bfloat": 16, "float": 32, "double": 64}

CALL_RE = re.compile(
    r'^(\s*(?:(%[\w.$-]+)\s*=\s*)?(?:tail\s+|notail\s+|musttail\s+)?call\s+)(.*?)\s+asm\s+'
    r'((?:sideeffect\s+|alignstack\s+|inteldialect\s+|unwind\s+)*)'
    r'"((?:[^"\\]|\\.)*)",\s*"((?:[^"\\]|\\.)*)"\((.*)$')


class Skip(Exception):
    pass


def unescape(s):
    return re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s)


def type_bits(t):
    t = t.strip()
    if re.fullmatch(r"i\d+", t):
        return int(t[1:])
    if t in FP_BITS:
        return FP_BITS[t]
    if t.startswith("ptr"):
        return 64
    m = re.fullmatch(r"<(\d+) x (.+)>", t)
    if m:
        b = type_bits(m.group(2))
        return int(m.group(1)) * b if b else None
    return None


def to_int(t, v, w, op):
    """LLVM value v of type t as an iw value."""
    if t.startswith("ptr"):
        return op("ptrtoint %s %s to i%d" % (t, v, w))
    b = type_bits(t)
    if b is None:
        raise Skip("type " + t)
    if not re.fullmatch(r"i\d+", t):
        v = op("bitcast %s %s to i%d" % (t, v, b))
    if b == w:
        return v
    return op("%s i%d %s to i%d" % ("trunc" if b > w else "zext", b, v, w))


def from_int(t, v, w, op):
    if t.startswith("ptr"):
        return op("inttoptr i%d %s to %s" % (w, v, t))
    b = type_bits(t)
    if b is None:
        raise Skip("type " + t)
    if b != w:
        v = op("%s i%d %s to i%d" % ("trunc" if w > b else "zext", w, v, b))
    if not re.fullmatch(r"i\d+", t):
        v = op("bitcast i%d %s to %s" % (b, v, t))
    return v


def parse_args(text):
    """[(type, value)] of a call's argument list (attributes dropped)."""
    out = []
    for a in split_top(text):
        a = a.strip()
        if not a:
            continue
        try:
            t, e = parse_type(a)
        except TypeParseError:
            raise Skip("arg " + a)
        rest = a[e:].strip()
        rest = re.sub(r"^((noundef|nonnull|signext|zeroext|inreg|elementtype\([^)]*\)|"
                      r"align \d+|dereferenceable\(\d+\))\s+)*", "", rest)
        out.append((t, rest))
    return out


def build(fname, rett, tmpl, cons, args, lifter_mod, parse_sig):
    cs = [c.strip() for c in split_top(cons) if c.strip()]
    cs = [c for c in cs if not c.startswith("~")]
    outs, ins = [], []
    for c in cs:
        if c.startswith("="):
            c = c[1:].lstrip("&")
            if c not in CLASS_BITS or c == "b":
                raise Skip("output constraint " + c)
            outs.append(c)
        else:
            ins.append(c)
    if len(ins) != len(args):
        raise Skip("operand count")
    if rett == "void":
        rtys = []
    elif outs and len(outs) == 1:
        rtys = [rett]
    elif rett.startswith("{"):
        rtys = [t.strip() for t in split_top(rett.strip()[1:-1])]
    else:
        raise Skip("return " + rett)
    if len(rtys) != len(outs):
        raise Skip("outputs")
    # operand k -> PTX text; registers per operand
    opnd, regs, params, loads = {}, [], [], []
    for k, c in enumerate(outs):
        w = CLASS_BITS[c]
        r = "%%ptxtv_o%d" % k
        regs.append((r, w))
        opnd[k] = r
    lname = fname + ".lifted"
    pidx = 0
    wargs = []          # (llvm value of wrapper param, width) passed to the lifted function
    for j, (c, (t, v)) in enumerate(zip(ins, args)):
        k = len(outs) + j
        if c in ("i", "n"):
            if not re.fullmatch(r"-?\d+", v):
                raise Skip("immediate " + v)
            opnd[k] = v
            continue
        if c.isdigit():
            tied = int(c)
            if tied >= len(outs):
                raise Skip("tie")
            w = CLASS_BITS[outs[tied]]
            r = opnd[tied]
        elif c in CLASS_BITS:
            w = CLASS_BITS[c]
            r = "%%ptxtv_i%d" % k
            regs.append((r, w))
            opnd[k] = r
        else:
            raise Skip("input constraint " + c)
        pw = 32 if w == 1 else max(w, 16) if w != 128 else 128
        pname = "%s_param_%d" % (lname, pidx)
        params.append(".param .b%d %s" % (pw, pname) if pw != 128 else
                      ".param .align 16 .b8 %s[16]" % pname)
        if w == 1:
            loads.append("ld.param.b32 %%ptxtv_t%d, [%s];" % (pidx, pname))
            loads.append("setp.ne.b32 %s, %%ptxtv_t%d, 0;" % (r, pidx))
            regs.append(("%%ptxtv_t%d" % pidx, 32))
        elif w == 8:
            loads.append("ld.param.b16 %%ptxtv_t%d, [%s];" % (pidx, pname))
            loads.append("cvt.u8.u16 %s, %%ptxtv_t%d;" % (r, pidx))
            regs.append(("%%ptxtv_t%d" % pidx, 16))
        else:
            loads.append("ld.param.b%d %s, [%s];" % (w, r, pname))
        wargs.append((j, t, v, w, pw))
        pidx += 1

    def sub(m):
        k = int(m.group(1) or m.group(2))
        mod = m.group(3)
        if k not in opnd:
            raise Skip("operand $%d" % k)
        s = opnd[k]
        if mod in (None, "c"):
            return s
        if mod == "n" and re.fullmatch(r"-?\d+", s):
            return str(-int(s))
        raise Skip("modifier " + mod)
    body = unescape(tmpl).replace("$$", "\0")
    body = re.sub(r"\$(?:(\d+)|\{(\d+)(?::(\w+))?\})", sub, body)
    if "$" in body:
        raise Skip("template")
    body = body.replace("\0", "$")
    # outputs: struct layout with natural alignment
    off, stores, rlayout = 0, [], []
    for k, c in enumerate(outs):
        w = CLASS_BITS[c]
        al = max(w // 8, 1)
        off = (off + al - 1) // al * al
        stores.append("st.param.b%d [func_retval0+%d], %s;" % (w, off, opnd[k]))
        rlayout.append(w)
        off += w // 8
    if outs:
        size = (off + 15) // 16 * 16 if len(outs) > 1 or rlayout[0] == 128 else off
        if len(outs) == 1 and rlayout[0] != 128:
            rdecl = "(.param .b%d func_retval0) " % rlayout[0]
        else:
            rdecl = "(.param .align 16 .b8 func_retval0[%d]) " % size
    else:
        rdecl = ""
    decl = []
    for r, w in regs:
        decl.append("\t.reg .%s %s;" % ("pred" if w == 1 else "b%d" % w, r))
    ptx = ".visible .func %s%s(\n%s\n)\n{\n%s\n%s\n%s\n%s\nret;\n}\n" % (
        rdecl, lname, ",\n".join("\t" + p for p in params), "\n".join(decl),
        "\n".join(loads), body, "\n".join(stores))
    inner_args = ", ".join("i%d %%x%d" % (pw, n) for n, (_, _, _, _, pw) in enumerate(wargs))
    if not outs:
        iret = "void"
    elif len(outs) == 1:
        iret = "i%d" % rlayout[0]
    else:
        iret = "{ %s }" % ", ".join("i%d" % w for w in rlayout)
    sig = parse_sig("define internal %s @%s(%s)" % (iret, lname, inner_args))
    try:
        L = lifter_mod.lift_function(ptx, sig, {}, {}, {}, lname)
    except lifter_mod.Unsupported as e:
        raise Skip("lift: %s" % e)
    inner = lifter_mod.render(L, "define internal %s @%s(%s) alwaysinline" % (iret, lname, inner_args))
    # wrapper with the source's types
    lines = []
    cnt = [0]

    def op(text):
        cnt[0] += 1
        t = "%%w%d" % cnt[0]
        lines.append("  %s = %s" % (t, text))
        return t
    wparams = ", ".join("%s %%a%d" % (t, j) for j, (t, _) in enumerate(args))
    cargs = []
    for (j, t, v, w, pw) in wargs:
        x = to_int(t, "%%a%d" % j, w if w > 1 else 1, op) if w != 1 else None
        if w == 1:
            if t != "i1":
                raise Skip("pred type " + t)
            x = op("zext i1 %%a%d to i32" % j)
        elif pw != w:
            x = op("zext i%d %s to i%d" % (w, x, pw))
        cargs.append("i%d %s" % (pw, x))
    call = "call %s @%s(%s)" % (iret, lname, ", ".join(cargs))
    if not outs:
        lines.append("  " + call)
        lines.append("  ret void")
    else:
        r = op(call)
        if len(outs) == 1:
            lines.append("  ret %s %s" % (rett, from_int(rtys[0], r, rlayout[0], op)))
        else:
            agg = "poison"
            for k, (t, w) in enumerate(zip(rtys, rlayout)):
                e = op("extractvalue %s %s, %d" % (iret, r, k))
                agg = op("insertvalue %s %s, %s %s, %d" % (rett, agg, t, from_int(t, e, w, op), k))
            lines.append("  ret %s %s" % (rett, agg))
    wrapper = "define internal %s @%s(%s) alwaysinline {\n%s\n}" % (rett, fname, wparams, "\n".join(lines))
    return wrapper + "\n" + inner, L.extra_decls


def expand(text, lifter_mod):
    from llvmsig import parse_sig
    out, defs, extra = [], [], {}
    changed = False
    for line in text.split("\n"):
        m = CALL_RE.match(line) if " asm " in line else None
        if not m:
            out.append(line)
            continue
        head, _res, rett, _flags, tmpl, cons, rest = m.groups()
        # split the argument list from what follows it
        depth, k = 1, 0
        while k < len(rest) and depth:
            depth += {"(": 1, ")": -1}.get(rest[k], 0)
            k += 1
        argtext, tail = rest[:k - 1], rest[k:]
        try:
            args = parse_args(argtext)
            fname = "ptxtv_asm_%d" % len(defs)
            body, decls = build(fname, rett.strip(), tmpl, cons, args, lifter_mod, parse_sig)
        except Skip:
            out.append(line)
            continue
        defs.append(body)
        extra.update(decls)
        tail = re.sub(r"\s*#\d+", "", tail)
        out.append("%s%s @%s(%s)%s" % (head, rett.strip(), fname,
                                        ", ".join("%s %s" % a for a in args), tail))
        changed = True
    if not changed:
        return text, False
    have = set(re.findall(r"^declare [^@]*@([\w.$\"-]+)\(", text, re.M))
    out += defs + [d for n, d in extra.items() if n not in have]
    return "\n".join(out), True
