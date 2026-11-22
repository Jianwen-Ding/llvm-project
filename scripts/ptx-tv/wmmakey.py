"""Warp-level matrix instructions (wmma.load/store/mma, mma.sync including
the sparse and block-scale forms, ldmatrix, stmatrix) as uninterpreted calls named
by a token set derived independently from the intrinsic name (the naming
scheme of IntrinsicsNVVM.td: WMMA_NAME_LDST, WMMA_NAME, MMA_NAME,
LDMATRIX_NAME, STMATRIX_NAME; NVPTXUsage does not describe these) and from
the PTX opcode.  Types are compared as the full (d, a, b, c) tuple: where
the intrinsic's signature names fewer types, the others follow the PTX
ISA's type rules (f16 forms are named by d and c with f16 inputs; integer
inputs accumulate in s32; bf16 and tf32 in f32; f64 in f64).  Fragments
are passed register by register as zero-extended i64 (packed f16x2/bf16x2
as one 32-bit register, element 0 in the low half) and results come back
as i64 each; addresses as i64 (space from the pointer's address space:
p1 = .global, p3 = .shared)."""
import re

INT = {"s8", "u8", "s4", "u4", "b1", "s32"}
SPACE = {"1": "global", "3": "shared"}


def name_of(tokens, nres, nops):
    return "ptx.wmma." + ".".join(sorted(tokens)).replace("::", "__") + ".r%d.n%d" % (nres, nops)


def _tuple(sig, ptx_f16_pair=False):
    """(d, a, b, c) from an intrinsic signature (MMA_SIGNATURE)."""
    t = sig.split(".")
    if len(t) == 4:
        return tuple(t)
    if len(t) == 2 and set(t) <= {"f16", "f32"}:
        return (t[0], "f16", "f16", t[1])
    a = t[0]
    b = t[1] if len(t) == 2 else a
    if a in INT:
        acc = "s32"
    elif a in ("bf16", "tf32"):
        acc = "f32"
    elif a == "f64":
        acc = "f64"
    else:
        return None
    return (acc, a, b, acc)


def _types(tup):
    return {"d_" + tup[0], "a_" + tup[1], "b_" + tup[2], "c_" + tup[3]}


def _space(p):
    if p is None or p == "0":
        return set()
    if p not in SPACE:
        return None
    return {"sp_" + SPACE[p]}


def intrinsic(name):
    """(tokens, kind) or None; kind is "load", "store" or "mma"."""
    m = re.fullmatch(r"wmma\.(m\d+n\d+k\d+)\.(load|store)\.([abcd])\.(row|col)(\.stride)?\.(\w+?)"
                     r"(?:\.p(\d+)(?:i8)?)?", name)
    if m:
        sp = _space(m.group(7))
        if sp is None:
            return None
        t = {"wmma", m.group(2), "frag_" + m.group(3), m.group(4), m.group(1), "t_" + m.group(6)} | sp
        if m.group(5):
            t.add("stride")
        return t, m.group(2)
    m = re.fullmatch(r"(ld|st)matrix\.sync\.aligned\.(m\d+n\d+)\.(x[124])(\.trans)?\.([\w.]+?)"
                     r"(?:\.p(\d+)(?:i8)?)?", name)
    if m:
        sp = _space(m.group(6))
        if sp is None:
            return None
        t = {m.group(1) + "matrix", m.group(2), m.group(3), "t_" + m.group(5).replace(".", "_")} | sp
        if m.group(4):
            t.add("trans")
        return t, "load" if m.group(1) == "ld" else "store"
    m = re.fullmatch(r"wmma\.(m\d+n\d+k\d+)\.mma(?:\.(xor|and)\.popc)?\.(row|col)\.(row|col)"
                     r"(?:\.(rn|rz|rm|rp))?\.([\w.]+?)(\.satfinite)?", name)
    if m:
        tup = _tuple(m.group(6))
        if tup is None:
            return None
        t = {"wmma", "mma", m.group(1), "a" + m.group(3), "b" + m.group(4)} | _types(tup)
        for g, tok in ((2, None), (5, None), (7, "satfinite")):
            if m.group(g):
                t.add(tok or (m.group(g) + "_popc" if g == 2 else m.group(g)))
        return t, "mma"
    m = re.fullmatch(r"mma(?:\.(xor|and)\.popc)?(\.sp(?:\.ordered\.metadata)?)?(\.block\.scale)?"
                     r"\.(m\d+n\d+k\d+)\.(row|col)\.(row|col)"
                     r"(?:\.kind\.(f8f6f4))?(?:\.(mxf8f6f4|mxf4nvf4|mxf4))?(?:\.scale\.([124])x)?"
                     r"(\.satfinite)?\.([\w.]+)", name)
    if m:
        sig = m.group(11)
        bs = bool(m.group(3))
        if bs != bool(m.group(8)):
            return None
        stype = None
        if bs:
            sig, _, stype = sig.rpartition(".")
            if stype not in ("ue8m0", "ue4m3"):
                return None
        tup = _tuple(sig)
        if tup is None:
            return None
        t = {"mma", m.group(4), "a" + m.group(5), "b" + m.group(6)} | _types(tup)
        if m.group(1):
            t.add(m.group(1) + "_popc")
        if m.group(2):
            t.add("sp_ordered" if "ordered" in m.group(2) else "sp")
        if m.group(7):
            t.add("kind_" + m.group(7))
        if bs:
            t |= {"block_scale", "kind_" + m.group(8), "s_" + stype}
            if m.group(9):
                t.add("sv%sx" % m.group(9))
        if m.group(10):
            t.add("satfinite")
        return t, "mma"
    return None


def ptx(base, mods):
    """(tokens, kind) or None for a PTX opcode."""
    ms = list(mods)
    if base == "wmma" and ms[:1] in (["load"], ["store"]):
        if len(ms) < 6 or ms[2:4] != ["sync", "aligned"] or ms[1] not in "abcd" or len(ms[1]) != 1:
            return None
        layout, geom, rest = ms[4], ms[5], ms[6:]
        t = {"wmma", ms[0], "frag_" + ms[1], layout, geom}
        if rest and rest[0] in ("global", "shared"):
            t.add("sp_" + rest[0])
            rest = rest[1:]
        if len(rest) != 1 or layout not in ("row", "col"):
            return None
        t.add("t_" + rest[0])
        return t, ms[0]
    if base in ("ldmatrix", "stmatrix"):
        if ms[:2] != ["sync", "aligned"] or len(ms) < 5:
            return None
        t = {base, ms[2], ms[3]}
        rest = ms[4:]
        if rest and rest[0] == "trans":
            t.add("trans")
            rest = rest[1:]
        if rest and rest[0] in ("shared", "shared::cta"):
            t.add("sp_shared")
            rest = rest[1:]
        if not rest:
            return None
        t.add("t_" + "_".join(rest))
        return t, "load" if base == "ldmatrix" else "store"
    if base == "wmma" and ms[:1] == ["mma"]:
        rest = ms[1:]
        t = {"wmma", "mma"}
        if rest[:2] in (["xor", "popc"], ["and", "popc"]):
            t.add(rest[0] + "_popc")
            rest = rest[2:]
        if rest[:2] != ["sync", "aligned"] or len(rest) < 5:
            return None
        t |= {"a" + rest[2], "b" + rest[3], rest[4]}
        rest = rest[5:]
        if rest and rest[0] in ("rn", "rz", "rm", "rp"):
            t.add(rest[0])
            rest = rest[1:]
        if rest and rest[-1] == "satfinite":
            t.add("satfinite")
            rest = rest[:-1]
        if len(rest) == 2:
            tup = (rest[0], "f16", "f16", rest[1])
        elif len(rest) == 4:
            tup = tuple(rest)
        else:
            return None
        return t | _types(tup), "mma"
    if base == "mma":
        t = {"mma"}
        if ms[:1] == ["sp"]:
            t.add("sp")
            ms = ms[1:]
        elif ms[:1] == ["sp::ordered_metadata"]:
            t.add("sp_ordered")
            ms = ms[1:]
        if ms[:2] != ["sync", "aligned"] or len(ms) < 9:
            return None
        t |= {ms[2], "a" + ms[3], "b" + ms[4]}
        rest = ms[5:]
        if rest and rest[0].startswith("kind::"):
            kind = rest[0][6:]
            rest = rest[1:]
            if rest and rest[0] == "block_scale":
                t |= {"block_scale", "kind_" + kind}
                rest = rest[1:]
                mm = re.fullmatch(r"scale_vec::([124])X", rest[0]) if rest else None
                if mm:
                    t.add("sv%sx" % mm.group(1))
                    rest = rest[1:]
                if not rest or rest[-1] not in ("ue8m0", "ue4m3"):
                    return None
                t.add("s_" + rest[-1])
                rest = rest[:-1]
            elif kind == "f8f6f4":
                t.add("kind_f8f6f4")
            else:
                return None
        if rest and rest[0] == "satfinite":
            t.add("satfinite")
            rest = rest[1:]
        if rest[-2:] in (["xor", "popc"], ["and", "popc"]):
            t.add(rest[-2] + "_popc")
            rest = rest[:-2]
        if len(rest) != 4:
            return None
        return t | _types(tuple(rest)), "mma"
    return None


def bits(ty):
    """Bit width of a scalar or vector IR type used by these intrinsics."""
    ty = ty.strip()
    m = re.fullmatch(r"<(\d+) x (\w+)>", ty)
    if m:
        b = bits(m.group(2))
        return None if b is None else int(m.group(1)) * b
    return {"half": 16, "bfloat": 16, "float": 32, "double": 64, "i8": 8, "i16": 16,
            "i32": 32, "i64": 64}.get(ty)


def helper(fname, ret, rtys, args, kname, short):
    """(fname, body, decls) of a helper replacing the intrinsic call."""
    w = ["define internal %s @%s(%s) alwaysinline {" % (
        ret, fname, ", ".join("%s %%y%d" % (t, k) for k, (t, _) in enumerate(args)))]
    cargs = []
    for k, (t, _) in enumerate(args):
        if t.startswith("ptr"):
            narrow = re.match(r"ptr addrspace\((3|6|7)\)", t) and short
            w.append("  %%p%d = ptrtoint %s %%y%d to %s" % (k, t, k, "i32" if narrow else "i64"))
            w.append(("  %%c%d = zext i32 %%p%d to i64" if narrow else "  %%c%d = add i64 %%p%d, 0") % (k, k))
        else:
            b = bits(t)
            w.append("  %%b%d = bitcast %s %%y%d to i%d" % (k, t, k, b))
            w.append(("  %%c%d = zext i%d %%b%d to i64" % (k, b, k)) if b < 64
                     else "  %%c%d = add i64 %%b%d, 0" % (k, k))
        cargs.append("i64 %%c%d" % k)
    n = len(rtys)
    rty = "void" if not n else "{ %s }" % ", ".join(["i64"] * n)
    call = "call %s @%s(%s)" % (rty, kname, ", ".join(cargs))
    if not n:
        w += ["  " + call, "  ret void", "}"]
    else:
        w.append("  %r = " + call)
        outs = []
        for k, t in enumerate(rtys):
            b = bits(t)
            w.append("  %%e%d = extractvalue %s %%r, %d" % (k, rty, k))
            w.append(("  %%t%d = trunc i64 %%e%d to i%d" % (k, k, b)) if b < 64
                     else "  %%t%d = add i64 %%e%d, 0" % (k, k))
            w.append("  %%o%d = bitcast i%d %%t%d to %s" % (k, b, k, t))
            outs.append("%%o%d" % k)
        if ret.startswith("{"):
            acc = "poison"
            for k, (t, o) in enumerate(zip(rtys, outs)):
                w.append("  %%a%d = insertvalue %s %s, %s %s, %d" % (k, ret, acc, t, o, k))
                acc = "%%a%d" % k
            w.append("  ret %s %s" % (ret, acc))
        else:
            w.append("  ret %s %s" % (ret, outs[0]))
        w.append("}")
    return fname, "\n".join(w), {kname: "declare %s @%s(%s) nounwind willreturn" % (
        rty, kname, ", ".join(["i64"] * len(args)))}
