"""Semantics for NVVM intrinsics that name a single PTX instruction.

An NVVM intrinsic such as `llvm.nvvm.fmin.ftz.nan.f` is documented as the
PTX instruction `min.ftz.NaN.f32`.  To check code that uses it, the source
call is replaced by an always-inline function whose body is produced by
lifting that one PTX instruction with ptx_lift.  The check then validates
the backend's instruction selection and the code around it against the
same instruction model the lifter uses for the backend's output.

Intrinsics with no entry here are left as uninterpreted calls.
"""
import re

from llvmsig import parse_sig, split_top

TSUF = {"f": "f32", "d": "f64", "f16": "f16", "f16x2": "f16x2",
        "bf16": "bf16", "bf16x2": "bf16x2"}
OVL = {"f32": "f32", "f64": "f64", "f16": "f16", "bf16": "bf16",
       "v2f16": "f16x2", "v2bf16": "bf16x2", "v2f32": "f32x2"}
RND_IMM = {0: "rz", 1: "rn", 2: "rp", 3: "rm"}

LLTY_BITS = {"float": 32, "double": 64, "half": 16, "bfloat": 16, "<2 x float>": 64,
             "<2 x half>": 32, "<2 x bfloat>": 32, "i16": 16, "i32": 32,
             "i64": 64}


def mapping(name, args):
    """name: intrinsic without 'llvm.nvvm.'; args: list of (type, value).

    Returns (ptx_instruction_template, kept_arg_indices) or None.
    The template uses %d for the result and %a<k> for kept args.
    """
    n = name
    imms = {}
    for k, (t, v) in enumerate(args):
        if re.fullmatch(r"-?\d+", v) or v in ("true", "false"):
            imms[k] = v

    m = re.fullmatch(r"f(min|max)(\.ftz)?(\.nan)?(\.xorsign\.abs)?\.(f|d|f16|f16x2|bf16|bf16x2)", n)
    if m:
        op, ftz, nan, xs, t = m.groups()
        return ("%s%s%s%s.%s %%d, %%a0, %%a1" % (op, ftz or "", ".NaN" if nan else "",
                                                 xs or "", TSUF[t]), [0, 1])
    m = re.fullmatch(r"f(add|mul)(\.ftz)?(\.sat)?\.(\w+)", n)
    if m and 2 in imms:
        op, ftz, sat, ovl = m.groups()
        if ovl not in OVL:
            return None
        rnd = RND_IMM.get(int(imms[2]))
        return ("%s.%s%s%s.%s %%d, %%a0, %%a1" % (op, rnd, ftz or "", sat or "", OVL[ovl]), [0, 1])
    m = re.fullmatch(r"fma(\.ftz)?(\.sat|\.relu)?\.(\w+)", n)
    if m and 3 in imms:
        ftz, var, ovl = m.groups()
        if ovl not in OVL:
            return None
        rnd = RND_IMM.get(int(imms[3]))
        return ("fma.%s%s%s.%s %%d, %%a0, %%a1, %%a2" % (rnd, ftz or "", var or "", OVL[ovl]), [0, 1, 2])
    m = re.fullmatch(r"(div|rcp|sqrt)\.(rn|rz|rm|rp)(\.ftz)?\.(f|d)", n)
    if m:
        op, rnd, ftz, t = m.groups()
        nargs = 2 if op == "div" else 1
        ops = ", ".join("%%a%d" % k for k in range(nargs))
        return ("%s.%s%s.%s %%d, %s" % (op, rnd, ftz or "", TSUF[t], ops), list(range(nargs)))
    m = re.fullmatch(r"div\.(approx|full)(\.ftz)?(\.f)?", n)
    if m:
        kind, ftz, _ = m.groups()
        return ("div.%s%s.f32 %%d, %%a0, %%a1" % (kind, ftz or ""), [0, 1])
    m = re.fullmatch(r"(rcp|sqrt|rsqrt|lg2|sin|cos)\.approx(\.ftz)?\.(f|d)", n)
    if m:
        op, ftz, t = m.groups()
        return ("%s.approx%s.%s %%d, %%a0" % (op, ftz or "", TSUF[t]), [0])
    m = re.fullmatch(r"ex2\.approx(\.ftz)?\.(\w+)", n)
    if m:
        ftz, ovl = m.groups()
        if ovl not in OVL:
            return None
        return ("ex2.approx%s.%s %%d, %%a0" % (ftz or "", OVL[ovl]), [0])
    m = re.fullmatch(r"(floor|ceil|trunc)(\.ftz)?\.(f|d)", n)
    if m:
        op, ftz, t = m.groups()
        r = {"floor": "rmi", "ceil": "rpi", "trunc": "rzi"}[op]
        tt = TSUF[t]
        return ("cvt.%s%s.%s.%s %%d, %%a0" % (r, ftz or "", tt, tt), [0])
    m = re.fullmatch(r"saturate(\.ftz)?\.(f|d)", n)
    if m:
        ftz, t = m.groups()
        tt = TSUF[t]
        return ("cvt.sat%s.%s.%s %%d, %%a0" % (ftz or "", tt, tt), [0])
    m = re.fullmatch(r"fabs(\.ftz)?\.(\w+)", n)
    if m:
        ftz, ovl = m.groups()
        if ovl not in OVL:
            return None
        return ("abs%s.%s %%d, %%a0" % (ftz or "", OVL[ovl]), [0])
    m = re.fullmatch(r"(neg|abs)\.(bf16|bf16x2)", n)
    if m:
        return ("%s.%s %%d, %%a0" % m.groups(), [0])
    m = re.fullmatch(r"d2f\.(rn|rz|rm|rp)(\.ftz)?", n)
    if m:
        rnd, ftz = m.groups()
        return ("cvt.%s%s.f32.f64 %%d, %%a0" % (rnd, ftz or ""), [0])
    m = re.fullmatch(r"([fd])2(u?)(i|ll)\.(rn|rz|rm|rp)(\.ftz)?", n)
    if m:
        src, u, w, rnd, ftz = m.groups()
        dt = ("u" if u else "s") + ("32" if w == "i" else "64")
        return ("cvt.%si%s.%s.%s %%d, %%a0" % (rnd, ftz or "", dt, TSUF[src]), [0])
    m = re.fullmatch(r"(u?)(i|ll)2([fd])\.(rn|rz|rm|rp)", n)
    if m:
        u, w, dst, rnd = m.groups()
        st = ("u" if u else "s") + ("32" if w == "i" else "64")
        return ("cvt.%s.%s.%s %%d, %%a0" % (rnd, TSUF[dst], st), [0])
    m = re.fullmatch(r"ff2(f16x2|bf16x2)\.(rn|rz)(\.relu)?", n)
    if m:
        t, rnd, relu = m.groups()
        return ("cvt.%s%s.%s.f32 %%d, %%a0, %%a1" % (rnd, relu or "", t), [0, 1])
    m = re.fullmatch(r"f2(f16|bf16)\.(rn|rz)(\.relu)?", n)
    if m:
        t, rnd, relu = m.groups()
        return ("cvt.%s%s.%s.f32 %%d, %%a0" % (rnd, relu or "", t), [0])
    m = re.fullmatch(r"f2h\.rn(\.ftz)?", n)
    if m:
        return ("cvt.rn%s.f16.f32 %%d, %%a0" % (m.group(1) or ""), [0])
    m = re.fullmatch(r"mul24\.(u?)i", n)
    if m:
        return ("mul24.lo.%s32 %%d, %%a0, %%a1" % ("u" if m.group(1) else "s"), [0, 1])
    m = re.fullmatch(r"mulhi\.(u?)(s|i|ll)", n)
    if m:
        u, w = m.groups()
        bits = {"s": 16, "i": 32, "ll": 64}[w]
        return ("mul.hi.%s%d %%d, %%a0, %%a1" % ("u" if u else "s", bits), [0, 1])
    m = re.fullmatch(r"sad\.(u?)(s|i|ll)", n)
    if m:
        u, w = m.groups()
        bits = {"s": 16, "i": 32, "ll": 64}[w]
        return ("sad.%s%d %%d, %%a0, %%a1, %%a2" % ("u" if u else "s", bits), [0, 1, 2])
    m = re.fullmatch(r"idp4a\.([su])\.([su])", n)
    if m:
        return ("dp4a.%s32.%s32 %%d, %%a0, %%a1, %%a2" % m.groups(), [0, 1, 2])
    m = re.fullmatch(r"idp2a\.([su])\.([su])", n)
    if m and 2 in imms:
        hi = imms[2] in ("true", "1", "-1")
        return ("dp2a.%s.%s32.%s32 %%d, %%a0, %%a1, %%a2" % ((("hi" if hi else "lo"),) + m.groups()),
                [0, 1, 3])
    m = re.fullmatch(r"flo\.([su])\.i(32|64)", n)
    if m and 1 in imms:
        sh = ".shiftamt" if imms[1] in ("true", "1", "-1") else ""
        return ("bfind%s.%s%s %%d, %%a0" % (sh, m.group(1), m.group(2)), [0])
    m = re.fullmatch(r"fsh([lr])\.clamp\.i32", n)
    if m:
        # llvm.nvvm.fshX.clamp(hi, lo, amt) -> shf.X.clamp d, lo, hi, amt
        return ("shf.%s.clamp.b32 %%d, %%a1, %%a0, %%a2" % m.group(1), [0, 1, 2])
    m = re.fullmatch(r"([sz])ext\.(wrap|clamp)", n)
    if m:
        return ("szext.%s.%s32 %%d, %%a0, %%a1" % (m.group(2), "s" if m.group(1) == "s" else "u"), [0, 1])
    m = re.fullmatch(r"bmsk\.(wrap|clamp)", n)
    if m:
        return ("bmsk.%s.b32 %%d, %%a0, %%a1" % m.group(1), [0, 1])
    m = re.fullmatch(r"prmt(\.(f4e|b4e|rc8|ecl|ecr|rc16))?", n)
    if m:
        mode = ("." + m.group(2)) if m.group(2) else ""
        if mode:
            return ("prmt.b32%s %%d, %%a0, 0, %%a1" % mode, [0, 1]) if len(args) == 2 else \
                ("prmt.b32%s %%d, %%a0, %%a1, %%a2" % mode, [0, 1, 2])
        return ("prmt.b32 %d, %a0, %a1, %a2", [0, 1, 2])
    return None


def ptx_param_bits(ty):
    return LLTY_BITS.get(ty)


def synth_ptx(fname, ret, argtys, instr):
    rb = ptx_param_bits(ret)
    abits = [ptx_param_bits(t) for t in argtys]
    if rb is None or None in abits:
        return None
    lines = [".visible .func (.param .b%d func_retval0) %s(" % (rb, fname)]
    lines.append(",\n".join("\t.param .b%d %s_param_%d" % (b, fname, k)
                            for k, b in enumerate(abits)))
    lines.append(")\n{")
    lines.append("\t.reg .b%d %%d;" % rb)
    for k, b in enumerate(abits):
        lines.append("\t.reg .b%d %%a%d;" % (b, k))
    for k, b in enumerate(abits):
        lines.append("\tld.param.b%d %%a%d, [%s_param_%d];" % (b, k, fname, k))
    lines.append("\t" + instr + ";")
    lines.append("\tst.param.b%d [func_retval0], %%d;" % rb)
    lines.append("\tret;\n}")
    return "\n".join(lines)


CALL_RE = re.compile(r"@llvm\.nvvm\.([\w.]+)\(")


def expand(text, lifter_mod):
    """Rewrite calls to modelled NVVM intrinsics; return new module text."""
    out_lines = []
    defs = {}
    removed = False
    for line in text.split("\n"):
        if line.lstrip().startswith("declare") or "call " not in line:
            out_lines.append(line)
            continue
        m = CALL_RE.search(line)
        if not m:
            out_lines.append(line)
            continue
        name = m.group(1)
        j = m.end() - 1
        depth, k = 0, j
        while True:
            c = line[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        argtext = line[j + 1:k]
        args = []
        for a in split_top(argtext):
            toks = a.split()
            # type is everything but the trailing value; drop attributes
            val = toks[-1]
            ty = a[:a.rfind(val)].strip()
            ty = re.sub(r"\b(noundef|immarg|signext|zeroext|inreg)\b", "", ty).strip()
            args.append((ty, val))
        # memory barriers: no-ops under single-thread semantics (as in tgt)
        if re.fullmatch(r"membar\.\w+|fence\.[\w.]+", name) \
                and not re.match(r"\s*%", line):
            removed = True
            continue
        # f32 shuffles are data movement: express them via the i32 form,
        # which is what the .b32 PTX instruction is lifted to.
        sm = re.fullmatch(r"shfl\.(sync\.)?(up|down|bfly|idx)\.f32(p?)", name)
        if sm:
            key = ("shfl", name)
            if key not in defs:
                fname = "ptxtv_sem_%d" % len(defs)
                sync, mode, p = sm.groups()
                iname = "llvm.nvvm.shfl.%s%s.i32%s" % (sync or "", mode, p)
                n = len(args)
                params = ", ".join("%s %%x%d" % (t, i) for i, (t, _) in enumerate(args))
                vi = 1 if sync else 0
                cargs = ", ".join("i32 %%x%d" % i if i != vi else "i32 %xi" for i in range(n))
                if p:
                    body = ("define internal { float, i1 } @%s(%s) alwaysinline {\n"
                            "  %%xi = bitcast float %%x%d to i32\n"
                            "  %%r = call { i32, i1 } @%s(%s)\n"
                            "  %%v = extractvalue { i32, i1 } %%r, 0\n"
                            "  %%q = extractvalue { i32, i1 } %%r, 1\n"
                            "  %%f = bitcast i32 %%v to float\n"
                            "  %%a = insertvalue { float, i1 } poison, float %%f, 0\n"
                            "  %%b = insertvalue { float, i1 } %%a, i1 %%q, 1\n"
                            "  ret { float, i1 } %%b\n}" % (fname, params, vi, iname, cargs))
                    decl = "declare { i32, i1 } @%s(%s) convergent nounwind" % (
                        iname, ", ".join(["i32"] * n))
                else:
                    body = ("define internal float @%s(%s) alwaysinline {\n"
                            "  %%xi = bitcast float %%x%d to i32\n"
                            "  %%r = call i32 @%s(%s)\n"
                            "  %%f = bitcast i32 %%r to float\n"
                            "  ret float %%f\n}" % (fname, params, vi, iname, cargs))
                    decl = "declare i32 @%s(%s) convergent nounwind" % (
                        iname, ", ".join(["i32"] * n))
                defs[key] = (fname, body, {iname: decl})
            line = line[:m.start()] + "@" + defs[key][0] + line[j:]
            out_lines.append(line)
            continue
        # ldu/ldg: a plain load of the result type with the given alignment
        lm = re.fullmatch(r"ld[ug]\.global\.[ifp]\.[\w.]+", name)
        if lm and len(args) == 2 and re.fullmatch(r"\d+", args[1][1]):
            cm = re.search(r"call\s+((?:<[^>]*>)|\S+)\s+@llvm\.nvvm\.", line)
            res = re.match(r"\s*(%[\w.$-]+)\s*=", line)
            if cm and res:
                out_lines.append("  %s = load %s, %s %s, align %s" % (
                    res.group(1), cm.group(1), args[0][0], args[0][1], args[1][1]))
                removed = True
                continue
        mp = mapping(name, args)
        cm = re.search(r"call\s+(?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\s+)*"
                       r"((?:<[^>]*>)|\S+)\s+@llvm\.nvvm\.", line)
        if not mp or not cm:
            out_lines.append(line)
            continue
        ret = cm.group(1)
        instr, keep = mp
        kept = [args[i] for i in keep]
        key = (name, instr, ret, tuple(t for t, _ in kept))
        if key not in defs:
            fname = "ptxtv_sem_%d" % len(defs)
            ptx = synth_ptx(fname, ret, [t for t, _ in kept], instr)
            if ptx is None:
                out_lines.append(line)
                continue
            sig = parse_sig("define internal %s @%s(%s)" % (
                ret, fname, ", ".join("%s %%x%d" % (t, i) for i, (t, _) in enumerate(kept))))
            try:
                L = lifter_mod.lift_function(ptx, sig, {}, {}, {}, fname)
            except lifter_mod.Unsupported:
                out_lines.append(line)
                continue
            body = lifter_mod.render(L, "define internal %s @%s(%s) alwaysinline" % (
                ret, fname, ", ".join("%s %%x%d" % (t, i) for i, (t, _) in enumerate(kept))))
            defs[key] = (fname, body, L.extra_decls)
        fname = defs[key][0]
        new_args = ", ".join("%s %s" % a for a in kept)
        line = line[:m.start()] + "@" + fname + "(" + new_args + line[k:]
        out_lines.append(line)
    if not defs:
        return ("\n".join(out_lines), True) if removed else (text, False)
    extra = {}
    for _, body, decls in defs.values():
        out_lines.append(body)
        extra.update(decls)
    have = set(re.findall(r"^declare [^@]*@([\w.$\"-]+)\(", text, re.M))
    for n, d in extra.items():
        if n not in have:
            out_lines.append(d)
    return "\n".join(out_lines), True
