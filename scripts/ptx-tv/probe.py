"""Probe-derived operand maps for intrinsics the lifter has no model for.

For an NVVM intrinsic call in one of the families below (tensor-core,
TMA/bulk copies, sparse (de)compression, ...), we compile a probe function

    define RET @probe(<non-immediate args as params>) {
      %r = call RET @llvm.nvvm.X(<args, immediates kept>)
      ret RET %r
    }

with the same llc flags, find the single PTX instruction it lowers to, and
trace each of its register operands back to (parameter, byte offset) and
each result register forward to (return value byte offset).

Both sides are then expressed as a call to an opaque function named after
the instruction's opcode and operand shape; scalar operands are passed as
i64 (zero-extended), address operands as pointers, in PTX operand order.

What this checks: that in the real function the backend emits that
instruction with the right values in the right operands and uses its
results correctly.  What it cannot check: the instruction selection of the
intrinsic itself (the probe tells us what the backend does in isolation).
"""
import hashlib
import os
import re
import subprocess
import tempfile

import ptx_lift
from llvmsig import split_top
import layout

FAMILIES = re.compile(r"(tcgen05|cp\.async\.bulk|cp\.reduce|spcompress|spdecompress|"
                      r"st\.async|multimem|tensormap|clusterlaunchcontrol|griddepcontrol|"
                      r"ldmatrix|stmatrix|wmma|mma|wgmma|cp\.async|red\.async|"
                      r"stacksave|stackrestore)\b")
PTX_FAMILIES = re.compile(r"^(tcgen05|cp|spcompress|spdecompress|st|multimem|tensormap|"
                          r"clusterlaunchcontrol|griddepcontrol|ldmatrix|stmatrix|wmma|mma|"
                          r"wgmma|red)$")
COPY_OPS = {"mov", "cvta"}
LLC = None
_cache = {}


def shape_of(ops):
    """Operand structure: brackets, braces and element counts (no values)."""
    out = []
    for o in ops:
        o = o.strip()
        if o.startswith("["):
            inner = split_top(o[1:-1])
            parts = []
            for p in inner:
                p = p.strip()
                if p.startswith("{"):
                    parts.append("{%d}" % len(split_top(p[1:-1])))
                else:
                    parts.append("A" if not parts else "S")
            out.append("[" + ",".join(parts) + "]")
        elif o.startswith("{"):
            out.append("{%d}" % len(split_top(o[1:-1])))
        elif "|" in o:
            out.append("S|S")
        else:
            out.append("S")
    return ",".join(out)


def op_name(opcode, ops):
    sig = opcode + "(" + shape_of(ops) + ")"
    return "ptx.op." + re.sub(r"[^\w.]", "_", opcode)[:60] + "." + \
        hashlib.md5(sig.encode()).hexdigest()[:10]


def flatten_operands(ops):
    """[(kind, text)] in order: kind is 'addr' (first item of a bracket),
    'addroff' (+offset inside a bracket address), 'val' (scalar), 'dst'."""
    items = []
    for o in ops:
        o = o.strip()
        if o.startswith("["):
            inner = split_top(o[1:-1])
            for k, p in enumerate(inner):
                p = p.strip()
                if p.startswith("{"):
                    items += [("val", x.strip()) for x in split_top(p[1:-1])]
                elif k == 0:
                    items.append(("addr", p))   # base{+offset}, one pointer
                else:
                    items.append(("val", p))
        elif o.startswith("{"):
            items += [("val", x.strip()) for x in split_top(o[1:-1])]
        elif "|" in o:
            items += [("val", x.strip()) for x in o.split("|")]
        else:
            items.append(("val", o))
    return items


class Mapping:
    pass


def immarg_positions(decl_line):
    m = re.search(r"@[\w.$\"-]+\((.*)\)", decl_line)
    if not m:
        return set()
    return {k for k, a in enumerate(split_top(m.group(1))) if "immarg" in a}


def _probe(name, ret, args, decl_line, llc_args, typedefs_text):
    params, call_args = [], []
    imm = immarg_positions(decl_line)
    for k, (t, v) in enumerate(args):
        if k in imm or v.startswith("@"):
            call_args.append("%s %s" % (t, v))
        else:
            params.append((k, t))
            call_args.append("%s %%p%d" % (t, k))
    proto = ", ".join("%s %%p%d" % (t, k) for k, t in params)
    if ret == "void":
        body = "  call void @llvm.nvvm.%s(%s)\n  ret void" % (name, ", ".join(call_args))
    else:
        body = "  %%r = call %s @llvm.nvvm.%s(%s)\n  ret %s %%r" % (ret, name, ", ".join(call_args), ret)
    mod = "%s\n%s\ndefine %s @probe(%s) {\n%s\n}\n" % (typedefs_text, decl_line, ret, proto, body)
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(mod)
        path = f.name
    try:
        p = subprocess.run([LLC] + llc_args + [path, "-o", "-"], capture_output=True,
                           text=True, timeout=120)
    finally:
        os.unlink(path)
    if p.returncode:
        return None
    funcs = ptx_lift.parse_module(p.stdout)
    f = funcs.get("probe")
    if not f:
        return None
    try:
        items = ptx_lift.parse_body(f.body)
    except ptx_lift.Unsupported:
        return None
    pidx = {pp.name: params[i][0] for i, pp in enumerate(f.params) if i < len(params)}
    if len(f.params) != len(params):
        return None
    src = {}          # reg -> ("arg", argindex, byteoff, bits)
    target = None
    results = {}      # reg -> retval byte offset
    regw = {}
    for it in items:
        if it[0] != "stmt":
            continue
        s = it[1]
        if s.startswith(".reg"):
            m = re.fullmatch(r"\.reg\s+\.(\w+)\s+(.+)", s)
            for item in m.group(2).split(","):
                item = item.strip()
                mm = re.fullmatch(r"(%?[\w$]+)<(\d+)>", item)
                for nm in (["%s%d" % (mm.group(1), q) for q in range(int(mm.group(2)))] if mm else [item]):
                    regw[nm] = ptx_lift.TBITS.get(m.group(1), 32)
            continue
        if s.startswith("."):
            continue
        pred, opc, ops = ptx_lift.decode(s)
        if pred:
            return None
        base = opc.split(".")[0]
        mods = opc.split(".")[1:]
        if base == "ld" and "param" in mods[0]:
            dsts = [d.strip() for d in (split_top(ops[0][1:-1]) if ops[0].startswith("{") else [ops[0]])]
            m = re.fullmatch(r"\[\s*([\w$]+)\s*(?:\+\s*(\d+))?\s*\]", ops[1].strip())
            if not m or m.group(1) not in pidx:
                return None
            ts = [x for x in mods if x in ptx_lift.TBITS]
            eb = ptx_lift.TBITS[ts[0]]
            for q, d in enumerate(dsts):
                src[d] = ("arg", pidx[m.group(1)], int(m.group(2) or 0) + q * eb // 8, eb)
            continue
        if base in COPY_OPS and len(ops) == 2 and ops[1].strip() in src:
            src[ops[0].strip()] = src[ops[1].strip()]
            continue
        # i1 argument -> predicate:  and.b16 x, a, 1 ; setp.ne.b16 p, x, 0
        if base == "and" and len(ops) == 3 and ops[1].strip() in src and ops[2].strip() == "1":
            s0 = src[ops[1].strip()]
            src[ops[0].strip()] = ("bit", s0[1], s0[2], 1)
            continue
        if base == "setp" and mods[:1] == ["ne"] and len(ops) == 3 and \
                src.get(ops[1].strip(), ("",))[0] == "bit" and ops[2].strip() == "0":
            s0 = src[ops[1].strip()]
            src[ops[0].strip()] = ("arg", s0[1], s0[2], 1)
            continue
        if base == "st" and "param" in mods[0]:
            m = re.fullmatch(r"\[\s*func_retval0\s*(?:\+\s*(\d+))?\s*\]", ops[0].strip())
            if not m:
                return None
            vals = [x.strip() for x in (split_top(ops[1][1:-1]) if ops[1].startswith("{") else [ops[1]])]
            ts = [x for x in mods if x in ptx_lift.TBITS]
            eb = ptx_lift.TBITS[ts[0]]
            for q, v in enumerate(vals):
                if v in results or target is None:
                    return None
                results[v] = (int(m.group(1) or 0) + q * eb // 8, eb)
            continue
        if base == "ret":
            continue
        if target is not None:
            return None   # more than one real instruction
        target = (opc, ops)
    if target is None:
        return None
    opc, ops = target
    flat = flatten_operands(ops)
    if flat is None:
        return None
    mp = Mapping()
    mp.opcode, mp.name = opc, op_name(opc, ops)
    has_dst = ret != "void"
    mp.ops, mp.dsts = [], []
    # destination operands: the leading operand(s) whose registers reach the result
    # destinations: leading flattened operands whose registers reach the
    # return value (possibly several operands, e.g. data and metadata)
    n_dst = 0
    if has_dst:
        for kind, text in flat:
            if kind == "val" and (text in results or text == "_"):
                n_dst += 1
            else:
                break
        if n_dst == 0:
            return None
    for q, (kind, text) in enumerate(flat):
        if q < n_dst:
            if text == "_":
                mp.dsts.append(None)
            elif text in results:
                mp.dsts.append(results[text] + (regw.get(text, 32),))
            else:
                return None
            continue
        if kind == "addr":
            if "+" in text:
                return None
            s = src.get(text)
            if not s or not args[s[1]][0].startswith("ptr"):
                return None
            if regw.get(text, 64) < 64:
                # 32-bit address (e.g. tensor memory): passed as its integer bits
                mp.ops.append(("piece", s[1], s[2], regw[text]))
            else:
                mp.ops.append(("ptr", s[1]))
        elif text in src:
            s = src[text]
            if s[0] != "arg":
                return None
            mp.ops.append(("piece", s[1], s[2], min(regw.get(text, s[3]), s[3]) if s[3] == 1 else regw.get(text, s[3])))
        else:
            v = ptx_lift.Lifter.imm(None, text)
            if v is None:
                return None
            mp.ops.append(("imm", v))
    return mp


def mapping_for(name, ret, args, decl_line, llc_args, typedefs_text):
    key = (name, ret, tuple(args), tuple(llc_args))
    if key not in _cache:
        try:
            _cache[key] = _probe(name, ret, args, decl_line, llc_args, typedefs_text)
        except Exception:
            _cache[key] = None
    return _cache[key]


def helper_body(fname, mp, ret, args, typedefs):
    """Source-side helper implementing the intrinsic through the opaque call."""
    params = [(k, t) for k, (t, v) in enumerate(args)]
    L = ["define internal %s @%s(%s) alwaysinline {" % (
        ret, fname, ", ".join("%s %%a%d" % (t, k) for k, t in params))]
    slots = {}
    n = [0]

    def tmp():
        n[0] += 1
        return "%%v%d" % n[0]
    cargs = []
    for o in mp.ops:
        if o[0] == "ptr":
            t = args[o[1]][0]
            cargs.append("%s %%a%d" % (t, o[1]))
        elif o[0] == "imm":
            cargs.append("i64 %d" % o[1])
        else:
            _, k, off, bits = o
            t = args[k][0]
            if t.startswith("ptr"):
                v = tmp()
                L.append("  %s = ptrtoint %s %%a%d to i64" % (v, t, k))
                if off:
                    v2 = tmp()
                    L.append("  %s = lshr i64 %s, %d" % (v2, v, off * 8))
                    v = v2
                if bits < 64:
                    v2 = tmp()
                    L.append("  %s = and i64 %s, %d" % (v2, v, (1 << bits) - 1))
                    v = v2
                cargs.append("i64 " + v)
                continue
            if k not in slots:
                sz = layout.alloc_size(layout.parse(t, typedefs))
                s = tmp()
                L.append("  %s = alloca [%d x i8], align 16" % (s, max(sz, 1) + 16))
                L.append("  store %s %%a%d, ptr %s" % (t, k, s))
                slots[k] = s
            p = tmp()
            L.append("  %s = getelementptr i8, ptr %s, i64 %d" % (p, slots[k], off))
            v = tmp()
            L.append("  %s = load i%d, ptr %s, align 1" % (v, bits, p))
            if bits < 64:
                v2 = tmp()
                L.append("  %s = zext i%d %s to i64" % (v2, bits, v))
                v = v2
            cargs.append("i64 " + v)
    nd = len(mp.dsts)
    rty = "void" if nd == 0 else "{ %s }" % ", ".join(["i64"] * nd)
    if rty == "void":
        L.append("  call void @%s(%s)" % (mp.name, ", ".join(cargs)))
    else:
        r = tmp()
        L.append("  %s = call %s @%s(%s)" % (r, rty, mp.name, ", ".join(cargs)))
    if ret == "void":
        L.append("  ret void")
    else:
        sz = layout.alloc_size(layout.parse(ret, typedefs))
        rs = tmp()
        L.append("  %s = alloca [%d x i8], align 16" % (rs, max(sz, 1) + 16))
        for q, d in enumerate(mp.dsts):
            if d is None:
                continue
            off, sbits, rbits = d
            e = tmp()
            L.append("  %s = extractvalue %s %s, %d" % (e, rty, r, q))
            t2 = tmp()
            L.append("  %s = trunc i64 %s to i%d" % (t2, e, sbits))
            p = tmp()
            L.append("  %s = getelementptr i8, ptr %s, i64 %d" % (p, rs, off))
            L.append("  store i%d %s, ptr %s, align 1" % (sbits, t2, p))
        rv = tmp()
        L.append("  %s = load %s, ptr %s" % (rv, ret, rs))
        L.append("  ret %s %s" % (ret, rv))
    L.append("}")
    decl = "declare %s @%s(%s) nounwind" % (rty, mp.name, ", ".join(a.rsplit(" ", 1)[0] for a in cargs))
    return "\n".join(L), decl
