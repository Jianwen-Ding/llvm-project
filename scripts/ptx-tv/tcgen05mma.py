"""tcgen05.mma (base forms: optional .sp, .scale_d, .ashift; shared or tensor
A) as uninterpreted calls named by a token set derived independently from
the intrinsic (NVPTXUsage.md flag tables, IntrinsicsNVVM.td argument
order, trailing flags defaulting to 0) and from the PTX opcode.  Operands
are passed in a canonical order: d, a, b, idesc, enable_input_d, [sp
metadata], [scale_d]."""
import re

KIND = {0: "f16", 1: "tf32", 2: "f8f6f4", 3: "i8", 4: "ti16"}
COLL = {0: "discard", 1: "lastuse", 2: "fill", 3: "use"}


def name_of(tokens):
    return "ptx.tcgen05.mma." + ".".join(sorted(tokens))


def _flags(v, n, want):
    f = v[n:]
    if len(f) > want or not all(re.fullmatch(r"\d+", x) for x in f):
        return None
    return [int(x) for x in f] + [0] * (want - len(f))


def _coll(t, ca, cb):
    if ca not in COLL or cb not in COLL:
        return False
    t.add("ca_" + COLL[ca])
    if cb:
        t.add("cb_" + COLL[cb])
    return True


def _block(kind, blk):
    """The PTX ISA: a missing scale-vector size means .block32 for mxf4;
    .block32 is the only size for mxf8f6f4."""
    if blk is None and kind in ("mxf4", "mxf8f6f4"):
        return "block32"
    return blk


def intrinsic(name, args):
    """(tokens, canonical arg indices) or None.  The .decompress_b forms
    (IntrinsicsNVVM.td: the decompression table pointer follows the common
    arguments / scale pointers; kind f8f6f4 or mxf8f6f4.block32; NVPTXUsage
    does not describe them) map onto the corresponding base forms, with the
    table pointer last in canonical order."""
    m = re.fullmatch(r"tcgen05\.mma\.(shared|tensor)\.(f8f6f4(?:\.disable_output_lane\.cg[12])?"
                     r"|mxf8f6f4\.block_scale\.block32)\.decompress_b", name)
    if m:
        sp, form = m.group(1), m.group(2)
        if form == "f8f6f4":
            if len(args) != 9:
                return None
            r = _intrinsic("tcgen05.mma." + sp, args[:5] + [("i32", "2")] + args[6:])
            back, lut = list(range(5)), 5
        elif form.startswith("f8f6f4.disable"):
            if len(args) != 9:
                return None
            r = _intrinsic("tcgen05.mma.%s.%s" % (sp, form[len("f8f6f4."):]),
                           args[:5] + [args[6], ("i32", "2")] + args[7:])
            back, lut = [0, 1, 2, 3, 4, 6], 5
        else:
            if len(args) != 11:
                return None
            r = _intrinsic("tcgen05.mma.%s.mxf8f6f4.block_scale.block32" % sp, args[:7] + args[8:])
            back, lut = list(range(7)), 7
        if r is None:
            return None
        t, keep = r
        return t | {"decompress_lut_b"}, [back[i] for i in keep] + [lut]
    return _intrinsic(name, args)


def _intrinsic(name, args):
    v = [x for _, x in args]
    m = re.fullmatch(r"tcgen05\.mma\.ws(\.sp)?\.(shared|tensor)(\.zero_col_mask)?", name)
    if m:
        sp, space, zcm = bool(m.group(1)), m.group(2), bool(m.group(3))
        nfix = 5 + sp + zcm
        f = _flags(v, nfix, 3)
        if f is None or len(args) < nfix + 1 or f[0] not in KIND or f[1] not in (0, 1, 2, 3) \
                or f[2] not in COLL:
            return None
        t = {"ws", "cg1", "kind_" + KIND[f[0]], "b%d_%s" % (f[1], COLL[f[2]])}
        for flag, tok in ((sp, "sp"), (zcm, "zcm"), (space == "tensor", "a_tensor")):
            if flag:
                t.add(tok)
        return t, list(range(nfix))
    m = re.fullmatch(r"tcgen05\.mma(\.sp)?\.(shared|tensor)\.(mxf8f6f4|mxf4nvf4|mxf4)\.block_scale"
                     r"(?:\.(block16|block32))?", name)
    if m:
        sp, space, kind, blk = bool(m.group(1)), m.group(2), m.group(3), m.group(4)
        nfix = 7 + sp
        f = _flags(v, nfix, 3)
        if f is None or len(args) < nfix + 1 or f[0] not in (1, 2):
            return None
        blk = _block(kind, blk)
        if blk is None:
            return None
        t = {"block_scale", "cg%d" % f[0], "kind_" + kind, blk}
        if not _coll(t, f[1], f[2]):
            return None
        if sp:
            t.add("sp")
        if space == "tensor":
            t.add("a_tensor")
        return t, list(range(nfix))
    m = re.fullmatch(r"tcgen05\.mma(\.sp)?\.(shared|tensor)(\.scale_d)?\.disable_output_lane\.cg([12])"
                     r"(\.ashift)?", name)
    if m:
        sp, space, sd, cg, ash = bool(m.group(1)), m.group(2), bool(m.group(3)), m.group(4), bool(m.group(5))
        nfix = 6 + sp + sd
        f = _flags(v, nfix, 3)
        if f is None or len(args) < nfix + 1 or f[0] not in KIND:
            return None
        t = {"dol", "cg" + cg, "kind_" + KIND[f[0]]}
        if not _coll(t, f[1], f[2]):
            return None
        for flag, tok in ((sp, "sp"), (sd, "scale_d"), (ash, "ashift"), (space == "tensor", "a_tensor")):
            if flag:
                t.add(tok)
        return t, list(range(nfix))
    m = re.fullmatch(r"tcgen05\.mma(\.sp)?\.(shared|tensor)(\.scale_d)?(\.ashift)?", name)
    if not m:
        return None
    sp, space, sd, ash = bool(m.group(1)), m.group(2), bool(m.group(3)), bool(m.group(4))
    nfix = 5 + sp + sd
    v = [x for _, x in args]
    flags = v[nfix:]
    if len(args) < nfix + 2 or len(flags) > 4 or not all(re.fullmatch(r"\d+", f) for f in flags):
        return None
    flags = [int(f) for f in flags] + [0] * (4 - len(flags))
    kind, cg, ca, cb = flags
    if kind not in KIND or cg not in (1, 2) or ca not in COLL or cb not in COLL:
        return None
    t = {"cg%d" % cg, "kind_" + KIND[kind], "ca_" + COLL[ca]}
    if cb:
        t.add("cb_" + COLL[cb])
    if sp:
        t.add("sp")
    if sd:
        t.add("scale_d")
    if ash:
        t.add("ashift")
    if space == "tensor":
        t.add("a_tensor")
    return t, list(range(nfix))


def ptx(mods, ops):
    """(tokens, operands in canonical order) or None."""
    if not mods or mods[0] != "mma":
        return None
    if "decompress::lut::b" in mods:
        # [d], a, b, [lut], idesc, ...: the table operand follows b
        if len(ops) < 5 or "ws" in mods or "sp" in mods:
            return None
        r = ptx([x for x in mods if x != "decompress::lut::b"], list(ops[:3]) + list(ops[4:]))
        if r is None:
            return None
        return r[0] | {"decompress_lut_b"}, r[1] + [ops[3].strip()]
    if "block_scale" in mods:
        return _ptx_block_scale(mods, ops)
    if len(mods) > 1 and mods[1] == "ws":
        return _ptx_ws(mods, ops)
    if any(o.strip().startswith("{") for o in ops):
        r = _ptx_base(mods, [o for o in ops if not o.strip().startswith("{")])
        if r is None:
            return None
        t, canon = r
        vec = [o for o in ops if o.strip().startswith("{")]
        if len(vec) != 1:
            return None
        # the disable-output-lane mask follows the base operands
        return t | {"dol"}, canon + [vec[0].strip()]
    return _ptx_base(mods, ops)


def _ptx_ws(mods, ops):
    rest = mods[2:]
    t = {"ws"}
    if rest and rest[0] == "sp":
        t.add("sp")
        rest = rest[1:]
    for x in rest:
        if x == "cta_group::1":
            t.add("cg1")
        elif re.fullmatch(r"kind::(\w+)", x) and x.split("::")[1] in KIND.values():
            t.add("kind_" + x.split("::")[1])
        else:
            mm = re.fullmatch(r"collector::b([0-3])::(\w+)", x)
            if not mm or mm.group(2) not in COLL.values():
                return None
            t.add("b%s_%s" % (mm.group(1), mm.group(2)))
    ops = [o.strip() for o in ops]
    # [d], a, b, [sp]?, idesc, p, zero_col_mask?
    n = 5 + ("sp" in t)
    if len(ops) == n + 1:
        t.add("zcm")
    elif len(ops) != n:
        return None
    if ops[1].startswith("["):
        t.add("a_tensor")
    k = 4 if "sp" in t else 3
    canon = ops[:3] + [ops[k], ops[k + 1]] + ([ops[3]] if "sp" in t else []) + \
        ([ops[k + 2]] if "zcm" in t else [])
    return t, canon


def _ptx_block_scale(mods, ops):
    rest = mods[1:]
    t = {"block_scale"}
    if rest and rest[0] == "sp":
        t.add("sp")
        rest = rest[1:]
    kind = blk = None
    seen_ca = False
    for x in rest:
        mm = re.fullmatch(r"cta_group::([12])", x)
        if mm:
            t.add("cg" + mm.group(1))
        elif re.fullmatch(r"kind::(mxf8f6f4|mxf4nvf4|mxf4)", x):
            kind = x.split("::")[1]
        elif x in ("block16", "block32"):
            blk = x
        elif x == "block_scale":
            pass
        else:
            mm = re.fullmatch(r"collector::([ab])::(\w+)", x)
            if not mm or mm.group(2) not in COLL.values():
                return None
            if mm.group(1) == "a":
                t.add("ca_" + mm.group(2))
                seen_ca = True
            elif mm.group(2) != "discard":
                t.add("cb_" + mm.group(2))
    if kind is None:
        return None
    blk = _block(kind, blk)
    if blk is None:
        return None
    t |= {"kind_" + kind, blk}
    if not seen_ca:
        t.add("ca_discard")
    ops = [o.strip() for o in ops]
    # [d], a, b, [sp]?, idesc, [scale_a], [scale_b], p
    n = 7 + ("sp" in t)
    if len(ops) != n:
        return None
    if ops[1].startswith("["):
        t.add("a_tensor")
    k = 4 if "sp" in t else 3
    canon = ops[:3] + [ops[k], ops[k + 3]] + ([ops[3]] if "sp" in t else []) + [ops[k + 1], ops[k + 2]]
    return t, canon


def _ptx_base(mods, ops):
    rest = mods[1:]
    t = set()
    if rest and rest[0] == "sp":
        t.add("sp")
        rest = rest[1:]
    seen_ca = False
    for x in rest:
        mm = re.fullmatch(r"cta_group::([12])", x)
        if mm:
            t.add("cg" + mm.group(1))
            continue
        mm = re.fullmatch(r"kind::(\w+)", x)
        if mm and mm.group(1) in KIND.values():
            t.add("kind_" + mm.group(1))
            continue
        mm = re.fullmatch(r"collector::([ab])::(\w+)", x)
        if mm and mm.group(2) in COLL.values():
            if mm.group(1) == "a":
                t.add("ca_" + mm.group(2))
                seen_ca = True
            elif mm.group(2) != "discard":
                t.add("cb_" + mm.group(2))
            continue
        if x == "ashift":
            t.add("ashift")
            continue
        return None
    if not seen_ca:
        t.add("ca_discard")
    ops = [o.strip() for o in ops]
    # [d], a, b, [sp]?, idesc, p, scale?
    n = 5 + ("sp" in t)
    if len(ops) == n + 1:
        t.add("scale_d")
    elif len(ops) != n:
        return None
    if ops[1].startswith("["):
        t.add("a_tensor")
    d, a, b = ops[0], ops[1], ops[2]
    k = 3
    spm = None
    if "sp" in t:
        spm = ops[3]
        k = 4
    idesc, p = ops[k], ops[k + 1]
    canon = [d, a, b, idesc, p] + ([spm] if spm else []) + ([ops[k + 2]] if "scale_d" in t else [])
    return t, canon
