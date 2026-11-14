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


def intrinsic(name, args):
    """(tokens, canonical arg indices) or None."""
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
