"""cp.async.bulk.tensor global->shared (TMA g2s, non-.cta forms, default
data-validity pattern) as uninterpreted calls named by a token set derived
independently from the intrinsic (NVPTXUsage.md: mode, dimension, the
flag_mc/flag_ch/cta_group flags, i16/i32 multicast mask) and from the PTX
opcode; operands in a canonical order: dst, tensor map, coordinates, mbar,
im2col offsets, [multicast mask], [cache hint].  The tensor map and the copy
itself stay opaque."""
import re

MODES = {"tile": "tile", "im2col": "im2col", "tile.gather4": "tile::gather4",
         "im2col.w": "im2col::w", "im2col.w.128": "im2col::w::128"}


def name_of(tokens):
    return "ptx.tma." + ".".join(sorted(tokens)).replace("::", "__")


def intrinsic(name, args):
    """(tokens, canonical arg indices) or None."""
    m = re.fullmatch(r"cp\.async\.bulk\.tensor\.g2s\.(tile|im2col|tile\.gather4|im2col\.w|im2col\.w\.128)"
                     r"\.([1-5])d\.(i16|i32)", name)
    if not m:
        return None
    mode, nd, mty = m.group(1), int(m.group(2)), m.group(3)
    ncoord = 5 if mode == "tile.gather4" else nd
    noff = {"tile": 0, "tile.gather4": 0, "im2col": nd - 2}.get(mode, 2)
    v = [x for _, x in args]
    base = 3 + ncoord + noff
    if len(args) != base + 6:
        return None
    fmc, fch, cg, vp = v[base + 2], v[base + 3], v[base + 4], v[base + 5]
    if vp != "0" or cg not in ("0", "1", "2"):
        return None
    t = {"g2s", "%dd" % nd, "mode_" + MODES[mode]}
    if cg != "0":
        t.add("cg" + cg)
    keep = [0, 2] + list(range(3, 3 + ncoord)) + [1] + list(range(3 + ncoord, base))
    if fmc in ("1", "true"):
        t.add("mc32" if mty == "i32" else "mc")
        keep.append(base)
    elif fmc not in ("0", "false"):
        return None
    if fch in ("1", "true"):
        t.add("ch")
        keep.append(base + 1)
    elif fch not in ("0", "false"):
        return None
    return t, keep


def ptx(mods, ops):
    """(tokens, canonical operand texts) or None for a cp opcode."""
    if mods[:3] != ["async", "bulk", "tensor"] or len(mods) < 8:
        return None
    m = re.fullmatch(r"([1-5])d", mods[3])
    if not m or mods[4:6] != ["shared::cluster", "global"]:
        return None
    nd = int(m.group(1))
    mode = mods[6]
    if mode not in MODES.values() or mods[7] != "mbarrier::complete_tx::bytes":
        return None
    t = {"g2s", "%dd" % nd, "mode_" + mode}
    for x in mods[8:]:
        mm = re.fullmatch(r"cta_group::([12])", x)
        if mm:
            t.add("cg" + mm.group(1))
        elif x == "multicast::cluster":
            t.add("mc")
        elif x == "multicast::cluster::32b":
            t.add("mc32")
        elif x == "L2::cache_hint":
            t.add("ch")
        else:
            return None
    ops = [o.strip() for o in ops]
    if len(ops) < 3:
        return None
    tm = re.fullmatch(r"\[\s*([%\w$]+)\s*,\s*\{([^}]*)\}\s*\]", ops[1])
    if not tm:
        return None
    coords = [c.strip() for c in tm.group(2).split(",")]
    rest = ops[3:]
    offs = []
    if rest and rest[0].startswith("{"):
        offs = [c.strip() for c in rest[0][1:-1].split(",")]
        rest = rest[1:]
    if len(rest) != ("mc" in t or "mc32" in t) + ("ch" in t):
        return None
    canon = [ops[0], tm.group(1)] + coords + [ops[2]] + offs + rest
    return t, canon
