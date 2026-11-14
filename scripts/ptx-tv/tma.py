"""TMA (cp.async.bulk.tensor / cp.reduce.async.bulk.tensor /
cp.async.bulk.prefetch.tensor) as uninterpreted calls named by a token set
derived independently from the intrinsic (NVPTXUsage.md: direction, mode,
dimension, flag arguments, red_op table, i16/i32 multicast mask) and from
the PTX opcode; operands in PTX textual order (tensor-map coordinates and
im2col offsets flattened).  The tensor map and the copy stay opaque.
Forms with a non-default data-validity pattern and the override forms are
not modelled (left probe-derived)."""
import re

G2S_MODES = {"tile": "tile", "im2col": "im2col", "tile.gather4": "tile::gather4",
             "im2col.w": "im2col::w", "im2col.w.128": "im2col::w::128"}
S2G_MODES = {"tile": "tile", "im2col": "im2col_no_offs", "tile.scatter4": "tile::scatter4",
             "im2col.w": "im2col_no_offs::w", "im2col.w.128": "im2col_no_offs::w::128"}
RED = ["add", "min", "max", "inc", "dec", "and", "or", "xor"]


def name_of(tokens):
    return "ptx.tma." + ".".join(sorted(tokens)).replace("::", "__")


def _counts(mode, nd):
    ncoord = 5 if mode in ("tile.gather4", "tile.scatter4") else nd
    noff = {"tile": 0, "tile.gather4": 0, "tile.scatter4": 0, "im2col": nd - 2}.get(mode, 2)
    return ncoord, noff


def _flag(x):
    return {"1": True, "true": True, "0": False, "false": False}.get(x)


def intrinsic(name, args):
    """(tokens, arg indices in PTX operand order) or None."""
    v = [x for _, x in args]
    modes = "tile|im2col|tile\\.gather4|tile\\.scatter4|im2col\\.w|im2col\\.w\\.128"
    m = re.fullmatch(r"cp\.async\.bulk\.tensor\.g2s(\.cta)?\.(%s)\.([1-5])d(?:\.(i16|i32))?" % modes, name)
    if m:
        cta, mode, nd, mty = bool(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if mode not in G2S_MODES:
            return None
        ncoord, noff = _counts(mode, nd)
        base = 3 + ncoord + noff
        t = {"g2s", "%dd" % nd, "mode_" + G2S_MODES[mode]}
        if cta:
            # (dst, bar, tmap, coords, offs, ch, flag_ch, valid_pattern)
            if len(args) != base + 3 or v[base + 2] != "0" or _flag(v[base + 1]) is None:
                return None
            t.add("dst_cta")
            keep = [0, 2] + list(range(3, 3 + ncoord)) + [1] + list(range(3 + ncoord, base))
            if _flag(v[base + 1]):
                t.add("ch")
                keep.append(base)
            return t, keep
        # (dst, bar, tmap, coords, offs, mc, ch, flag_mc, flag_ch, cta_group, valid_pattern)
        if mty is None or len(args) != base + 6 or v[base + 5] != "0" or v[base + 4] not in ("0", "1", "2"):
            return None
        fmc, fch = _flag(v[base + 2]), _flag(v[base + 3])
        if fmc is None or fch is None:
            return None
        if v[base + 4] != "0":
            t.add("cg" + v[base + 4])
        keep = [0, 2] + list(range(3, 3 + ncoord)) + [1] + list(range(3 + ncoord, base))
        if fmc:
            t.add("mc32" if mty == "i32" else "mc")
            keep.append(base)
        if fch:
            t.add("ch")
            keep.append(base + 1)
        return t, keep
    m = re.fullmatch(r"cp\.async\.bulk\.tensor\.(s2g|reduce)\.(%s)\.([1-5])d" % modes, name)
    if m:
        kind, mode, nd = m.group(1), m.group(2), int(m.group(3))
        if mode not in S2G_MODES:
            return None
        ncoord = 5 if mode == "tile.scatter4" else nd
        # (src, tmap, coords, ch, [red_op], flag_ch)
        base = 2 + ncoord
        want = base + 2 + (kind == "reduce")
        if len(args) != want or _flag(v[-1]) is None:
            return None
        t = {kind, "%dd" % nd, "mode_" + S2G_MODES[mode]}
        if kind == "reduce":
            if not re.fullmatch(r"[0-7]", v[base + 1]):
                return None
            t.add("op_" + RED[int(v[base + 1])])
        keep = [1] + list(range(2, base)) + [0]
        if _flag(v[-1]):
            t.add("ch")
            keep.append(base)
        return t, keep
    m = re.fullmatch(r"cp\.async\.bulk\.tensor\.prefetch\.evict\.priority\.(%s)\.([1-5])d" % modes, name)
    if m:
        # (tmap, coords, offs, evict_policy): 0 = evict_normal (no qualifier), 1 = evict_last
        mode, nd = m.group(1), int(m.group(2))
        if mode not in G2S_MODES:
            return None
        ncoord, noff = _counts(mode, nd)
        base = 1 + ncoord + noff
        if len(args) != base + 1 or v[-1] not in ("0", "1"):
            return None
        t = {"prefetch", "%dd" % nd, "mode_" + G2S_MODES[mode]}
        if v[-1] == "1":
            t.add("evict_last")
        return t, list(range(0, base))
    m = re.fullmatch(r"cp\.async\.bulk\.tensor\.prefetch\.(%s)\.([1-5])d" % modes, name)
    if m:
        mode, nd = m.group(1), int(m.group(2))
        if mode not in G2S_MODES:
            return None
        ncoord, noff = _counts(mode, nd)
        # (tmap, coords, offs, ch, flag_ch)
        base = 1 + ncoord + noff
        if len(args) != base + 2 or _flag(v[-1]) is None:
            return None
        t = {"prefetch", "%dd" % nd, "mode_" + G2S_MODES[mode]}
        keep = list(range(0, base))
        if _flag(v[-1]):
            t.add("ch")
            keep.append(base)
        return t, keep
    return None


def _flatten(ops):
    """Operand texts in PTX order with tensor coordinates / vectors flattened."""
    out = []
    for o in ops:
        o = o.strip()
        tm = re.fullmatch(r"\[\s*([%\w$]+)\s*,\s*\{([^}]*)\}\s*\]", o)
        if tm:
            out.append(tm.group(1))
            out += [c.strip() for c in tm.group(2).split(",")]
        elif o.startswith("{"):
            out += [c.strip() for c in o[1:-1].split(",")]
        else:
            out.append(o)
    return out


def ptx(mods, ops):
    """(tokens, operand texts) or None for a cp opcode's modifiers."""
    t = set()
    if mods[:4] == ["async", "bulk", "prefetch", "tensor"]:
        m = re.fullmatch(r"([1-5])d", mods[4]) if len(mods) > 7 else None
        if not m or mods[5:7] != ["L2", "global"] or mods[7] not in G2S_MODES.values():
            return None
        t = {"prefetch", m.group(1) + "d", "mode_" + mods[7]}
        rest = mods[8:]
    elif mods[:3] == ["async", "bulk", "tensor"] or mods[:4] == ["reduce", "async", "bulk", "tensor"]:
        red = mods[0] == "reduce"
        mods = mods[1:] if red else mods
        m = re.fullmatch(r"([1-5])d", mods[3]) if len(mods) > 6 else None
        if not m:
            return None
        nd = m.group(1) + "d"
        if mods[4:6] in (["shared::cluster", "global"], ["shared::cta", "global"]) and not red:
            if mods[6] not in G2S_MODES.values() or len(mods) < 8 or mods[7] != "mbarrier::complete_tx::bytes":
                return None
            t = {"g2s", nd, "mode_" + mods[6]}
            if mods[4] == "shared::cta":
                t.add("dst_cta")
            rest = mods[8:]
        elif mods[4:6] == ["global", "shared::cta"]:
            k = 6
            if red:
                if mods[6] not in RED:
                    return None
                t.add("op_" + mods[6])
                k = 7
            if len(mods) <= k + 1 or mods[k] not in S2G_MODES.values() or mods[k + 1] != "bulk_group":
                return None
            t |= {"reduce" if red else "s2g", nd, "mode_" + mods[k]}
            rest = mods[k + 2:]
        else:
            return None
    else:
        return None
    for x in rest:
        mm = re.fullmatch(r"cta_group::([12])", x)
        if mm:
            t.add("cg" + mm.group(1))
        elif x == "multicast::cluster":
            t.add("mc")
        elif x == "multicast::cluster::32b":
            t.add("mc32")
        elif x == "L2::cache_hint":
            t.add("ch")
        elif x == "L2::evict_last" and "prefetch" in t:
            t.add("evict_last")
        else:
            return None
    return t, _flatten(ops)
