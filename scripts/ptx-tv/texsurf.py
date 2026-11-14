"""Texture/surface instructions as uninterpreted calls keyed by their
documented modifiers, derived independently from the NVVM intrinsic name and
from the PTX opcode (the image state stays opaque): txq/suq queries,
suld/sust (.b, by geometry, vector width, bit size and clamp mode) and the
unified tex fetches (geometry, result and coordinate types, .level/.grad).
Operands in PTX order as zero-extended i64; results returned as i64 each."""
import re

GEOM = {"1d", "2d", "3d", "a1d", "a2d", "cube", "acube", "2dms", "a2dms"}
PTX_GEOM = {"a1d": "a1d", "a2d": "a2d", "1d": "1d", "2d": "2d", "3d": "3d", "cube": "cube",
            "acube": "acube", "2dms": "2dms", "a2dms": "a2dms"}
TYBITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}


def name_of(tokens, nres, nops):
    return "ptx.texsurf." + ".".join(sorted(tokens)) + ".r%d.n%d" % (nres, nops)


def intrinsic(name):
    """(tokens, nres) or None."""
    m = re.fullmatch(r"(txq|suq)\.([\w.]+)", name)
    if m:
        return {m.group(1), "q_" + m.group(2).replace(".", "_")}, 1
    m = re.fullmatch(r"suld\.(\w+)\.(v[24])?(i8|i16|i32|i64)\.(trap|clamp|zero)", name)
    if m and m.group(1) in GEOM:
        n = int(m.group(2)[1]) if m.group(2) else 1
        return {"suld", "g_" + m.group(1), "b%d" % TYBITS[m.group(3)], "v%d" % n, m.group(4)}, n
    m = re.fullmatch(r"sust\.b\.(\w+)\.(v[24])?(i8|i16|i32|i64)\.(trap|clamp|zero)", name)
    if m and m.group(1) in GEOM:
        n = int(m.group(2)[1]) if m.group(2) else 1
        return {"sust", "g_" + m.group(1), "b%d" % TYBITS[m.group(3)], "v%d" % n, m.group(4)}, 0
    m = re.fullmatch(r"tex\.unified\.(\w+)\.v4(f32|s32|u32)\.(s32|f32)(\.level|\.grad)?", name)
    if m and m.group(1) in GEOM:
        t = {"tex", "g_" + m.group(1), "r_" + m.group(2), "c_" + m.group(3)}
        if m.group(4):
            t.add(m.group(4)[1:])
        return t, 4
    return None


def ptx(base, mods):
    """(tokens, result_is_first_operand) or None."""
    if base in ("txq", "suq") and len(mods) == 2 and mods[1] == "b32":
        return {base, "q_" + mods[0]}, True
    if base in ("suld", "sust") and mods and mods[0] == "b" and len(mods) in (4, 5):
        rest = mods[1:]
        g = rest[0]
        n = 1
        if rest[1] in ("v2", "v4"):
            n = int(rest[1][1])
            rest = rest[:1] + rest[2:]
        if g not in PTX_GEOM.values() or len(rest) != 3 or not re.fullmatch(r"b(8|16|32|64)", rest[1]) \
                or rest[2] not in ("trap", "clamp", "zero"):
            return None
        return {base, "g_" + g, rest[1], "v%d" % n, rest[2]}, base == "suld"
    if base == "tex":
        rest = list(mods)
        extra = None
        if rest and rest[0] in ("level", "grad"):
            extra = rest.pop(0)
        if len(rest) != 4 or rest[0] not in PTX_GEOM.values() or rest[1] != "v4":
            return None
        t = {"tex", "g_" + rest[0], "r_" + rest[2], "c_" + rest[3]}
        if extra:
            t.add(extra)
        return t, True
    return None
