"""Narrow floating-point conversions (FP8/FP6/FP4/ue8m0/ue5m3/tf32, .pzo,
.rs, .scaled) as uninterpreted functions with a canonical name.

Both the PTX `cvt` and the NVVM intrinsic that names it map to
`@ptx.cvt.<dst>.<src>.<modifiers>`, with operands passed as integers of
their register widths in instruction order.  This checks instruction
selection, modifiers, operand order and packing, not the numerics of the
narrow formats.
"""
import re

NARROW = {"e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2", "ue8m0x2",
          "ue5m3x2", "s2f6x2", "tf32", "e4m3x4", "e5m2x4", "e2m3x4",
          "e3m2x4", "e2m1x4", "ue8m0"}
# destinations for which PTX requires .satfinite (implied by intrinsics)
SATFINITE_IMPLIED = {"e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2",
                     "s2f6x2", "e4m3x4", "e5m2x4", "e2m3x4", "e3m2x4", "e2m1x4"}
WIDE = {"f32", "f16x2", "bf16x2", "f16", "bf16", "f32x4"}
ROUND = {"rn", "rz", "rm", "rp", "rna", "rs"}


def key(dst, src, mods):
    mods = set(mods)
    if dst in SATFINITE_IMPLIED:
        mods.discard("satfinite")
    if src in NARROW and dst in WIDE:      # up-conversion: exact, no rounding
        mods -= ROUND
    return "ptx.cvt.%s.%s.%s" % (dst, src, ".".join(sorted(mods)) or "none")


def is_narrow(types, mods):
    return any(t in NARROW for t in types) or any(
        m == "pzo" or m == "rs" or m.startswith("scaled") for m in mods)


def ptx_key(opcode):
    """Key from a PTX cvt opcode, e.g. cvt.rn.satfinite.e4m3x2.f32."""
    parts = opcode.split(".")[1:]
    types = [p for p in parts if p in NARROW or p in WIDE or
             re.fullmatch(r"[usb]\d+|f\d+(x\d)?|bf16x2", p)]
    if len(types) != 2:
        return None
    mods = [p.replace("::", "_") for p in parts if p not in types]
    return key(types[0], types[1], mods)


INTR = re.compile(r"(ff|f16x2|bf16x2|f32x4|e\dm\dx2|ue8m0x2|ue5m3x2|s2f6x2)\.to\."
                  r"(f16x2|bf16x2|e\dm\dx[24]|ue8m0x2|ue5m3x2|s2f6x2)((?:\.\w+)*)")


def intrinsic_key(name, args):
    """(key, kept_arg_indices) for an NVVM narrow-conversion intrinsic."""
    m = INTR.fullmatch(name)
    tf = re.fullmatch(r"f2tf32((?:\.\w+)*)", name)
    if m:
        src, dst, rest = m.group(1), m.group(2), m.group(3)
        src = "f32" if src == "ff" else src
    elif tf:
        src, dst, rest = "f32", "tf32", tf.group(1)
    else:
        return None
    toks = [t for t in rest.split(".") if t]
    mods = []
    i = 0
    while i < len(toks):
        if toks[i] == "scale" and i + 2 < len(toks):
            mods.append("scaled_%s_%s" % (toks[i + 1], toks[i + 2]))
            i += 3
            continue
        mods.append(toks[i])
        i += 1
    keep = list(range(len(args)))
    # trailing i1 immediate: .pzo flag
    if args and args[-1][0] == "i1" and args[-1][1] in ("true", "false", "0", "1"):
        if args[-1][1] in ("true", "1"):
            mods.append("pzo")
        keep = keep[:-1]
    # logical widths of 8-bit operands (PTX uses .b8 registers for them):
    # an e2m1x2 value, and a single ue8m0 scale (scale.n1)
    argw = [None] * len(keep)
    if src == "e2m1x2" and argw:
        argw[0] = 8
    if any(m.startswith("scaled_n1") for m in mods) and len(argw) >= 2:
        argw[-1] = 8
    retw = 8 if dst == "e2m1x2" else None
    return key(dst, src, mods), keep, argw, retw
