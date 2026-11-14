"""tcgen05.ld / tcgen05.st (not .red) as uninterpreted calls whose names are
derived independently on both sides: from the NVVM intrinsic as NVPTXUsage
documents it (llvm.nvvm.tcgen05.{ld,st}.<shape>.<num>, the i1 pack/unpack
flag, the extra i64 offset of 16x32bx2) and from the PTX opcode
(tcgen05.{ld,st}.sync.aligned.<shape>.<num>[.pack::16b|.unpack::16b].b32).
Tensor memory stays opaque, but the check covers instruction selection
(shape, repeat factor, packing), the register count (NVPTXUsage's table) and
the operands, unlike a probe-derived opaque instruction."""
import re

BASE = {"16x32bx2": 1, "16x64b": 1, "32x32b": 1, "16x128b": 2, "16x256b": 4}
NUMS = {1, 2, 4, 8, 16, 32, 64, 128}


def nregs(shape, num):
    return BASE[shape] * num


def key(kind, shape, num, packed):
    return "ptx.tcgen05.%s.%s.x%d%s" % (kind, shape, num, ".packed" if packed else "")


def decl(kind, shape, num, packed):
    n = nregs(shape, num)
    args = ["i32"] + (["i64"] if shape == "16x32bx2" else []) + \
        (["<%d x i32>" % n] if kind == "st" else [])
    ret = "<%d x i32>" % n if kind == "ld" else "void"
    return "declare %s @%s(%s) nounwind" % (ret, key(kind, shape, num, packed), ", ".join(args))


def intrinsic(name):
    m = re.fullmatch(r"tcgen05\.(ld|st)\.(16x32bx2|16x64b|32x32b|16x128b|16x256b)\.x(\d+)", name)
    if not m or int(m.group(3)) not in NUMS:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def ptx(mods):
    """(kind, shape, num, packed) for a tcgen05 opcode's modifiers, or None."""
    if len(mods) < 5 or mods[0] not in ("ld", "st") or mods[1:3] != ["sync", "aligned"]:
        return None
    kind, shape = mods[0], mods[3]
    m = re.fullmatch(r"x(\d+)", mods[4])
    if shape not in BASE or not m or int(m.group(1)) not in NUMS:
        return None
    rest = mods[5:]
    packed = False
    if rest and rest[0] == ("pack::16b" if kind == "ld" else "unpack::16b"):
        packed, rest = True, rest[1:]
    if rest != ["b32"]:
        return None
    return kind, shape, int(m.group(1)), packed
