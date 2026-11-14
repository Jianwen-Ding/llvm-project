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


# ---- tcgen05.ld.red -------------------------------------------------------
# IntrinsicsNVVM.td (NVVM_TCGEN05_LD_RED): returns {<n x T>, T} (the loaded
# values and the reduction result; T = i32 or float), arguments (tmem,
# [i64 offset for 16x32bx2], i32 redOp (0 min, 1 max), [i1 abs, i1 nan for
# f32]).  PTX: tcgen05.ld.red.sync.aligned.<shape>.x<num>.<min|max>[.abs]
# [.NaN].<u32|f32> {regs}, red, [taddr]{, offset}.

def red_key(shape, num, ty, op, ab, nan):
    return "ptx.tcgen05.ld_red.%s.x%d.%s.%s%s%s" % (shape, num, ty, op, ".abs" if ab else "",
                                                   ".nan" if nan else "")


def red_decl(shape, num, ty, op, ab, nan):
    n = nregs(shape, num)
    args = ["i32"] + (["i64"] if shape == "16x32bx2" else [])
    return "declare { <%d x i32>, i32 } @%s(%s) nounwind" % (
        n, red_key(shape, num, ty, op, ab, nan), ", ".join(args))


def red_intrinsic(name, args):
    m = re.fullmatch(r"tcgen05\.ld\.red\.(16x32bx2|32x32b)\.x(\d+)\.(i32|f32)", name)
    if not m or int(m.group(2)) not in NUMS:
        return None
    shape, num, t = m.group(1), int(m.group(2)), m.group(3)
    v = [x for _, x in args]
    k = 1 + (shape == "16x32bx2")
    want = k + 1 + (2 if t == "f32" else 0)
    if len(args) != want or v[k] not in ("0", "1"):
        return None
    op = "min" if v[k] == "0" else "max"
    ab = nan = False
    if t == "f32":
        if v[k + 1] not in ("0", "1", "true", "false") or v[k + 2] not in ("0", "1", "true", "false"):
            return None
        ab, nan = v[k + 1] in ("1", "true"), v[k + 2] in ("1", "true")
    return shape, num, ("f32" if t == "f32" else "u32"), op, ab, nan


def red_ptx(mods):
    if mods[:4] != ["ld", "red", "sync", "aligned"] or len(mods) < 8:
        return None
    shape = mods[4]
    m = re.fullmatch(r"x(\d+)", mods[5])
    if shape not in ("16x32bx2", "32x32b") or not m or int(m.group(1)) not in NUMS:
        return None
    rest = mods[6:]
    if rest[0] not in ("min", "max") or rest[-1] not in ("u32", "f32"):
        return None
    mid = rest[1:-1]
    ab, nan = "abs" in mid, "NaN" in mid
    if [x for x in mid if x not in ("abs", "NaN")] or (rest[-1] == "u32" and mid):
        return None
    return shape, int(m.group(1)), rest[-1], rest[0], ab, nan
