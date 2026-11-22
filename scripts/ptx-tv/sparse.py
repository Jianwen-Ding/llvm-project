"""Semantics of PTX spdecompress (PTX ISA 9.4, "Data Movement and Conversion
Instructions: spdecompress"), shared by the lifter and the NVVM intrinsic.

    for N in .num: for S in num_src: data[N][mdata[N][S]] = cdata[N][S]
    (positions not written are zero)

Operands are bundles of .b32 registers.  Elements (and metadata indices)
are taken to be packed contiguously from bit 0 of the first register
upwards (lane 0 in the low bits), which is how the operand sizes in the
ISA's table are computed and how LLVM packs vector lanes; this packing
order is an assumption, not stated explicitly by the ISA.  An index that
does not name a position in its target group is implementation-specific;
it is modelled as taken modulo the target group size (a deterministic
choice, the same on both sides of the check; poison would let the
source's optimiser fold it away).
"""
import re

SPF = re.compile(r"sp::(\d+):(\d+)")
VALID = {(1, 2), (1, 4), (1, 8), (1, 16), (2, 4), (2, 8), (2, 16), (4, 8), (4, 16)}


def regs(bits):
    return (bits + 31) // 32


def check(E, I, X, Y, num):
    return (E in (8, 16) and I in (2, 4) and (X, Y) in VALID and X * E <= 32 and
            (I != 2 or Y <= 4) and 32 <= Y * E * num <= 4096 and
            num in (1, 2, 4, 8, 16, 32, 64) and
            regs(X * I * num) + regs(X * E * num) + regs(Y * E * num) <= 253)


def spdecompress_ir(E, I, X, Y, num, mregs, cregs, fresh):
    """mregs, cregs: the i32 registers of mdata and cdata (lists of values).
    Returns (lines, result) with result a <Y*num x iE> value.  Index and
    element fields never straddle registers (I, E divide 32)."""
    L = []

    def op(text):
        t = fresh()
        L.append("%s = %s" % (t, text))
        return t

    def field(rs, k, w):
        r, sh = divmod(k * w, 32)
        v = op("lshr i32 %s, %d" % (rs[r], sh)) if sh else rs[r]
        return op("trunc i32 %s to i%d" % (v, w))
    vt = "<%d x i%d>" % (Y * num, E)
    data = "zeroinitializer"
    for n in range(num):
        for s in range(X):
            k = n * X + s
            ix = op("zext i%d %s to i32" % (I, field(mregs, k, I)))
            el = field(cregs, k, E)
            if (1 << I) > Y:
                ix = op("and i32 %s, %d" % (ix, Y - 1))
            pos = op("add i32 %s, %d" % (ix, n * Y)) if n else ix
            data = op("insertelement %s %s, i%d %s, i32 %s" % (vt, data, E, el, pos))
    return L, data


def intrinsic_params(name, ret, args):
    """(E, I, X, Y, num) for a call of llvm.nvvm.spdecompress.* returning ret
    (`<R x iE>`) with args [(mty, m), (cty, c), (i32, idx), (i32, ntgt)]."""
    if not name.startswith("spdecompress.") or len(args) != 4:
        return None
    rm = re.fullmatch(r"<(\d+) x i(8|16)>", ret)
    cm = re.fullmatch(r"<(\d+) x i(8|16)>", args[1][0])
    if not rm or not cm or rm.group(2) != cm.group(2):
        return None
    try:
        I, Y = int(args[2][1]), int(args[3][1])
    except ValueError:
        return None
    R, C, E = int(rm.group(1)), int(cm.group(1)), int(rm.group(2))
    if R % Y:
        return None
    num = R // Y
    if C % num:
        return None
    X = C // num
    if not check(E, I, X, Y, num):
        return None
    mr = regs(X * I * num)
    if args[0][0] != ("i32" if mr == 1 else "<%d x i32>" % mr):
        return None
    return E, I, X, Y, num


# ---- spcompress: keyed model ------------------------------------------------
# NVPTXUsage: llvm.nvvm.spcompress(DataTy %data, i32 %spdesc, i32 immarg
# %idx_size, i32 immarg %num_tgt = 4) -> {MDataTy, CDataTy}; data = 2*num
# registers, cdata = num, mdata = ceil(num * idx_size / elem_size).  The
# selection (implementation-specific) is not modelled: an uninterpreted
# call keyed by element size, index size and num, on 32-bit register bundles.

def spc_key(E, I, num):
    return "ptx.spcompress.b%d.b%d.x%d" % (E, I, num)


def spc_regs(E, I, num):
    return 2 * num, num, (num * I + E - 1) // E


def spc_decl(E, I, num):
    d, c, mr = spc_regs(E, I, num)
    return "declare { <%d x i32>, <%d x i32> } @%s(<%d x i32>, i32) nounwind willreturn memory(none)" % (
        mr, c, spc_key(E, I, num), d)


def spc_intrinsic(name, ret, args):
    m = re.fullmatch(r"spcompress\.[\w.]+", name)
    if not m or len(args) != 4 or args[3][1] != "4" or args[2][1] not in ("2", "4"):
        return None
    dm = re.fullmatch(r"<(\d+) x i(8|16)>", args[0][0])
    if not dm or args[1][0] != "i32":
        return None
    K, E, I = int(dm.group(1)), int(dm.group(2)), int(args[2][1])
    if (K * E) % 64:
        return None
    num = K * E // 64
    if num not in (1, 2, 4, 8, 16, 32, 64):
        return None
    return E, I, num


def spc_ptx(mods):
    if len(mods) != 4 or mods[0] not in ("b8", "b16") or mods[1] not in ("b2", "b4") \
            or mods[2] != "sp::2:4" or not re.fullmatch(r"x\d+", mods[3]):
        return None
    return int(mods[0][1:]), int(mods[1][1:]), int(mods[3][1:])
