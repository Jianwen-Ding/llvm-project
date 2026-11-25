"""tcgen05 alloc/dealloc/relinquish/commit/cp/shift/wait/fence as
uninterpreted calls named by a canonical token set, derived independently
from the NVVM intrinsic (NVPTXUsage.md) and from the PTX opcode, as mbar.py
does for mbarrier.  Operands are passed as integers: pointers to shared or
tensor memory (32-bit in PTX) as i32, generic pointers as i64."""
import re

CP_SHAPES = {"4x256b": "4x256b", "128x256b": "128x256b", "128x128b": "128x128b",
             "32x128b_warpx4": "32x128b.warpx4",
             "64x128b_warpx2_02_13": "64x128b.warpx2::02_13",
             "64x128b_warpx2_01_23": "64x128b.warpx2::01_23"}
CP_FMT = {"b6x16_p32": "b6x16_p32", "b4x16_p64": "b4x16_p64"}


def name_of(tokens):
    return "ptx.tcgen05." + ".".join(sorted(tokens)).replace("::", "__")


def intrinsic(name, args):
    """(tokens, kept_arg_indices) for llvm.nvvm.<name>(args), or None."""
    v = [x for _, x in args]
    m = re.fullmatch(r"tcgen05\.alloc\.(cg[12])\.p([03])", name)
    if m and len(args) == 3 and v[2] in ("0", "1", "true", "false"):
        t = {"alloc", m.group(1)}
        if m.group(2) == "3":
            t.add("shared")
        if v[2] in ("1", "true"):
            t.add("exclusive")
        return t, [0, 1]
    m = re.fullmatch(r"tcgen05\.dealloc\.(cg[12])", name)
    if m and len(args) == 3 and v[2] in ("0", "1", "true", "false"):
        t = {"dealloc", m.group(1)}
        if v[2] in ("1", "true"):
            t.add("exclusive")
        return t, [0, 1]
    m = re.fullmatch(r"tcgen05\.relinq\.alloc\.permit\.(cg[12])", name)
    if m and not args:
        return {"relinquish", m.group(1)}, []
    m = re.fullmatch(r"tcgen05\.commit(\.smem\.a\.read)?(\.mc)?\.(cg[12])\.p[03](\.i16|\.i32)?", name)
    if m and len(args) == (2 if m.group(2) else 1):
        t = {"commit", m.group(3)}
        if m.group(1):
            t.add("smem_a_read")
        if m.group(2):
            t.add("mc32" if m.group(4) == ".i32" else "mc")
        return t, list(range(len(args)))
    m = re.fullmatch(r"tcgen05\.cp\.(\w+?)(?:\.(b6x16_p32|b4x16_p64))?\.(cg[12])", name)
    if m and m.group(1) in CP_SHAPES and len(args) == 2:
        t = {"cp", m.group(3), "shape_" + CP_SHAPES[m.group(1)]}
        if m.group(2):
            t.add("src_" + m.group(2))
        return t, [0, 1]
    m = re.fullmatch(r"tcgen05\.shift\.down\.(cg[12])", name)
    if m and len(args) == 1:
        return {"shift_down", m.group(1)}, [0]
    m = re.fullmatch(r"tcgen05\.wait\.(ld|st)", name)
    if m and not args:
        return {"wait_" + m.group(1)}, []
    m = re.fullmatch(r"tcgen05\.fence\.(before|after)\.thread\.sync", name)
    if m and not args:
        return {"fence_" + m.group(1)}, []
    return None


def ptx(mods):
    """Token set for a tcgen05 opcode's modifiers, or None."""
    if not mods:
        return None
    op = mods[0]
    rest = mods[1:]
    cg = [m for m in rest if m.startswith("cta_group::")]
    cgt = {"cg" + cg[0].split("::")[1]} if len(cg) == 1 else set()
    rest = [m for m in rest if not m.startswith("cta_group::")]
    if op in ("alloc", "dealloc"):
        t = {op} | cgt
        for m in rest:
            if m == "exclusive":
                t.add("exclusive")
            elif m == "shared::cta" and op == "alloc":
                t.add("shared")
            elif m not in ("sync", "aligned", "b32"):
                return None
        return t if cgt else None
    if op == "relinquish_alloc_permit":
        return ({"relinquish"} | cgt) if cgt and set(rest) <= {"sync", "aligned"} else None
    if op == "commit":
        t = {"commit"} | cgt
        for m in rest:
            if m == "sync_restrict::shared::read::mma::a":
                t.add("smem_a_read")
            elif m == "multicast::cluster":
                t.add("mc")
            elif m == "multicast::cluster::32b":
                t.add("mc32")
            elif m not in ("mbarrier::arrive::one", "shared::cluster", "b64"):
                return None
        return t if cgt else None
    if op == "cp":
        s = ".".join(rest)
        for k, v in CP_SHAPES.items():
            for f in [None] + list(CP_FMT):
                want = v + ("" if f is None else ".b8x16." + f)
                if s == want:
                    t = {"cp", "shape_" + v} | cgt
                    if f:
                        t.add("src_" + f)
                    return t if cgt else None
        return None
    if op == "shift":
        return ({"shift_down"} | cgt) if cgt and rest == ["down"] else None
    if op in ("wait::ld", "wait::st"):
        return {"wait_" + op.split("::")[1]} if set(rest) <= {"sync", "aligned"} else None
    if op in ("fence::before_thread_sync", "fence::after_thread_sync"):
        return {"fence_" + op.split("::")[1].split("_")[0]} if not rest else None
    return None
