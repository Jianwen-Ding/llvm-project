"""Non-tensor bulk copies and cp.async as uninterpreted calls keyed by their
documented modifiers (NVPTXUsage: flag_mc/flag_ch/flag_oob, flag_scope
0..3 = .cta/.cluster/.gpu/.sys, flag_valid_pattern table, i16/i32 mask;
cp.async size and src-size), derived independently from the intrinsic and
from the PTX opcode; operands in PTX order as zero-extended i64.  Used where
the copy-semantics model does not apply (relaxed, ignore_oob, cp_mask,
multicast, data validity) and, with PTXTV_OPAQUE_COPIES=1, for all copies."""
import os
import re

import tma

OPAQUE = os.environ.get("PTXTV_OPAQUE_COPIES") == "1"

SCOPE = {"0": "cta", "1": "cluster", "2": "gpu", "3": "sys"}


def name_of(tokens, n):
    return "ptx.bulk." + ".".join(sorted(tokens)).replace("::", "__") + ".n%d" % n


def _f(x):
    return {"1": True, "true": True, "0": False, "false": False}.get(x)


def intrinsic(name, args):
    """(tokens, arg indices in PTX operand order) or None."""
    v = [x for _, x in args]
    m = re.fullmatch(r"cp\.async\.bulk\.global\.to\.shared\.cluster(\.relaxed)?(?:\.(i16|i32))?", name)
    if m:
        rel = bool(m.group(1))
        # (dst, mbar, src, size, mc, ch, flag_mc, flag_ch, [scope], vp)
        if len(args) == 8 and not rel:
            v = v + ["0"]
        elif len(args) != 9 + rel:
            return None
        fmc, fch, vp = _f(v[6]), _f(v[7]), v[-1]
        if fmc is None or fch is None or (vp != "0" and vp not in tma.VALIDITY):
            return None
        t = {"g2s_cluster"}
        if rel:
            if v[8] not in SCOPE:
                return None
            t |= {"relaxed", "scope_" + SCOPE[v[8]]}
        if vp != "0":
            t.add("valid" + vp)
        keep = [0, 2, 3, 1]
        if fmc:
            t.add("mc32" if m.group(2) == "i32" else "mc")
            keep.append(4)
        if fch:
            t.add("ch")
            keep.append(5)
        return t, keep
    m = re.fullmatch(r"cp\.async\.bulk\.global\.to\.shared\.cta(\.relaxed)?", name)
    if m:
        rel = bool(m.group(1))
        # (dst, mbar, src, size, ibl, ibr, ch, flag_ch, flag_oob, [scope], vp)
        if len(args) != 10 + rel:
            return None
        fch, foob, vp = _f(v[7]), _f(v[8]), v[-1]
        if fch is None or foob is None or (vp != "0" and vp not in tma.VALIDITY):
            return None
        t = {"g2s_cta"}
        if rel:
            if v[9] not in SCOPE:
                return None
            t |= {"relaxed", "scope_" + SCOPE[v[9]]}
        if vp != "0":
            t.add("valid" + vp)
        keep = [0, 2, 3]
        if foob:
            t.add("ignore_oob")
            keep += [4, 5]
        keep.append(1)
        if fch:
            t.add("ch")
            keep.append(6)
        return t, keep
    m = re.fullmatch(r"cp\.async\.bulk\.shared\.cta\.to\.global(\.bytemask)?", name)
    if m:
        bm = bool(m.group(1))
        # (dst, src, size, ch, flag_ch, [mask])
        if len(args) != 5 + bm or _f(v[4]) is None:
            return None
        t = {"s2g"} | ({"cp_mask"} if bm else set())
        keep = [0, 1, 2]
        if _f(v[4]):
            t.add("ch")
            keep.append(3)
        if bm:
            keep.append(5)
        return t, keep
    if name == "cp.async.bulk.shared.cta.to.cluster" and len(args) == 4:
        return {"s2s"}, [0, 2, 3, 1]
    m = re.fullmatch(r"cp\.async\.(ca|cg)\.shared\.global\.(4|8|16)(\.s)?", name)
    if m:
        return {"cp_async", m.group(1), "size" + m.group(2)} | ({"srcsize"} if m.group(3) else set()), \
            list(range(len(args)))
    return None


def ptx(mods, ops):
    """(tokens, operand texts) or None for a cp opcode's modifiers."""
    flat = tma._flatten(ops)
    if mods[:2] == ["async", "bulk"] and "tensor" not in mods and "prefetch" not in mods:
        rest = mods[2:]
        t = set()
        if rest and rest[0] == "relaxed":
            if len(rest) < 2 or rest[1] not in SCOPE.values():
                return None
            t |= {"relaxed", "scope_" + rest[1]}
            rest = rest[2:]
        if rest[:3] == ["shared::cluster", "global", "mbarrier::complete_tx::bytes"]:
            t.add("g2s_cluster")
            rest = rest[3:]
        elif rest[:3] == ["shared::cta", "global", "mbarrier::complete_tx::bytes"]:
            t.add("g2s_cta")
            rest = rest[3:]
        elif rest[:3] == ["global", "shared::cta", "bulk_group"]:
            t.add("s2g")
            rest = rest[3:]
        elif rest[:3] == ["shared::cluster", "shared::cta", "mbarrier::complete_tx::bytes"]:
            t.add("s2s")
            rest = rest[3:]
        else:
            return None
        for x in rest:
            if x in tma.VALIDITY_TOK:
                t.add(tma.VALIDITY_TOK[x])
            elif x == "multicast::cluster":
                t.add("mc")
            elif x == "multicast::cluster::32b":
                t.add("mc32")
            elif x == "L2::cache_hint":
                t.add("ch")
            elif x in ("ignore_oob", "cp_mask"):
                t.add(x)
            elif x == "b128" and "relaxed" in t:
                pass
            else:
                return None
        return t, flat
    m = None
    if mods[:1] == ["async"] and len(mods) >= 4 and mods[1] in ("ca", "cg") and \
            mods[2:4] in (["shared", "global"], ["shared::cta", "global"]) and len(mods) == 4:
        if len(flat) not in (3, 4) or flat[2] not in ("4", "8", "16"):
            return None
        t = {"cp_async", mods[1], "size" + flat[2]} | ({"srcsize"} if len(flat) == 4 else set())
        return t, flat[:2] + flat[3:]
    return None


def semantic(tokens):
    """Whether the copy-semantics model (ptx_lift.bulk_copy_ir/cp_async_ir,
    nvvm_sem.bulk_copy_operands/cp_async_helper) covers this form."""
    if "cp_async" in tokens:
        return True
    return tokens == {"s2s"} or (len(tokens & {"g2s_cluster", "g2s_cta", "s2g"}) == 1 and
                                 len(tokens) <= 2 and tokens - {"g2s_cluster", "g2s_cta", "s2g"} <= {"ch"})
