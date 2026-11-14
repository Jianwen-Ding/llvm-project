"""applypriority, eviction-priority prefetches and cp.async.mbarrier.arrive
as uninterpreted calls named by a token set derived independently from the
intrinsic (NVPTXUsage.md: evict_policy 0 = evict_normal (no qualifier on
prefetch), 1 = evict_last; applypriority accepts evict_normal only) and from
the PTX opcode; operands in PTX textual order (tensor coordinates and
offsets flattened, immediates kept).  The cache state stays opaque."""
import re

import tma

MODES = {"tile": "tile", "im2col": "im2col", "tile.gather4": "tile::gather4",
         "im2col.w": "im2col::w", "im2col.w.128": "im2col::w::128"}


def name_of(tokens):
    return "ptx.prio." + ".".join(sorted(tokens)).replace("::", "__")


def intrinsic(name, args):
    """(tokens, arg indices in PTX operand order, constant operands) or None."""
    v = [x for _, x in args]
    m = re.fullmatch(r"applypriority(\.global)?\.L2\.evict\.normal", name)
    if m and len(args) == 2 and v[1] == "128":
        return {"applyprio", "global" if m.group(1) else "generic", "size128"}, [0]
    if name == "applypriority.async.bulk.evict.priority" and len(args) == 3 and v[2] == "0":
        return {"applyprio", "bulk"}, [0, 1]
    if name == "cp.async.bulk.prefetch.L2" and len(args) == 4 and v[3] in ("0", "1", "true", "false"):
        ch = v[3] in ("1", "true")
        return {"prefetch_bulk"} | ({"ch"} if ch else set()), [0, 1] + ([2] if ch else [])
    if name == "cp.async.bulk.prefetch.evict.priority" and len(args) == 3 and v[2] in ("0", "1"):
        return {"prefetch_bulk"} | ({"evict_last"} if v[2] == "1" else set()), [0, 1]
    m = re.fullmatch(r"applypriority\.async\.bulk\.tensor"
                     r"\.evict\.priority\.(tile|im2col|tile\.gather4|im2col\.w|im2col\.w\.128)\.([1-5])d", name)
    if m:
        mode, nd = m.group(1), int(m.group(2))
        ncoord, noff = tma._counts(mode, nd)
        if len(args) != 1 + ncoord + noff + 1 or v[-1] != "0":
            return None
        return {"applyprio", "tensor", "%dd" % nd, "mode_" + MODES[mode]}, list(range(len(args) - 1))
    m = re.fullmatch(r"cp\.async\.mbarrier\.arrive(\.noinc)?(\.shared)?", name)
    if m and len(args) == 1:
        t = {"cp_async_arrive"}
        if m.group(1):
            t.add("noinc")
        if m.group(2):
            t.add("shared")
        return t, [0]
    return None


def ptx(base, mods, ops):
    """(tokens, operand texts in PTX order) or None."""
    flat = tma._flatten(ops)
    if base == "applypriority":
        if mods in (["L2::evict_normal"], ["global", "L2::evict_normal"]):
            if len(flat) == 2 and flat[1] == "128":
                return {"applyprio", "global" if mods[0] == "global" else "generic", "size128"}, flat[:1]
            return None
        if mods == ["async", "bulk", "global", "bulk_group", "L2::evict_normal"]:
            return {"applyprio", "bulk"}, flat
        if len(mods) == 8 and mods[:3] == ["async", "bulk", "tensor"] and re.fullmatch(r"[1-5]d", mods[3]) \
                and mods[4:7] == ["global", "bulk_group", "L2::evict_normal"] and mods[7] in MODES.values():
            return {"applyprio", "tensor", mods[3], "mode_" + mods[7]}, flat
        return None
    if base == "cp":
        if mods[:3] == ["async", "mbarrier", "arrive"]:
            rest = mods[3:]
            if not rest or rest[-1] != "b64" or not set(rest[:-1]) <= {"noinc", "shared"}:
                return None
            return {"cp_async_arrive"} | set(rest[:-1]), flat
        if mods[:5] == ["async", "bulk", "prefetch", "L2", "global"] and len(flat) >= 2:
            t = {"prefetch_bulk"}
            for x in mods[5:]:
                if x == "L2::evict_last":
                    t.add("evict_last")
                elif x == "L2::cache_hint":
                    t.add("ch")
                else:
                    return None
            return t, flat
    return None
