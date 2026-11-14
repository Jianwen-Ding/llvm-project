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
    """(tokens, arg indices in PTX operand order) or None; tensor-map
    override forms as in tma.intrinsic (operands after the tensor map)."""
    m = re.fullmatch(r"(applypriority\.async\.bulk\.tensor\.evict\.priority\.[\w.]*?)"
                     r"\.override\.(addr(?:\.dim(?:\.stride)?)?)(\.[1-5]d)", name)
    if m:
        ov = m.group(2)
        nd = int(m.group(3)[1])
        nov = {"addr": 1, "addr.dim": 1 + nd, "addr.dim.stride": 1 + nd + (nd - 1) + 1}[ov]
        r = _intrinsic(m.group(1) + m.group(3), args[:1] + args[1 + nov:])
        if r is None:
            return None
        t, keep = r
        keep = [i if i == 0 else i + nov for i in keep]
        keep = keep[:1] + list(range(1, 1 + nov)) + keep[1:]
        return t | {"ov_" + x.split("::")[1] for x in tma.OVERRIDE[ov]}, keep
    return _intrinsic(name, args)


def _intrinsic(name, args):
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
        if len(mods) >= 8 and mods[:3] == ["async", "bulk", "tensor"] and re.fullmatch(r"[1-5]d", mods[3]) \
                and mods[4:7] == ["global", "bulk_group", "L2::evict_normal"] and mods[7] in MODES.values():
            t = {"applyprio", "tensor", mods[3], "mode_" + mods[7]}
            for x in mods[8:]:
                if x not in ("override::global_address", "override::global_dim",
                             "override::global_dim_stride"):
                    return None
                t.add("ov_" + x.split("::")[1])
            return t, flat
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
