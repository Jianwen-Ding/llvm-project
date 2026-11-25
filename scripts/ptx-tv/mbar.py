"""mbarrier operations as uninterpreted calls with a canonical name.

The PTX opcode (mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64)
and the NVVM intrinsic (llvm.nvvm.mbarrier.try.wait.parity.relaxed.scope.
cluster.space.cta) are normalised to the same token set, so the check
covers instruction selection, modifiers and operands; mbarrier semantics
(cross-thread synchronisation) are not modelled.
"""
import re

SPLIT = {"arrive_drop": ["arrive", "drop"], "test_wait": ["test", "wait"],
         "try_wait": ["try", "wait"], "pending_count": ["pending", "count"],
         "expect_tx": ["expect", "tx"], "complete_tx": ["complete", "tx"],
         "noComplete": ["nocomplete"]}


def _name(tokens):
    t = set(tokens)
    # defaults that one side spells out and the other leaves implicit
    t -= {"scope_cta", "layout_v0", "tl"}
    if "arrive" in t:
        t.discard("release")
    if "wait" in t:
        t.discard("acquire")
    if "tx" in t and "arrive" not in t:
        t.discard("relaxed")
    return "ptx.mbarrier." + ".".join(sorted(t))


def ptx_key(opcode):
    parts = opcode.split(".")[1:]
    out = []
    for p in parts:
        if p in ("b64", "b32"):
            continue
        if p in ("shared", "shared::cta"):
            out.append("space_cta")
        elif p == "shared::cluster":
            out.append("space_cluster")
        elif p in ("cta", "cluster", "gpu", "sys"):
            out.append("scope_" + p)
        elif "::" in p:
            out.append(p.replace("::", "_"))
        else:
            out.extend(SPLIT.get(p, [p.lower()]))
    return _name(out)


def intrinsic_key(name, args=None):
    """(key, kept_arg_indices) for an llvm.nvvm.mbarrier.* intrinsic."""
    if not name.startswith("mbarrier."):
        return None
    args = args or []
    keep = list(range(len(args)))
    extra = []
    # init/check_layout take the layout as a trailing immediate
    if re.match(r"mbarrier\.(init|check\.layout|check_layout)", name) and args and \
            args[-1][0] == "i32" and re.fullmatch(r"\d+", args[-1][1]):
        extra.append("layout_v" + args[-1][1])
        keep = keep[:-1]
    k = _intrinsic_tokens(name) + extra
    return _name(k), keep


def _intrinsic_tokens(name):
    toks = name.split(".")[1:]
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("scope", "space") and i + 1 < len(toks):
            out.append("%s_%s" % (t, toks[i + 1]))
            i += 2
            continue
        if re.fullmatch(r"p\d+", t):   # overload suffix: p3 = shared space
            if t == "p3":
                out.append("space_cta")
            i += 1
            continue
        if t == "shared":
            out.append("space_cta")
        elif t == "check" and i + 1 < len(toks) and toks[i + 1] == "layout":
            out.append("check_layout")
            i += 2
            continue
        else:
            out.extend(SPLIT.get(t, [t.lower()]))
        i += 1
    return out
