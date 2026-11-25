#!/usr/bin/env python3
"""Audit proofs that involve directed rounding.

The lifter models .rz/.rm/.rp/.rna floating-point results with LLVM's
constrained operations, which Alive2 (following LangRef) makes poison unless
the free dynamic rounding mode matches.  A proved function whose PTX uses
two different directed modes would be poison everywhere on both sides, so
its proof would be vacuous.  This lists, for each proved function of a
results file, the directed modes its PTX uses, and reports any with more
than one.

Usage: rounding_audit.py results.jsonl [NVPTX test dir]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tv  # noqa: E402

FP = re.compile(r"^\s*(?:@!?%\w+\s+)?(?:add|sub|mul|fma|mad|div|rcp|sqrt|cvt)\.([\w.:]+)", re.M)


def main(results, root):
    corr = collections.defaultdict(set)
    for l in open(results):
        r = json.loads(l)
        if r.get("status") == "correct" and r["file"].endswith(".ll"):
            corr[r["file"]].add(r["function"])
    single, multi = collections.Counter(), []
    for f in sorted(corr):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        a = tv.llc_args(open(p).read())
        if a is None:
            continue
        rc, ptx, _ = tv.run([tv.BIN + "/llc"] + a + [p, "-o", "-"], timeout=300)
        if rc:
            continue
        for fm in re.finditer(r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\((.*?)^}", ptx, re.S | re.M):
            modes = {x for m in FP.finditer(fm.group(2)) for x in m.group(1).split(".")
                     if x in ("rz", "rm", "rp", "rna")}
            fn = [g for g in corr[f] if g == fm.group(1) or tv.ptx_name(g) == fm.group(1)]
            if not modes or not fn:
                continue
            if len(modes) > 1:
                multi.append((f, fn[0], sorted(modes)))
            else:
                single[next(iter(modes))] += 1
    print("proved functions with one directed mode:", dict(single))
    print("proved functions with several (vacuous):", len(multi))
    for m in multi:
        print("  ", *m)
    return 1 if multi else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llvm/test/CodeGen/NVPTX")))
