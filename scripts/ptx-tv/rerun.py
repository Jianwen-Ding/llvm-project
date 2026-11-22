#!/usr/bin/env python3
"""Re-check selected functions of a results file with the current lifter.

Usage: rerun.py RESULTS.jsonl OUT.jsonl [--status S ...] [-j N] [--smt-to MS]
Rows whose status is in --status (default: everything but `correct`) are
re-run; all other rows are copied.  The output has one row per function.
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tv  # noqa: E402


def work(item):
    path, funcs, opts = item
    ns = argparse.Namespace(root=opts["root"], workdir=opts["workdir"],
                            max_funcs=0, func=funcs, smt_to=opts["smt_to"],
                            keep=False)
    return tv.process_file(path, ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--status", action="append")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("--smt-to", type=int, default=5000)
    ap.add_argument("--root", default="/home/user/llvm-project/llvm/test/CodeGen/NVPTX")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    keep, redo = [], collections.defaultdict(list)
    for r in rows:
        st = r["status"]
        sel = (st in args.status) if args.status else (st != "correct")
        if "function" in r and sel:
            redo[r["file"]].append(r["function"])
        else:
            keep.append(r)
    opts = {"root": args.root, "workdir": args.workdir, "smt_to": args.smt_to}
    items = [(os.path.join(args.root, f), fs, opts) for f, fs in redo.items()]
    with open(args.out, "w") as out:
        for r in keep:
            out.write(json.dumps(r) + "\n")
        with ProcessPoolExecutor(args.j) as ex:
            futs = [ex.submit(work, it) for it in items]
            for fu in as_completed(futs):
                for r in fu.result():
                    out.write(json.dumps(r) + "\n")
                out.flush()
    print("re-ran %d functions in %d files" % (sum(map(len, redo.values())), len(redo)))


if __name__ == "__main__":
    main()
