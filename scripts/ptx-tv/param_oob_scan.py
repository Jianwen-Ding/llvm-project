#!/usr/bin/env python3
"""Scan llc output for .param accesses outside the declared parameter size.

For every test file, compile with its first nvptx64 llc RUN line and report
each ld.param/st.param whose [sym+off] access extends past the size of the
.param it names (function params, return value, and call-site params).
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptx_lift  # noqa: E402
import tv  # noqa: E402


def scan(path):
    text = open(path).read()
    args = tv.llc_args(text)
    if args is None:
        return path, None, []
    try:
        p = subprocess.run([tv.BIN + "/llc"] + args + [path, "-o", "-"],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return path, args, []
    if p.returncode:
        return path, args, []
    res = []
    for name in ptx_lift.parse_module(p.stdout):
        oob = tv.param_oob(p.stdout, name)
        if oob:
            res.append((name, oob))
    return path, args, res


def main():
    files = sys.argv[1:]
    total = 0
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for path, args, res in ex.map(scan, files):
            for name, oob in res:
                total += 1
                print("%s :: %s  [llc %s]" % (os.path.basename(path), name, " ".join(args)))
                for o in oob[:8]:
                    print("    " + o)
    print("functions with out-of-bounds .param accesses: %d" % total)


if __name__ == "__main__":
    main()
