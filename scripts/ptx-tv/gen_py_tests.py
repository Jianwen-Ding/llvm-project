#!/usr/bin/env python3
"""Materialise the NVPTX tests that lit generates from Python scripts
(`wmma*.py`): run each script with the arguments of its `%python` RUN line,
prepend the first plain `llc` RUN line (with %t... replaced by %s) so that
tv.py can process the result, and write a function list that checks each
distinct function once, in the file with the newest PTX version.

Usage: gen_py_tests.py OUTDIR [NVPTX test dir]
Writes OUTDIR/<name>.ll and OUTDIR/funcs.json ({file: [functions]})."""
import json
import os
import re
import subprocess
import sys


def runs(text):
    out, cur = [], ""
    for line in text.splitlines():
        m = re.match(r"#\s*RUN:\s?(.*)", line)
        if not m:
            continue
        s = m.group(1).rstrip()
        if s.endswith("\\"):
            cur += s[:-1] + " "
            continue
        out.append(cur + s)
        cur = ""
    return out


def main(outdir, testdir):
    os.makedirs(outdir, exist_ok=True)
    best = {}
    for f in sorted(os.listdir(testdir)):
        if not (f.startswith("wmma") and f.endswith(".py")):
            continue
        rl = runs(open(os.path.join(testdir, f)).read())
        py = next((r for r in rl if r.startswith("%python %s")), None)
        llc = next((r for r in rl if r.startswith("llc <") and "%ptxas" not in r), None)
        if not py or not llc:
            continue
        args = py[len("%python %s"):].split(">")[0].split()
        ir = subprocess.run([sys.executable, os.path.join(testdir, f)] + args,
                            capture_output=True, text=True, check=True).stdout
        llc = re.sub(r"%t[-\w.]*", "%s", llc.split("|")[0]).strip() + " | FileCheck %s"
        name = f[:-3] + ".ll"
        open(os.path.join(outdir, name), "w").write("; RUN: " + llc + "\n" + ir)
        ptx = int(re.search(r"--ptx=(\d+)", py).group(1))
        for fn in re.findall(r"^define [^@]*@([\w.]+)\(", ir, re.M):
            if fn not in best or ptx > best[fn][0]:
                best[fn] = (ptx, name)
    sel = {}
    for fn, (_, name) in sorted(best.items()):
        sel.setdefault(name, []).append(fn)
    json.dump(sel, open(os.path.join(outdir, "funcs.json"), "w"), indent=0)
    print(len(best), "functions in", len(sel), "files")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llvm/test/CodeGen/NVPTX"))
