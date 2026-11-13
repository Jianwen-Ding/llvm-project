#!/usr/bin/env python3
"""Find .entry functions whose body uses func_retval0 (never declared for
kernels), i.e. kernels with a non-void LLVM return type."""
import os, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptx_lift, tv

def scan(path):
    args = tv.llc_args(open(path).read())
    if args is None:
        return path, []
    p = subprocess.run([tv.BIN + "/llc"] + args + [path, "-o", "-"], capture_output=True, text=True)
    if p.returncode:
        return path, []
    out = []
    for name, f in ptx_lift.parse_module(p.stdout).items():
        if f.kind == "entry" and not f.ret and re.search(r"\bfunc_retval0\b", f.body):
            out.append(name)
    return path, out

with ProcessPoolExecutor(os.cpu_count()) as ex:
    n = 0
    for path, names in ex.map(scan, sys.argv[1:]):
        for nm in names:
            n += 1
            print("%s :: %s" % (os.path.basename(path), nm))
    print("kernels storing to an undeclared func_retval0: %d" % n)
