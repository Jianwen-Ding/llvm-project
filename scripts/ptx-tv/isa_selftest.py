#!/usr/bin/env python3
"""Check the lifter's model of PTX instructions the backend does not emit
(so the test suite cannot reach them) against reference IR written from the
PTX ISA text.

Each case is isa_tests/NAME.ll (functions whose bodies are inline PTX, which
llc copies into its output) and isa_tests/NAME.ref.ll (the same functions as
plain IR following the ISA's description).  The PTX from the first is lifted
and Alive2 checks that it refines the second, as tv.py does for the suite.

Usage: isa_selftest.py [-j N] [NAME ...]   (prints one JSON line per function)
"""
import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tv  # noqa: E402

LLC = ["-mtriple=nvptx64", "-mcpu=sm_90", "-mattr=+ptx80"]


def case(name, smt_to=10000):
    asm = os.path.join(HERE, "isa_tests", name + ".ll")
    ref = os.path.join(HERE, "isa_tests", name + ".ref.ll")
    wd = tempfile.mkdtemp(prefix="isa_")
    tv.LLC_FLAGS[os.getpid()] = " ".join(LLC)
    tv.PROBE_LLC[os.getpid()] = LLC
    rc, ptx, err = tv.run([tv.BIN + "/llc"] + LLC + [asm, "-o", "-"])
    if rc:
        return [{"case": name, "status": "llc-failed", "detail": err[-300:]}]
    src_path, src_text = tv.prepare_source(ref, wd)
    defines, decls, globals_, typedefs = tv.module_symbols(src_text)
    tv.RENAMES.clear()
    tv.RENAMES.update(tv.nvptx_renames(src_text))
    name_map = tv.build_name_map(defines, decls, globals_)
    out = []
    for n, (line, sig) in defines.items():
        r = tv.check_function(n, line, sig, src_path, src_text, ptx, name_map, decls,
                              globals_, wd, smt_to, False, typedefs)
        r["case"] = name
        out.append({k: v for k, v in r.items() if k != "detail" or r["status"] != "correct"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    names = a.names or sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "isa_tests"))
                              if f.endswith(".ll") and not f.endswith(".ref.ll"))
    bad = 0
    with ProcessPoolExecutor(a.j) as ex:
        for rs in ex.map(case, names):
            for r in rs:
                bad += r["status"] != "correct"
                print(json.dumps(r))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
