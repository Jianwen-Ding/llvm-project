#!/usr/bin/env python3
"""The code llvm-air2agx makes from AIR, against the other ways this repository compiles AIR for AGX.

Every kernel of the AIR hardware suite (llvm/utils/air-gpu/air_gpu_suite.py, one module per case) is compiled:

  portable  llc -mtriple=air -air-portable, then llc -mtriple=agx (the AIR the AGX backend was built for)
  llc-air   the .air in the .metallib the AIR target writes for Metal, by llc -mtriple=agx (llc reads .air)
  air2agx   that .metallib by llvm-air2agx (agx-lower-air, -O3 with the AGX passes, the code generator)

and costed by the AGX Performance Simulator's agx_cost (one SIMD-group, zero launch inputs: the suite's kernels are
straight-line per thread), with the instructions (without trap padding), waits, GPRs and stack bytes of each.

  air2agx_compare.py --bin build/bin --cost agx_cost [--mcpu agx-g14g] [-j N] [--json out.json] [--group G ...]
"""

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, "..", "..", "air-gpu", "air_gpu_suite.py")
PATHS = ["portable", "llc-air", "air2agx"]


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def asm_metrics(asm):
    m = {}
    for key, pat in (("gprs", r"general purpose registers: (\d+)"), ("stack", r"stack size: (\d+)")):
        r = re.search(pat, asm)
        m[key] = int(r.group(1)) if r else 0
    insts = [l.strip() for l in asm.splitlines()
             if l.startswith("\t") and not l.strip().startswith(("//", ".")) and l.strip() != "trap"]
    m["instructions"] = len(insts)
    m["waits"] = sum(1 for i in insts if i.startswith("wait"))
    return m


def metallib_air(lib, out):
    d = open(lib, "rb").read()
    off, size = struct.unpack_from("<QQ", d, 72)
    open(out, "wb").write(d[off:off + size])


def cost(costbin, code):
    try:
        r = run([costbin, code, "--allow-unknown"])
        return json.loads(r.stdout.strip().splitlines()[-1]).get("cycles")
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def one(args, name, ir):
    b = args.bin
    with tempfile.TemporaryDirectory() as t:
        ll = os.path.join(t, "k.ll")
        open(ll, "w").write(ir)
        res = {}
        try:
            run([b + "/llc", "-mtriple=air", "-filetype=obj", ll, "-o", t + "/k.metallib"])
            metallib_air(t + "/k.metallib", t + "/k.air")
            run([b + "/llc", "-mtriple=air", "-air-portable", ll, "-o", t + "/k.portable.ll"])
        except subprocess.CalledProcessError as e:
            return name, {"error": e.stderr.strip()[-300:]}
        jobs = {
            "portable": [[b + "/llc", "-mtriple=agx", "-mcpu=" + args.mcpu, t + "/k.portable.ll", "-o", t + "/p.s"],
                         [b + "/llc", "-mtriple=agx", "-mcpu=" + args.mcpu, "-filetype=obj", t + "/k.portable.ll",
                          "-o", t + "/p.o"]],
            "llc-air": [[b + "/llc", "-mtriple=agx", "-mcpu=" + args.mcpu, t + "/k.air", "-o", t + "/l.s"],
                        [b + "/llc", "-mtriple=agx", "-mcpu=" + args.mcpu, "-filetype=obj", t + "/k.air",
                         "-o", t + "/l.o"]],
            "air2agx": [[b + "/llvm-air2agx", "-mcpu=" + args.mcpu, "--emit=asm", t + "/k.metallib", "-o", t + "/a.s"],
                        [b + "/llvm-air2agx", "-mcpu=" + args.mcpu, t + "/k.metallib", "-o", t + "/a.o"]],
        }
        for path, cmds in jobs.items():
            tag = path[0]
            try:
                for c in cmds:
                    run(c)
                run([b + "/llvm-objcopy", "-O", "binary", "--only-section=.text", t + "/%s.o" % tag,
                     t + "/%s.bin" % tag])
            except subprocess.CalledProcessError as e:
                res[path] = {"error": e.stderr.strip()[-300:]}
                continue
            m = asm_metrics(open(t + "/%s.s" % tag).read())
            m["cycles"] = cost(args.cost, t + "/%s.bin" % tag)
            res[path] = m
        return name, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the LLVM build's bin directory")
    ap.add_argument("--cost", required=True, help="agx_cost")
    ap.add_argument("--mcpu", default="agx-g14g")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("--group", action="append")
    ap.add_argument("--json")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("air_gpu_suite", SUITE)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    work = []
    for g in args.group or sorted(suite.GROUPS):
        for c in suite.GROUPS[g]():
            if not c.metal_only and not c.absent:
                work.append(("%s/%s" % (g, c.name), suite.module_ir([c])))

    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.j) as ex:
        for name, res in ex.map(lambda w: one(args, *w), work):
            results[name] = res
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)

    def ok(r, p):
        return p in r and "error" not in r[p] and r[p].get("cycles")

    print("%d kernels" % len(results))
    for p in PATHS:
        print("  %-9s compiled and costed: %d" % (p, sum(1 for r in results.values() if ok(r, p))))
    for base in ("portable", "llc-air"):
        both = [r for r in results.values() if ok(r, base) and ok(r, "air2agx")]
        if not both:
            continue
        print("air2agx against %s (%d kernels both compile):" % (base, len(both)))
        for metric in ("cycles", "instructions", "waits", "gprs", "stack"):
            a = sum(r["air2agx"][metric] for r in both)
            b = sum(r[base][metric] for r in both)
            better = sum(1 for r in both if r["air2agx"][metric] < r[base][metric])
            worse = sum(1 for r in both if r["air2agx"][metric] > r[base][metric])
            ratio = [r["air2agx"][metric] / r[base][metric] for r in both if r[base][metric]]
            print("  %-12s total %8d -> %8d (%+.1f%%), geomean ratio %.3f, %d better, %d worse" % (
                metric, b, a, 100.0 * (a - b) / b if b else 0.0,
                statistics.geometric_mean(ratio) if ratio and all(x > 0 for x in ratio) else float("nan"),
                better, worse))
    return 0


if __name__ == "__main__":
    sys.exit(main())
