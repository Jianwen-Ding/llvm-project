#!/usr/bin/env python3
"""Random device atomic additions on the AGX simulator, against a model.

Each seed is a kernel whose threads add a value to a word of a buffer with
atomicrmw add (32-bit, monotonic), K times each, their addresses from an index
buffer of one of several patterns: every thread its own word, one word for
all, neighbouring threads in pairs, equal words that are never neighbours (a
stride), a few words at random, or a mix. The old values are stored. Compiled
with llc, run with agx_run on 1024 threads; every word's final value must be
its start plus the sum of the values added to it, and the old values of a
word's additions must be a serialization of them: sorted, each the one before
plus that addition's value (values from 1, no wrap).

Usage:
  agx-atomic-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy ...] [--seeds A:B]
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

N = 1024
WORDS = 2048
IDX, VAL, ACC, OLD = 0x10000000, 0x20000000, 0x30000000, 0x40000000
PATTERNS = ("own", "one", "pairs", "stride", "few", "mixed")


def gen(seed):
    rng = np.random.default_rng(seed)
    return {'pattern': PATTERNS[seed % len(PATTERNS)], 'k': int(rng.integers(1, 4)),
            'used': bool(rng.integers(0, 4)), 'guard': bool(rng.integers(0, 3) == 0)}


def emit(k):
    L = ['target triple = "agx"',
         'declare i32 @llvm.agx.thread.position.in.grid.x()',
         'define agx_kernel void @k(ptr addrspace(1) %idx, ptr addrspace(1) %val, '
         'ptr addrspace(1) %acc, ptr addrspace(1) %old) '
         '"agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" {',
         'entry:',
         '  %tid = call i32 @llvm.agx.thread.position.in.grid.x()']
    last = 'entry'
    for r in range(k['k']):
        L += ['  %%ix%d = mul nuw nsw i32 %%tid, %d' % (r, k['k']),
              '  %%ia%d = add nuw nsw i32 %%ix%d, %d' % (r, r, r),
              '  %%pi%d = getelementptr i32, ptr addrspace(1) %%idx, i32 %%ia%d' % (r, r),
              '  %%i%d = load i32, ptr addrspace(1) %%pi%d, align 4' % (r, r),
              '  %%pv%d = getelementptr i32, ptr addrspace(1) %%val, i32 %%ia%d' % (r, r),
              '  %%v%d = load i32, ptr addrspace(1) %%pv%d, align 4' % (r, r),
              '  %%pa%d = getelementptr i32, ptr addrspace(1) %%acc, i32 %%i%d' % (r, r)]
        if k['guard']:
            # (threads whose value is a multiple of 7 skip it: a divergent atomic)
            L += ['  %%m%d = urem i32 %%v%d, 7' % (r, r),
                  '  %%g%d = icmp ne i32 %%m%d, 0' % (r, r),
                  '  br i1 %%g%d, label %%do%d, label %%skip%d' % (r, r, r),
                  'do%d:' % r]
        L += ['  %%o%d = atomicrmw add ptr addrspace(1) %%pa%d, i32 %%v%d monotonic, align 4' % (r, r, r)]
        if k['used']:
            L += ['  %%po%d = getelementptr i32, ptr addrspace(1) %%old, i32 %%ia%d' % (r, r),
                  '  store i32 %%o%d, ptr addrspace(1) %%po%d, align 4' % (r, r)]
        if k['guard']:
            L += ['  br label %%skip%d' % r, 'skip%d:' % r]
    L += ['  ret void', '}']
    return '\n'.join(L) + '\n'


def indices(k, rng):
    n = N * k['k']
    p = k['pattern']
    t = np.arange(n)
    if p == 'own':
        ix = t % WORDS
    elif p == 'one':
        ix = np.full(n, int(rng.integers(0, WORDS)))
    elif p == 'pairs':
        ix = (t // 2) % WORDS
    elif p == 'stride':
        ix = (t % 16) * 3 + 5
    elif p == 'few':
        ix = rng.integers(0, 5, n)
    else:
        ix = np.where(rng.random(n) < 0.5, t % WORDS, rng.integers(0, 3, n))
    return ix.astype(np.uint32)


def run_seed(seed, a, tmp):
    k = gen(seed)
    src = emit(k)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or [""])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed + 1)
    n = N * k['k']
    ix = indices(k, rng)
    val = rng.integers(1, 1000, n).astype(np.uint32)
    acc = rng.integers(0, 1000, WORDS).astype(np.uint32)
    old = np.zeros(n, np.uint32)
    files = {}
    for nm, arr in (("idx", ix), ("val", val), ("acc", acc), ("old", old)):
        files[nm] = os.path.join(tmp, nm + ".bin")
        arr.tofile(files[nm])
    fa, fo = os.path.join(tmp, "acc.out"), os.path.join(tmp, "old.out")
    try:
        r = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "256,1,1",
                            "--map", "%#x=%s" % (IDX, files["idx"]), "--map", "%#x=%s" % (VAL, files["val"]),
                            "--map", "%#x=%s" % (ACC, files["acc"]), "--map", "%#x=%s" % (OLD, files["old"]),
                            "--dump", "%#x=%d:%s" % (ACC, acc.nbytes, fa),
                            "--dump", "%#x=%d:%s" % (OLD, old.nbytes, fo),
                            "--uniform", "0=%d" % IDX, "--uniform", "1=0", "--uniform", "2=%d" % VAL,
                            "--uniform", "3=0", "--uniform", "4=%d" % ACC, "--uniform", "5=0",
                            "--uniform", "6=%d" % OLD, "--uniform", "7=0"],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, "agx_run timed out"
    if r.returncode:
        return src, "agx_run failed: " + ((r.stderr + r.stdout).strip().splitlines() or [""])[-1][:200]
    got_acc, got_old = np.fromfile(fa, np.uint32), np.fromfile(fo, np.uint32)
    done = (val % 7 != 0) if k['guard'] else np.ones(n, bool)
    ref = acc.astype(np.int64).copy()
    np.add.at(ref, ix[done], val[done].astype(np.int64))
    bad = np.nonzero(got_acc.astype(np.int64) != ref)[0]
    if bad.size:
        e = int(bad[0])
        return src, "word %d: %d, the model %d (%d words differ)" % (e, int(got_acc[e]), int(ref[e]), bad.size)
    if k['used']:
        for w in np.unique(ix[done]):
            sel = np.nonzero(done & (ix == w))[0]
            o = got_old[sel].astype(np.int64)
            order = np.argsort(o, kind='stable')
            o, v = o[order], val[sel][order].astype(np.int64)
            if o[0] != acc[w] or np.any(o[1:] != o[:-1] + v[:-1]):
                return src, "word %d: its old values are no serialization of its additions" % w
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", required=True)
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default="llvm-objcopy")
    ap.add_argument("--mcpu", default="agx-g14c")
    ap.add_argument("--seeds", default="0:60")
    ap.add_argument("--keep", default=None, help="directory to keep failing kernels in")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(lo, hi):
            src, err = run_seed(seed, a, tmp)
            if err:
                fails += 1
                print("seed %d: %s" % (seed, err), flush=True)
                if a.keep:
                    os.makedirs(a.keep, exist_ok=True)
                    open(os.path.join(a.keep, "atomic-%d.ll" % seed), "w").write(src)
    print("%d seeds, %d failing" % (hi - lo, fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
