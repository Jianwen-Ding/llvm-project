#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's row loops that load and store device memory, against a Python model, on
the AGX Performance Simulator.

Each seed generates a kernel whose threads each update a row of 256 elements of an output buffer b (32-, 16- or
8-bit; or a column, element (i + o) * 1024 + tid): a loop of one to three statements b[x + i + o] = f(b[x + i + o'],
c[...]) in program order, x the thread's row, i the counter, the offsets small constants (so a statement may read
what an earlier one, or an earlier iteration, stored). c is an input buffer at i + o'' (every thread's same
elements) or x + i + o'', or b itself at x + i + o'' (one buffer passed twice: alias versioning's overlapping copy).
The trip is a constant or an input word's low bits, the loop guarded or not; f is an exact integer op of the
element's width (add, sub, xor, or, and, a multiply or shift by a constant, umax). Compiled with llc (and every other seed with opt -O2 first), run with agx_run on 1024 threads
over random rows and compared word for word with the model: alias and alignment versioning, the unroller's
runtime remainders, LSR, word indices and rebased indices (L15-E769: a gep a load and a store shared, a crash)
with stores in the loop.

  agx-rmw-fuzz.py --llc build/bin/llc --run path/to/agx_run [--opt build/bin/opt] [--objcopy ...] [--seeds A:B]
                  [--mcpu agx-g14c] [--keep DIR]

Exits with 1 if a kernel fails to compile or gives a wrong result (kept in DIR with --keep).
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

M = 0xffffffff
N = 1024
ROW = 256
IN, OUT = 0x1000000, 0x2000000
WORDS = 65536 * 4


def gen(seed):
    R = random.Random(seed)
    alias = R.random() < 0.3
    k = dict(alias=alias,
             w=R.choice([32, 32, 32, 16, 8]),
             col=R.random() < 0.25,
             # (with b passed as c, b's first word may change under the
             # other threads' loads: a constant trip)
             trip=R.choice(['const'] if alias else ['const', 'const', 'input']),
             tconst=R.choice([1, 3, 8, 16, 17, 33, 64, 100, 128]),
             tmask=R.choice([7, 31, 63, 127]),
             tadd=R.choice([0, 1, 5]),
             guarded=R.random() < 0.6,
             step=R.choice([1, 1, 1, 2]),
             stmts=[])
    for _ in range(R.randint(1, 3)):
        k['stmts'].append(dict(
            st=R.randrange(0, 64),                     # b's store offset
            lb=R.choice([None, R.randrange(0, 64)]),   # b's load offset (None: no b load)
            same=R.random() < 0.5,                     # b's load at the store's address
            cw=R.choice(['row', 'shared'] if not alias else ['row']),
            lc=R.randrange(0, 64),
            op=R.choice(['add', 'sub', 'xor', 'or', 'and', 'mul', 'shl', 'lshr', 'umax']),
            k=R.choice([1, 3, 7, 0x9E3779B1, 0xFFFF0000]),
        ))
        s = k['stmts'][-1]
        if s['same']:
            s['lb'] = s['st']
    return k


def emit(k):
    w = k['w']
    T, al = 'i%d' % w, w // 8
    L = ['target triple = "agx"',
         'declare i32 @llvm.agx.thread.position.in.grid.x()',
         'declare %s @llvm.umax.%s(%s, %s)' % (T, T, T, T),
         'define agx_kernel void @k(ptr addrspace(1) %c, ptr addrspace(1) %b) '
         '"agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" {',
         'entry:',
         '  %tid = call i32 @llvm.agx.thread.position.in.grid.x()',
         '  %x = shl nuw nsw i32 %tid, 8']
    if k['trip'] == 'const':
        L.append('  %%t = add i32 0, %d' % k['tconst'])
    else:
        L += ['  %tw = load i32, ptr addrspace(1) %c, align 4',
              '  %%tm = and i32 %%tw, %d' % k['tmask'],
              '  %%t = add nuw nsw i32 %%tm, %d' % k['tadd']]
    if k['guarded']:
        L += ['  %g = icmp ugt i32 %t, 0', '  br i1 %g, label %loop, label %exit']
    else:
        L += ['  br label %loop']
    L += ['loop:', '  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]']
    for n, s in enumerate(k['stmts']):
        def addr(base, off, own, nm):
            idx = '%%%s.i' % nm
            if own and k['col']:
                # a column: (i + o) * 1024 + tid
                L.append('  %%%s.r = add nuw nsw i32 %%i, %d' % (nm, off))
                L.append('  %%%s.m = mul nuw nsw i32 %%%s.r, %d' % (nm, nm, N))
                L.append('  %s = add nuw nsw i32 %%%s.m, %%tid' % (idx, nm))
            elif own:
                L.append('  %%%s.r = add nuw nsw i32 %%x, %%i' % nm)
                L.append('  %s = add nuw nsw i32 %%%s.r, %d' % (idx, nm, off))
            else:
                L.append('  %s = add nuw nsw i32 %%i, %d' % (idx, off))
            L.append('  %%%s.z = zext i32 %s to i64' % (nm, idx))
            L.append('  %%%s.p = getelementptr inbounds %s, ptr addrspace(1) %s, i64 %%%s.z' % (nm, T, base, nm))
            return '%%%s.p' % nm
        pc = addr('%b' if k['alias'] else '%c', s['lc'], s['cw'] == 'row', 'c%d' % n)
        L.append('  %%vc%d = load %s, ptr addrspace(1) %s, align %d' % (n, T, pc, al))
        a = '%%vc%d' % n
        pst = addr('%b', s['st'], True, 's%d' % n)
        kk = s['k'] & ((1 << w) - 1)
        if s['lb'] is not None:
            pb = pst if s['same'] else addr('%b', s['lb'], True, 'b%d' % n)
            L.append('  %%vb%d = load %s, ptr addrspace(1) %s, align %d' % (n, T, pb, al))
            bv = '%%vb%d' % n
        else:
            bv = str(kk)
        op, r = s['op'], '%%r%d' % n
        if op in ('add', 'sub', 'xor', 'or', 'and'):
            L.append('  %s = %s %s %s, %s' % (r, op, T, bv, a))
        elif op == 'mul':
            L.append('  %%m%d = mul %s %s, %d' % (n, T, a, kk))
            L.append('  %s = add %s %s, %%m%d' % (r, T, bv, n))
        elif op in ('shl', 'lshr'):
            L.append('  %%m%d = %s %s %s, %d' % (n, op, T, a, s['k'] & (w - 1)))
            L.append('  %s = xor %s %s, %%m%d' % (r, T, bv, n))
        else:
            L.append('  %s = call %s @llvm.umax.%s(%s %s, %s %s)' % (r, T, T, T, bv, T, a))
        L.append('  store %s %s, ptr addrspace(1) %s, align %d' % (T, r, pst, al))
    L += ['  %%i1 = add nuw nsw i32 %%i, %d' % k['step'],
          '  %e = icmp ult i32 %i1, %t',
          '  br i1 %e, label %loop, label %exit',
          'exit:', '  ret void', '}']
    return '\n'.join(L) + '\n'


def dtype(w):
    return {32: np.uint32, 16: np.uint16, 8: np.uint8}[w]


def model(k, t, cin, bout):
    """b's elements (bout, updated in place) after every thread's loop of trip t, cin c's elements (b itself where
    aliased)."""
    w = k['w']
    Mw = (1 << w) - 1
    b = bout.astype(np.uint64)
    c = b if k['alias'] else cin.astype(np.uint64)
    if t == 0 and k['guarded']:
        return bout
    tid = np.arange(N, dtype=np.int64)
    x = tid * ROW
    def own(i, o):
        return (i + o) * N + tid if k['col'] else x + i + o
    i = 0
    while True:
        for s in k['stmts']:
            ci = own(i, s['lc']) if s['cw'] == 'row' else np.full(N, i + s['lc'])
            a = c[ci].copy()
            kk = s['k'] & Mw
            bv = b[own(i, s['lb'])].copy() if s['lb'] is not None else np.full(N, kk, np.uint64)
            op = s['op']
            if op == 'add':
                r = bv + a
            elif op == 'sub':
                r = bv - a
            elif op == 'xor':
                r = bv ^ a
            elif op == 'or':
                r = bv | a
            elif op == 'and':
                r = bv & a
            elif op == 'mul':
                r = bv + ((a * kk) & Mw)
            elif op == 'shl':
                r = bv ^ ((a << (s['k'] & (w - 1))) & Mw)
            elif op == 'lshr':
                r = bv ^ (a >> (s['k'] & (w - 1)))
            else:
                r = np.maximum(bv, a)
            b[own(i, s['st'])] = r & Mw
        i += k['step']
        if not i < t:   # (unsigned: i and t are small)
            break
    return b.astype(dtype(w))


def run_seed(seed, a, tmp):
    k = gen(seed)
    src = emit(k)
    ll, ll2, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", "o.ll", ".o", ".bin"))
    open(ll, "w").write(src)
    inp_ll = ll
    if seed % 2 and a.opt:
        r = subprocess.run([a.opt, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O2", ll, "-S", "-o", ll2],
                           capture_output=True, text=True)
        if r.returncode:
            return src, "opt failure: " + (r.stderr.strip().splitlines() or [""])[0][:200]
        inp_ll = ll2
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", inp_ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or [""])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    w = k['w']
    cin = rng.integers(0, 2**32, WORDS, dtype=np.uint64).astype(np.uint32)
    bw = rng.integers(0, 2**32, N * ROW * w // 32, dtype=np.uint64).astype(np.uint32)
    m = rng.random(bw.size) < 0.2
    bw[m] = rng.choice(np.array([0, 1, M, 0x80000000, 0x7fffffff], np.uint32), m.sum())
    bout = bw.view(dtype(w))
    fin, fb, fout = (os.path.join(tmp, n) for n in ("in.bin", "b.bin", "out.bin"))
    cin.tofile(fin)
    bout.tofile(fb)
    nbytes = bout.nbytes
    cva = OUT if k['alias'] else IN
    try:
        r = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "256,1,1", "--map", "%#x=%s" % (IN, fin),
                            "--map", "%#x=%s" % (OUT, fb), "--dump", "%#x=%d:%s" % (OUT, nbytes, fout),
                            "--uniform", "0=%d" % cva, "--uniform", "1=0", "--uniform", "2=%d" % OUT,
                            "--uniform", "3=0"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return src, "agx_run timed out"
    if r.returncode:
        return src, "agx_run failed: " + ((r.stderr + r.stdout).strip().splitlines() or [""])[-1][:200]
    got = np.fromfile(fout, dtype(w))
    t = k['tconst'] if k['trip'] == 'const' else (int(cin[0]) & k['tmask']) + k['tadd']
    ref = model(k, t, bout if k['alias'] else cin.view(dtype(w)), bout.copy())
    bad = np.nonzero(got != ref)[0]
    if bad.size:
        e = int(bad[0])
        return src, "element %d of %d bits: %#x, the model %#x (%d elements differ)" % (
            e, w, int(got[e]), int(ref[e]), bad.size)
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--opt", default=None, help="opt for the -O2 seeds (default: next to llc)")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--seeds", default="0:100")
    ap.add_argument("--mcpu", default="agx-g14c")
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    d = os.path.dirname(a.llc)
    a.objcopy = a.objcopy or os.path.join(d, "llvm-objcopy")
    if a.opt is None and os.path.exists(os.path.join(d, "opt")):
        a.opt = os.path.join(d, "opt")
    lo, hi = (int(x) for x in a.seeds.split(":"))
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(lo, hi):
            src, err = run_seed(seed, a, tmp)
            if err:
                bad += 1
                print("seed %d: %s" % (seed, err))
                if a.keep:
                    os.makedirs(a.keep, exist_ok=True)
                    open(os.path.join(a.keep, "seed%d.ll" % seed), "w").write(src)
    print("%d seeds, %d failing" % (hi - lo, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
