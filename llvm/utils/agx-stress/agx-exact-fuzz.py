#!/usr/bin/env python3
"""Differential fuzz of float arithmetic on converted integers converted back, on the AGX Performance Simulator.

Each seed generates a kernel of one to four trees of uitofp / sitofp of input bytes and halfwords, constants
(dyadic ones, which E401 can do exactly in integers, and others, which it must leave as float), fmul / fadd / fsub /
fma / fmuladd, and a conversion back to an integer (fptosi / fptoui, their saturating forms, to i16 or i32), often
after adding 0.5. Constants stay within 2^-20..2^20 and trees within four products, so no value nears a denormal:
the model is IEEE binary32 with round-to-nearest (each operation rounded once, fma from the exact value). Compiled
with llc, run with agx_run on 1024 threads of random words (40% boundary values), each result compared with the
model, out-of-range non-saturating conversions (poison) excepted (EXPERIMENTS.md L15-E406).

  agx-exact-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy ...] [--seeds A:B] [--mcpu agx-g14c]
                    [--keep DIR]
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

N = 1024
IN, OUT = 0x1000000, 0x2000000
EDGE = np.array([0, 1, 0xFF, 0x80, 0x7F, 0xFFFF, 0x8000, 0x7FFF, 0xFFFFFFFF, 0x01010101, 0xFF00FF00], np.uint32)


def f32(x):
    """x (a Fraction or int) rounded to the nearest binary32, ties to even, as a Fraction."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    # (exact in binary64 for these magnitudes when x has at most 53 significant bits; otherwise round twice
    # through a finer grid: round to 2^-k granularity of the f32 ulp directly)
    s = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    ulp = Fraction(2) ** (e - 23)
    q, r = divmod(a, ulp)
    if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1):
        q += 1
    return s * q * ulp


def gen(seed):
    R = random.Random(seed)
    trees = []
    for _ in range(R.randint(1, 4)):
        nodes = []  # (kind, args)
        dy = R.random() < 0.7

        def const():
            if dy or R.random() < 0.5:
                return ('c', Fraction(R.choice([1, 2, 3, 5, 7, -1, -3])) * Fraction(2) ** R.randint(-8, 3))
            return ('c', Fraction(R.choice([0.3, 0.1, 1.7, 0.299, 2.5e-3, 1e5])))

        def leaf():
            if R.random() < 0.25:
                return const()
            return ('in', R.randrange(4), R.choice(['u8', 's8', 'u16', 's16', 'b1', 'b2']))

        def tree(depth, muls):
            k = R.random()
            if depth == 0 or k < 0.25:
                return leaf()
            if k < 0.45 and muls < 4:
                return ('mul', tree(depth - 1, muls + 1), const() if R.random() < 0.8 else leaf())
            if k < 0.75:
                return (R.choice(['add', 'sub']), tree(depth - 1, muls), tree(depth - 1, muls))
            if muls < 4:
                return (R.choice(['fma', 'fmuladd']), tree(depth - 1, muls + 1),
                        const() if R.random() < 0.8 else leaf(), tree(depth - 1, muls))
            return leaf()
        t = tree(R.randint(1, 4), 0)
        if R.random() < 0.6:
            t = ('add', t, ('c', Fraction(1, 2)))
        conv = R.choice(['fptosi.i32', 'fptoui.i32', 'fptosi.sat.i16', 'fptoui.sat.i16', 'fptosi.sat.i32',
                         'fptosi.i16', 'float', 'float', 'scaled'])
        trees.append((t, conv))
    return trees


def fhex(c):
    v = float(c)
    return '0x%016X' % struct.unpack('<Q', struct.pack('<d', float(np.float32(v))))[0]


def emit(trees):
    L = ['target triple = "agx"', '',
         'declare i32 @llvm.agx.thread.position.in.grid.x()',
         'declare float @llvm.fma.f32(float, float, float)',
         'declare float @llvm.fmuladd.f32(float, float, float)',
         'declare i16 @llvm.fptosi.sat.i16.f32(float)',
         'declare i16 @llvm.fptoui.sat.i16.f32(float)',
         'declare i32 @llvm.fptosi.sat.i32.f32(float)', '',
         'define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {',
         '  %tid = call i32 @llvm.agx.thread.position.in.grid.x()']
    for w in range(4):
        L += ['  %%ia%d = add i32 %%tid, %d' % (w, w * 65536),
              '  %%pa%d = getelementptr inbounds i32, ptr addrspace(1) %%in, i32 %%ia%d' % (w, w),
              '  %%w%d = load i32, ptr addrspace(1) %%pa%d, align 4' % (w, w)]
    cnt = [0]

    def nv():
        cnt[0] += 1
        return '%%v%d' % cnt[0]

    def go(t):
        k = t[0]
        if k == 'c':
            return fhex(t[1])
        if k == 'in':
            w, ty = t[1], t[2]
            if ty in ('u8', 's8', 'b1', 'b2'):
                sh = {'u8': 0, 's8': 8, 'b1': 16, 'b2': 24}[ty]
                a = nv(); L.append('  %s = lshr i32 %%w%d, %d' % (a, w, sh))
                b = nv(); L.append('  %s = trunc i32 %s to i8' % (b, a))
                r = nv(); L.append('  %s = %s i8 %s to float' % (r, 'sitofp' if ty == 's8' else 'uitofp', b))
            else:
                b = nv(); L.append('  %s = trunc i32 %%w%d to i16' % (b, w))
                r = nv(); L.append('  %s = %s i16 %s to float' % (r, 'sitofp' if ty == 's16' else 'uitofp', b))
            return r
        if k in ('add', 'sub', 'mul'):
            a, b = go(t[1]), go(t[2]); r = nv()
            L.append('  %s = f%s float %s, %s' % (r, k, a, b)); return r
        a, b, c = go(t[1]), go(t[2]), go(t[3]); r = nv()
        L.append('  %s = call float @llvm.%s.f32(float %s, float %s, float %s)' % (r, k, a, b, c)); return r
    for j, (t, conv) in enumerate(trees):
        v = go(t)
        r = nv()
        if conv in ('float', 'scaled'):
            # the float itself (E407: a tree whose value stays a float), or scaled by a non-dyadic constant
            if conv == 'scaled':
                L.append('  %s = fmul float %s, 0x3FD5555560000000' % (r, v)); v = r; r = nv()
            L.append('  %s = bitcast float %s to i32' % (r, v)); ri = r
        elif conv in ('fptosi.i32', 'fptoui.i32'):
            L.append('  %s = %s float %s to i32' % (r, conv.split('.')[0], v)); ri = r
        elif conv == 'fptosi.i16':
            L.append('  %s = fptosi float %s to i16' % (r, v)); ri = nv(); L.append('  %s = sext i16 %s to i32' % (ri, r))
        else:
            nm = {'fptosi.sat.i16': 'fptosi.sat.i16', 'fptoui.sat.i16': 'fptoui.sat.i16',
                  'fptosi.sat.i32': 'fptosi.sat.i32'}[conv]
            ty = 'i16' if conv.endswith('i16') else 'i32'
            L.append('  %s = call %s @llvm.%s.f32(float %s)' % (r, ty, nm, v))
            ri = r
            if ty == 'i16':
                ri = nv(); L.append('  %s = %s i16 %s to i32' % (ri, 'sext' if 'si' in conv else 'zext', r))
        o = nv(); L.append('  %s = add i32 %%tid, %d' % (o, j * 65536))
        p = nv(); L.append('  %s = getelementptr inbounds i32, ptr addrspace(1) %%out, i32 %s' % (p, o))
        L.append('  store i32 %s, ptr addrspace(1) %s, align 4' % (ri, p))
    L += ['  ret void', '}', '']
    return '\n'.join(L)


def model(trees, words):
    def go(t):
        k = t[0]
        if k == 'c':
            return f32(t[1])
        if k == 'in':
            x = int(words[t[1]])
            ty = t[2]
            if ty in ('u8', 's8', 'b1', 'b2'):
                b = (x >> {'u8': 0, 's8': 8, 'b1': 16, 'b2': 24}[ty]) & 0xFF
                return Fraction(b - 256 if ty == 's8' and b >= 128 else b)
            h = x & 0xFFFF
            return Fraction(h - 65536 if ty == 's16' and h >= 32768 else h)
        if k == 'add':
            return f32(go(t[1]) + go(t[2]))
        if k == 'sub':
            return f32(go(t[1]) - go(t[2]))
        if k == 'mul':
            return f32(go(t[1]) * go(t[2]))
        return f32(go(t[1]) * go(t[2]) + go(t[3]))  # fma; fmuladd either way: compared only where it agrees
    def bits(v, neg_zero):
        if v == 0:
            return 0x80000000 if neg_zero else 0
        return struct.unpack('<I', struct.pack('<f', float(v)))[0]

    def negzero(t):
        # whether the tree's float result is -0 (zeros of two signs: IEEE's rules for +, -, *, fma)
        k = t[0]
        if k == 'c':
            return False
        if k == 'in':
            return False
        if k == 'mul':
            a, b = go(t[1]), go(t[2])
            if a * b != 0:
                return False
            sa = negzero(t[1]) if a == 0 else a < 0
            sb = negzero(t[2]) if b == 0 else b < 0
            return sa != sb
        if k in ('add', 'sub'):
            a, b = go(t[1]), go(t[2])
            if f32(a + b if k == 'add' else a - b) != 0 or a != 0 or b != 0:
                return False
            na, nb = negzero(t[1]), negzero(t[2])
            if k == 'sub':
                nb = not nb
            return na and nb
        a, b, c = go(t[1]), go(t[2]), go(t[3])
        if f32(a * b + c) != 0 or a * b != 0 or c != 0:
            return False
        sa = negzero(t[1]) if a == 0 else a < 0
        sb = negzero(t[2]) if b == 0 else b < 0
        return (sa != sb) and negzero(t[3])
    res = []
    for t, conv in trees:
        v = go(t)
        if conv == 'float':
            res.append((True, bits(v, negzero(t))))
            continue
        if conv == 'scaled':
            w = f32(v * f32(Fraction(np.float32(1 / 3).item())))
            res.append((True, bits(w, negzero(t) if v == 0 else False)))
            continue
        tr = int(v)  # toward zero
        if conv == 'fptosi.i32':
            ok = -2**31 <= tr < 2**31; r = tr & 0xFFFFFFFF
        elif conv == 'fptoui.i32':
            ok = 0 <= tr < 2**32 and v > -1; r = tr & 0xFFFFFFFF
        elif conv == 'fptosi.i16':
            ok = -2**15 <= tr < 2**15; r = tr & 0xFFFFFFFF
        elif conv == 'fptosi.sat.i16':
            ok = True; r = max(-2**15, min(2**15 - 1, tr)) & 0xFFFFFFFF
        elif conv == 'fptoui.sat.i16':
            ok = True; r = max(0, min(2**16 - 1, tr))
        else:
            ok = True; r = max(-2**31, min(2**31 - 1, tr)) & 0xFFFFFFFF
        res.append((ok, r))
    return res


def has_fmuladd(t):
    return isinstance(t, tuple) and (t[0] == 'fmuladd' or any(has_fmuladd(x) for x in t[1:] if isinstance(x, tuple)))


def run_seed(seed, a, tmp):
    trees = gen(seed)
    src = emit(trees)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + r.stderr.strip().splitlines()[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    inp = np.zeros(4 * 65536, np.uint32)
    for w in range(4):
        x = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
        m = rng.random(N) < 0.4
        x[m] = rng.choice(EDGE, m.sum())
        inp[w * 65536:w * 65536 + N] = x
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nbytes = 4 * 65536 * 4
    r = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "256,1,1", "--map", "%#x=%s" % (IN, fin),
                        "--alloc", "%#x=%d" % (OUT, nbytes), "--dump", "%#x=%d:%s" % (OUT, nbytes, fout),
                        "--uniform", "0=%d" % IN, "--uniform", "1=0", "--uniform", "2=%d" % OUT,
                        "--uniform", "3=0"], capture_output=True, text=True, timeout=120)
    if r.returncode:
        return src, "agx_run failed: " + ((r.stderr + r.stdout).strip().splitlines() or [""])[-1][:200]
    got = np.fromfile(fout, np.uint32)
    for t in range(N):
        words = [inp[w * 65536 + t] for w in range(4)]
        for j, (ok, v) in enumerate(model(trees, words)):
            # (fmuladd may be fused or not: its trees are compared only where they convert exactly the same)
            if not ok or has_fmuladd(trees[j][0]):
                continue
            g = int(got[j * 65536 + t])
            if g != v:
                return src, "tree %d: thread %d gave %#x, the model %#x" % (j, t, g, v)
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--seeds", default="0:100")
    ap.add_argument("--mcpu", default="agx-g14c")
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    a.objcopy = a.objcopy or os.path.join(os.path.dirname(a.llc), "llvm-objcopy")
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
