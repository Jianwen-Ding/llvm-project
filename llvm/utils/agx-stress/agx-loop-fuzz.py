#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's loop induction code against a Python model, on the AGX Performance Simulator.

Each seed generates a kernel of one to three strided loops, the shape MLIR's scf.for from a lane in steps of s
lowers to: a counter c = 0, s, 2s, ... (i32, or i64 against a zero-extended trip) beside an index i = start + c
that a load and a running hash use; the trip t a bounded unsigned minimum of an input, the exit c' <u t (or c' != t
for a trip that is a multiple of s); a guarded do-while or a plain while loop. Starts are thread positions
(bounded), shifted and offset ones, or input words that wrap; steps are powers of two and not, constants or an input's
(a & 63) + 1 (invariant, E417's merged counters); some hashes take a mask c + (tid & 31) <= (b & 2047) of
the counter in its own width (E426), some an i32 twin of the counter stepped by s or another constant
(E431's congruent counters), some a constant trip that opt unrolls (E439's counters down by the factor). Each
loop's hash is
stored per thread. Compiled with llc (and every other seed with opt -O2 first), run with agx_run on 1024 threads
of random words (40% boundary values) and compared with the model: this covers AGXNarrowInductions (narrowed
counters, E386's up-counters merged into their index, E206's down-counters), loop strength reduction and the
structurizer's loop exits. Some loops also keep a running maximumnum with the PTX front end's default-NaN select,
from a number or an input word (E388's NaN tests folded through the loop's phi), whose bits are stored
(EXPERIMENTS.md L15-E393).

  agx-loop-fuzz.py --llc build/bin/llc --run path/to/agx_run [--opt build/bin/opt] [--objcopy ...] [--seeds A:B]
                   [--mcpu agx-g14c] [--keep DIR]

Exits with 1 if a kernel fails to compile or gives a wrong result (kept in DIR with --keep).
"""

import argparse
import os
import random
import subprocess
import sys
import struct
import tempfile

import numpy as np

M = 0xffffffff
N = 1024
IN, OUT = 0x1000000, 0x2000000
WORDS = 4 * 65536          # input words (the loads index them masked)
EDGE = np.array([0, 1, 2, 3, 31, 32, 33, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000000, 0x7FFFFFFF, 0xFFFFFFE0, 0x100,
                 0xFF, 1000, 1023, 1024, 4096], np.uint32)


def gen(seed):
    R = random.Random(seed)
    loops = []
    for _ in range(R.randint(1, 3)):
        loops.append(dict(
            start=R.choice(['tid', 'tid_shl', 'row_lane', 'a', 'a_tid', 'neg']),
            shl=R.choice([1, 3, 5, 10]),
            k=R.choice([1, 7, 0x9E3779B1, 0xFFFFFFF0, 256]),
            step=R.choice([1, 2, 3, 4, 8, 16, 24, 32, 32, 64, 128]),
            lim=R.choice([1, 7, 31, 32, 100, 1024, 3000]),
            mask=R.choice([0xff, 0x3ff, 0xfff, 0xffffffff]),
            wide=R.random() < 0.6,
            exact=R.random() < 0.25,
            guarded=R.random() < 0.7,
            usec=R.random() < 0.15,
            fmax=R.random() < 0.4,
            fstart=R.choice(['num', 'a']),
            vstep=R.random() < 0.3,  # s = (a & 63) + 1, invariant, not a constant (E417)
            cmask=R.random() < 0.35,  # (c + (tid & 31) <= (b & 2047)) ? 1 : 0 into the hash (E426)
            # an i32 twin d = 0, e, 2e, ... of the counter into the hash, e = s or not (E431)
            twin=R.random() < 0.35,
            tsame=R.random() < 0.6,
            # a constant trip in steps of 1 (opt unrolls it: LSR's counter
            # then counts down by the factor, E439)
            ctrip=R.choice([None, None, None, 8, 16, 64, 96, 256]),
            # the running maximum's kind: maximumnum, or maximum / minimum
            # (NaN-propagating, -0 below +0: E444's deferred NaN and zeros)
            fop=R.choice(['maximumnum', 'maximumnum', 'maximum', 'minimum']),
            # the running maximum also used in the loop: compared with the
            # value (no zero's sign reaches the hash: E446 gives it the
            # deferred value) or its bits hashed (a zero's sign does)
            fuse=R.choice([None, None, 'cmp', 'bits']),
        ))
    for p in loops:
        if p['ctrip']:
            p.update(step=1, exact=False, vstep=False)
        if not p['fmax'] or (p['fuse'] == 'bits' and p['fop'] != 'maximumnum'):
            p['fuse'] = None   # (bits only of default NaNs)
    return loops


def sanitize(name, v):
    """IR for v with denormals made zeros of their sign (the model leaves flushing out)."""
    return ['  %%%se = and i32 %s, 2139095040' % (name, v),
            '  %%%sz = icmp eq i32 %%%se, 0' % (name, name),
            '  %%%ss = and i32 %s, -2147483648' % (name, v),
            '  %%%s = select i1 %%%sz, i32 %%%ss, i32 %s' % (name, name, name, v)]


def san(x):
    return x & 0x80000000 if x & 0x7f800000 == 0 else x


def isnan(x):
    return x & 0x7f800000 == 0x7f800000 and x & 0x7fffff != 0


def maximumnum(x, y):
    """maximumnum of two float bit patterns (no denormals), then the default NaN where it is one."""
    if isnan(x) and isnan(y):
        return 0x7fffffff
    if isnan(x):
        return y
    if isnan(y):
        return x
    fx, fy = (struct.unpack('<f', struct.pack('<I', v))[0] for v in (x, y))
    if fx == fy:
        return x & y          # (-0 below +0)
    return x if fx > fy else y


def twin_step(p):
    """The twin counter's constant step: the counter's own where it is one, or another."""
    return p['step'] if p['tsame'] and not p.get('vstep') else p['step'] * 2 + 1


def maximum(x, y, mn=False):
    """maximum (minimum) of two float bit patterns (no denormals): a NaN if either is one, -0 below +0, then
    the default NaN where it is one."""
    if isnan(x) or isnan(y):
        return 0x7fffffff
    fx, fy = (struct.unpack('<f', struct.pack('<I', v))[0] for v in (x, y))
    if fx == fy:
        return (x | y) if mn else (x & y)
    return (x if fx < fy else y) if mn else (x if fx > fy else y)


def minimum(x, y):
    return maximum(x, y, True)


def emit(loops):
    L = ['target triple = "agx"', '',
         'declare i32 @llvm.agx.thread.position.in.grid.x()',
         'declare i32 @llvm.umin.i32(i32, i32)',
         'declare float @llvm.maximumnum.f32(float, float)',
         'declare float @llvm.maximum.f32(float, float)',
         'declare float @llvm.minimum.f32(float, float)', '',
         'define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {',
         'entry:',
         '  %tid = call i32 @llvm.agx.thread.position.in.grid.x()',
         '  %pa = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %tid',
         '  %a = load i32, ptr addrspace(1) %pa, align 4',
         '  %tidb = add i32 %tid, 65536',
         '  %pb = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %tidb',
         '  %b = load i32, ptr addrspace(1) %pb, align 4',
         '  br label %l0.pre0']
    for j, p in enumerate(loops):
        s, nxt = p['step'], 'l%d.pre0' % (j + 1)
        q = 'l%d.' % j
        L.append('%spre0:' % q)
        if p.get('vstep'):
            L += ['  %%%ssv0 = and i32 %%a, 63' % q, '  %%%ssv = add i32 %%%ssv0, 1' % (q, q)]
            if p['wide']:
                L.append('  %%%ssv64 = zext i32 %%%ssv to i64' % (q, q))
            s = '%%%ssv' % q
        st = {'tid': '%tid', 'a': '%a'}.get(p['start'])
        if p['start'] == 'tid_shl':
            L.append('  %%%sst = shl i32 %%tid, %d' % (q, p['shl'])); st = '%%%sst' % q
        elif p['start'] == 'row_lane':
            L += ['  %%%srw = lshr i32 %%tid, 5' % q, '  %%%srw2 = shl i32 %%%srw, 10' % (q, q),
                  '  %%%sln = and i32 %%tid, 31' % q, '  %%%sst = add i32 %%%srw2, %%%sln' % (q, q, q)]
            st = '%%%sst' % q
        elif p['start'] == 'a_tid':
            L += ['  %%%sm = mul i32 %%a, %d' % (q, p['k'] - (1 << 32) if p['k'] >> 31 else p['k']),
                  '  %%%sst = add i32 %%%sm, %%tid' % (q, q)]
            st = '%%%sst' % q
        elif p['start'] == 'neg':
            L.append('  %%%sst = sub i32 -64, %%tid' % q); st = '%%%sst' % q
        L += ['  %%%sbm = and i32 %%b, %d' % (q, p['mask'] - (1 << 32) if p['mask'] >> 31 else p['mask']),
              '  %%%stu = call i32 @llvm.umin.i32(i32 %%%sbm, i32 %d)' % (q, q, p['lim'])]
        trip = '%%%stu' % q
        if p.get('ctrip'):
            trip = str(p['ctrip'])
        if p['exact']:
            L += ['  %%%stu1 = add i32 %%%stu, 1' % (q, q), '  %%%stx = mul i32 %%%stu1, %s' % (q, q, s)]
            trip = '%%%stx' % q
        ty = 'i64' if p['wide'] else 'i32'
        if p['fmax']:
            # a running maximumnum then the PTX front end's default NaN where it is one (E388's fold)
            if p['fstart'] == 'num':
                fs = '-3.000000e+38'
            else:
                L += sanitize(q + 'fa', '%a') + ['  %%%sfs = bitcast i32 %%%sfa to float' % (q, q)]
                fs = '%%%sfs' % q
        if p['wide']:
            L.append('  %%%st64 = zext i32 %s to i64' % (q, trip)); tripv = '%%%st64' % q
        else:
            tripv = trip
        pred = 'ne' if p['exact'] else 'ult'
        if p['guarded']:
            L += ['  %%%snone = icmp eq i32 %s, 0' % (q, trip),
                  '  br i1 %%%snone, label %%%sexit, label %%%sbody' % (q, q, q)]
            entry_pred = 'pre0'
        else:
            L += ['  br label %%%shead' % q, '%shead:' % q,
                  '  %%%sch = phi %s [ 0, %%%spre0 ], [ %%%scn, %%%sbody ]' % (q, ty, q, q, q),
                  '  %%%sah = phi i32 [ %%a, %%%spre0 ], [ %%%san, %%%sbody ]' % (q, q, q, q)]
            if p['fmax']:
                L.append('  %%%sfh = phi float [ %s, %%%spre0 ], [ %%%sfn, %%%sbody ]' % (q, fs, q, q, q))
            if p.get('twin'):
                L.append('  %%%sd = phi i32 [ 0, %%%spre0 ], [ %%%sdn, %%%sbody ]' % (q, q, q, q))
            L += ['  %%%sgo = icmp ult %s %%%sch, %s' % (q, ty, q, tripv),
                  '  br i1 %%%sgo, label %%%sbody, label %%%sexit' % (q, q, q)]
            entry_pred = 'head'
        L.append('%sbody:' % q)
        if p['guarded']:
            L += ['  %%%sc = phi %s [ 0, %%%spre0 ], [ %%%scn, %%%sbody ]' % (q, ty, q, q, q),
                  '  %%%sacc = phi i32 [ %%a, %%%spre0 ], [ %%%san, %%%sbody ]' % (q, q, q, q)]
            c, acc = '%%%sc' % q, '%%%sacc' % q
            if p['fmax']:
                L.append('  %%%sfm = phi float [ %s, %%%spre0 ], [ %%%sfn, %%%sbody ]' % (q, fs, q, q, q))
            if p.get('twin'):
                L.append('  %%%sd = phi i32 [ 0, %%%spre0 ], [ %%%sdn, %%%sbody ]' % (q, q, q, q))
            fm = '%%%sfm' % q
        else:
            c, acc = '%%%sch' % q, '%%%sah' % q
            fm = '%%%sfh' % q
        c32 = c
        if p['wide']:
            L.append('  %%%sc32 = trunc %s %s to i32' % (q, ty, c)); c32 = '%%%sc32' % q
        L += ['  %%%si = add i32 %s, %s' % (q, st, c32),
              '  %%%sim = and i32 %%%si, %d' % (q, q, WORDS - 1),
              '  %%%sp = getelementptr inbounds i32, ptr addrspace(1) %%in, i32 %%%sim' % (q, q),
              '  %%%sv = load i32, ptr addrspace(1) %%%sp, align 4' % (q, q),
              '  %%%sh1 = mul i32 %s, 31' % (q, acc),
              '  %%%sh2 = lshr i32 %%%si, 7' % (q, q),
              '  %%%sh3 = xor i32 %%%si, %%%sh2' % (q, q, q),
              '  %%%sh4 = add i32 %%%sh1, %%%sh3' % (q, q, q)]
        h = '%%%sh4' % q
        if p['usec']:  # the counter feeds the hash too (it is no longer the exit test's alone)
            L.append('  %%%sh5 = xor i32 %%%sh4, %s' % (q, q, c32))
            h = '%%%sh5' % q
        if p.get('cmask'):
            # a mask of the counter (in its own width) against a bound: a
            # compare that selects, as attention's j <= i
            L += ['  %%%smt = and i32 %%tid, 31' % q, '  %%%smb = and i32 %%b, 2047' % q]
            if p['wide']:
                L += ['  %%%smt6 = zext i32 %%%smt to i64' % (q, q), '  %%%smb6 = zext i32 %%%smb to i64' % (q, q),
                      '  %%%smk = add i64 %s, %%%smt6' % (q, c, q),
                      '  %%%smc = icmp ule i64 %%%smk, %%%smb6' % (q, q, q)]
            else:
                L += ['  %%%smk = add i32 %s, %%%smt' % (q, c, q), '  %%%smc = icmp ule i32 %%%smk, %%%smb' % (q, q, q)]
            L += ['  %%%sms = select i1 %%%smc, i32 1, i32 0' % (q, q), '  %%%sh6 = xor i32 %s, %%%sms' % (q, h, q)]
            h = '%%%sh6' % q
        if p.get('twin'):
            L += ['  %%%sh7 = xor i32 %s, %%%sd' % (q, h, q),
                  '  %%%sdn = add i32 %%%sd, %d' % (q, q, twin_step(p))]
            h = '%%%sh7' % q
        def fmax_ops():
            F = sanitize(q + 'fv', '%%%sv' % q) + ['  %%%sff = bitcast i32 %%%sfv to float' % (q, q)]
            if p.get('fop', 'maximumnum') == 'maximumnum':
                # (the PTX front end's default NaN after each step)
                F += ['  %%%sft = call float @llvm.maximumnum.f32(float %s, float %%%sff)' % (q, fm, q),
                      '  %%%sfu = fcmp uno float %%%sft, 0.000000e+00' % (q, q),
                      '  %%%sfn = select i1 %%%sfu, float 0x7FFFFFFFE0000000, float %%%sft' % (q, q, q)]
            else:
                # (maximum / minimum bare in the loop, as MLIR's maximumf: the
                # default NaN only for the store)
                F.append('  %%%sfn = call float @llvm.%s.f32(float %s, float %%%sff)' % (q, p['fop'], fm, q))
            return F
        if p['fmax'] and p.get('fuse'):
            L += fmax_ops()
            if p['fuse'] == 'cmp':
                L += ['  %%%sfc = fcmp ogt float %%%sfn, %%%sff' % (q, q, q),
                      '  %%%sfz = zext i1 %%%sfc to i32' % (q, q)]
            else:
                L.append('  %%%sfz = bitcast float %%%sfn to i32' % (q, q))
            L.append('  %%%sh8 = xor i32 %s, %%%sfz' % (q, h, q))
            h = '%%%sh8' % q
        L += ['  %%%san = add i32 %s, %%%sv' % (q, h, q),
              '  %%%scn = add nuw %s %s, %s' % (q, ty, c,
                                                (('%%%ssv64' % q) if p['wide'] else s) if p.get('vstep') else str(s))]
        if p['fmax'] and not p.get('fuse'):
            L += fmax_ops()
        if p['guarded']:
            L += ['  %%%sgo = icmp %s %s %%%scn, %s' % (q, pred, ty, q, tripv),
                  '  br i1 %%%sgo, label %%%sbody, label %%%sexit' % (q, q, q)]
        else:
            L += ['  br label %%%shead' % q]
        L += ['%sexit:' % q]
        if p['guarded']:
            L.append('  %%%sr = phi i32 [ %%a, %%%spre0 ], [ %%%san, %%%sbody ]' % (q, q, q, q))
            if p['fmax']:
                L.append('  %%%sfr = phi float [ %s, %%%spre0 ], [ %%%sfn, %%%sbody ]' % (q, fs, q, q, q))
        else:
            L.append('  %%%sr = phi i32 [ %%%sah, %%%shead ]' % (q, q, q))
            if p['fmax']:
                L.append('  %%%sfr = phi float [ %%%sfh, %%%shead ]' % (q, q, q))
        if p['fmax']:
            if p.get('fop', 'maximumnum') != 'maximumnum':
                L += ['  %%%sfru = fcmp uno float %%%sfr, 0.000000e+00' % (q, q),
                      '  %%%sfrc = select i1 %%%sfru, float 0x7FFFFFFFE0000000, float %%%sfr' % (q, q, q)]
                fr = '%%%sfrc' % q
            else:
                fr = '%%%sfr' % q
            L += ['  %%%sfb = bitcast float %s to i32' % (q, fr),
                  '  %%%sfo = add i32 %%tid, %d' % (q, j * 65536 + N),
                  '  %%%sfp = getelementptr inbounds i32, ptr addrspace(1) %%out, i32 %%%sfo' % (q, q),
                  '  store i32 %%%sfb, ptr addrspace(1) %%%sfp, align 4' % (q, q)]
        L += ['  %%%so = add i32 %%tid, %d' % (q, j * 65536),
              '  %%%sop = getelementptr inbounds i32, ptr addrspace(1) %%out, i32 %%%so' % (q, q),
              '  store i32 %%%sr, ptr addrspace(1) %%%sop, align 4' % (q, q),
              '  br label %%%s' % nxt]
    L += ['l%d.pre0:' % len(loops), '  ret void', '}', '']
    return '\n'.join(L)


def model(loops, tid, a, b, inp):
    """(output word, value) for each loop's hash and float maximum."""
    res = []
    for j, p in enumerate(loops):
        s = ((a & 63) + 1) if p.get('vstep') else p['step']
        st = {'tid': tid, 'a': a, 'tid_shl': (tid << p['shl']) & M, 'row_lane': ((tid >> 5) << 10) + (tid & 31),
              'a_tid': (a * p['k'] + tid) & M, 'neg': (-64 - tid) & M}[p['start']]
        t = min(b & p['mask'], p['lim'])
        if p.get('ctrip'):
            t = p['ctrip']
        if p['exact']:
            t = ((t + 1) * s) & M
        # wide counters compare 64-bit values, i32 ones may wrap (not here: t + s < 2^32)
        acc, c, d = a, 0, 0
        fm = 0xff61b1e6 if p['fstart'] == 'num' else san(a)   # -3e38
        first = True
        while True:
            if p['guarded']:
                if first and t == 0:
                    break
            elif not c < t:
                break
            first = False
            i = (st + c) & M
            v = int(inp[i & (WORDS - 1)])
            h = (acc * 31) & M
            h = (h + (i ^ (i >> 7))) & M if not p['usec'] else ((h + (i ^ (i >> 7))) & M) ^ (c & M)
            if p.get('cmask'):
                k = (c + (tid & 31)) if p['wide'] else ((c + (tid & 31)) & M)
                h ^= 1 if k <= (b & 2047) else 0
            if p.get('twin'):
                h ^= d
                d = (d + twin_step(p)) & M
            fmop = {'maximumnum': maximumnum, 'maximum': maximum, 'minimum': minimum}[p.get('fop', 'maximumnum')]
            if p.get('fuse'):
                fm = fmop(fm, san(v))
                if p['fuse'] == 'cmp':
                    ffm, ffv = (struct.unpack('<f', struct.pack('<I', x))[0] for x in (fm, san(v)))
                    h ^= 1 if (not isnan(fm) and not isnan(san(v)) and ffm > ffv) else 0
                else:
                    h ^= fm
            acc = (h + v) & M
            if not p.get('fuse'):
                fm = fmop(fm, san(v))
            c += s
            if p['guarded']:
                if not ((c != t) if p['exact'] else (c < t)):
                    break
        res.append((j * 65536 + tid, acc, j))
        if p['fmax']:
            if p.get('fop', 'maximumnum') != 'maximumnum' and isnan(fm):
                fm = 0x7fffffff   # (the default NaN of the store's select)
            res.append((j * 65536 + N + tid, fm, j))
    return res


def run_seed(seed, a, tmp):
    loops = gen(seed)
    src = emit(loops)
    ll, ll2, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", "o.ll", ".o", ".bin"))
    open(ll, "w").write(src)
    inp_ll = ll
    if seed % 2 and a.opt:
        r = subprocess.run([a.opt, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O2", ll, "-S", "-o", ll2],
                           capture_output=True, text=True)
        if r.returncode:
            return src, "opt failure: " + r.stderr.strip().splitlines()[0][:200]
        inp_ll = ll2
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", inp_ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + r.stderr.strip().splitlines()[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, 2**32, WORDS, dtype=np.uint64).astype(np.uint32)
    sp = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001, 0xff61b1e6, 0x3f800000], np.uint32)
    m = rng.random(WORDS) < 0.03
    inp[m] = rng.choice(sp, m.sum())
    for base in (0, 65536):
        x = inp[base:base + N]
        m = rng.random(N) < 0.4
        x[m] = rng.choice(EDGE, m.sum())
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nbytes = 4 * 65536 * 4
    # (a wrong exit test can loop past agx_run's limit: a failure, not a crash)
    try:
        r = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "256,1,1", "--map", "%#x=%s" % (IN, fin),
                            "--alloc", "%#x=%d" % (OUT, nbytes), "--dump", "%#x=%d:%s" % (OUT, nbytes, fout),
                            "--uniform", "0=%d" % IN, "--uniform", "1=0", "--uniform", "2=%d" % OUT,
                            "--uniform", "3=0"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return src, "agx_run timed out"
    if r.returncode:
        return src, "agx_run failed: " + ((r.stderr + r.stdout).strip().splitlines() or [""])[-1][:200]
    got = np.fromfile(fout, np.uint32)
    for t in range(N):
        ref = model(loops, t, int(inp[t]), int(inp[65536 + t]), inp)
        for w, v, j in ref:
            g = int(got[w])
            if g != v:
                return src, "loop %d: thread %d gave %#x, the model %#x (a %#x, b %#x)" % (
                    j, t, g, v, int(inp[t]), int(inp[65536 + t]))
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
