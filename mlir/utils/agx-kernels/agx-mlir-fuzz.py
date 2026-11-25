#!/usr/bin/env python3
"""Differential fuzz of MLIR row and column loops through convert-gpu-to-air and the AGX backend, against a Python
model, on the AGX Performance Simulator.

Each seed generates a gpu.func whose threads each update their row of a 1024x256 memref b (or their column of a
256x1024 one) in an scf.for of one to three statements, in program order:

  b[t, j + o] = f(b[t, j + o'] or a constant, c[...])     (some under scf.if g(...) { ... } else { b's old value })

c a second memref at the thread's row (column) or at j + o'' alone (every thread's same elements), or b itself
passed twice; on rows, a third of the seeds by vector<4> loads and stores in steps of 4 or 8 (the reduction a
vector, then vector.reduction); the trip a constant or a runtime i32, steps of 1 or 2; the elements i32, i16, f32 or f16 (small integers,
exact sums and products); f an add, sub, xor, a multiply by a constant, a max or min (maximumf / minimumf / maxnumf / minnumf), g an unsigned or ordered compare,
and a guarded statement's c load inside its scf.if (speculate-guarded-loads). A sixteenth of the seeds instead exit early: an scf.while from j = 0 while j < T and pred(c[t, j], K), its load
guarded by an scf.if (speculate-guarded-loads) or clamped, a running f of the elements passed, j and the value
stored after it (the structurizer's loop exits, E735's merged exits). A sixteenth divide: b[t, j] = op(x, d) + k, op divui / divsi / remui / remsi / floordivsi /
ceildivsi / ceildivui, x the element (or its low bits), d a constant (powers of two, odd, negative), the kernel's
argument, an element or j + 1 (the division expansions; division by zero and INT_MIN / -1 left out). An eighth
share a threadgroup tile: each of rounds tile[l] = f(c[g, l + 16 r]), a
gpu.barrier, another thread's place (l + sh mod 128 or 256) combined with the thread's own value, a barrier, and
the result stored (threadgroup accesses, barriers, block barriers of 128 threads). An eighth compute float math over a row (exp, e^(x - k), x > k ? e^(x - k) : x, tanh, the
sigmoid, log(abs(x) + 1), e^x k + 1, and e^(x - m) of the row's maximum m from a loop before: E731 / E732 / E743's
range analyses), against double with each arithmetic step rounded to float and 16 ulps of each function's value,
over inputs up to 100 in magnitude (overflows and flushes). A quarter of the seeds instead run a SIMD-group a row: each lane's running value
over j = lane, lane + 32, ... (scf.for from the lane id), gpu.subgroup_reduce of them, and the row rewritten with
the result (the online softmax's shape, exact ops). Lowered as bench.py does
(convert-gpu-to-air, mlir-translate, the AIR target's portable passes, opt -O3, llc -O3, the constants in
uniforms on every other seed), run with agx_run on 1024 threads and compared element for element with the model
(NaNs alike): narrow-indices, narrow-loops, linearize-indices, column-no-wrap, constant-trip-loops and
speculate-guarded-loads, then the backend's loop passes, with stores in the loop.

  agx-mlir-fuzz.py --bin BUILD/bin --run path/to/agx_run [--seeds A:B] [--mcpu agx-g14c] [--keep DIR]

Exits with 1 if a kernel fails to compile or gives a wrong result (kept in DIR with --keep).
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench'))
import bench  # noqa: E402

N, ROW = 1024, 256
FLOATS = ('f32', 'f16')
BASE = 0x100000000


def gen(seed):
    R = random.Random(seed)
    k = dict(ty=R.choice(['i32', 'i32', 'i16', 'f32', 'f16']),
             col=R.random() < 0.3,
             alias=R.random() < 0.25,
             trip=R.choice(['const', 'const', 'arg']),
             tconst=R.choice([1, 3, 8, 16, 17, 33, 64, 100, 128, 192]),
             targ=R.choice([0, 1, 5, 31, 64, 129, 192]),
             step=R.choice([1, 1, 1, 2]),
             stmts=[])
    # a vector<4> of the row's elements an access (rows only: a column's
    # vector would span the other threads' elements), in steps of 4 or 8 at
    # offsets of 4
    k['vec'] = not k['col'] and R.random() < 0.3
    if k['vec']:
        k['step'] = R.choice([4, 4, 8])
    # a running value carried by the loop (iter_args), f(acc, c[...]) each
    # iteration, stored after the loop at the row's last element
    k['red'] = R.random() < 0.35
    k['rop'] = R.choice(['add', 'sub', 'mul', 'max', 'min'] +
                        (['xor'] if k['ty'] not in FLOATS else ['maxnum', 'minnum']))
    k['rlc'] = R.randrange(0, 64) & (~3 if k['vec'] else ~0)
    k['rinit'] = R.choice([0, 1, 5])
    # c at the thread's row (column) or at j + o alone, for every statement
    cw = 'own' if k['alias'] or R.random() < 0.7 else 'shared'
    ops = ['add', 'sub', 'mul', 'max', 'min'] + (['xor'] if k['ty'] not in FLOATS else ['maxnum', 'minnum'])
    for _ in range(R.randint(1, 3)):
        k['stmts'].append(dict(
            st=R.randrange(0, 64), lb=R.choice([None, R.randrange(0, 64)]),
            same=R.random() < 0.5, cw=cw,
            lc=R.randrange(0, 64), op=R.choice(ops), kc=R.choice([1, 3, 7, -2]),
            guard=R.random() < 0.35 and not k['vec']))
        s = k['stmts'][-1]
        if k['vec']:
            for f in ('st', 'lb', 'lc'):
                if s[f] is not None:
                    s[f] &= ~3
        if s['same']:
            s['lb'] = s['st']
    return k


def emit(k):
    T = k['ty']
    E = T                                # the scalar
    V = 'vector<4x%s>' % T if k['vec'] else T
    flt = T in FLOATS
    bshape = '256x1024' if k['col'] else '1024x256'
    cshape = bshape if k['alias'] or any(s['cw'] == 'own' for s in k['stmts']) else None
    cty = 'memref<%sx%s, 1>' % (cshape or '256', T)
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: memref<%sx%s, 1>, %%n: i32) kernel' % (cty, bshape, T),
         '      attributes {known_block_size = array<i32: 256, 1, 1>} {',
         '    %c0 = arith.constant 0 : index',
         '    %%st = arith.constant %d : index' % k['step'],
         '    %t = gpu.global_id x upper_bound 1024']
    if k['trip'] == 'const':
        L.append('    %%T = arith.constant %d : index' % k['tconst'])
    else:
        L.append('    %T = arith.index_cast %n : i32 to index')
    def lit(v):
        x = ('%d.0' % v) if flt else str(v)
        return 'dense<%s> : %s' % (x, V) if k['vec'] else '%s : %s' % (x, V)
    if k['red']:
        L.append('    %%init = arith.constant %s' % lit(k['rinit']))
        L.append('    %%res = scf.for %%j = %%c0 to %%T step %%st iter_args(%%acc = %%init) -> (%s) {' % V)
    else:
        L.append('    scf.for %j = %c0 to %T step %st {')
    n = [0]

    def nv():
        n[0] += 1
        return '%%v%d' % n[0]

    def idx(off):
        o, v = nv(), nv()
        L.append('      %s = arith.constant %d : index' % (o, off))
        L.append('      %s = arith.addi %%j, %s : index' % (v, o))
        return v

    def access(mem, off, own, shape):
        j = idx(off)
        vt = ', %s' % V if k['vec'] else ''
        if not own:
            return '%s[%s] : memref<256x%s, 1>%s' % (mem, j, E, vt)
        sub = '%s, %%t' % j if k['col'] else '%%t, %s' % j
        return '%s[%s] : memref<%sx%s, 1>%s' % (mem, sub, shape, E, vt)

    LOAD = 'vector.load' if k['vec'] else 'memref.load'
    STORE = 'vector.store' if k['vec'] else 'memref.store'

    def const(v):
        c = nv()
        L.append('      %s = arith.constant %s' % (c, lit(v)))
        return c

    def op(o, a, b, kc):
        r = nv()
        if o == 'mul':
            m = nv()
            L.append('      %s = arith.%s %s, %s : %s' % (m, 'mulf' if flt else 'muli', b, const(kc), V))
            L.append('      %s = arith.%s %s, %s : %s' % (r, 'addf' if flt else 'addi', a, m, V))
        elif o in ('max', 'min', 'maxnum', 'minnum'):
            name = {'max': 'maximumf', 'min': 'minimumf', 'maxnum': 'maxnumf', 'minnum': 'minnumf'}[o] if flt \
                else {'max': 'maxui', 'min': 'minui'}[o]
            L.append('      %s = arith.%s %s, %s : %s' % (r, name, a, b, V))
        else:
            name = {'add': 'addf', 'sub': 'subf'}[o] if flt else {'add': 'addi', 'sub': 'subi', 'xor': 'xori'}[o]
            L.append('      %s = arith.%s %s, %s : %s' % (r, name, a, b, V))
        return r

    cmem = '%b' if k['alias'] else '%c'
    for s in k['stmts']:
        own = s['cw'] == 'own'
        if s['lb'] is not None:
            bv = nv()
            L.append('      %s = %s %s' % (bv, LOAD, access('%b', s['lb'], True, bshape)))
        else:
            bv = const(s['kc'])
        if s['guard']:
            # g(b's value, a constant): the c load and f inside the if
            g = nv()
            L.append('      %s = arith.%s %s, %s, %s : %s' % (
                g, 'cmpf' if flt else 'cmpi', 'olt' if flt else 'ult', bv, const(16), T))
            r = nv()
            L.append('      %s = scf.if %s -> (%s) {' % (r, g, T))
            cv = nv()
            L.append('      %s = %s %s' % (cv, LOAD, access(cmem, s['lc'], own, bshape)))
            y = op(s['op'], bv, cv, s['kc'])
            L.append('      scf.yield %s : %s' % (y, T))
            L.append('      } else {')
            L.append('      scf.yield %s : %s' % (bv, T))
            L.append('      }')
        else:
            cv = nv()
            L.append('      %s = %s %s' % (cv, LOAD, access(cmem, s['lc'], own, bshape)))
            r = op(s['op'], bv, cv, s['kc'])
        L.append('      %s %s, %s' % (STORE, r, access('%b', s['st'], True, bshape)))
    if k['red']:
        cv = nv()
        L.append('      %s = %s %s' % (cv, LOAD, access(cmem, k['rlc'], k['stmts'][0]['cw'] == 'own', bshape)))
        acc = op(k['rop'], '%acc', cv, 3)
        L.append('      scf.yield %s : %s' % (acc, V))
        L.append('    }')
        res = '%res'
        if k['vec']:
            kind = {'add': 'add', 'sub': 'add', 'mul': 'add', 'xor': 'xor',
                    'max': 'maximumf' if flt else 'maxui', 'min': 'minimumf' if flt else 'minui',
                    'maxnum': 'maxnumf', 'minnum': 'minnumf'}[k['rop']]
            res = nv()
            L.append('    %s = vector.reduction <%s>, %%res : %s into %s' % (res, kind, V, E))
        last = nv()
        L.append('    %s = arith.constant 255 : index' % last)
        sub = '%s, %%t' % last if k['col'] else '%%t, %s' % last
        L.append('    memref.store %s, %%b[%s] : memref<%sx%s, 1>' % (res, sub, bshape, E))
    else:
        L.append('    }')
    L += ['    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


DT = {'i32': np.uint32, 'i16': np.uint16, 'f32': np.float32, 'f16': np.float16}


def fminmax(mx, a, b):
    """llvm.maximum / minimum: a NaN if either is one, -0 below +0."""
    a, b = np.broadcast_arrays(a, b)
    r = np.where(a > b, a, b) if mx else np.where(a < b, a, b)
    eq = a == b
    zs = np.where(np.signbit(a), b, a) if mx else np.where(np.signbit(a), a, b)
    r = np.where(eq, zs, r)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, r)


def model(k, t, c, b):
    """b (updated in place) after every thread's loop; c is b where aliased."""
    with np.errstate(all='ignore'):
        return model_(k, t, c, b)


def model_(k, t, c, b):
    T = k['ty']
    flt = T in FLOATS
    FT = DT[T]
    W = 32 if T == 'i32' else 16
    MW = (1 << W) - 1
    tid = np.arange(N)

    lanes = np.arange(4) if k.get('vec') else 0
    shape = (N, 4) if k.get('vec') else (N,)

    def at(arr, off, own, j):
        if not own:
            return (j + off + lanes,)
        if k['col']:
            return (j + off, tid)
        return (tid[:, None], j + off + lanes) if k.get('vec') else (tid, j + off)

    def f(o, a, x, kc):
        if flt:
            with np.errstate(all='ignore'):
                if o == 'add':
                    return (a + x).astype(FT)
                if o == 'sub':
                    return (a - x).astype(FT)
                if o == 'mul':
                    return (a + (x * FT(kc)).astype(FT)).astype(FT)
                # (values that grow past the type's range give infinities and NaNs: maxnumf / minnumf
                # ignore a NaN, maximumf / minimumf give it and order -0 below +0)
                if o in ('maxnum', 'minnum'):
                    return (np.fmax if o == 'maxnum' else np.fmin)(a, x).astype(FT)
                return fminmax(o == 'max', a, x).astype(FT)
        a64, x64 = a.astype(np.int64), x.astype(np.int64)
        if o == 'add':
            r = a64 + x64
        elif o == 'sub':
            r = a64 - x64
        elif o == 'xor':
            r = a64 ^ x64
        elif o == 'mul':
            r = a64 + x64 * kc
        elif o == 'min':
            r = np.minimum(a64, x64)
        else:
            r = np.maximum(a64, x64)
        return (r & MW).astype(DT[T])

    j = 0
    acc = np.full(shape, k['rinit'], DT[T]) if k.get('red') else None
    last = (255, tid) if k['col'] else (tid, 255)

    def reduced(acc):
        if not k.get('vec'):
            return acc
        rop = k['rop']
        if rop in ('add', 'sub', 'mul'):
            if flt:
                r = acc[:, 0]
                for l in range(1, 4):
                    r = (r + acc[:, l]).astype(FT)
                return r
            return (acc.astype(np.int64).sum(axis=1) & MW).astype(DT[T])
        if rop == 'xor':
            return np.bitwise_xor.reduce(acc, axis=1)
        if flt:
            r = acc[:, 0]
            for l in range(1, 4):
                r = f(rop, r, acc[:, l], 0)
            return r
        return (acc.max(axis=1) if rop in ('max', 'maxnum') else acc.min(axis=1)).astype(DT[T])
    if not 0 < t:
        if k.get('red'):
            b[last] = reduced(acc)
        return b
    while j < t:
        for s in k['stmts']:
            own = s['cw'] == 'own'
            if s['lb'] is not None:
                bv = b[at(b, s['lb'], True, j)].copy()
            else:
                bv = np.full(shape, s['kc'] if flt else s['kc'] & MW, DT[T])
            cm = b if k['alias'] else c
            cv = cm[at(cm, s['lc'], own, j)]
            cv = cv.copy() if own else np.broadcast_to(cv, shape).astype(DT[T])
            r = f(s['op'], bv, cv, s['kc'])
            if s['guard']:
                g = (bv < FT(16)) if flt else (bv.astype(np.int64) < 16)
                r = np.where(g, r, bv).astype(DT[T])
            b[at(b, s['st'], True, j)] = r
        if k.get('red'):
            own = k['stmts'][0]['cw'] == 'own'
            cm = b if k['alias'] else c
            cv = cm[at(cm, k['rlc'], own, j)]
            cv = cv.copy() if own else np.broadcast_to(cv, shape).astype(DT[T])
            acc = f(k['rop'], acc, cv, 3)
        j += k['step']
    if k.get('red'):
        b[last] = reduced(acc)
    return b


def compile_kernel(a, src, tmp, uniform_constants):
    ll = bench.lower(os.path.join(a.bin, 'mlir-opt'), a.bin, src, tmp)
    llc, opt = os.path.join(a.bin, 'llc'), os.path.join(a.bin, 'opt')
    air, agx, obj, code = (os.path.join(tmp, x) for x in ('air.ll', 'agx.ll', 'k.o', 'k.bin'))
    uc = ['-agx-uniform-constants'] if uniform_constants else []
    bench.run([llc, '-mtriple=air', '-air-portable', ll, '-o', air])
    bench.run([opt, '-mtriple=agx', '-mcpu=' + a.mcpu, '-O3'] + uc + ['-S', air, '-o', agx])
    bench.run([llc, '-mtriple=agx', '-mcpu=' + a.mcpu, '-O3'] + uc + ['-filetype=obj', agx, '-o', obj])
    bench.run([os.path.join(a.bin, 'llvm-objcopy'), '-O', 'binary', '--only-section=.text', obj, code])
    consts = []
    if uniform_constants:
        consts = bench._agx_bench().constant_uniforms(obj, 'k', os.path.join(a.bin, 'llvm-objcopy'))
    return code, consts


def gen_lanes(seed):
    R = random.Random(seed)
    k = dict(ty=R.choice(['i32', 'i32', 'f32']),
             trip=R.choice(['const', 'arg']),
             tconst=R.choice([32, 64, 100, 192, 256]),
             targ=R.choice([0, 7, 32, 33, 200, 256]),
             init=R.choice([0, 1, 5]),
             second=R.random() < 0.7,
             op2=R.choice(['add', 'sub', 'max', 'min']))
    k['op'] = R.choice(['add', 'max', 'min'] + (['xor'] if k['ty'] != 'f32' else ['maxnum', 'minnum']))
    return k


def emit_lanes(k):
    """One SIMD-group a row: each lane's running value over j = lane, lane + 32, ..., the SIMD-group's reduction
    of them, stored at the row's element 256, and (second) the row rewritten with it."""
    T = k['ty']
    flt = T == 'f32'
    lit = (lambda v: '%d.0' % v) if flt else str
    M = 'memref<1024x257x%s, 1>' % T
    fop = {'add': 'addf' if flt else 'addi', 'sub': 'subf' if flt else 'subi', 'xor': 'xori',
           'max': 'maximumf' if flt else 'maxui', 'min': 'minimumf' if flt else 'minui',
           'maxnum': 'maxnumf', 'minnum': 'minnumf'}
    kind = {'add': 'add', 'xor': 'xor', 'max': 'maximumf' if flt else 'maxui',
            'min': 'minimumf' if flt else 'minui', 'maxnum': 'maxnumf', 'minnum': 'minnumf'}[k['op']]
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: %s, %%n: i32) kernel' % (M, M),
         '      attributes {known_block_size = array<i32: 32, 1, 1>} {',
         '    %c32 = arith.constant 32 : index',
         '    %c256 = arith.constant 256 : index',
         '    %row = gpu.block_id x upper_bound 1024',
         '    %lane = gpu.lane_id upper_bound 32']
    if k['trip'] == 'const':
        L.append('    %%T = arith.constant %d : index' % k['tconst'])
    else:
        L.append('    %T = arith.index_cast %n : i32 to index')
    L += ['    %%init = arith.constant %s : %s' % (lit(k['init']), T),
          '    %%p = scf.for %%j = %%lane to %%T step %%c32 iter_args(%%a = %%init) -> (%s) {' % T,
          '      %%v = memref.load %%c[%%row, %%j] : %s' % M,
          '      %%a2 = arith.%s %%a, %%v : %s' % (fop[k['op']], T),
          '      scf.yield %%a2 : %s' % T,
          '    }',
          '    %%r = gpu.subgroup_reduce %s %%p : (%s) -> %s' % (kind, T, T)]
    if k['second']:
        L += ['    scf.for %j = %lane to %T step %c32 {',
              '      %%v = memref.load %%c[%%row, %%j] : %s' % M,
              '      %%w = arith.%s %%v, %%r : %s' % (fop[k['op2']], T),
              '      memref.store %%w, %%b[%%row, %%j] : %s' % M,
              '    }']
    L += ['    memref.store %%r, %%b[%%row, %%c256] : %s' % M,
          '    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


def model_lanes(k, t, c, b):
    T = k['ty']
    flt = T == 'f32'
    MW = 0xffffffff

    def f(o, x, y):
        if flt:
            if o == 'add':
                return (x + y).astype(np.float32)
            if o == 'sub':
                return (x - y).astype(np.float32)
            return (np.maximum if o in ('max', 'maxnum') else np.minimum)(x, y).astype(np.float32)
        x64, y64 = x.astype(np.int64), y.astype(np.int64)
        r = {'add': lambda: x64 + y64, 'sub': lambda: x64 - y64, 'xor': lambda: x64 ^ y64,
             'max': lambda: np.maximum(x64, y64), 'min': lambda: np.minimum(x64, y64)}[o]()
        return (r & MW).astype(np.uint32)
    acc = np.full((1024, 32), k['init'], DT[T])
    for j0 in range(0, max(t, 0), 32):
        js = j0 + np.arange(32)
        v = c[:, np.minimum(js, 255)]
        acc = np.where(js[None, :] < t, f(k['op'], acc, v), acc).astype(DT[T])
    r = acc[:, 0]
    for l in range(1, 32):
        r = f('add' if k['op'] in ('add',) else k['op'], r, acc[:, l])
    if k['second']:
        for j in range(min(max(t, 0), 256)):
            b[:, j] = f(k['op2'], c[:, j], r)
    b[:, 256] = r
    return b


def run_lanes(seed, a, tmp):
    k = gen_lanes(seed)
    src = emit_lanes(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    T = k['ty']
    if T == 'f32':
        c = rng.integers(-40, 40, (1024, 257)).astype(np.float32)
        b = rng.integers(-40, 40, (1024, 257)).astype(np.float32)
    else:
        c = rng.integers(0, 1 << 32, (1024, 257), dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 1 << 32, (1024, 257), dtype=np.uint64).astype(np.uint32)
    t = k['tconst'] if k['trip'] == 'const' else k['targ']
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    cmd = [a.run, code, '--grid', '%d,1,1' % (1024 * 32), '--wg', '32,1,1',
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (vc & 0xffffffff), '--uniform', '1=%d' % (vc >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=%d' % t] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, DT[T]).reshape(b.shape)
    ref = model_lanes(k, t, c, b.copy())
    bad = np.argwhere(got != ref)
    if bad.size:
        e = tuple(bad[0])
        return src, 'lanes: element %s: %r, the model %r (%d elements differ)' % (e, got[e], ref[e], len(bad))
    return src, None


MATH = ['exp', 'expk', 'condexp', 'tanh', 'sigmoid', 'log1', 'expscale', 'maxexp', 'online', 'online', 'tanh1',
        'gelu']


def gen_math(seed):
    R = random.Random(seed)
    return dict(f=R.choice(MATH), k=R.choice([0.5, 3.0, -2.0, 10.0]), trip=R.choice([16, 64, 100, 256]),
                lo=R.choice([-100.0, -30.0, -10.0, -1.0]), hi=R.choice([1.0, 10.0, 30.0, 100.0]))


def emit_math(k):
    """b[t, j] = g(c[t, j]) over a thread's row of floats; for maxexp the row's maximum m first (a loop of
    maximumf), then e^(c - m) (E731 / E743's exponent below a loop's maximum)."""
    M = 'memref<1024x256xf32, 1>'
    K = '%r' % float(k['k'])
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: %s, %%n: i32) kernel' % (M, M),
         '      attributes {known_block_size = array<i32: 256, 1, 1>} {',
         '    %c0 = arith.constant 0 : index',
         '    %c1 = arith.constant 1 : index',
         '    %%T = arith.constant %d : index' % k['trip'],
         '    %one = arith.constant 1.0 : f32',
         '    %%kk = arith.constant %s : f32' % K,
         '    %t = gpu.global_id x upper_bound 1024']
    if k['f'] == 'maxexp':
        L += ['    %ninf = arith.constant 0xFF800000 : f32',
              '    %m = scf.for %j = %c0 to %T step %c1 iter_args(%a = %ninf) -> (f32) {',
              '      %%v = memref.load %%c[%%t, %%j] : %s' % M,
              '      %a2 = arith.maximumf %a, %v : f32',
              '      scf.yield %a2 : f32',
              '    }']
    if k['f'] == 'online':
        # the online softmax's running maximum m and sum s of e^(x - m), rescaled by e^(m - m') (E446 / E681),
        # stored at the row's elements 0 and 1
        L += ['    %ninf = arith.constant 0xFF800000 : f32',
              '    %zero = arith.constant 0.0 : f32',
              '    %r:2 = scf.for %j = %c0 to %T step %c1 iter_args(%m = %ninf, %s = %zero) -> (f32, f32) {',
              '      %%x = memref.load %%c[%%t, %%j] : %s' % M,
              '      %m2 = arith.maximumf %m, %x : f32',
              '      %d1 = arith.subf %m, %m2 : f32', '      %e1 = math.exp %d1 : f32',
              '      %d2 = arith.subf %x, %m2 : f32', '      %e2 = math.exp %d2 : f32',
              '      %s2 = math.fma %s, %e1, %e2 : f32',
              '      scf.yield %m2, %s2 : f32, f32',
              '    }',
              '    %i0 = arith.constant 0 : index', '    %i1 = arith.constant 1 : index',
              '    memref.store %%r#0, %%b[%%t, %%i0] : %s' % M,
              '    memref.store %%r#1, %%b[%%t, %%i1] : %s' % M,
              '    gpu.return', '  }', '}']
        return '\n'.join(L) + '\n'
    L += ['    scf.for %j = %c0 to %T step %c1 {',
          '      %%x = memref.load %%c[%%t, %%j] : %s' % M]
    f = k['f']
    if f == 'exp':
        L += ['      %y = math.exp %x : f32']
    elif f == 'expk':
        L += ['      %d = arith.subf %x, %kk : f32', '      %y = math.exp %d : f32']
    elif f == 'condexp':
        L += ['      %g = arith.cmpf ogt, %x, %kk : f32',
              '      %y = scf.if %g -> (f32) {',
              '        %d = arith.subf %x, %kk : f32', '        %e = math.exp %d : f32', '        scf.yield %e : f32',
              '      } else {', '        scf.yield %x : f32', '      }']
    elif f == 'tanh':
        L += ['      %y = math.tanh %x : f32']
    elif f == 'sigmoid':
        L += ['      %ng = arith.negf %x : f32', '      %e = math.exp %ng : f32',
              '      %d = arith.addf %e, %one : f32', '      %y = arith.divf %one, %d : f32']
    elif f == 'log1':
        L += ['      %a = math.absf %x : f32', '      %d = arith.addf %a, %one : f32', '      %y = math.log %d : f32']
    elif f == 'tanh1':
        L += ['      %h = math.tanh %x : f32', '      %y = arith.addf %h, %one : f32']
    elif f == 'gelu':
        # 0.5 x (1 + tanh(0.79788 (x + 0.044715 x^3)))
        L += ['      %x2 = arith.mulf %x, %x : f32', '      %x3 = arith.mulf %x2, %x : f32',
              '      %ca = arith.constant 0.044715 : f32', '      %cb = arith.constant 0.7978845608 : f32',
              '      %half = arith.constant 0.5 : f32',
              '      %q = arith.mulf %x3, %ca : f32', '      %s = arith.addf %x, %q : f32',
              '      %tt = arith.mulf %s, %cb : f32', '      %h = math.tanh %tt : f32',
              '      %u = arith.addf %h, %one : f32', '      %hx = arith.mulf %x, %half : f32',
              '      %y = arith.mulf %hx, %u : f32']
    elif f == 'expscale':
        L += ['      %e = math.exp %x : f32', '      %h = arith.mulf %e, %kk : f32', '      %y = arith.addf %h, %one : f32']
    else:
        L += ['      %d = arith.subf %x, %m : f32', '      %y = math.exp %d : f32']
    L += ['      memref.store %%y, %%b[%%t, %%j] : %s' % M, '    }', '    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


def model_math(k, c):
    """The reference and its tolerance: each arithmetic step rounded to float as the kernel does, the functions
    in double; 16 ulps of the function's own value carried through the steps after it (a sum that cancels keeps
    the error of its larger term)."""
    f32 = np.float32
    x = c[:, :k['trip']]
    K = f32(k['k'])
    one = f32(1)

    def F(v):
        # (a double's value as the float the kernel has: past the float range, an infinity)
        return v.astype(np.float32).astype(np.float64)

    def ulp(v):
        u = np.spacing(np.abs(v.astype(np.float32))).astype(np.float64)
        return np.where(np.isfinite(u), u, 0)
    with np.errstate(all='ignore'):
        f = k['f']
        if f == 'online':
            m = x.max(axis=1)
            ssum = np.exp(F(x) - F(m)[:, None]).sum(axis=1)
            y = np.zeros(x.shape)
            e = np.full(x.shape, np.inf)
            y[:, 0], e[:, 0] = F(m), 0.0
            y[:, 1], e[:, 1] = ssum, 1e-4 * ssum
            return y, e + 2.0 ** -126
        if f == 'exp':
            y = F(np.exp(F(x))); e = 16 * ulp(y)
        elif f in ('expk', 'maxexp'):
            m = K if f == 'expk' else x.max(axis=1, keepdims=True)
            d = (x - m).astype(f32)
            y = F(np.exp(F(d))); e = 16 * ulp(y)
        elif f == 'condexp':
            d = (x - K).astype(f32)
            y = F(np.where(x > K, np.exp(F(d)), F(x))); e = np.where(x > K, 16 * ulp(y), 0)
        elif f == 'tanh':
            y = np.tanh(F(x)); e = 16 * ulp(y)
        elif f == 'sigmoid':
            ex = F(np.exp(F((-x).astype(f32))))
            d = (ex.astype(f32) + one).astype(f32)
            y = 1 / F(d); e = 16 * ulp(ex) / np.maximum(F(d), 1) ** 2 + 4 * ulp(y)
        elif f == 'log1':
            d = (np.abs(x) + one).astype(f32)
            y = np.log(F(d)); e = 16 * ulp(y) + ulp(d)
        elif f == 'tanh1':
            # (1 + tanh in double cancels too: the quotient form; results near the smallest normal may flush)
            y = F(2 / (1 + np.exp(-2 * F(x)))); e = 16 * ulp(y) + 2.0 ** -124
        elif f == 'gelu':
            s = (x + ((x * x).astype(f32) * x).astype(f32) * f32(0.044715)).astype(f32)
            tt = (s * f32(0.7978845608)).astype(f32)
            u = 2 / (1 + np.exp(-2 * F(tt)))
            y = F(F((x * f32(0.5)).astype(f32)) * u); e = 16 * ulp(y) + 2.0 ** -124
            # (1 + tanh below 2^-125 flushes in its reciprocal, as tanh's -1 did: a zero result there)
            e = np.where(u < 2.0 ** -125, np.abs(y) + 2.0 ** -124, e)
        else:
            ex = F(np.exp(F(x)))
            h = (ex.astype(f32) * K).astype(f32)
            y = F((h + one).astype(f32)); e = 16 * ulp(ex) * abs(float(K)) + ulp(h) + ulp(y)
    return y, e + 2.0 ** -126


def run_math(seed, a, tmp):
    k = gen_math(seed)
    src = emit_math(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    c = rng.uniform(k['lo'], k['hi'], (1024, 256)).astype(np.float32)
    m = rng.random(c.shape) < 0.05
    c[m] = rng.choice(np.array([0.0, -0.0, 1.0, -1.0, k['k']], np.float32), m.sum())
    b = np.zeros((1024, 256), np.float32)
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    cmd = [a.run, code, '--grid', '1024,1,1', '--wg', '256,1,1',
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (vc & 0xffffffff), '--uniform', '1=%d' % (vc >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=0'] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, np.float32).reshape(b.shape)[:, :k['trip']].astype(np.float64)
    ref, tol = model_math(k, c)
    with np.errstate(invalid='ignore'):
        ok = (np.abs(got - ref) <= tol) | (np.isinf(got) & np.isinf(ref) & (np.sign(got) == np.sign(ref)))
    bad = np.argwhere(~ok)
    if bad.size:
        e = tuple(bad[0])
        return src, 'math %s: element %s of c %r: %r, the reference %r (%d elements off)' % (
            k['f'], e, float(c[e]), got[e], ref[e], len(bad))
    return src, None


def gen_tg(seed):
    R = random.Random(seed)
    return dict(ty=R.choice(['i32', 'f32']), bs=R.choice([128, 256]), rounds=R.choice([1, 2, 3]),
                sh=R.choice([1, 5, 31, 64, 127]), op=R.choice(['add', 'sub', 'max', 'xor']),
                op2=R.choice(['add', 'sub', 'max']), kc=R.choice([1, 3]))


def emit_tg(k):
    """A threadgroup tile: each round tile[l] = f(c[g, l + 16 r]) by every thread, a barrier, the tile at another
    thread's place (l + sh mod bs) combined with the thread's own value, a barrier, the result stored at
    b[g, l + 16 r]."""
    T = k['ty']
    flt = T == 'f32'
    if flt and k['op'] == 'xor':
        k['op'] = 'add'
    fop = {'add': 'addf' if flt else 'addi', 'sub': 'subf' if flt else 'subi', 'xor': 'xori',
           'max': 'maximumf' if flt else 'maxui'}
    M = 'memref<256x512x%s, 1>' % T
    W = 'memref<%dx%s, 3>' % (k['bs'], T)
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: %s, %%n: i32) workgroup(%%tile: %s) kernel' % (M, M, W),
         '      attributes {known_block_size = array<i32: %d, 1, 1>} {' % k['bs'],
         '    %c0 = arith.constant 0 : index', '    %c1 = arith.constant 1 : index',
         '    %c16 = arith.constant 16 : index',
         '    %%R = arith.constant %d : index' % k['rounds'],
         '    %%sh = arith.constant %d : index' % k['sh'],
         '    %%bs = arith.constant %d : index' % k['bs'],
         '    %g = gpu.block_id x upper_bound 256',
         '    %%l = gpu.thread_id x upper_bound %d' % k['bs'],
         '    %%kc = arith.constant %s : %s' % (('%d.0' % k['kc']) if flt else str(k['kc']), T),
         '    scf.for %r = %c0 to %R step %c1 {',
         '      %o = arith.muli %r, %c16 : index',
         '      %j = arith.addi %l, %o : index',
         '      %%v = memref.load %%c[%%g, %%j] : %s' % M,
         '      %%u = arith.%s %%v, %%kc : %s' % (fop[k['op']], T),
         '      memref.store %%u, %%tile[%%l] : %s' % W,
         '      gpu.barrier',
         '      %ls = arith.addi %l, %sh : index',
         '      %li = arith.remui %ls, %bs : index',
         '      %%w = memref.load %%tile[%%li] : %s' % W,
         '      gpu.barrier',
         '      %%y = arith.%s %%w, %%v : %s' % (fop[k['op2']], T),
         '      memref.store %%y, %%b[%%g, %%j] : %s' % M,
         '    }', '    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


def model_tg(k, c, b):
    T = k['ty']
    flt = T == 'f32'
    FT = DT[T]
    bs = k['bs']
    l = np.arange(bs)

    def f(o, x, y):
        if flt:
            return {'add': x + y, 'sub': x - y, 'max': np.maximum(x, y)}[o].astype(FT)
        x64, y64 = x.astype(np.int64), y.astype(np.int64)
        return ({'add': x64 + y64, 'sub': x64 - y64, 'xor': x64 ^ y64, 'max': np.maximum(x64, y64)}[o]
                & 0xffffffff).astype(FT)
    kc = np.full((256, bs), k['kc'], FT)
    for r in range(k['rounds']):
        j = l + 16 * r
        v = c[:, j]
        tile = f(k['op'], v, kc)
        w = tile[:, (l + k['sh']) % bs]
        b[:, j] = f(k['op2'], w, v)
    return b


def run_tg(seed, a, tmp):
    k = gen_tg(seed)
    src = emit_tg(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    T = k['ty']
    if T == 'f32':
        c = rng.integers(-40, 40, (256, 512)).astype(np.float32)
        b = rng.integers(-40, 40, (256, 512)).astype(np.float32)
    else:
        c = rng.integers(0, 1 << 32, (256, 512), dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 1 << 32, (256, 512), dtype=np.uint64).astype(np.uint32)
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    # (the tile's bytes of threadgroup memory)
    cmd = [a.run, code, '--grid', '%d,1,1' % (256 * k['bs']), '--wg', '%d,1,1' % k['bs'],
           '--shared', str(4 * k['bs']),
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (vc & 0xffffffff), '--uniform', '1=%d' % (vc >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=0'] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, DT[T]).reshape(b.shape)
    ref = model_tg(k, c, b.copy())
    bad = np.argwhere(got != ref)
    if bad.size:
        e = tuple(bad[0])
        return src, 'tg: element %s: %r, the model %r (%d elements differ)' % (e, got[e], ref[e], len(bad))
    return src, None


IDIV = ['divui', 'divsi', 'remui', 'remsi', 'floordivsi', 'ceildivsi', 'ceildivui']


def gen_idiv(seed):
    R = random.Random(seed)
    k = dict(ty=R.choice(['i32', 'i32', 'i16']), op=R.choice(IDIV),
             den=R.choice(['const', 'const', 'arg', 'elem', 'index']),
             d=R.choice([1, 2, 3, 7, 10, 16, 255, 256, 1000, -1, -3, -16, 65535]),
             narrow=R.random() < 0.4, add=R.choice([0, 1, 100]))
    if k['ty'] == 'i16' and abs(k['d']) > 30000:
        k['d'] = 7
    return k


def emit_idiv(k):
    """b[t, j] = op(c[t, j] (or narrowed to 0..65535 by an and), the divisor) + add over a thread's row: the
    divisor a constant, the kernel's argument n, an element c[t, 255], or the index j + 1."""
    T = k['ty']
    M = 'memref<1024x256x%s, 1>' % T
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: %s, %%n: i32) kernel' % (M, M),
         '      attributes {known_block_size = array<i32: 256, 1, 1>} {',
         '    %c0 = arith.constant 0 : index', '    %c1 = arith.constant 1 : index',
         '    %c255 = arith.constant 255 : index',
         '    %t = gpu.global_id x upper_bound 1024',
         '    %%add = arith.constant %d : %s' % (k['add'], T)]
    if k['den'] == 'const':
        L.append('    %%d = arith.constant %d : %s' % (k['d'], T))
    elif k['den'] == 'arg':
        L.append('    %%d = arith.%s %%n : i32 to %s' % ('trunci', T) if T != 'i32' else '    %d = arith.addi %n, %add : i32')
        if T == 'i32':
            L[-1] = '    %d = arith.addi %n, %c0i : i32'
            L.insert(-1, '    %c0i = arith.constant 0 : i32')
    elif k['den'] == 'elem':
        L.append('    %%d = memref.load %%c[%%t, %%c255] : %s' % M)
    L += ['    scf.for %j = %c0 to %c255 step %c1 {',
          '      %%x0 = memref.load %%c[%%t, %%j] : %s' % M]
    if k['narrow']:
        L += ['      %%m = arith.constant %d : %s' % (65535 if T == 'i32' else 255, T),
              '      %%x = arith.andi %%x0, %%m : %s' % T]
    else:
        L += ['      %%x = arith.addi %%x0, %%add : %s' % T]
    if k['den'] == 'index':
        L += ['      %j1 = arith.addi %j, %c1 : index',
              '      %%d = arith.index_cast %%j1 : index to %s' % T]
    L += ['      %%q = arith.%s %%x, %%d : %s' % (k['op'], T),
          '      %%y = arith.addi %%q, %%add : %s' % T,
          '      memref.store %%y, %%b[%%t, %%j] : %s' % M,
          '    }', '    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


def model_idiv(k, c, n, b):
    T = k['ty']
    W = 32 if T == 'i32' else 16
    MW = (1 << W) - 1

    def s(v):   # signed
        v = v.astype(np.int64) & MW
        return np.where(v >= 1 << (W - 1), v - (1 << W), v)

    def u(v):
        return v.astype(np.int64) & MW
    x = c[:, :255]
    x = (u(x) & (65535 if T == 'i32' else 255)) if k['narrow'] else u(u(x) + k['add'])
    if k['den'] == 'const':
        d = np.full(x.shape, k['d'] & MW, np.int64)
    elif k['den'] == 'arg':
        d = np.full(x.shape, n & MW, np.int64)
    elif k['den'] == 'elem':
        d = np.broadcast_to(u(c[:, 255:256]), x.shape)
    else:
        d = np.broadcast_to(np.arange(1, 256, dtype=np.int64)[None, :], x.shape)
    op = k['op']
    ok = d != 0
    dd = np.where(ok, d, 1)
    if op in ('divui', 'remui', 'ceildivui'):
        q = {'divui': x // dd, 'remui': x % dd, 'ceildivui': -(-x // dd)}[op]
    else:
        xs, ds = s(x), s(dd)
        # (signed overflow: INT_MIN / -1 is undefined, as division by zero)
        ok &= ~((xs == -(1 << (W - 1))) & (ds == -1))
        ds = np.where(ok, ds, 1)
        tq = np.trunc(xs / ds).astype(np.int64)
        tq = np.where(np.abs(xs) < 2 ** 52, tq, xs // ds)   # (exact for these widths)
        tr = xs - tq * ds
        if op == 'divsi':
            q = tq
        elif op == 'remsi':
            q = tr
        elif op == 'floordivsi':
            q = np.where((tr != 0) & ((tr < 0) != (ds < 0)), tq - 1, tq)
        else:
            q = np.where((tr != 0) & ((tr < 0) == (ds < 0)), tq + 1, tq)
    y = (q + k['add']) & MW
    return y.astype(DT[T]), ok


def run_idiv(seed, a, tmp):
    k = gen_idiv(seed)
    src = emit_idiv(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    T = k['ty']
    W = 32 if T == 'i32' else 16
    c = rng.integers(0, 1 << W, (1024, 256), dtype=np.uint64).astype(DT[T])
    m = rng.random(c.shape) < 0.2
    edge = np.array([0, 1, 2, 3, 7, (1 << W) - 1, (1 << W) - 2, 1 << (W - 1), (1 << (W - 1)) - 1, 100, 1000], np.uint64)
    c[m] = rng.choice(edge, m.sum()).astype(DT[T])
    # (an element divisor never zero)
    c[:, 255] = np.where(c[:, 255] == 0, 3, c[:, 255])
    n = int(rng.choice([1, 3, 7, 16, 100, 1000, (1 << W) - 1, (1 << W) - 3]))
    b = np.zeros((1024, 256), DT[T])
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    cmd = [a.run, code, '--grid', '1024,1,1', '--wg', '256,1,1',
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (vc & 0xffffffff), '--uniform', '1=%d' % (vc >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=%d' % n] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, DT[T]).reshape(b.shape)[:, :255]
    ref, ok = model_idiv(k, c, n, b)
    bad = np.argwhere(ok & (got != ref))
    if bad.size:
        e = tuple(bad[0])
        return src, 'idiv %s: element %s of c %#x: %#x, the model %#x (%d elements differ)' % (
            k['op'], e, int(c[e]), int(got[e]), int(ref[e]), len(bad))
    return src, None


def gen_while(seed):
    R = random.Random(seed)
    k = dict(ty=R.choice(['i32', 'i32', 'f32']), T=R.choice([0, 1, 16, 100, 255, 256]),
             pred=R.choice(['lt', 'le', 'ne', 'gt']), K=R.choice([0, 3, 20, 35]),
             op=R.choice(['add', 'xor', 'max', 'sub']), init=R.choice([0, 7]), guard=R.random() < 0.7)
    if k['ty'] == 'f32' and k['op'] == 'xor':
        k['op'] = 'add'
    return k


def emit_while(k):
    """An early exit: j from 0 while j < T and pred(c[t, j], K), a running f(acc, c[t, j]) of the elements passed;
    j and acc stored at b[t, 0] and b[t, 1] (scf.while, its load guarded by j < T in an scf.if or not)."""
    T = k['ty']
    flt = T == 'f32'
    M = 'memref<1024x256x%s, 1>' % T
    lit = (lambda v: '%d.0' % v) if flt else str
    cmp = ('arith.cmpf %s,' % {'lt': 'olt', 'le': 'ole', 'ne': 'une', 'gt': 'ogt'}[k['pred']]) if flt else \
        ('arith.cmpi %s,' % {'lt': 'ult', 'le': 'ule', 'ne': 'ne', 'gt': 'ugt'}[k['pred']])
    fop = {'add': 'addf' if flt else 'addi', 'sub': 'subf' if flt else 'subi', 'xor': 'xori',
           'max': 'maximumf' if flt else 'maxui'}[k['op']]
    L = ['gpu.module @k {',
         '  gpu.func @k(%%c: %s, %%b: %s, %%n: i32) kernel' % (M, M),
         '      attributes {known_block_size = array<i32: 256, 1, 1>} {',
         '    %c0 = arith.constant 0 : index', '    %c1 = arith.constant 1 : index',
         '    %%T = arith.constant %d : index' % k['T'],
         '    %%K = arith.constant %s : %s' % (lit(k['K']), T),
         '    %%init = arith.constant %s : %s' % (lit(k['init']), T),
         '    %%zero = arith.constant %s : %s' % (lit(0), T),
         '    %t = gpu.global_id x upper_bound 1024',
         '    %%res:3 = scf.while (%%j = %%c0, %%acc = %%init) : (index, %s) -> (index, %s, %s) {' % (T, T, T),
         '      %in = arith.cmpi ult, %j, %T : index']
    if k['guard']:
        L += ['      %%v = scf.if %%in -> (%s) {' % T,
              '        %%l = memref.load %%c[%%t, %%j] : %s' % M,
              '        scf.yield %%l : %s' % T,
              '      } else {',
              '        scf.yield %%zero : %s' % T,
              '      }']
    else:
        # (the load at min(j, 255): in bounds whatever j)
        L += ['      %c255 = arith.constant 255 : index',
              '      %jc = arith.minui %j, %c255 : index',
              '      %%v = memref.load %%c[%%t, %%jc] : %s' % M]
    L += ['      %%p = %s %%v, %%K : %s' % (cmp, T),
          '      %go = arith.andi %in, %p : i1',
          '      scf.condition(%%go) %%j, %%acc, %%v : index, %s, %s' % (T, T),
          '    } do {',
          '    ^bb0(%%j: index, %%acc: %s, %%v: %s):' % (T, T),
          '      %j1 = arith.addi %j, %c1 : index',
          '      %%a1 = arith.%s %%acc, %%v : %s' % (fop, T),
          '      scf.yield %%j1, %%a1 : index, %s' % T,
          '    }',
          '    %i0 = arith.constant 0 : index', '    %i1 = arith.constant 1 : index',
          '    %%rj = arith.index_cast %%res#0 : index to %s' % ('i32'),
          ]
    if flt:
        L += ['    %rjf = arith.uitofp %rj : i32 to f32',
              '    memref.store %%rjf, %%b[%%t, %%i0] : %s' % M]
    else:
        L += ['    memref.store %%rj, %%b[%%t, %%i0] : %s' % M]
    L += ['    memref.store %%res#1, %%b[%%t, %%i1] : %s' % M, '    gpu.return', '  }', '}']
    return '\n'.join(L) + '\n'


def model_while(k, c, b):
    T = k['ty']
    flt = T == 'f32'
    FT = DT[T]
    tid = np.arange(1024)
    j = np.zeros(1024, np.int64)
    acc = np.full(1024, k['init'], FT)
    alive = np.ones(1024, bool)
    K = FT(k['K'])
    for _ in range(k['T'] + 1):
        inb = j < k['T']
        v = np.where(inb, c[tid, np.minimum(j, 255)], FT(0)).astype(FT)
        if flt:
            p = {'lt': v < K, 'le': v <= K, 'ne': ~(v == K), 'gt': v > K}[k['pred']]
        else:
            p = {'lt': v < K, 'le': v <= K, 'ne': v != K, 'gt': v > K}[k['pred']]
        go = alive & inb & p
        if flt:
            a1 = {'add': acc + v, 'sub': acc - v, 'max': np.maximum(acc, v)}[k['op']].astype(FT)
        else:
            a64, v64 = acc.astype(np.int64), v.astype(np.int64)
            a1 = ({'add': a64 + v64, 'sub': a64 - v64, 'xor': a64 ^ v64, 'max': np.maximum(a64, v64)}[k['op']]
                  & 0xffffffff).astype(FT)
        acc = np.where(go, a1, acc).astype(FT)
        j = np.where(go, j + 1, j)
        alive = go
        if not alive.any():
            break
    b[:, 0] = j.astype(FT)
    b[:, 1] = acc
    return b


def run_while(seed, a, tmp):
    k = gen_while(seed)
    src = emit_while(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    T = k['ty']
    c = rng.integers(0, 40, (1024, 256)).astype(DT[T])
    b = np.zeros((1024, 256), DT[T])
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    cmd = [a.run, code, '--grid', '1024,1,1', '--wg', '256,1,1',
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (vc & 0xffffffff), '--uniform', '1=%d' % (vc >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=0'] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, DT[T]).reshape(b.shape)
    ref = model_while(k, c, b.copy())
    bad = np.argwhere(got != ref)
    if bad.size:
        e = tuple(bad[0])
        return src, 'while: element %s: %r, the model %r (%d elements differ)' % (e, got[e], ref[e], len(bad))
    return src, None


def run_seed(seed, a, tmp):
    if seed % 16 == 8:
        return run_while(seed, a, tmp)
    if seed % 16 == 0:
        return run_idiv(seed, a, tmp)
    if seed % 8 == 5:
        return run_tg(seed, a, tmp)
    # (a quarter of the seeds: a SIMD-group a row; an eighth: float math)
    if seed % 4 == 3:
        return run_lanes(seed, a, tmp)
    if seed % 8 == 1:
        return run_math(seed, a, tmp)
    k = gen(seed)
    src = emit(k)
    sp = os.path.join(tmp, 'k.mlir.src')
    open(sp, 'w').write(src)
    try:
        code, consts = compile_kernel(a, sp, tmp, seed % 2 == 1)
    except subprocess.CalledProcessError as e:
        return src, 'compile failure: ' + ((e.stderr or '').strip().splitlines() or [''])[-1][:200]
    rng = np.random.default_rng(seed)
    T = k['ty']
    if T in FLOATS:
        b = rng.integers(-40, 40, N * ROW).astype(DT[T])
        c = rng.integers(-40, 40, N * ROW).astype(DT[T])
        # (a few NaNs, infinities and -0s: E444's deferred NaN and zero sign)
        sp = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], DT[T])
        for arr in (b, c):
            m = rng.random(arr.size) < 0.02
            arr[m] = rng.choice(sp, m.sum())
    else:
        top = 1 << (32 if T == 'i32' else 16)
        b = rng.integers(0, top, N * ROW, dtype=np.uint64).astype(DT[T])
        c = rng.integers(0, top, N * ROW, dtype=np.uint64).astype(DT[T])
        m = rng.random(N * ROW) < 0.15
        b[m] = rng.integers(0, 24, m.sum()).astype(DT[T])   # (under the guards' 16)
    shape = (256, 1024) if k['col'] else (1024, 256)
    b = b.reshape(shape)
    shared = not k['alias'] and all(s['cw'] == 'shared' for s in k['stmts'])
    c = c[:256].copy() if shared else c.reshape(shape)
    t = k['tconst'] if k['trip'] == 'const' else k['targ']
    fb, fc, fo = (os.path.join(tmp, x) for x in ('b.bin', 'c.bin', 'o.bin'))
    b.tofile(fb)
    c.tofile(fc)
    vb, vc = BASE, BASE + 0x1000000
    cva = vb if k['alias'] else vc
    cmd = [a.run, code, '--grid', '%d,1,1' % N, '--wg', '256,1,1',
           '--map', '%#x=%s' % (vb, fb), '--map', '%#x=%s' % (vc, fc), '--dump', '%#x=%d:%s' % (vb, b.nbytes, fo),
           '--uniform', '0=%d' % (cva & 0xffffffff), '--uniform', '1=%d' % (cva >> 32),
           '--uniform', '2=%d' % (vb & 0xffffffff), '--uniform', '3=%d' % (vb >> 32),
           '--uniform', '4=%d' % t] + consts
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return src, 'agx_run timed out'
    if r.returncode:
        return src, 'agx_run failed: ' + ((r.stderr + r.stdout).strip().splitlines() or [''])[-1][:200]
    got = np.fromfile(fo, DT[T]).reshape(shape)
    ref = model(k, t, c, b.copy())
    if T in FLOATS:
        diff = ~((got == ref) | (np.isnan(got) & np.isnan(ref)))
    else:
        diff = got != ref
    bad = np.argwhere(diff)
    if bad.size:
        e = tuple(bad[0])
        return src, 'element %s: %r, the model %r (%d elements differ)' % (e, got[e], ref[e], len(bad))
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True, help='the LLVM build bin directory (mlir-opt, mlir-translate, opt, llc)')
    ap.add_argument('--run', required=True, help='agx_run from the AGX Performance Simulator')
    ap.add_argument('--seeds', default='0:50')
    ap.add_argument('--mcpu', default='agx-g14c')
    ap.add_argument('--keep', default=None)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(':'))
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(lo, hi):
            src, err = run_seed(seed, a, tmp)
            if err:
                bad += 1
                print('seed %d: %s' % (seed, err))
                if a.keep:
                    os.makedirs(a.keep, exist_ok=True)
                    open(os.path.join(a.keep, 'seed%d.mlir' % seed), 'w').write(src)
    print('%d seeds, %d failing' % (hi - lo, bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
