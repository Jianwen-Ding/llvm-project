#!/usr/bin/env python3
"""Differential fuzz of MLIR row and column loops through convert-gpu-to-air and the AGX backend, against a Python
model, on the AGX Performance Simulator.

Each seed generates a gpu.func whose threads each update their row of a 1024x256 memref b (or their column of a
256x1024 one) in an scf.for of one to three statements, in program order:

  b[t, j + o] = f(b[t, j + o'] or a constant, c[...])     (some under scf.if g(...) { ... } else { b's old value })

c a second memref at the thread's row (column) or at j + o'' alone (every thread's same elements), or b itself
passed twice; on rows, a third of the seeds by vector<4> loads and stores in steps of 4 or 8 (the reduction a
vector, then vector.reduction); the trip a constant or a runtime i32, steps of 1 or 2; the elements i32, i16 or f32 (small integers,
exact sums and products); f an add, sub, xor, a multiply by a constant, a max or min (maximumf / minimumf / maxnumf / minnumf), g an unsigned or ordered compare,
and a guarded statement's c load inside its scf.if (speculate-guarded-loads). An eighth of the seeds instead compute float math over a row (exp, e^(x - k), x > k ? e^(x - k) : x, tanh, the
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
BASE = 0x100000000


def gen(seed):
    R = random.Random(seed)
    k = dict(ty=R.choice(['i32', 'i32', 'i16', 'f32']),
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
                        (['xor'] if k['ty'] != 'f32' else ['maxnum', 'minnum']))
    k['rlc'] = R.randrange(0, 64) & (~3 if k['vec'] else ~0)
    k['rinit'] = R.choice([0, 1, 5])
    # c at the thread's row (column) or at j + o alone, for every statement
    cw = 'own' if k['alias'] or R.random() < 0.7 else 'shared'
    ops = ['add', 'sub', 'mul', 'max', 'min'] + (['xor'] if k['ty'] != 'f32' else ['maxnum', 'minnum'])
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
    flt = T == 'f32'
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


DT = {'i32': np.uint32, 'i16': np.uint16, 'f32': np.float32}


def model(k, t, c, b):
    """b (updated in place) after every thread's loop; c is b where aliased."""
    T = k['ty']
    flt = T == 'f32'
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
                    return (a + x).astype(np.float32)
                if o == 'sub':
                    return (a - x).astype(np.float32)
                if o == 'mul':
                    return (a + (x * np.float32(kc)).astype(np.float32)).astype(np.float32)
                # (no NaN arises: the values are small integers)
                return (np.maximum if o in ('max', 'maxnum') else np.minimum)(a, x).astype(np.float32)
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
                    r = (r + acc[:, l]).astype(np.float32)
                return r
            return (acc.astype(np.int64).sum(axis=1) & MW).astype(DT[T])
        if rop == 'xor':
            return np.bitwise_xor.reduce(acc, axis=1)
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
                g = (bv < np.float32(16)) if flt else (bv.astype(np.int64) < 16)
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


MATH = ['exp', 'expk', 'condexp', 'tanh', 'sigmoid', 'log1', 'expscale', 'maxexp', 'online', 'online']


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


def run_seed(seed, a, tmp):
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
    if T == 'f32':
        b = rng.integers(-40, 40, N * ROW).astype(np.float32)
        c = rng.integers(-40, 40, N * ROW).astype(np.float32)
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
    if T == 'f32':
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
