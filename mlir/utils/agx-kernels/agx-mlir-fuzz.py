#!/usr/bin/env python3
"""Differential fuzz of MLIR row and column loops through convert-gpu-to-air and the AGX backend, against a Python
model, on the AGX Performance Simulator.

Each seed generates a gpu.func whose threads each update their row of a 1024x256 memref b (or their column of a
256x1024 one) in an scf.for of one to three statements, in program order:

  b[t, j + o] = f(b[t, j + o'] or a constant, c[...])     (some under scf.if g(...) { ... } else { b's old value })

c a second memref at the thread's row (column) or at j + o'' alone (every thread's same elements), or b itself
passed twice; the trip a constant or a runtime i32, steps of 1 or 2; the elements i32, i16 or f32 (small integers,
exact sums and products); f an add, sub, xor, a multiply by a constant or a max, g an unsigned or ordered compare,
and a guarded statement's c load inside its scf.if (speculate-guarded-loads). Lowered as bench.py does
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
    # a running value carried by the loop (iter_args), f(acc, c[...]) each
    # iteration, stored after the loop at the row's last element
    k['red'] = R.random() < 0.35
    k['rop'] = R.choice(['add', 'sub', 'mul', 'max'] + (['xor'] if k['ty'] != 'f32' else []))
    k['rlc'] = R.randrange(0, 64)
    k['rinit'] = R.choice([0, 1, 5])
    # c at the thread's row (column) or at j + o alone, for every statement
    cw = 'own' if k['alias'] or R.random() < 0.7 else 'shared'
    ops = ['add', 'sub', 'mul', 'max'] + (['xor'] if k['ty'] != 'f32' else [])
    for _ in range(R.randint(1, 3)):
        k['stmts'].append(dict(
            st=R.randrange(0, 64), lb=R.choice([None, R.randrange(0, 64)]),
            same=R.random() < 0.5, cw=cw,
            lc=R.randrange(0, 64), op=R.choice(ops), kc=R.choice([1, 3, 7, -2]),
            guard=R.random() < 0.35))
        s = k['stmts'][-1]
        if s['same']:
            s['lb'] = s['st']
    return k


def emit(k):
    T = k['ty']
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
    if k['red']:
        L.append('    %%init = arith.constant %s : %s' % (('%d.0' % k['rinit']) if flt else str(k['rinit']), T))
        L.append('    %%res = scf.for %%j = %%c0 to %%T step %%st iter_args(%%acc = %%init) -> (%s) {' % T)
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
        if not own:
            return '%s[%s] : memref<256x%s, 1>' % (mem, j, T)
        sub = '%s, %%t' % j if k['col'] else '%%t, %s' % j
        return '%s[%s] : memref<%sx%s, 1>' % (mem, sub, shape, T)

    def const(v):
        c = nv()
        L.append('      %s = arith.constant %s : %s' % (c, ('%d.0' % v) if flt else str(v), T))
        return c

    def op(o, a, b, kc):
        r = nv()
        if o == 'mul':
            m = nv()
            L.append('      %s = arith.%s %s, %s : %s' % (m, 'mulf' if flt else 'muli', b, const(kc), T))
            L.append('      %s = arith.%s %s, %s : %s' % (r, 'addf' if flt else 'addi', a, m, T))
        elif o == 'max':
            L.append('      %s = arith.%s %s, %s : %s' % (r, 'maximumf' if flt else 'maxui', a, b, T))
        else:
            name = {'add': 'addf', 'sub': 'subf'}[o] if flt else {'add': 'addi', 'sub': 'subi', 'xor': 'xori'}[o]
            L.append('      %s = arith.%s %s, %s : %s' % (r, name, a, b, T))
        return r

    cmem = '%b' if k['alias'] else '%c'
    for s in k['stmts']:
        own = s['cw'] == 'own'
        if s['lb'] is not None:
            bv = nv()
            L.append('      %s = memref.load %s' % (bv, access('%b', s['lb'], True, bshape)))
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
            L.append('      %s = memref.load %s' % (cv, access(cmem, s['lc'], own, bshape)))
            y = op(s['op'], bv, cv, s['kc'])
            L.append('      scf.yield %s : %s' % (y, T))
            L.append('      } else {')
            L.append('      scf.yield %s : %s' % (bv, T))
            L.append('      }')
        else:
            cv = nv()
            L.append('      %s = memref.load %s' % (cv, access(cmem, s['lc'], own, bshape)))
            r = op(s['op'], bv, cv, s['kc'])
        L.append('      memref.store %s, %s' % (r, access('%b', s['st'], True, bshape)))
    if k['red']:
        cv = nv()
        L.append('      %s = memref.load %s' % (cv, access(cmem, k['rlc'], k['stmts'][0]['cw'] == 'own', bshape)))
        acc = op(k['rop'], '%acc', cv, 3)
        L.append('      scf.yield %s : %s' % (acc, T))
        L.append('    }')
        last = nv()
        L.append('    %s = arith.constant 255 : index' % last)
        sub = '%s, %%t' % last if k['col'] else '%%t, %s' % last
        L.append('    memref.store %%res, %%b[%s] : memref<%sx%s, 1>' % (sub, bshape, T))
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

    def at(arr, off, own, j):
        if not own:
            return (j + off,)
        return (j + off, tid) if k['col'] else (tid, j + off)

    def f(o, a, x, kc):
        if flt:
            with np.errstate(all='ignore'):
                if o == 'add':
                    return (a + x).astype(np.float32)
                if o == 'sub':
                    return (a - x).astype(np.float32)
                if o == 'mul':
                    return (a + (x * np.float32(kc)).astype(np.float32)).astype(np.float32)
                return np.where(np.isnan(a) | np.isnan(x), np.float32('nan'), np.maximum(a, x)).astype(np.float32)
        a64, x64 = a.astype(np.int64), x.astype(np.int64)
        if o == 'add':
            r = a64 + x64
        elif o == 'sub':
            r = a64 - x64
        elif o == 'xor':
            r = a64 ^ x64
        elif o == 'mul':
            r = a64 + x64 * kc
        else:
            r = np.maximum(a64, x64)
        return (r & MW).astype(DT[T])

    j = 0
    acc = np.full(N, k['rinit'], DT[T]) if k.get('red') else None
    last = (255, tid) if k['col'] else (tid, 255)
    if not 0 < t:
        if k.get('red'):
            b[last] = acc
        return b
    while j < t:
        for s in k['stmts']:
            own = s['cw'] == 'own'
            if s['lb'] is not None:
                bv = b[at(b, s['lb'], True, j)].copy()
            else:
                bv = np.full(N, s['kc'] if flt else s['kc'] & MW, DT[T])
            cm = b if k['alias'] else c
            cv = cm[at(cm, s['lc'], own, j)]
            cv = cv.copy() if own else np.full(N, cv, DT[T])
            r = f(s['op'], bv, cv, s['kc'])
            if s['guard']:
                g = (bv < np.float32(16)) if flt else (bv.astype(np.int64) < 16)
                r = np.where(g, r, bv).astype(DT[T])
            b[at(b, s['st'], True, j)] = r
        if k.get('red'):
            own = k['stmts'][0]['cw'] == 'own'
            cm = b if k['alias'] else c
            cv = cm[at(cm, k['rlc'], own, j)]
            cv = cv.copy() if own else np.full(N, cv, DT[T])
            acc = f(k['rop'], acc, cv, 3)
        j += k['step']
    if k.get('red'):
        b[last] = acc
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


def run_seed(seed, a, tmp):
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
