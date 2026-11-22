#!/usr/bin/env python3
"""Check convert-gpu-to-air's small-float lowerings on the AGX simulator.

Compiles kernels of arith.extf / arith.truncf / arithmetic on the f8, f6 and
f4 types with bench.py's pipeline, runs them on agx_run and compares every
result with LLVM's APFloat (apfloat-ref.cpp, built against LLVMSupport):

  * extf of every code to f32;
  * truncf to the type of every value, its neighbours, the midpoints of
    neighbouring values and theirs, random and special inputs, from f32 and
    from f64;
  * addf / mulf / divf of every pair of codes, and (x * y) + x.

A NaN result compares as any NaN where LLVM does not specify its payload
or sign (an extension's, an IEEE-like format's truncation, arithmetic's);
finite-only formats get
no NaN or infinity inputs, and no division (APFloat leaves those undefined).
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench'))
import bench  # noqa: E402

TYPES = {'f8E4M3FN': 8, 'f8E5M2': 8, 'f8E4M3FNUZ': 8, 'f8E5M2FNUZ': 8,
         'f8E4M3B11FNUZ': 8, 'f8E3M4': 8, 'f8E4M3': 8, 'f6E2M3FN': 6,
         'f6E3M2FN': 6, 'f4E2M1FN': 4}
FINITE = {'f6E2M3FN', 'f6E3M2FN', 'f4E2M1FN'}
IEEE = {'f8E5M2', 'f8E3M4', 'f8E4M3'}

KERNEL = '''gpu.module @k {
  gpu.func @k(%%a: memref<65536x%s, 1>, %%o: memref<65536x%s, 1>) kernel {
    %%i = gpu.global_id x
    %%v = memref.load %%a[%%i] : memref<65536x%s, 1>
%s
    memref.store %%r, %%o[%%i] : memref<65536x%s, 1>
    gpu.return
  }
}
'''


class Checker:
  def __init__(self, a):
    self.a = a
    self.bad = 0

  def ref(self, t, mode, xs):
    out = subprocess.run([self.a.ref, t, mode],
                         input='\n'.join('%x' % int(x) for x in xs),
                         capture_output=True, text=True, check=True).stdout
    return [int(l.split()[0], 16) for l in out.splitlines() if l]

  def run(self, body, tin, tout, inp, dt_in, dt_out, n):
    src = KERNEL % (tin, tout, tin, body, tout)
    with tempfile.TemporaryDirectory() as t:
      f = os.path.join(t, 'k.mlir')
      open(f, 'w').write(src)
      ll = bench.lower(os.path.join(self.a.bin, 'mlir-opt'), self.a.bin, f, t)
      grid = (n + 255) // 256 * 256
      r = bench.cost(self.a.bin, self.a.cost, ll, t, self.a.mcpu,
                     ['--grid', '%d,1,1' % grid, '--wg', '256,1,1'])
      extra = []
      args = r['args']
      for k in range(0, len(args), 2):
        if args[k] == '--uniform' and args[k + 1].split('=')[0] not in ('1', '3'):
          extra += args[k:k + 2]
      buf = np.zeros(1 << 16, dt_in)
      buf[:len(inp)] = inp
      buf.tofile(os.path.join(t, 'a.bin'))
      nb = (1 << 16) * 8
      subprocess.run(
          [self.a.run, os.path.join(t, 'k.bin'), '--grid', '%d,1,1' % grid,
           '--wg', '256,1,1', '--map', '0x1000000=' + os.path.join(t, 'a.bin'),
           '--alloc', '0x3000000=%d' % nb,
           '--dump', '0x3000000=%d:%s' % (nb, os.path.join(t, 'o.bin')),
           '--uniform', '0=16777216', '--uniform', '1=0',
           '--uniform', '2=50331648', '--uniform', '3=0'] + extra,
          check=True, capture_output=True)
      return np.fromfile(os.path.join(t, 'o.bin'), dt_out)[:n].astype(np.int64)

  def report(self, what, got, ref, ok_nan=None):
    m = [k for k in np.nonzero(got != np.array(ref))[0]
         if not (ok_nan and ok_nan(int(got[k]), int(ref[k])))]
    self.bad += len(m)
    print('%-34s %6d checked, %d wrong%s' % (
        what, len(ref), len(m),
        ''.join(' [%d: %#x not %#x]' % (k, got[k], ref[k]) for k in m[:3])))

  def check(self, t, w):
    n = 1 << w
    f32nan = lambda g, r: (g & 0x7fffffff) > 0x7f800000 and (r & 0x7fffffff) > 0x7f800000
    # extension of every code
    cv = '%v' if w == 8 else '%t'
    body = (('    %%t = arith.trunci %%v : i8 to i%d\n' % w) if w < 8 else '') + (
        '    %%f = arith.bitcast %s : i%d to %s\n    %%e = arith.extf %%f : %s to f32\n'
        '    %%r = arith.bitcast %%e : f32 to i32' % (cv, w, t, t))
    ref = self.ref(t, 'ext', range(n))
    got = self.run(body, 'i8', 'i32', np.arange(n, dtype=np.uint8), np.uint8,
                   np.uint32, n)
    self.report(t + ' extf', got, ref, f32nan)
    nancodes = {c for c in range(n) if (ref[c] & 0x7fffffff) > 0x7f800000}
    code_nan = (lambda g, r: g in nancodes and r in nancodes and
                (g >> (w - 1)) == (r >> (w - 1))) if t in IEEE else None
    # truncation from f32
    vals = np.array(ref, np.uint32).view(np.float32)
    fin = np.unique(np.abs(vals[np.isfinite(vals)]))
    cand = np.concatenate([fin, (fin[1:] + fin[:-1]) / 2, [fin[-1] * 1.5, fin[-1] * 2]])
    bits = cand.astype(np.float32).view(np.uint32).astype(np.int64)
    rng = np.random.default_rng(w)
    xs = set()
    for d in range(-3, 4):
      xs.update(((bits + d) & 0xffffffff).tolist())
    xs.update(rng.integers(0, 1 << 31, 1500).tolist())
    xs.update([0, 1, 0x7fffff, 0x800000, 0x7f7fffff, 0x7f800000, 0x7fc00000,
               0x7f800001])
    xs = sorted(xs)
    xs = xs + [x | 0x80000000 for x in xs]
    if t in FINITE:
      xs = [x for x in xs if (x & 0x7fffffff) < 0x7f800000]
    body = ('    %%f = arith.bitcast %%v : i32 to f32\n    %%t = arith.truncf %%f : f32 to %s\n'
            '    %%b = arith.bitcast %%t : %s to i%d\n    %%r = arith.extui %%b : i%d to i32'
            % (t, t, w, w))
    got = self.run(body, 'i32', 'i32', np.array(xs, np.uint32), np.uint32,
                   np.uint32, len(xs))
    self.report(t + ' truncf from f32', got, self.ref(t, 'trunc32', xs), code_nan)
    # truncation from f64
    d = np.array(xs, np.uint32).view(np.float32).astype(np.float64)
    ds = sorted({(int(x) + dd) & 0xffffffffffffffff
                 for x in d.view(np.uint64).tolist() for dd in (-1, 0, 1)})[:65536]
    if t in FINITE:
      ds = [x for x in ds if (x & 0x7fffffffffffffff) < 0x7ff0000000000000]
    body = ('    %%f = arith.bitcast %%v : i64 to f64\n    %%t = arith.truncf %%f : f64 to %s\n'
            '    %%b = arith.bitcast %%t : %s to i%d\n    %%r = arith.extui %%b : i%d to i32'
            % (t, t, w, w))
    got = self.run(body, 'i64', 'i32', np.array(ds, np.uint64), np.uint64,
                   np.uint32, len(ds))
    self.report(t + ' truncf from f64', got, self.ref(t, 'trunc64', ds), code_nan)
    # arithmetic of every pair
    ops = ('add', 'mul', 'mad') + (() if t in FINITE else ('div',))
    pairs = np.arange(n * n)
    x, y = pairs % n, pairs // n
    for op in ops:
      got = []
      for c0 in range(0, n * n, 32768):
        cx, cy = x[c0:c0 + 32768], y[c0:c0 + 32768]
        buf = np.zeros(1 << 16, np.uint8)
        buf[:len(cx)] = cx
        buf[32768:32768 + len(cy)] = cy
        tr = lambda v, o: ('    %%%s = arith.trunci %s : i8 to i%d\n' % (o, v, w)) if w < 8 else ''
        fx = '%tx' if w < 8 else '%v'
        fy = '%ty' if w < 8 else '%w'
        body = (tr('%v', 'tx') +
                '    %o2 = arith.constant 32768 : index\n    %k = arith.addi %i, %o2 : index\n'
                '    %w = memref.load %a[%k] : memref<65536xi8, 1>\n' + tr('%w', 'ty') +
                '    %%fx = arith.bitcast %s : i%d to %s\n    %%fy = arith.bitcast %s : i%d to %s\n'
                % (fx, w, t, fy, w, t))
        if op == 'mad':
          body += ('    %%p = arith.mulf %%fx, %%fy : %s\n    %%s = arith.addf %%p, %%fx : %s\n' % (t, t))
        else:
          body += '    %%s = arith.%sf %%fx, %%fy : %s\n' % (op, t)
        body += '    %%rb = arith.bitcast %%s : %s to i%d\n' % (t, w)
        body += ('    %%r = arith.extui %%rb : i%d to i8' % w) if w < 8 else '    %r = arith.ori %rb, %rb : i8'
        got += list(self.run(body, 'i8', 'i8', buf, np.uint8, np.uint8, len(cx)))
      ref = self.ref(t, op, [int(a) | (int(b) << w) for a, b in zip(x, y)])
      self.report('%s %sf of every pair' % (t, op), np.array(got), ref,
                  lambda g, r: g in nancodes and r in nancodes)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('--bin', required=True, help='LLVM build bin directory')
  ap.add_argument('--run', required=True, help="the simulator's agx_run")
  ap.add_argument('--cost', required=True, help="the simulator's agx_cost")
  ap.add_argument('--ref', required=True, help='apfloat-ref, built')
  ap.add_argument('-mcpu', default='agx-g14c')
  ap.add_argument('types', nargs='*', default=list(TYPES))
  a = ap.parse_args()
  c = Checker(a)
  for t in a.types:
    c.check(t, TYPES[t])
  print('%d wrong' % c.bad)
  return 1 if c.bad else 0


if __name__ == '__main__':
  sys.exit(main())
