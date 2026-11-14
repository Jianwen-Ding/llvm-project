#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's software double precision against IEEE 754 (Python floats), on the AGX
Performance Simulator.

Each seed generates a kernel of 6-16 operations on doubles: fadd / fsub / fmul / fdiv, fma, sqrt, fcmp of all
fourteen predicates into selects, fneg / fabs, sitofp / uitofp of i32, fptosi.sat / fptoui.sat to i32, and a
round trip through float (fptrunc, fpext). It runs on 256 threads of doubles over the whole exponent range,
denormals included, with 35% special values (zeros, the smallest denormal, the largest finite, infinities,
NaNs, 2^52, 2^31 and 2^32, values that round to float's limits) and small integers, and every output is compared
with Python's IEEE double arithmetic (math.fma, a correctly rounded sqrt, float conversions with denormals).
NaN payloads are not checked (EXPERIMENTS.md L15-E384).

  agx-f64-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
                  [--mcpu agx-g14c] [--keep DIR]
"""

import argparse
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import warnings

import numpy as np

PRED=['oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno']
def d2b(x): return struct.unpack('<Q',struct.pack('<d',x))[0]
def b2d(b): return struct.unpack('<d',struct.pack('<Q',b))[0]
def isnand(b): return (b>>52)&0x7ff==0x7ff and b&((1<<52)-1)
def gen(seed):
  R=random.Random(seed); decl=set(); ops=[]; dv=[]; iv=[]; n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()']
  for i in range(4):
    w=nv(); d=nv()
    body+=[f'  %g{i} = getelementptr i64, ptr addrspace(1) %p, i32 {i*65536}', f'  %g{i}b = getelementptr i64, ptr addrspace(1) %g{i}, i32 %t', f'  {w} = load i64, ptr addrspace(1) %g{i}b', f'  {d} = bitcast i64 {w} to double']
    ops.append((w,'in',i)); ops.append((d,'id',w)); dv.append(d)
  for i in range(2):
    w=nv(); body+=[f'  {w} = trunc i64 %v{1+2*i} to i32']; ops.append((w,'lo',f'%v{1+2*i}')); iv.append(w)
  for _ in range(R.randint(6,16)):
    k=R.random(); r=nv(); a=R.choice(dv); b=R.choice(dv)
    if k<0.35:
      op=R.choice(['fadd','fsub','fmul','fdiv']); body.append(f'  {r} = {op} double {a}, {b}'); ops.append((r,op,a,b)); dv.append(r)
    elif k<0.45:
      c=R.choice(dv); decl.add('declare double @llvm.fma.f64(double, double, double)'); body.append(f'  {r} = call double @llvm.fma.f64(double {a}, double {b}, double {c})'); ops.append((r,'fma',a,b,c)); dv.append(r)
    elif k<0.52:
      decl.add('declare double @llvm.sqrt.f64(double)'); body.append(f'  {r} = call double @llvm.sqrt.f64(double {a})'); ops.append((r,'sqrt',a)); dv.append(r)
    elif k<0.64:
      p=R.choice(PRED); c=nv(); x=R.choice(dv); y=R.choice(dv)
      body+=[f'  {c} = fcmp {p} double {a}, {b}', f'  {r} = select i1 {c}, double {x}, double {y}']; ops.append((r,'fsel',p,a,b,x,y)); dv.append(r)
    elif k<0.7:
      fn=R.choice(['fneg','fabs'])
      if fn=='fneg': body.append(f'  {r} = fneg double {a}')
      else: decl.add('declare double @llvm.fabs.f64(double)'); body.append(f'  {r} = call double @llvm.fabs.f64(double {a})')
      ops.append((r,fn,a)); dv.append(r)
    elif k<0.78:
      i=R.choice(iv); s=R.choice(['sitofp','uitofp']); body.append(f'  {r} = {s} i32 {i} to double'); ops.append((r,s,i)); dv.append(r)
    elif k<0.86:
      s=R.choice(['fptosi','fptoui']); decl.add(f'declare i32 @llvm.{s}.sat.i32.f64(double)')
      body.append(f'  {r} = call i32 @llvm.{s}.sat.i32.f64(double {a})'); ops.append((r,s,a)); iv.append(r)
    else:
      f=nv(); body+=[f'  {f} = fptrunc double {a} to float', f'  {r} = fpext float {f} to double']; ops.append((r,'roundtrip',a)); dv.append(r)
  outs=dv[4:][-5:]+iv[2:][-2:]
  for j,o in enumerate(outs):
    if o in dv: c=nv(); body.append(f'  {c} = bitcast double {o} to i64')
    else: c=nv(); body.append(f'  {c} = zext i32 {o} to i64')
    body+=[f'  %o{j} = getelementptr i64, ptr addrspace(1) %q, i32 {j*65536}', f'  %o{j}b = getelementptr i64, ptr addrspace(1) %o{j}, i32 %t', f'  store i64 {c}, ptr addrspace(1) %o{j}b']
  L=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()']+sorted(decl)+['define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {']+body+['  ret void','}']
  return '\n'.join(L)+'\n', ops, outs, dv
def fcmp(p,x,y):
  un=math.isnan(x) or math.isnan(y)
  base={'eq':x==y,'gt':x>y,'ge':x>=y,'lt':x<y,'le':x<=y}
  if p=='ord': return not un
  if p=='uno': return un
  if p=='one': return (not un) and x!=y
  if p=='une': return un or x!=y
  return ((not un) and base[p[1:]]) if p[0]=='o' else (un or base[p[1:]])
def f32round(x):
  # double -> float, IEEE RNE with float denormals
  try: f=struct.unpack('<f',struct.pack('<f',x))[0]
  except OverflowError: f=math.copysign(math.inf,x)
  return f
def sat(x,signed):
  if math.isnan(x): return 0
  lo,hi=(-2**31,2**31-1) if signed else (0,2**32-1)
  n=hi if x==math.inf else lo if x==-math.inf else max(lo,min(hi,math.trunc(x)))
  return n&0xffffffff
def evaluate(ops,outs,dv,ins,N):
  res=[]
  for lane in range(N):
    e={}
    for op in ops:
      r,k=op[0],op[1]
      if k=='in': v=int(ins[op[2]][lane]); e[r]=v; continue
      if k=='id': e[r]=b2d(e[op[2]]); continue
      if k=='lo': e[r]=e[op[2]]&0xffffffff; continue
      g=lambda x:e[x]
      try:
        if k=='fadd': v=g(op[2])+g(op[3])
        elif k=='fsub': v=g(op[2])-g(op[3])
        elif k=='fmul': v=g(op[2])*g(op[3])
        elif k=='fdiv':
          a,b=g(op[2]),g(op[3])
          if b==0: v=math.nan if (a==0 or math.isnan(a)) else math.copysign(math.inf,a)*math.copysign(1,b)
          else: v=a/b
        elif k=='fma':
          a,b,c=g(op[2]),g(op[3]),g(op[4])
          try: v=math.fma(a,b,c)
          except OverflowError:
            from fractions import Fraction
            v=-math.inf if Fraction(a)*Fraction(b)+Fraction(c)<0 else math.inf
        elif k=='sqrt': a=g(op[2]); v=math.nan if (a<0 or math.isnan(a)) else math.sqrt(a)
        elif k=='fsel': v=g(op[5]) if fcmp(op[2],g(op[3]),g(op[4])) else g(op[6])
        elif k=='fneg': v=b2d(d2b(g(op[2]))^(1<<63))
        elif k=='fabs': v=b2d(d2b(g(op[2]))&~(1<<63))
        elif k in ('sitofp','uitofp'): x=g(op[2]); v=float(x-(1<<32) if k=='sitofp' and x>>31 else x)
        elif k in ('fptosi','fptoui'): v=sat(g(op[2]),k=='fptosi')
        elif k=='roundtrip': v=f32round(g(op[2]))
      except OverflowError: v=math.inf
      except ValueError: v=math.nan
      e[r]=v
    res.append([e[o] if isinstance(e[o],int) else d2b(e[o]) for o in outs])
  return res


N = 256
EDGE = np.array([0, 1 << 63, 1, (1 << 63) | 1, (1 << 52) - 1, 1 << 52, 0x3FF0000000000000, 0xBFF0000000000000,
                 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001,
                 0x4330000000000000, 0x41E0000000000000, 0xC1E0000000000000, 0x41F0000000000000, 0x3FE0000000000000,
                 0x36A0000000000000, 0x3810000000000000, 0x47EFFFFFE0000000], np.uint64)


def run_seed(seed, a, tmp):
    src, ops, outs, dv = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    ins = []
    inp = np.zeros(4 * 65536, np.uint64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(4):
            x = (rng.standard_normal(N) * np.exp2(rng.integers(-1060, 1030, N).astype(np.float64))).view(np.uint64)
            m = rng.random(N) < 0.35
            x[m] = rng.choice(EDGE, m.sum())
            if i == 3:
                x[::4] = rng.integers(-3000000, 3000000, len(x[::4])).astype(np.int64).astype(np.uint64)
            ins.append(x)
            inp[i * 65536:i * 65536 + N] = x
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nb = 7 * 65536 * 8
    rr = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "64,1,1", "--map", "0x1000000=" + fin,
                         "--alloc", "0x3000000=%d" % nb, "--dump", "0x3000000=%d:%s" % (nb, fout),
                         "--uniform", "0=16777216", "--uniform", "1=0", "--uniform", "2=50331648", "--uniform", "3=0"],
                        capture_output=True, text=True, timeout=900)
    if rr.returncode:
        return src, "agx_run failure: " + (rr.stdout + rr.stderr)[-200:]
    got = np.fromfile(fout, np.uint64)
    ref = evaluate(ops, outs, dv, ins, N)
    for j, o in enumerate(outs):
        for th in range(N):
            g, ex = int(got[j * 65536 + th]), ref[th][j]
            if g != ex and not (o in dv and isnand(g) and isnand(ex)):
                return src, "%s: thread %d gave %#x, IEEE %#x" % (o, th, g, ex)
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--seeds", default="0:200")
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
