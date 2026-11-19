#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's float code against a model of the hardware, on the AGX Performance Simulator.

Each seed generates a random kernel of 8-22 operations on f32 values: fadd / fsub / fmul (with registers and the
constants -0.5, 3, 0.1, inf), fma, fcmp of all fourteen predicates into selects, maximumnum / minimumnum, fneg /
fabs / copysign, sitofp / uitofp and fptosi.sat / fptoui.sat, and maximum / minimum (also nsz, L15-E452). It runs on 512 threads of floats over the whole
exponent range with 35% special values (zeros, denormals, normals at the edges, infinities, quiet and signalling
NaNs) and every output word is compared with the model's. The model is the hardware's, as the simulator's
recorded semantics give it: denormal sources read as zero and results below the smallest normal flushed (the
AGX backend keeps fp32 flushing, L15 notes), exact operations rounded once, min / max returning an operand as it
is in the IEEE output denormal mode. Not checked: NaN payloads (any NaN is a NaN), the sign of copysign from a
NaN, and min / max of two zeros or denormals that compare equal (the flushed zero, or an operand where LLVM
folded min(x, x) or min(x, min(x, y))) (EXPERIMENTS.md L15-E378).

  agx-fp-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
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
from fractions import Fraction

import numpy as np

MINN=2.0**-126
def bits(f): return struct.unpack('<I',struct.pack('<f',f))[0]
def fl(b): return struct.unpack('<f',struct.pack('<I',b&0xffffffff))[0]
def daz(b):
  if (b>>23)&0xff==0: return -0.0 if b>>31 else 0.0
  return fl(b)
def to_f32(x):
  # x: python float (f64) or Fraction; hardware: NaN canonical, |x|<MINN -> signed zero, else RNE
  if isinstance(x,Fraction):
    if x==0: return 0
    if abs(x)<MINN: return 0x80000000 if x<0 else 0
    f=float(x)  # RNE to f64 (exact enough? Fraction->float rounds correctly)
    # round Fraction to f32 correctly: use numpy on exact via two-step is double rounding; do it exactly:
    return round_frac_f32(x)
  if math.isnan(x): return 0x7fc00000
  if x!=0 and abs(x)<MINN: return 0x80000000 if math.copysign(1,x)<0 else 0
  try: return bits(x)
  except OverflowError: return 0x7f800000 if x>0 else 0xff800000
def round_frac_f32(x):
  s=x<0; a=-x if s else x
  e=math.floor(math.log2(a)) if a>0 else 0
  # adjust e so that 2^e <= a < 2^(e+1)
  while Fraction(2)**e>a: e-=1
  while Fraction(2)**(e+1)<=a: e+=1
  q=a/ (Fraction(2)**(e-23)); n=math.floor(q); r=q-n
  if r>Fraction(1,2) or (r==Fraction(1,2) and n%2==1): n+=1
  if n==2**24: n//=2; e+=1
  if e>127: return 0xff800000 if s else 0x7f800000
  if e<-126: return 0x80000000 if s else 0
  b=((e+127)<<23)|(n-2**23)
  return b|(0x80000000 if s else 0)
def isnanb(b): return (b>>23)&0xff==0xff and b&0x7fffff
def fadd(a,b): 
  x,y=daz(a),daz(b)
  if math.isnan(x) or math.isnan(y): return 0x7fc00000
  if math.isinf(x) or math.isinf(y):
    r=x+y; return 0x7fc00000 if math.isnan(r) else bits(r)
  if x==0 and y==0: return bits(x+y)
  ex=Fraction(x)+Fraction(y)
  if ex==0: return 0
  return round_frac_f32(ex) if abs(ex)>=MINN else (0x80000000 if ex<0 else 0)
def fmul(a,b):
  x,y=daz(a),daz(b)
  if math.isnan(x) or math.isnan(y): return 0x7fc00000
  if math.isinf(x) or math.isinf(y):
    r=x*y; return 0x7fc00000 if math.isnan(r) else bits(r)
  ex=Fraction(x)*Fraction(y)
  if ex==0: return bits(x*y)
  return round_frac_f32(ex) if abs(ex)>=MINN else (0x80000000 if ex<0 else 0)
def ffma(a,b,c):
  x,y,z=daz(a),daz(b),daz(c)
  if any(math.isnan(v) for v in (x,y,z)): return 0x7fc00000
  if any(math.isinf(v) for v in (x,y,z)):
    r=x*y+z if not (math.isinf(x*y) and math.isinf(z)) else (x*y)+z
    return 0x7fc00000 if math.isnan(r) else bits(r)
  ex=Fraction(x)*Fraction(y)+Fraction(z)
  if ex==0:
    p=x*y; return bits(p+z) if True else 0
  return round_frac_f32(ex) if abs(ex)>=MINN else (0x80000000 if ex<0 else 0)
def fcmp(p,a,b):
  x,y=daz(a),daz(b); un=math.isnan(x) or math.isnan(y)
  base={'eq':x==y,'gt':x>y,'ge':x>=y,'lt':x<y,'le':x<=y,'ne':x!=y and not un}
  if p=='ord': return not un
  if p=='uno': return un
  if p[0]=='o': return (not un) and base[p[1:]]
  return un or base[p[1:]] if p[1:]!='ne' else (un or x!=y)
def maxnumber(a,b,mx,same=False):
  if same: return a
  x,y=daz(a),daz(b)
  if math.isnan(x) and math.isnan(y): return 0x7fc00000
  if math.isnan(x): return b
  if math.isnan(y): return a
  if x==y==0: 
    neg=(math.copysign(1,x)<0, math.copysign(1,y)<0)
    z = (neg[0] and neg[1]) if mx else (neg[0] or neg[1])
    return 0x80000000 if z else 0
  # (IEEE output denormal mode: the operand's bits as they are)
  if mx: return a if x>y else b
  return a if x<y else b
def sitofp(v,signed):
  n=v-(1<<32) if signed and v>>31 else v
  return round_frac_f32(Fraction(n)) if n else 0
def fptoi_sat(a,signed):
  x=daz(a)
  if math.isnan(x): return 0
  lo,hi=(-2**31,2**31-1) if signed else (0,2**32-1)
  if math.isinf(x): n=hi if x>0 else lo
  else: n=max(lo,min(hi,math.trunc(x)))
  return n & 0xffffffff

PRED=['oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno']
def gen(seed):
  R=random.Random(seed)
  decl=set(); ops=[]; fv=[]; iv=[]; n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()']
  for i in range(4):
    v=nv(); f=nv()
    body+=[f'  %g{i} = getelementptr i32, ptr addrspace(1) %p, i32 {i*65536}', f'  %g{i}b = getelementptr i32, ptr addrspace(1) %g{i}, i32 %t', f'  {v} = load i32, ptr addrspace(1) %g{i}b', f'  {f} = bitcast i32 {v} to float']
    ops.append((v,'in',i)); ops.append((f,'id',v)); iv.append(v); fv.append(f)
  for _ in range(R.randint(8,22)):
    k=R.random(); r=nv(); a=R.choice(fv); b=R.choice(fv)
    if k<0.35:
      op=R.choice(['fadd','fsub','fmul'])
      if R.random()<0.25: b=R.choice(['-5.000000e-01','3.000000e+00','0x3FB99999A0000000','0x7FF0000000000000'])
      body.append(f'  {r} = {op} float {a}, {b}'); ops.append((r,op,a,b)); fv.append(r)
    elif k<0.5:
      c=R.choice(fv); decl.add('declare float @llvm.fma.f32(float, float, float)')
      body.append(f'  {r} = call float @llvm.fma.f32(float {a}, float {b}, float {c})'); ops.append((r,'fma',a,b,c)); fv.append(r)
    elif k<0.65:
      p=R.choice(PRED); c=nv(); x=R.choice(fv); y=R.choice(fv)
      body+=[f'  {c} = fcmp {p} float {a}, {b}', f'  {r} = select i1 {c}, float {x}, float {y}']; ops.append((r,'fsel',p,a,b,x,y)); fv.append(r)
    elif k<0.73:
      # (maximum / minimum, also nsz: one fcmpsel, L15-E452)
      fn=R.choice(['maximumnum','minimumnum','maximum','minimum']); decl.add(f'declare float @llvm.{fn}.f32(float, float)')
      nsz=fn in ('maximum','minimum') and R.random()<0.4
      body.append(f'  {r} = call {"nsz " if nsz else ""}float @llvm.{fn}.f32(float {a}, float {b})'); ops.append((r,fn+('_nsz' if nsz else ''),a,b)); fv.append(r)
    elif k<0.81:
      fn=R.choice(['fneg','fabs','copysign'])
      if fn=='fneg': body.append(f'  {r} = fneg float {a}'); ops.append((r,'fneg',a))
      elif fn=='fabs': decl.add('declare float @llvm.fabs.f32(float)'); body.append(f'  {r} = call float @llvm.fabs.f32(float {a})'); ops.append((r,'fabs',a))
      else: decl.add('declare float @llvm.copysign.f32(float, float)'); body.append(f'  {r} = call float @llvm.copysign.f32(float {a}, float {b})'); ops.append((r,'copysign',a,b))
      fv.append(r)
    elif k<0.88:
      i=R.choice(iv); s=R.choice(['sitofp','uitofp'])
      body.append(f'  {r} = {s} i32 {i} to float'); ops.append((r,s,i)); fv.append(r)
    else:
      s=R.choice(['fptosi','fptoui']); decl.add(f'declare i32 @llvm.{s}.sat.i32.f32(float)')
      body.append(f'  {r} = call i32 @llvm.{s}.sat.i32.f32(float {a})'); ops.append((r,s,a)); iv.append(r)
  outs=fv[4:][-6:]+iv[4:][-2:]
  for j,o in enumerate(outs):
    if o in fv:
      c=nv(); body.append(f'  {c} = bitcast float {o} to i32'); o2=c
    else: o2=o
    body+=[f'  %o{j} = getelementptr i32, ptr addrspace(1) %q, i32 {j*65536}', f'  %o{j}b = getelementptr i32, ptr addrspace(1) %o{j}, i32 %t', f'  store i32 {o2}, ptr addrspace(1) %o{j}b']
  L=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()']+sorted(decl)+['define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {']+body+['  ret void','}']
  return '\n'.join(L)+'\n', ops, outs
CONST={'1.000000e+00':bits(1.0),'-5.000000e-01':bits(-0.5),'3.000000e+00':bits(3.0),'0x3FB99999A0000000':bits(0.1),'0.000000e+00':0,'-0.000000e+00':0x80000000,'0x7FF0000000000000':0x7f800000}
def evaluate(ops,outs,ins,N):
  """Per thread, the outputs as (bits, don't-care mask): a NaN's sign (copysign from a NaN) is unspecified, and
  anything computed from an unspecified bit is not checked."""
  res=[]
  for lane in range(N):
    e={}; dc={}
    g=lambda x: CONST[x] if x in CONST else e[x]
    d=lambda x: 0 if x in CONST else dc[x]
    for op in ops:
      r,k=op[0],op[1]; m=0
      srcs=[x for x in op[2:] if isinstance(x,str) and (x in e or x in CONST)]
      if k=='in': v=int(ins[op[2]][lane])
      elif k=='id': v=e[op[2]]; m=dc[op[2]]
      elif k=='fneg': v=g(op[2])^0x80000000; m=d(op[2])
      elif k=='fabs': v=g(op[2])&0x7fffffff; m=d(op[2])&0x7fffffff
      elif k=='copysign':
        v=(g(op[2])&0x7fffffff)|(g(op[3])&0x80000000)
        m=(d(op[2])&0x7fffffff)|(0x80000000 if (d(op[3]) or isnanb(g(op[3]))) else 0)
      elif k=='fsel':
        c=fcmp(op[2],g(op[3]),g(op[4]))
        if d(op[3]) or d(op[4]): m=0xffffffff
        v=g(op[5]) if c else g(op[6]); m|= d(op[5]) if c else d(op[6])
      else:
        if any(d(x) for x in srcs): m=0xffffffff
        if k=='fadd': v=fadd(g(op[2]),g(op[3]))
        elif k=='fsub': v=fadd(g(op[2]),g(op[3])^0x80000000)
        elif k=='fmul': v=fmul(g(op[2]),g(op[3]))
        elif k=='fma': v=ffma(g(op[2]),g(op[3]),g(op[4]))
        elif k in ('maximumnum','minimumnum'):
          A,B=g(op[2]),g(op[3]); v=maxnumber(A,B,k=='maximumnum',op[2]==op[3])
          # two zeros / denormals that compare equal with different bits: the
          # flushed zero or an operand as it is (folds of min(x, min(x, y)),
          # select(c, x, x)): not checked
          if op[2]!=op[3] and daz(A)==0 and daz(B)==0 and ((A|B)&0x7fffff): m=0xffffffff
        elif k in ('maximum','minimum','maximum_nsz','minimum_nsz'):
          A,B=g(op[2]),g(op[3])
          if isnanb(A) or isnanb(B): v=0x7fc00000
          else:
            v=maxnumber(A,B,k.startswith('maximum'),op[2]==op[3])
            if op[2]!=op[3] and daz(A)==0 and daz(B)==0 and ((A|B)&0x7fffff): m=0xffffffff
            # (nsz: either zero for two of different signs)
            if k.endswith('_nsz') and op[2]!=op[3] and daz(A)==0 and daz(B)==0: m|=0x80000000
        elif k in ('sitofp','uitofp'): v=sitofp(e[op[2]],k=='sitofp')
        elif k in ('fptosi','fptoui'): v=fptoi_sat(g(op[2]),k=='fptosi')
      e[r]=v&0xffffffff; dc[r]=m
    res.append([(e[o],dc[o]) for o in outs])
  return res


EDGE = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
                 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x4F000000, 0xCF000000,
                 0x4F800000, 0x3F000000, 0x3FC00000, 0x40490FDB, 0x33800000, 0x0C000000], np.uint32)
N = 512
IN, OUT = 0x1000000, 0x2000000


def run_seed(seed, a, tmp):
    src, ops, outs = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    ins = []
    with np.errstate(over="ignore", under="ignore"):
        for _ in range(4):
            x = (rng.standard_normal(N) * np.exp2(rng.integers(-130, 130, N))).astype(np.float32).view(np.uint32)
            m = rng.random(N) < 0.35
            x[m] = rng.choice(EDGE, m.sum())
            ins.append(x)
    inp = np.zeros(4 * 65536, np.uint32)
    for i in range(4):
        inp[i * 65536:i * 65536 + N] = ins[i]
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nbytes = 8 * 65536 * 4
    subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "256,1,1", "--map", "%#x=%s" % (IN, fin),
                    "--alloc", "%#x=%d" % (OUT, nbytes), "--dump", "%#x=%d:%s" % (OUT, nbytes, fout),
                    "--uniform", "0=%d" % IN, "--uniform", "1=0", "--uniform", "2=%d" % OUT, "--uniform", "3=0"],
                   check=True, capture_output=True, timeout=600)
    got = np.fromfile(fout, np.uint32)
    ref = evaluate(ops, outs, ins, N)
    for j, o in enumerate(outs):
        g = got[j * 65536:j * 65536 + N]
        for i in range(N):
            ex, dm = ref[i][j]
            gg = int(g[i])
            if dm == 0xFFFFFFFF or (isnanb(gg) and isnanb(ex)):
                continue
            if (gg & ~dm & 0xFFFFFFFF) != (ex & ~dm & 0xFFFFFFFF):
                return src, "%s: thread %d gave %#x, the model %#x (inputs %s)" % (
                    o, i, gg, ex, [hex(int(x[i])) for x in ins])
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
