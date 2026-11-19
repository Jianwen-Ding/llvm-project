#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's half precision code against a model of the hardware, on the AGX
Performance Simulator.

Each seed generates a kernel of 8-20 operations on half values: fadd / fsub / fmul, fma, fcmp of all fourteen
predicates into selects, maximumnum / minimumnum, fneg / fabs, fpext to float with a float add or multiply and
fptrunc back, sitofp / uitofp of i32 and fptosi.sat / fptoui.sat. It runs on 512 threads (40% special halves:
zeros, denormals, the largest finite, infinities, NaNs; small and out-of-range integers) and every output is
compared with the model: half keeps denormals and rounds each exact result once, float flushes its denormals
(the hardware's, as the simulator's recorded semantics give it), min / max of equal halves combine their bits.
NaN payloads are not checked (EXPERIMENTS.md L15-E383).

  agx-half-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
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

daz32, bits32 = daz, bits
def h2f(h):
  s=-1 if h>>15 else 1; e=(h>>10)&31; m=h&1023
  if e==31: return float('nan') if m else s*float('inf')
  if e==0: return math.copysign(m*2.0**-24, s)
  return math.copysign((1+m/1024)*2.0**(e-15), s)
def isnanh(h): return (h>>10)&31==31 and h&1023
def round_frac_f16(x):
  # RNE to half with denormals; x Fraction nonzero
  s=x<0; a=-x if s else x
  e=math.floor(math.log2(a))
  while Fraction(2)**e>a: e-=1
  while Fraction(2)**(e+1)<=a: e+=1
  if e<-14: e=-14  # denormal range: quantum 2^-24
  q=a/(Fraction(2)**(e-10)); n=math.floor(q); r=q-n
  if r>Fraction(1,2) or (r==Fraction(1,2) and n%2==1): n+=1
  if n>=2048: n//=2; e+=1   # (n==2048 exactly)
  if e>15: return 0xfc00 if s else 0x7c00
  if n<1024: b=n  # denormal
  else: b=((e+15)<<10)|(n-1024)
  return b|(0x8000 if s else 0)
def to_h(x):
  # from a python float / Fraction exact value
  if isinstance(x,float):
    if math.isnan(x): return 0x7e00
    if math.isinf(x): return 0x7c00 if x>0 else 0xfc00
    if x==0: return 0x8000 if math.copysign(1,x)<0 else 0
    x=Fraction(x)
  if x==0: return 0
  return round_frac_f16(x)
def hop(op,a,b,c=None):
  x,y=h2f(a),h2f(b); z=h2f(c) if c is not None else None
  vals=[v for v in (x,y,z) if v is not None]
  if any(math.isnan(v) for v in vals): return 0x7e00
  if any(math.isinf(v) for v in vals):
    r={'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'fma':lambda:x*y+z}[op]()
    return to_h(r)
  F=Fraction
  if op=='add': ex=F(x)+F(y); zr=x+y
  elif op=='sub': ex=F(x)-F(y); zr=x-y
  elif op=='mul': ex=F(x)*F(y); zr=x*y
  else: ex=F(x)*F(y)+F(z); zr=x*y+z
  if ex==0: return to_h(zr) if zr==0 else 0
  return round_frac_f16(ex)
def fptrunc(b32):
  x=daz32(b32)
  if math.isnan(x): return 0x7e00
  if math.isinf(x) or x==0: return to_h(x)
  return round_frac_f16(Fraction(x))
def fpext(h):
  x=h2f(h)
  if math.isnan(x): return 0x7fc00000
  if math.isinf(x) or x==0: return bits32(x)
  return round_frac_f32(Fraction(x))

fadd32, fmul32 = fadd, fmul


def fmax32_0(s):
  """maximumnum(s, +0) of a float's bits: +0 for a NaN, -0 or anything below."""
  if (s & 0x7f800000) == 0x7f800000 and s & 0x7fffff: return 0
  if s >> 31: return 0
  return s
PRED=['oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno']
def hcmp(p,a,b):
  x,y=h2f(a),h2f(b); un=math.isnan(x) or math.isnan(y)
  base={'eq':x==y,'gt':x>y,'ge':x>=y,'lt':x<y,'le':x<=y}
  if p=='ord': return not un
  if p=='uno': return un
  if p=='one': return (not un) and x!=y
  if p=='une': return un or x!=y
  return ((not un) and base[p[1:]]) if p[0]=='o' else (un or base[p[1:]])
def hmax(a,b,mx,same):
  if same: return a
  x,y=h2f(a),h2f(b)
  if math.isnan(x) and math.isnan(y): return 0x7e00
  if math.isnan(x): return b
  if math.isnan(y): return a
  if x==y: return (a&b) if mx else (a|b)
  return (a if x>y else b) if mx else (a if x<y else b)
def gen(seed):
  R=random.Random(seed); decl=set(); ops=[]; hv=[]; iv=[]; n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()']
  for i in range(4):
    w=nv(); h=nv(); x=nv()
    body+=[f'  %g{i} = getelementptr i32, ptr addrspace(1) %p, i32 {i*65536}', f'  %g{i}b = getelementptr i32, ptr addrspace(1) %g{i}, i32 %t', f'  {w} = load i32, ptr addrspace(1) %g{i}b', f'  {x} = trunc i32 {w} to i16', f'  {h} = bitcast i16 {x} to half']
    ops.append((w,'in',i)); ops.append((h,'lo16',w)); iv.append(w); hv.append(h)
  for _ in range(R.randint(8,20)):
    k=R.random(); r=nv(); a=R.choice(hv); b=R.choice(hv)
    if k<0.3:
      op=R.choice(['fadd','fsub','fmul']); body.append(f'  {r} = {op} half {a}, {b}'); ops.append((r,op,a,b)); hv.append(r)
    elif k<0.42:
      c=R.choice(hv); decl.add('declare half @llvm.fma.f16(half, half, half)'); body.append(f'  {r} = call half @llvm.fma.f16(half {a}, half {b}, half {c})'); ops.append((r,'fma',a,b,c)); hv.append(r)
    elif k<0.55:
      p=R.choice(PRED); c=nv(); x=R.choice(hv); y=R.choice(hv)
      body+=[f'  {c} = fcmp {p} half {a}, {b}', f'  {r} = select i1 {c}, half {x}, half {y}']; ops.append((r,'fsel',p,a,b,x,y)); hv.append(r)
    elif k<0.63:
      fn=R.choice(['maximumnum','minimumnum']); decl.add(f'declare half @llvm.{fn}.f16(half, half)')
      body.append(f'  {r} = call half @llvm.{fn}.f16(half {a}, half {b})'); ops.append((r,fn,a,b)); hv.append(r)
    elif k<0.75:
      # widen, f32 arithmetic, narrow
      ea=nv(); eb=nv(); s=nv(); op=R.choice(['fadd','fmul'])
      body+=[f'  {ea} = fpext half {a} to float', f'  {eb} = fpext half {b} to float', f'  {s} = {op} float {ea}, {eb}', f'  {r} = fptrunc float {s} to half']
      ops.append((r,'ext32',op,a,b)); hv.append(r)
    elif k<0.78:
      # two f32 operations before the narrowing (its float intermediate is
      # not a half: no half operations for it, L15-E440), or one under a
      # maximumnum with 0
      c=R.choice(hv); ea=nv(); eb=nv(); ec=nv(); s=nv(); t=nv(); op=R.choice(['fadd','fmul']); op2=R.choice(['fadd','fmul','max0'])
      body+=[f'  {ea} = fpext half {a} to float', f'  {eb} = fpext half {b} to float', f'  {ec} = fpext half {c} to float', f'  {s} = {op} float {ea}, {eb}']
      if op2=='max0':
        decl.add('declare float @llvm.maximumnum.f32(float, float)'); body.append(f'  {t} = call float @llvm.maximumnum.f32(float {s}, float 0.000000e+00)')
      else:
        body.append(f'  {t} = {op2} float {s}, {ec}')
      body.append(f'  {r} = fptrunc float {t} to half')
      ops.append((r,'ext32b',op,op2,a,b,c)); hv.append(r)
    elif k<0.82:
      fn=R.choice(['fneg','fabs'])
      if fn=='fneg': body.append(f'  {r} = fneg half {a}')
      else: decl.add('declare half @llvm.fabs.f16(half)'); body.append(f'  {r} = call half @llvm.fabs.f16(half {a})')
      ops.append((r,fn,a)); hv.append(r)
    elif k<0.9:
      i=R.choice(iv); s=R.choice(['sitofp','uitofp']); body.append(f'  {r} = {s} i32 {i} to half'); ops.append((r,s,i)); hv.append(r)
    else:
      s=R.choice(['fptosi','fptoui']); decl.add(f'declare i32 @llvm.{s}.sat.i32.f16(half)')
      body.append(f'  {r} = call i32 @llvm.{s}.sat.i32.f16(half {a})'); ops.append((r,s,a)); iv.append(r)
  outs=hv[4:][-6:]+iv[4:][-2:]
  for j,o in enumerate(outs):
    if o in hv:
      c=nv(); z=nv(); body+=[f'  {c} = bitcast half {o} to i16', f'  {z} = zext i16 {c} to i32']; o2=z
    else: o2=o
    body+=[f'  %o{j} = getelementptr i32, ptr addrspace(1) %q, i32 {j*65536}', f'  %o{j}b = getelementptr i32, ptr addrspace(1) %o{j}, i32 %t', f'  store i32 {o2}, ptr addrspace(1) %o{j}b']
  L=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()']+sorted(decl)+['define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {']+body+['  ret void','}']
  return '\n'.join(L)+'\n', ops, outs, hv
def sat_h(a,signed):
  x=h2f(a)
  if math.isnan(x): return 0
  lo,hi=(-2**31,2**31-1) if signed else (0,2**32-1)
  n=hi if x==math.inf else lo if x==-math.inf else max(lo,min(hi,math.trunc(x)))
  return n&0xffffffff
def i2h(v,signed):
  n=v-(1<<32) if signed and v>>31 else v
  return to_h(Fraction(n)) if n else 0
def evaluate(ops,outs,hv,ins,N):
  res=[]
  for lane in range(N):
    e={}
    for op in ops:
      r,k=op[0],op[1]
      if k=='in': v=int(ins[op[2]][lane])
      elif k=='lo16': v=e[op[2]]&0xffff
      elif k in ('fadd','fsub','fmul'): v=hop({'fadd':'add','fsub':'sub','fmul':'mul'}[k],e[op[2]],e[op[3]])
      elif k=='fma': v=hop('fma',e[op[2]],e[op[3]],e[op[4]])
      elif k=='fsel': v=e[op[5]] if hcmp(op[2],e[op[3]],e[op[4]]) else e[op[6]]
      elif k in ('maximumnum','minimumnum'): v=hmax(e[op[2]],e[op[3]],k=='maximumnum',op[2]==op[3])
      elif k=='ext32':
        x,y=fpext(e[op[3]]),fpext(e[op[4]]); s=fadd32(x,y) if op[2]=='fadd' else fmul32(x,y); v=fptrunc(s)
      elif k=='ext32b':
        x,y,z=fpext(e[op[4]]),fpext(e[op[5]]),fpext(e[op[6]]); s=fadd32(x,y) if op[2]=='fadd' else fmul32(x,y)
        if op[3]=='max0': t=fmax32_0(s)
        else: t=fadd32(s,z) if op[3]=='fadd' else fmul32(s,z)
        v=fptrunc(t)
      elif k=='fneg': v=e[op[2]]^0x8000
      elif k=='fabs': v=e[op[2]]&0x7fff
      elif k in ('sitofp','uitofp'): v=i2h(e[op[2]],k=='sitofp')
      elif k in ('fptosi','fptoui'): v=sat_h(e[op[2]],k=='fptosi')
      e[r]=v
    res.append([e[o] for o in outs])
  return res


N = 512
EDGE = np.array([0, 0x8000, 1, 0x8001, 0x3FF, 0x83FF, 0x400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00,
                 0xFE00, 0x7C01, 0x3800, 0x3555, 0x0200, 0x6800, 0x5BFF], np.uint32)


def run_seed(seed, a, tmp):
    src, ops, outs, hv = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    ins = []
    inp = np.zeros(4 * 65536, np.uint32)
    for i in range(4):
        x = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
        m = rng.random(N) < 0.4
        x[m] = (x[m] & 0xFFFF0000) | rng.choice(EDGE, m.sum())
        if i == 3:
            x[::3] = rng.integers(-70000, 70000, len(x[::3])).astype(np.int64).astype(np.uint32)
        ins.append(x)
        inp[i * 65536:i * 65536 + N] = x
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nb = 8 * 65536 * 4
    rr = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "64,1,1", "--map", "0x1000000=" + fin,
                         "--alloc", "0x2000000=%d" % nb, "--dump", "0x2000000=%d:%s" % (nb, fout),
                         "--uniform", "0=16777216", "--uniform", "1=0", "--uniform", "2=33554432", "--uniform", "3=0"],
                        capture_output=True, text=True, timeout=600)
    if rr.returncode:
        return src, "agx_run failure: " + (rr.stdout + rr.stderr)[-200:]
    got = np.fromfile(fout, np.uint32)
    ref = evaluate(ops, outs, hv, ins, N)
    for j, o in enumerate(outs):
        for th in range(N):
            g, ex = int(got[j * 65536 + th]), ref[th][j]
            if o in hv and g >> 16 == 0 and isnanh(g) and isnanh(ex):
                continue
            if g != ex:
                return src, "%s: thread %d gave %#x, the model %#x" % (o, th, g, ex)
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
