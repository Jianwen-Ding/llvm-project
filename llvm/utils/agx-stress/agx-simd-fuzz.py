#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's SIMD-group shuffles against a lane model, on the AGX Performance Simulator.

Each seed generates a kernel of 6-18 operations over all 32 threads of each SIMD-group: shuffles by constants
(simd / quad, xor, up, down, rotate up), shuffles by an index computed from the thread index ((lane op c) & 31),
selects on comparisons of the thread index, integer add / xor / and / or / sub, Hillis-Steele add scans (of earlier
results too) and xor butterflies of every integer reduction op in any distance order; a branch on a data-dependent
condition holding SIMD-group and quad reductions, exclusive prefixes (quad: add, the simulator's only one) and
ballots of the active threads; then a last value used only under `lane == c`, which the optimizer sinks into that
branch. With an llc that sets -agx-simd-reductions it also checks the butterflies as reductions (L15-E381).
The model computes each SIMD-group's 32 lanes with the backend's own lane rules (LanePermutation::source; up and
down keep a thread's own value where the source is outside). This covers the shuffle recognitions (lane
permutations, own-value shifts, quad xors, shuffle selects, scans) and their placement (EXPERIMENTS.md L15-E365,
L15-E380).

  agx-simd-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
                   [--mcpu agx-g14c] [--keep DIR]
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

M=0xffffffff
def src_lane(kind,L,K):
  Q=L&~3; I=L&3
  if kind=='simd.shuffle': return K&31
  if kind=='simd.shuffle.xor': return L^K if (L^K)<32 else L
  if kind=='simd.shuffle.up': return L-K if L>=K else L
  if kind=='simd.shuffle.down': return L+K if L+K<32 else L
  if kind=='simd.shuffle.rotate.up': return (L-K)&31
  if kind=='quad.shuffle': return Q|K if K<4 else L
  if kind=='quad.shuffle.xor': return Q|(I^K) if (I^K)<4 else L
  if kind=='quad.shuffle.up': return L-K if I>=K else L
  if kind=='quad.shuffle.down': return L+K if I+K<4 else L
  if kind=='quad.shuffle.rotate.up': return Q|((I-K)&3)
def gen(seed):
  R=random.Random(seed)
  decl={'declare i32 @llvm.agx.thread.position.in.grid.x()','declare i32 @llvm.agx.thread.index.in.simdgroup()'}
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()','  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()',
        '  %ga = getelementptr i32, ptr addrspace(1) %p, i32 %t','  %x0 = load i32, ptr addrspace(1) %ga',
        '  %gb = getelementptr i32, ptr addrspace(1) %ga, i32 65536','  %x1 = load i32, ptr addrspace(1) %gb']
  ops=[('%x0','in',0),('%x1','in',1),('%lane','lane')]; vals=['%x0','%x1']; n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  def lane_expr():
    v='%lane'; e=[]
    for _ in range(R.randint(1,2)):
      op=R.choice(['xor','add','sub','and','or','shl','lshr']); c=R.choice([1,2,3,4,5,7,8,15,16,17,31]) if op not in ('shl','lshr') else R.choice([1,2])
      r=nv(); body.append(f'  {r} = {op} i32 {v}, {c}'); ops.append((r,op,v,c)); v=r
    r=nv(); body.append(f'  {r} = and i32 {v}, 31'); ops.append((r,'and',v,31)); return r
  for _ in range(R.randint(6,18)):
    k=R.random(); r=nv(); a=R.choice(vals)
    if k<0.35:
      kind=R.choice(['simd.shuffle.xor','simd.shuffle.up','simd.shuffle.down','simd.shuffle.rotate.up','quad.shuffle','quad.shuffle.xor','quad.shuffle.up','quad.shuffle.down','quad.shuffle.rotate.up'])
      K=R.randrange(1,32) if kind.startswith('simd') else R.randrange(0,4)
      decl.add(f'declare i32 @llvm.agx.{kind}.i32(i32, i32)')
      body.append(f'  {r} = call i32 @llvm.agx.{kind}.i32(i32 {a}, i32 {K})'); ops.append((r,'shfl',kind,a,K))
    elif k<0.5:
      ix=lane_expr(); decl.add('declare i32 @llvm.agx.simd.shuffle.i32(i32, i32)')
      body.append(f'  {r} = call i32 @llvm.agx.simd.shuffle.i32(i32 {a}, i32 {ix})'); ops.append((r,'shfli',a,ix))
    elif k<0.68:
      p=R.choice(['eq','ne','ult','ugt','uge','ule']); c=R.choice([0,1,3,4,7,8,15,16,24,31]); cv=nv(); b=R.choice(vals)
      body+=[f'  {cv} = icmp {p} i32 %lane, {c}', f'  {r} = select i1 {cv}, i32 {a}, i32 {b}']; ops.append((r,'sel',p,c,a,b))
    elif k<0.85:
      b=R.choice(vals); op=R.choice(['add','xor','and','or','sub'])
      body.append(f'  {r} = {op} i32 {a}, {b}'); ops.append((r,op,a,b))
    elif k<0.92:
      # an xor butterfly over all 32 lanes: op(v, shuffle_xor(v, d)) for d in some order
      dists=[1,2,4,8,16]; R.shuffle(dists); opn=R.choice(['add','and','or','xor','smin','smax','umin','umax']); v=a
      for d in dists:
        kind='quad.shuffle.xor' if d<=2 and R.random()<0.5 else 'simd.shuffle.xor'
        decl.add(f'declare i32 @llvm.agx.{kind}.i32(i32, i32)'); sh=nv(); w=nv()
        body.append(f'  {sh} = call i32 @llvm.agx.{kind}.i32(i32 {v}, i32 {d})'); ops.append((sh,'shfl',kind,v,d))
        x,y=(v,sh) if R.random()<0.5 else (sh,v)
        if opn in ('add','and','or','xor'): body.append(f'  {w} = {opn} i32 {x}, {y}'); ops.append((w,opn,x,y))
        else:
          decl.add(f'declare i32 @llvm.{opn}.i32(i32, i32)'); body.append(f'  {w} = call i32 @llvm.{opn}.i32(i32 {x}, i32 {y})'); ops.append((w,opn,x,y))
        v=w
      body.append(f'  {r} = add i32 {v}, 0'); ops.append((r,'add',v,'0'))
    else:
      # Hillis-Steele inclusive add scan
      decl.add('declare i32 @llvm.agx.simd.shuffle.up.i32(i32, i32)'); v=a
      for d in (1,2,4,8,16):
        s=nv(); c=nv(); z=nv(); w=nv()
        body+=[f'  {s} = call i32 @llvm.agx.simd.shuffle.up.i32(i32 {v}, i32 {d})', f'  {c} = icmp uge i32 %lane, {d}', f'  {z} = select i1 {c}, i32 {s}, i32 0', f'  {w} = add i32 {v}, {z}']
        ops+=[(s,'shfl','simd.shuffle.up',v,d),(c,'cmpd',d),(z,'selz',c,s),(w,'add',v,z)]; v=w
      body.append(f'  {r} = add i32 {v}, 0'); ops.append((r,'add',v,'0'))
    vals.append(r)
  outs=vals[-4:]
  for j,o in enumerate(outs):
    body+=[f'  %o{j} = getelementptr i32, ptr addrspace(1) %q, i32 {j*65536}', f'  %o{j}b = getelementptr i32, ptr addrspace(1) %o{j}, i32 %t', f'  store i32 {o}, ptr addrspace(1) %o{j}b']
  # a divergent region: SIMD-group reductions / exclusive prefixes / ballots over the active threads
  dv=R.choice(vals); dk=R.choice([1,3,0x55,0x80000000,7]); dc=nv(); dt=nv()
  body+=[f'  {dt} = and i32 {dv}, {dk}', f'  {dc} = icmp ne i32 {dt}, 0', f'  br i1 {dc}, label %div, label %join', 'div:']
  divops=[]
  for j in range(R.randint(1,4)):
    kind=R.choice(['simd.reduce','simd.prefix','quad.reduce','quad.prefix','ballot'])
    r=nv(); a_=R.choice(vals)
    if kind=='ballot':
      bc=nv(); decl.add('declare i32 @llvm.agx.simd.ballot(i1)')
      body+=[f'  {bc} = icmp ult i32 {a_}, {R.choice([0x40000000,0x80000000,0xc0000000])}', f'  {r} = call i32 @llvm.agx.simd.ballot(i1 {bc})']
      divops.append((r,'ballot',a_,int(body[-2].split(', ')[-1])))
    else:
      opc=16 if kind=='quad.prefix' else R.choice([16,0,2,4,20,22,28,30]); decl.add(f'declare i32 @llvm.agx.{kind}.i32(i32, i32)')
      body.append(f'  {r} = call i32 @llvm.agx.{kind}.i32(i32 {a_}, i32 {opc})'); divops.append((r,kind,a_,opc))
    body+=[f'  %d{j} = getelementptr i32, ptr addrspace(1) %q, i32 {(5+j)*65536}', f'  %d{j}b = getelementptr i32, ptr addrspace(1) %d{j}, i32 %t', f'  store i32 {r}, ptr addrspace(1) %d{j}b']
  body+=['  br label %join','join:']
  ops.append(('#div',(dv,dk),divops))
  # a last value used only under lane == c (sunk into the branch)
  cl=R.randrange(32); last=R.choice(vals); s=nv()
  decl.add('declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)')
  body+=[f'  {s} = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 {last}, i32 {R.choice([1,2,4,8,16])})', f'  %only = icmp eq i32 %lane, {cl}', '  br i1 %only, label %st, label %done', 'st:',
         f'  %fin = add i32 {last}, {s}', '  %o9 = getelementptr i32, ptr addrspace(1) %q, i32 262144', '  %o9b = getelementptr i32, ptr addrspace(1) %o9, i32 %t', '  store i32 %fin, ptr addrspace(1) %o9b', '  br label %done', 'done:', '  ret void', '}']
  sk=[x for x in body if x.startswith(f'  {s} =')][0].split('i32 ')[-1].rstrip(')')
  ops.append((s,'shfl','simd.shuffle.xor',last,int(sk)))
  L=['target triple = "agx"']+sorted(decl)+['define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {']+body
  return '\n'.join(L)+'\n', ops, outs, (cl,last,s)
def evaluate(ops, outs, fin, x0, x1):
  # x0, x1: lists of 32 lanes for one SIMD-group
  e={}; div=None
  for op in ops:
    if op[0]=='#div': div=op; continue
    r,k=op[0],op[1]
    if k=='in': v=[ (x0 if op[2]==0 else x1)[l] for l in range(32)]
    elif k=='lane': v=list(range(32))
    elif k=='shfl': kind,a,K=op[2],op[3],op[4]; v=[e[a][src_lane(kind,l,K)] for l in range(32)]
    elif k=='shfli': a,ix=op[2],op[3]; v=[e[a][e[ix][l]&31] for l in range(32)]
    elif k=='sel':
      p,c,a,b=op[2],op[3],op[4],op[5]
      f={'eq':lambda l:l==c,'ne':lambda l:l!=c,'ult':lambda l:l<c,'ugt':lambda l:l>c,'uge':lambda l:l>=c,'ule':lambda l:l<=c}[p]
      v=[e[a][l] if f(l) else e[b][l] for l in range(32)]
    elif k=='cmpd': v=[l>=op[2] for l in range(32)]
    elif k=='selz': v=[e[op[3]][l] if e[op[2]][l] else 0 for l in range(32)]
    else:
      a=e[op[2]]; b=e[op[3]] if isinstance(op[3],str) and op[3] in e else [int(op[3])]*32
      sx_=lambda v: v-(1<<32) if v>>31 else v
      f={'smin':lambda x,y:x if sx_(x)<=sx_(y) else y,'smax':lambda x,y:x if sx_(x)>=sx_(y) else y,'umin':min,'umax':max,'add':lambda x,y:x+y,'xor':lambda x,y:x^y,'and':lambda x,y:x&y,'or':lambda x,y:x|y,'sub':lambda x,y:x-y,'shl':lambda x,y:x<<y,'lshr':lambda x,y:x>>y}[k]
      v=[f(a[l],b[l]) for l in range(32)]
    e[r]=[x&M if isinstance(x,int) else x for x in v]
  cl,last,s=fin
  dres=[]
  if div:
    (dv,dk),divops=div[1],div[2]
    act=[(e[dv][l]&dk)!=0 for l in range(32)]
    ID={16:0,0:M,2:0,4:0,20:0x7fffffff,22:0x80000000,28:M,30:0}
    def comb(o,x,y):
      sx=lambda v: v-(1<<32) if v>>31 else v
      if o==16: return (x+y)&M
      if o==0: return x&y
      if o==2: return x|y
      if o==4: return x^y
      if o==20: return x if sx(x)<=sx(y) else y
      if o==22: return x if sx(x)>=sx(y) else y
      if o==28: return min(x,y)
      return max(x,y)
    for (r,kind,a_,p) in divops:
      v=[None]*32
      if kind=='ballot':
        m=sum(1<<l for l in range(32) if act[l] and e[a_][l]<p)
        for l in range(32):
          if act[l]: v[l]=m
      else:
        w=4 if kind.startswith('quad') else 32
        for base in range(0,32,w):
          lanes=[l for l in range(base,base+w) if act[l]]
          if kind.endswith('reduce'):
            t=ID[p]
            for l in lanes: t=comb(p,t,e[a_][l])
            for l in lanes: v[l]=t
          else:
            t=ID[p]
            for l in lanes: v[l]=t; t=comb(p,t,e[a_][l])
      dres.append(v)
  return [e[o] for o in outs], (e[last][cl]+e[s][cl])&M, dres

N = 256


def run_seed(seed, a, tmp):
    src, ops, outs, fin = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    inp = np.zeros(2 * 65536, np.uint32)
    inp[:N] = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
    inp[65536:65536 + N] = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
    fin_, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin_)
    nb = 9 * 65536 * 4
    subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "64,1,1", "--map", "0x1000000=" + fin_,
                    "--alloc", "0x2000000=%d" % nb, "--dump", "0x2000000=%d:%s" % (nb, fout),
                    "--uniform", "0=16777216", "--uniform", "1=0", "--uniform", "2=33554432", "--uniform", "3=0"],
                   check=True, capture_output=True, timeout=600)
    got = np.fromfile(fout, np.uint32)
    for gi in range(N // 32):
        x0 = [int(v) for v in inp[gi * 32:gi * 32 + 32]]
        x1 = [int(v) for v in inp[65536 + gi * 32:65536 + gi * 32 + 32]]
        res, fv, dres = evaluate(ops, outs, fin, x0, x1)
        for j, o in enumerate(outs):
            g = [int(v) for v in got[j * 65536 + gi * 32:j * 65536 + gi * 32 + 32]]
            if g != res[j]:
                bad = [l for l in range(32) if g[l] != res[j][l]]
                return src, "SIMD-group %d, %s: lanes %s differ" % (gi, o, bad[:8])
        for j, dv in enumerate(dres):
            g = [int(v) for v in got[(5 + j) * 65536 + gi * 32:(5 + j) * 65536 + gi * 32 + 32]]
            bad = [l for l in range(32) if dv[l] is not None and g[l] != dv[l]]
            if bad:
                return src, "SIMD-group %d, divergent operation %d: lanes %s differ" % (gi, j, bad[:8])
        g9 = int(got[4 * 65536 + gi * 32 + fin[0]])
        if g9 != fv:
            return src, "SIMD-group %d: the store under lane == %d wrote %#x, the model %#x" % (gi, fin[0], g9, fv)
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--seeds", default="0:300")
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
