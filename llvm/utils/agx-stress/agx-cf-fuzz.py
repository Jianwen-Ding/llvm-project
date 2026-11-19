#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's control flow against an interpreter, on the AGX Performance Simulator.

Each seed generates a structured program over five i32 variables: assignments of integer expressions, nested
if / else (and pairs of ifs on the same condition), loops whose trip count depends on a variable ((v & 7) + 1, capped), break and continue on data-dependent
conditions, nested up to four deep. It is emitted as LLVM IR with private variables and compiled the way the
benchmark kernels are (opt -O2, which promotes them to phis, then llc -O3; every third seed llc alone, which promotes them
itself), run on 512 threads with 40% boundary
inputs, and each thread's final variables are compared with an interpreter of the program. This covers the
structurizer, the execution-mask code, loop exits and the loop optimizations of opt and the backend
(EXPERIMENTS.md L15-E382).

  agx-cf-fuzz.py --bin build/bin --run path/to/agx_run [--seeds A:B] [--mcpu agx-g14c] [--keep DIR]
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

M=0xffffffff
NV=5
def sx(v): return v-(1<<32) if v>>31 else v
class G:
  def __init__(s,seed):
    s.R=random.Random(seed); s.lines=[]; s.n=0; s.depth=0
  def nv(s): s.n+=1; return f'%t{s.n}'
  def nb(s): s.n+=1; return f'b{s.n}'
  def expr(s):
    R=s.R; k=R.random()
    a=R.randrange(NV)
    if k<0.3: return ('var',a)
    if k<0.4: return ('const',R.choice([0,1,2,3,7,100,-1,0x80000000,12345]))
    op=R.choice(['add','sub','mul','xor','and','or','shl','lshr','ashr','umin','smax'])
    return (op,('var',a),('var',R.randrange(NV)) if R.random()<0.6 else ('const',R.choice([1,3,5,31,255,-7])))
  def cond(s):
    p=s.R.choice(['eq','ne','slt','sgt','ult','ugt','sle','uge'])
    return (p,s.expr(),s.expr())
  def stmt(s,inloop):
    R=s.R; k=R.random()
    if inloop and k<0.18: return ('break' if R.random()<0.5 else 'continue',s.cond())
    if s.depth>=4 or k<0.45: return ('assign',R.randrange(NV),s.expr())
    s.depth+=1
    if k<0.52:
      r=('if',s.cond(),s.block(inloop),s.block(inloop) if R.random()<0.5 else [])
    elif k<0.62:
      # an if / else whose sides both end with the same assignment of an
      # operation of variables neither side assigns (one operation of the
      # same values on each side, then a phi: L15-E421)
      c=s.cond(); b1=[s.stmt(inloop)]; b2=[s.stmt(inloop)] if R.random()<0.7 else []
      busy=assigned(b1)|assigned(b2)
      free=[v for v in range(NV) if v not in busy]
      if free:
        x=('var',R.choice(free))
        y=('var',R.choice(free)) if R.random()<0.6 else ('const',R.choice([1,3,5,31,255,-7]))
        a=('assign',R.randrange(NV),(R.choice(['add','sub','mul','xor','and','or']),x,y))
        r=('if',c,b1+[a],b2+[a])
      else:
        r=('if',c,b1,b2)
    elif k<0.7:
      # two ifs on the same condition, one after the other, each loading a
      # table entry (which opt cannot speculate) into variables the
      # condition does not read (L15-E413)
      c=s.cond()
      used=vars_of(c); free=[v for v in range(NV) if v not in used] or [0]
      def body():
        return [('assign',R.choice(free),('tab',s.expr())) for _ in range(R.randint(1,2))]
      r=('seq',[('if',c,body(),[]),('if',c,body(),[])])
    elif k<0.9:
      r=('loop',R.randrange(NV),R.choice([3,5,8,13]),s.block(True))
    else:
      r=('break' if R.random()<0.5 else 'continue',s.cond()) if inloop else ('assign',R.randrange(NV),s.expr())
    s.depth-=1
    return r
  def block(s,inloop):
    return [s.stmt(inloop) for _ in range(s.R.randint(1,3))]
def assigned(block):
  out=set()
  for st in block:
    if st[0]=='assign': out.add(st[1])
    elif st[0]=='if': out|=assigned(st[2])|assigned(st[3])
    elif st[0]=='loop': out|=assigned(st[3])
    elif st[0]=='seq': out|=assigned(st[1])
  return out
def vars_of(e):
  if e[0]=='var': return {e[1]}
  if e[0]=='const': return set()
  return set().union(*(vars_of(x) for x in e[1:] if isinstance(x,tuple)))
# a table of 256 words in memory after the threads' inputs of variable 0
TAB=[(i*2654435761+12345)&0xffffffff for i in range(256)]
def gen(seed):
  g=G(seed); prog=g.block(False)+[g.stmt(False) for _ in range(2)]
  return prog
# ---- evaluation
def ev(e,V):
  k=e[0]
  if k=='var': return V[e[1]]
  if k=='const': return e[1]&M
  if k=='tab': return TAB[ev(e[1],V)&255]
  a,b=ev(e[1],V),ev(e[2],V)
  if k=='add': return (a+b)&M
  if k=='sub': return (a-b)&M
  if k=='mul': return (a*b)&M
  if k=='xor': return a^b
  if k=='and': return a&b
  if k=='or': return a|b
  if k=='shl': return (a<<(b&31))&M
  if k=='lshr': return a>>(b&31)
  if k=='ashr': return (sx(a)>>(b&31))&M
  if k=='umin': return min(a,b)
  if k=='smax': return a if sx(a)>=sx(b) else b
def evc(c,V):
  p,a,b=c[0],ev(c[1],V),ev(c[2],V)
  return {'eq':a==b,'ne':a!=b,'slt':sx(a)<sx(b),'sgt':sx(a)>sx(b),'ult':a<b,'ugt':a>b,'sle':sx(a)<=sx(b),'uge':a>=b}[p]
class Brk(Exception): pass
class Cnt(Exception): pass
def run(prog,V):
  for st in prog:
    k=st[0]
    if k=='assign': V[st[1]]=ev(st[2],V)
    elif k=='seq': run(st[1],V)
    elif k=='if':
      run(st[2] if evc(st[1],V) else st[3],V)
    elif k=='loop':
      # trip count: (v & 7) + 1 capped by the bound
      n=min((V[st[1]]&7)+1,st[2])
      for i in range(n):
        try: run(st[3],V)
        except Cnt: continue
        except Brk: break
    elif k=='break':
      if evc(st[1],V): raise Brk()
    elif k=='continue':
      if evc(st[1],V): raise Cnt()
# ---- IR emission with allocas
def emit(prog):
  L=[]; n=[0]; allocas=[]  # (every alloca in the entry block: static, promoted by llc alone too)
  def t(): n[0]+=1; return f'%t{n[0]}'
  def b(): n[0]+=1; return f'b{n[0]}'
  def ee(e):
    k=e[0]
    if k=='var': r=t(); L.append(f'  {r} = load i32, ptr addrspace(5) %a{e[1]}'); return r
    if k=='const': return str(e[1] if e[1]<0x80000000 else e[1]-(1<<32))
    if k=='tab':
      x=ee(e[1]); m=t(); L.append(f'  {m} = and i32 {x}, 255'); o=t(); L.append(f'  {o} = add i32 {m}, {N}')
      g=t(); L.append(f'  {g} = getelementptr i32, ptr addrspace(1) %p, i32 {o}'); r=t()
      L.append(f'  {r} = load i32, ptr addrspace(1) {g}'); return r
    x,y=ee(e[1]),ee(e[2]); r=t()
    if k in ('shl','lshr','ashr'):
      m=t(); L.append(f'  {m} = and i32 {y}, 31'); y=m
    if k in ('umin','smax'): L.append(f'  {r} = call i32 @llvm.{k}.i32(i32 {x}, i32 {y})')
    else: L.append(f'  {r} = {k} i32 {x}, {y}')
    return r
  def ec(c):
    x,y=ee(c[1]),ee(c[2]); r=t(); L.append(f'  {r} = icmp {c[0]} i32 {x}, {y}'); return r
  def blk(prog,brk,cnt):
    for st in prog:
      k=st[0]
      if k=='assign':
        v=ee(st[2]); L.append(f'  store i32 {v}, ptr addrspace(5) %a{st[1]}')
      elif k=='seq':
        blk(st[1],brk,cnt)
      elif k=='if':
        c=ec(st[1]); bt,bf,bj=b(),b(),b()
        L.append(f'  br i1 {c}, label %{bt}, label %{bf}'); L.append(f'{bt}:'); blk(st[2],brk,cnt); L.append(f'  br label %{bj}')
        L.append(f'{bf}:'); blk(st[3],brk,cnt); L.append(f'  br label %{bj}'); L.append(f'{bj}:')
      elif k=='loop':
        nvv=ee(('var',st[1])); m=t(); L.append(f'  {m} = and i32 {nvv}, 7'); m2=t(); L.append(f'  {m2} = add i32 {m}, 1')
        m3=t(); L.append(f'  {m3} = call i32 @llvm.umin.i32(i32 {m2}, i32 {st[2]})')
        ic=t().replace('%t','%i'); bh,bb,bl,bx=b(),b(),b(),b()
        allocas.append(f'  {ic} = alloca i32, addrspace(5)'); L.append(f'  store i32 0, ptr addrspace(5) {ic}'); L.append(f'  br label %{bh}')
        L.append(f'{bh}:'); iv=t(); L.append(f'  {iv} = load i32, ptr addrspace(5) {ic}'); cc=t(); L.append(f'  {cc} = icmp ult i32 {iv}, {m3}')
        L.append(f'  br i1 {cc}, label %{bb}, label %{bx}'); L.append(f'{bb}:'); blk(st[3],bx,bl); L.append(f'  br label %{bl}')
        L.append(f'{bl}:'); iv2=t(); L.append(f'  {iv2} = load i32, ptr addrspace(5) {ic}'); iv3=t(); L.append(f'  {iv3} = add i32 {iv2}, 1')
        L.append(f'  store i32 {iv3}, ptr addrspace(5) {ic}'); L.append(f'  br label %{bh}'); L.append(f'{bx}:')
      elif k in ('break','continue'):
        c=ec(st[1]); bn=b(); tgt=brk if k=='break' else cnt
        L.append(f'  br i1 {c}, label %{tgt}, label %{bn}'); L.append(f'{bn}:')
  head=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()','declare i32 @llvm.umin.i32(i32, i32)','declare i32 @llvm.smax.i32(i32, i32)',
        'define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {','entry:','  %t = call i32 @llvm.agx.thread.position.in.grid.x()']
  for i in range(NV):
    head+=[f'  %a{i} = alloca i32, addrspace(5)', f'  %g{i} = getelementptr i32, ptr addrspace(1) %p, i32 {i*65536}', f'  %g{i}b = getelementptr i32, ptr addrspace(1) %g{i}, i32 %t',
           f'  %x{i} = load i32, ptr addrspace(1) %g{i}b', f'  store i32 %x{i}, ptr addrspace(5) %a{i}']
  blk(prog,None,None)
  tail=[]
  for i in range(NV):
    tail+=[f'  %r{i} = load i32, ptr addrspace(5) %a{i}', f'  %o{i} = getelementptr i32, ptr addrspace(1) %q, i32 {i*65536}', f'  %o{i}b = getelementptr i32, ptr addrspace(1) %o{i}, i32 %t', f'  store i32 %r{i}, ptr addrspace(1) %o{i}b']
  head=head[:7]+allocas+head[7:]
  return '\n'.join(head+L+tail+['  ret void','}'])+'\n'


N = 512
EDGE = [0, 1, 2, 7, 8, 15, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 100]


def run_seed(seed, a, tmp):
    prog = gen(seed)
    src = emit(prog)
    ll, oll, obj, code = (os.path.join(tmp, s) for s in ("k.ll", "o.ll", "k.o", "k.bin"))
    open(ll, "w").write(src)
    if seed % 3 == 2:
        # (every third seed straight to llc, which promotes the variables
        # itself: the shapes opt -O2 would have merged reach the backend)
        oll = ll
    else:
        r = subprocess.run([os.path.join(a.bin, "opt"), "-mtriple=agx", "-mcpu=" + a.mcpu, "-O2", ll, "-S", "-o",
                            oll], capture_output=True, text=True)
        if r.returncode:
            return src, "opt failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    r = subprocess.run([os.path.join(a.bin, "llc"), "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", oll,
                        "-o", obj], capture_output=True, text=True)
    if r.returncode:
        return src, "llc failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([os.path.join(a.bin, "llvm-objcopy"), "-O", "binary", "--only-section=.text", obj, code],
                   check=True)
    rng = np.random.default_rng(seed)
    inp = np.zeros(NV * 65536, np.uint32)
    for i in range(NV):
        x = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
        m = rng.random(N) < 0.4
        x[m] = rng.choice(EDGE, m.sum())
        inp[i * 65536:i * 65536 + N] = x
    inp[N:N + 256] = TAB
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    inp.tofile(fin)
    nb = NV * 65536 * 4
    rr = subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "64,1,1", "--map", "0x1000000=" + fin,
                         "--alloc", "0x2000000=%d" % nb, "--dump", "0x2000000=%d:%s" % (nb, fout),
                         "--uniform", "0=16777216", "--uniform", "1=0", "--uniform", "2=33554432", "--uniform", "3=0"],
                        capture_output=True, text=True, timeout=600)
    if rr.returncode:
        return src, "agx_run failure: " + (rr.stdout + rr.stderr)[-200:]
    got = np.fromfile(fout, np.uint32)
    for th in range(N):
        V = [int(inp[i * 65536 + th]) for i in range(NV)]
        run(prog, V)
        g = [int(got[i * 65536 + th]) for i in range(NV)]
        if g != V:
            return src, "thread %d ended with %s, the interpreter %s" % (th, [hex(x) for x in g], [hex(x) for x in V])
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bin", help="the directory of opt, llc and llvm-objcopy")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--seeds", default="0:300")
    ap.add_argument("--mcpu", default="agx-g14c")
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
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
