#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's integer code against a Python model, on the AGX Performance Simulator.

Each seed generates a random kernel of 10-28 integer operations on i8 / i16 / i32 / i64 values (add, sub, mul,
and, or, xor with registers and constants; shifts by constants and masked registers; icmp of every predicate
into selects; smin / smax / umin / umax; udiv / urem / sdiv / srem by constants; ctpop / ctlz / cttz /
bitreverse / bswap / abs; fshl / fshr; the four saturating adds; trunc / zext / sext; runs of additions of wide
constants K + i d to one value; 64-bit compares of masked, offset values), compiles it with llc,
runs it with agx_run on 1024 threads of random words (40% boundary values) and compares every output word with
the model's (EXPERIMENTS.md L15-E376).

  agx-int-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
                  [--mcpu agx-g14c] [--keep DIR]

Exits with 1 if a kernel fails to compile or gives a wrong result (kept in DIR with --keep).
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

M={32:0xffffffff,16:0xffff,8:0xff,64:(1<<64)-1}
def sx(v,w): v&=M[w]; return v-(1<<w) if v>>(w-1) else v
def gen(seed):
  R=random.Random(seed)
  L=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()']
  decl=set(); ops=[]  # python lambdas over env
  vals=[]  # (name, width)
  n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()']
  for i in range(4):
    g=f'%g{i}'; v=nv()
    body+= [f'  {g} = getelementptr i32, ptr addrspace(1) %p, i32 {i*65536}', f'  {g}b = getelementptr i32, ptr addrspace(1) {g}, i32 %t', f'  {v} = load i32, ptr addrspace(1) {g}b']
    ops.append((v,32,('in',i))); vals.append((v,32))
  def pick(w=None):
    c=[x for x in vals if w is None or x[1]==w]
    return R.choice(c) if c else None
  for _ in range(R.randint(10,28)):
    k=R.random(); a=pick()
    w=a[1]; T=f'i{w}'
    r=nv()
    if k<0.3:
      op=R.choice(['add','sub','mul','and','or','xor'])
      b=pick(w) if R.random()<0.7 else None
      bs = b[0] if b else str(R.choice([1,2,3,7,8,15,16,31,255,256,1000,-1,-2,65535,0x7fff,-32768,123456789]+([1<<32,-(1<<33),0xffffffff,0x100000001] if w==64 else [])) if w>=32 else R.choice([1,2,3,7,255,-1,0x7f]))
      body.append(f'  {r} = {op} {T} {a[0]}, {bs}'); ops.append((r,w,(op,a[0],b[0] if b else int(bs)))); 
    elif k<0.42:
      op=R.choice(['shl','lshr','ashr']); s=R.randrange(0,w)
      if R.random()<0.3:
        b=pick(w); m=nv(); body.append(f'  {m} = and {T} {b[0]}, {w-1}'); ops.append((m,w,('and',b[0],w-1)))
        body.append(f'  {r} = {op} {T} {a[0]}, {m}'); ops.append((r,w,(op,a[0],m)))
      else:
        body.append(f'  {r} = {op} {T} {a[0]}, {s}'); ops.append((r,w,(op,a[0],s)))
    elif k<0.55:
      b=pick(w); p=R.choice(['eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge'])
      cst=R.choice([0,-1,1,100,0x8000]) if R.random()<0.4 else None
      c=nv(); body.append(f'  {c} = icmp {p} {T} {a[0]}, {b[0] if cst is None else cst}')
      x=pick(w); y=pick(w)
      body.append(f'  {r} = select i1 {c}, {T} {x[0]}, {T} {y[0]}'); ops.append((r,w,('sel',p,a[0],b[0] if cst is None else cst,x[0],y[0])))
    elif k<0.65:
      fn=R.choice(['smin','smax','umin','umax'])
      b=pick(w); decl.add(f'declare {T} @llvm.{fn}.{T}({T}, {T})')
      body.append(f'  {r} = call {T} @llvm.{fn}.{T}({T} {a[0]}, {T} {b[0]})'); ops.append((r,w,(fn,a[0],b[0])))
    elif k<0.72:
      op=R.choice(['udiv','urem','sdiv','srem']); d=R.choice([3,5,7,10,12,100,255,1000,641,65537] if w>=32 else [3,5,7,10,100])
      body.append(f'  {r} = {op} {T} {a[0]}, {d}'); ops.append((r,w,(op,a[0],d)))
    elif k<0.80:
      fn=R.choice(['ctpop','ctlz','cttz','bitreverse','bswap','abs']) if w!=8 else 'ctpop'
      if fn=='bswap' and w==8: fn='ctpop'
      if fn in ('ctlz','cttz','abs'):
        decl.add(f'declare {T} @llvm.{fn}.{T}({T}, i1)'); body.append(f'  {r} = call {T} @llvm.{fn}.{T}({T} {a[0]}, i1 false)')
      else:
        decl.add(f'declare {T} @llvm.{fn}.{T}({T})'); body.append(f'  {r} = call {T} @llvm.{fn}.{T}({T} {a[0]})')
      ops.append((r,w,(fn,a[0])))
    elif k<0.86:
      fn=R.choice(['fshl','fshr']); b=pick(w); s=R.randrange(0,w)
      decl.add(f'declare {T} @llvm.{fn}.{T}({T}, {T}, {T})'); body.append(f'  {r} = call {T} @llvm.{fn}.{T}({T} {a[0]}, {T} {b[0]}, {T} {s})'); ops.append((r,w,(fn,a[0],b[0],s)))
    elif k<0.90:
      fn=R.choice(['uadd.sat','usub.sat','sadd.sat','ssub.sat']); b=pick(w)
      decl.add(f'declare {T} @llvm.{fn}.{T}({T}, {T})'); body.append(f'  {r} = call {T} @llvm.{fn}.{T}({T} {a[0]}, {T} {b[0]})'); ops.append((r,w,(fn,a[0],b[0])))
    elif k<0.93:
      # division by a variable positive odd divisor (no division by zero, no INT_MIN / -1)
      b=pick(w); op=R.choice(['udiv','urem','sdiv','srem']); dm=nv(); d=nv()
      body+=[f'  {dm} = and {T} {b[0]}, {(1<<(w-2))-1}', f'  {d} = or {T} {dm}, 1', f'  {r} = {op} {T} {a[0]}, {d}']
      ops+=[(dm,w,('and',b[0],(1<<(w-2))-1)),(d,w,('or',dm,1)),(r,w,(op,a[0],d))]
    elif k<0.945 and w==32:
      # the high word of a 64-bit product (signed or unsigned)
      b=pick(32); e=R.choice(['zext','sext']); x=nv(); y=nv(); p_=nv(); h=nv()
      body+=[f'  {x} = {e} i32 {a[0]} to i64', f'  {y} = {e} i32 {b[0]} to i64', f'  {p_} = mul i64 {x}, {y}', f'  {h} = lshr i64 {p_}, 32', f'  {r} = trunc i64 {h} to i32']
      ops+=[(x,64,(e,a[0],32)),(y,64,(e,b[0],32)),(p_,64,('mul',x,y)),(h,64,('lshr',p_,32)),(r,32,('trunc',h))]
    elif k<0.965 and w==32:
      # a 64-bit compare of masked, offset values (MLIR's index bounds
      # checks, x + d - 1 against W; sometimes or-combined, as InstCombine
      # makes (x' | y') <u W), which fit 32 signed bits or not (L15-E427)
      b=pick(32); m1=R.choice([255,4095,0x7fffffff,0xffffffff]); m2=R.choice([255,1023,0x7fffffff])
      k1=R.choice([1,-1,-3,1024,-(1<<31),(1<<31)]); k2=R.choice([1,-2,7,-4096])
      ma=nv(); mb=nv(); za=nv(); zb=nv(); xa=nv(); xb=nv(); c=nv()
      e1=R.choice(['zext','sext'])
      # (half the time the thread position, bounded only by the kernel's
      # upper-bounds attribute, which the DAG does not see)
      src=('%t' if R.random()<0.5 else a[0])
      if src=='%t' and not any(o[0]=='%t' for o in ops):
        ops.insert(0,('%t',32,('tid',)))
      body+=[f'  {ma} = and i32 {src}, {m1-(1<<32) if m1>>31 else m1}', f'  {mb} = and i32 {b[0]}, {m2}',
             f'  {za} = {e1} i32 {ma} to i64', f'  {zb} = zext i32 {mb} to i64',
             f'  {xa} = add i64 {za}, {k1}', f'  {xb} = sub i64 {zb}, {k2}']
      ops+=[(ma,32,('and',src,m1-(1<<32) if m1>>31 else m1)),(mb,32,('and',b[0],m2)),(za,64,(e1,ma,32)),(zb,64,('zext',mb,32)),
            (xa,64,('add',za,k1)),(xb,64,('sub',zb,k2))]
      lhs=xa
      if R.random()<0.4:
        o=nv(); body.append(f'  {o} = or i64 {xa}, {xb}'); ops.append((o,64,('or',xa,xb))); lhs=o; rhs=R.choice([1024,4096,-1,0])
        rs=str(rhs)
      else:
        rs=xb
      p=R.choice(['eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge'])
      x=pick(32); y=pick(32)
      body+=[f'  {c} = icmp {p} i64 {lhs}, {rs}', f'  {r} = select i1 {c}, i32 {x[0]}, i32 {y[0]}']
      ops.append((r,32,('sel64',p,lhs,int(rs) if not rs.startswith('%') else rs,x[0],y[0])))
    elif k<0.97 and w==32:
      # additions of wide constants K + i d to one value (chained by their
      # differences, L15-E420), each an output
      K=R.choice([123456789,-1640531527,8192,1000,-70000,0x7fff0000]); d=R.choice([1,200,-3,0x9E3779B9-(1<<32),8192,K])
      prev=None
      for i in range(R.randint(2,3)):
        c=((K+i*d+(1<<31))&M[32])-(1<<31)
        if prev is not None:
          vals.append((prev,32))
        body.append(f'  {r} = add i32 {a[0]}, {c}'); ops.append((r,32,('add',a[0],c)))
        prev=r; r=nv()
      r=prev
    elif k<0.98 and w==32:
      # a 64-bit trip umin(c - zext(x & m), k) or umax (MLIR's loop bound of
      # a lane: umin(n - lane, n)), whose subtraction may borrow or not
      # (L15-E435), truncated or kept 64-bit
      src=('%t' if R.random()<0.5 else a[0])
      if src=='%t' and not any(o[0]=='%t' for o in ops):
        ops.insert(0,('%t',32,('tid',)))
      m1=R.choice([31,1023,0xffff]); c0=R.choice([5,1000,1023,1024,2000,0x10000]); k1=R.choice([7,1000,1023,0x10000,(1<<33)-1])
      fn=R.choice(['umin','umin','umax']); ma=nv(); za=nv(); xs=nv(); um=nv()
      decl.add(f'declare i64 @llvm.{fn}.i64(i64, i64)')
      body+=[f'  {ma} = and i32 {src}, {m1}', f'  {za} = zext i32 {ma} to i64', f'  {xs} = sub i64 {c0}, {za}',
             f'  {um} = call i64 @llvm.{fn}.i64(i64 {xs}, i64 {k1})']
      ops+=[(ma,32,('and',src,m1)),(za,64,('zext',ma,32)),(xs,64,('sub',c0,za)),(um,64,(fn,xs,k1))]
      if R.random()<0.5:
        body.append(f'  {r} = trunc i64 {um} to i32'); ops.append((r,32,('trunc',um)))
      else:
        r=um; w=64
    elif k<0.985 and len(vals)>=2:
      # an and / or of two compares into a select
      b=pick(w); c1=nv(); c2=nv(); c=nv(); x=pick(w); y=pick(w)
      p1,p2=R.choice(['slt','ult','eq','ne','sgt']),R.choice(['sle','ugt','eq','ne','uge']); lg=R.choice(['and','or'])
      body+=[f'  {c1} = icmp {p1} {T} {a[0]}, {b[0]}', f'  {c2} = icmp {p2} {T} {b[0]}, {x[0]}', f'  {c} = {lg} i1 {c1}, {c2}', f'  {r} = select i1 {c}, {T} {x[0]}, {T} {y[0]}']
      ops.append((r,w,('sel2',lg,p1,a[0],b[0],p2,b[0],x[0],x[0],y[0])))
    else:
      nw=R.choice([x for x in (8,16,32,64,64) if x!=w])
      if nw<w: body.append(f'  {r} = trunc {T} {a[0]} to i{nw}'); ops.append((r,nw,('trunc',a[0])))
      else:
        e=R.choice(['zext','sext']); body.append(f'  {r} = {e} {T} {a[0]} to i{nw}'); ops.append((r,nw,(e,a[0],w)))
      w=nw
    vals.append((r,w))
  outs=[]
  for (v,w) in [x for x in vals[4:]][-8:]:
    if w==64:
      h=nv(); f=nv(); t=nv()
      body+=[f'  {h} = lshr i64 {v}, 32', f'  {f} = xor i64 {v}, {h}', f'  {t} = trunc i64 {f} to i32']
      ops+= [(h,64,('lshr',v,32)),(f,64,('xor',v,h)),(t,32,('trunc',f))]
      outs.append((t,32))
    else: outs.append((v,w))
  for i,(v,w) in enumerate(outs):
    o=v if w==32 else nv()
    if w!=32: body.append(f'  {o} = zext i{w} {v} to i32')
    body+=[f'  %o{i} = getelementptr i32, ptr addrspace(1) %q, i32 {i*65536}', f'  %o{i}b = getelementptr i32, ptr addrspace(1) %o{i}, i32 %t', f'  store i32 {o}, ptr addrspace(1) %o{i}b']
  L+=sorted(decl)+['define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {']+body+['  ret void','}',
     'attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="%d,0,0" }' % N]
  return '\n'.join(L)+'\n', ops, outs

def evaluate(ops, outs, inputs):
  import numpy as np
  env={}; W={}
  for (r,w,op) in ops:
    W[r]=w
  N=len(inputs[0]); res=[]
  for lane in range(N):
    e={}
    for (r,w,op) in ops:
      m=M[w]; g=lambda x: e[x] if isinstance(x,str) else x&m
      k=op[0]
      if k=='in': v=int(inputs[op[1]][lane])
      elif k=='tid': v=lane
      elif k in ('add','sub','mul','and','or','xor'):
        a,b=g(op[1]),g(op[2]); v={'add':a+b,'sub':a-b,'mul':a*b,'and':a&b,'or':a|b,'xor':a^b}[k]
      elif k in ('shl','lshr','ashr'):
        a,s=g(op[1]),g(op[2])
        v= a<<s if k=='shl' else (a>>s if k=='lshr' else sx(a,w)>>s)
      elif k=='sel':
        p,a,b=op[1],g(op[2]),g(op[3]); sa,sb=sx(a,w),sx(b,w)
        c={'eq':a==b,'ne':a!=b,'slt':sa<sb,'sle':sa<=sb,'sgt':sa>sb,'sge':sa>=sb,'ult':a<b,'ule':a<=b,'ugt':a>b,'uge':a>=b}[p]
        v=g(op[4]) if c else g(op[5])
      elif k=='sel64':
        # (a compare of 64-bit values selecting 32-bit ones)
        a=e[op[2]]; b=e[op[3]] if isinstance(op[3],str) else op[3]&M[64]; sa,sb=sx(a,64),sx(b,64)
        c={'eq':a==b,'ne':a!=b,'slt':sa<sb,'sle':sa<=sb,'sgt':sa>sb,'sge':sa>=sb,'ult':a<b,'ule':a<=b,'ugt':a>b,'uge':a>=b}[op[1]]
        v=g(op[4]) if c else g(op[5])
      elif k=='sel2':
        def cmp(p,a,b):
          sa,sb=sx(a,w),sx(b,w)
          return {'eq':a==b,'ne':a!=b,'slt':sa<sb,'sle':sa<=sb,'sgt':sa>sb,'sge':sa>=sb,'ult':a<b,'ule':a<=b,'ugt':a>b,'uge':a>=b}[p]
        c1=cmp(op[2],g(op[3]),g(op[4])); c2=cmp(op[5],g(op[6]),g(op[7]))
        c= (c1 and c2) if op[1]=='and' else (c1 or c2)
        v=g(op[8]) if c else g(op[9])
      elif k in ('smin','smax','umin','umax'):
        a,b=g(op[1]),g(op[2])
        if k[0]=='s': v=(min if k=='smin' else max)(sx(a,w),sx(b,w))
        else: v=(min if k=='umin' else max)(a,b)
      elif k in ('udiv','urem','sdiv','srem'):
        a,d=g(op[1]),g(op[2])
        if k=='udiv': v=a//d
        elif k=='urem': v=a%d
        else:
          q=abs(sx(a,w))//d; q= q if sx(a,w)>=0 else -q
          v= q if k=='sdiv' else sx(a,w)-q*d
      elif k=='ctpop': v=bin(g(op[1])).count('1')
      elif k=='ctlz': a=g(op[1]); v=w-a.bit_length()
      elif k=='cttz': a=g(op[1]); v=w if a==0 else (a&-a).bit_length()-1
      elif k=='bitreverse': a=g(op[1]); v=int(format(a,f'0{w}b')[::-1],2)
      elif k=='bswap': a=g(op[1]); v=int.from_bytes(a.to_bytes(w//8,'little'),'big')
      elif k=='abs': v=abs(sx(g(op[1]),w))
      elif k in ('fshl','fshr'):
        a,b,s=g(op[1]),g(op[2]),op[3]%w; cat=(a<<w)|b
        v= (cat<<s)>>w if k=='fshl' else cat>>s
      elif k.endswith('.sat'):
        a,b=g(op[1]),g(op[2])
        if k=='uadd.sat': v=min(a+b,m)
        elif k=='usub.sat': v=max(a-b,0)
        else:
          sa,sb=sx(a,w),sx(b,w); s= sa+sb if k=='sadd.sat' else sa-sb
          v=max(min(s,(1<<(w-1))-1),-(1<<(w-1)))
      elif k=='trunc': v=g(op[1])
      elif k=='zext': v=e[op[1]]
      elif k=='sext': v=sx(e[op[1]],op[2])
      e[r]=v&m
    res.append([e[v] for v,_ in outs])
  return res


EDGE = np.array([0, 1, 2, 3, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000000, 0x7FFFFFFF, 0x8000, 0x7FFF, 0xFFFF, 0x10000, 0xFF,
                 0x80, 0x7F, 0x100, 0xFFFF8000, 0xFF80, 0x55555555, 0xAAAAAAAA], np.uint32)
N = 1024
IN, OUT = 0x1000000, 0x2000000


def run_seed(seed, a, tmp):
    src, ops, outs = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + r.stderr.strip().splitlines()[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    ins = [rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32) for _ in range(4)]
    for x in ins:
        m = rng.random(N) < 0.4
        x[m] = rng.choice(EDGE, m.sum())
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
    ref = evaluate(ops, outs, ins)
    for j, (v, _) in enumerate(outs):
        exp = np.array([r[j] for r in ref], dtype=np.uint64).astype(np.uint32)
        g = got[j * 65536:j * 65536 + N]
        if (g != exp).any():
            i = int(np.argmax(g != exp))
            return src, "%s: thread %d gave %#x, the model %#x (inputs %s)" % (
                v, i, int(g[i]), int(exp[i]), [hex(int(x[i])) for x in ins])
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
