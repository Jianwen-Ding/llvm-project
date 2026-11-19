#!/usr/bin/env python3
"""Differential fuzz of the AGX backend's narrow memory accesses against a byte-array model, on the AGX Performance
Simulator.

Each seed generates a kernel in which every thread owns 32 input and 32 output bytes: 10-30 random i8 / i16 / i32
loads at aligned offsets (some declared align 1), add / xor / or / and / shl / lshr / mul, trunc / zext / sext, and
i8 / i16 / i32 stores at aligned offsets that overlap each other, in program order. The output buffer starts with
random bytes; agx_run's output is compared byte for byte with the model's, which exercises the backend's byte
and halfword load / store combining, byte permutes and bit-field inserts (EXPERIMENTS.md L15-E379). Some kernels
also load bursts of 9-14 words at data-dependent offsets, more requests in flight than eight (L15-E437).

  agx-mem-fuzz.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--seeds A:B]
                  [--mcpu agx-g14c] [--keep DIR]
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

W={8:0xff,16:0xffff,32:0xffffffff}
def gen(seed):
  R=random.Random(seed)
  body=['  %t = call i32 @llvm.agx.thread.position.in.grid.x()','  %base = mul i32 %t, 32',
        '  %pin = getelementptr i8, ptr addrspace(1) %p, i32 %base','  %pout = getelementptr i8, ptr addrspace(1) %q, i32 %base']
  ops=[]; vals=[]; n=[0]
  def nv(): n[0]+=1; return f'%v{n[0]}'
  for _ in range(R.randint(10,30)):
    k=R.random()
    if k<0.35 or len(vals)<2:
      w=R.choice([8,8,16,16,32]); off=R.randrange(0,32,w//8); g=nv(); v=nv()
      body+=[f'  {g} = getelementptr i8, ptr addrspace(1) %pin, i32 {off}', f'  {v} = load i{w}, ptr addrspace(1) {g}, align {R.choice([w//8,w//8,1]) if w>8 else 1}']
      ops.append((v,w,('load',off))); vals.append((v,w))
    elif k<0.6:
      a=R.choice(vals); w=a[1]; T=f'i{w}'; r=nv()
      op=R.choice(['add','xor','or','and','shl','lshr','mul'])
      if op in ('shl','lshr'): c=R.randrange(0,w)
      else:
        b=[x for x in vals if x[1]==w]; c=R.choice(b)[0] if b and R.random()<0.6 else R.choice([1,3,0x55,0x7f,0xff])&W[w]
      body.append(f'  {r} = {op} {T} {a[0]}, {c}'); ops.append((r,w,(op,a[0],c))); vals.append((r,w))
    elif k<0.67 and any(x[1]==32 for x in vals):
      # a burst of 9-14 i32 loads at offsets ((a >> s) & 7) * 4 (distinct
      # shifts: no two the same address expression) that no combining
      # merges, all in flight before their xor (more than eight requests
      # outstanding: L15-E437)
      a=R.choice([x for x in vals if x[1]==32]); loaded=[]
      for c in R.sample(range(29),R.randint(9,14)):
        x=nv(); m=nv(); o=nv(); g=nv(); v=nv()
        body+=[f'  {x} = lshr i32 {a[0]}, {c}', f'  {m} = and i32 {x}, 7', f'  {o} = shl i32 {m}, 2',
               f'  {g} = getelementptr i8, ptr addrspace(1) %pin, i32 {o}', f'  {v} = load i32, ptr addrspace(1) {g}, align 4']
        ops.append((v,32,('dload',a[0],c))); loaded.append(v)
      acc=loaded[0]
      for v in loaded[1:]:
        r=nv(); body.append(f'  {r} = xor i32 {acc}, {v}'); ops.append((r,32,('xor',acc,v))); acc=r
      vals.append((acc,32))
      # (stored, so that the burst is live)
      off=R.randrange(0,32,4); g=nv()
      body+=[f'  {g} = getelementptr i8, ptr addrspace(1) %pout, i32 {off}', f'  store i32 {acc}, ptr addrspace(1) {g}, align 4']
      ops.append((None,32,('store',off,acc)))
    elif k<0.72:
      a=R.choice(vals); nw=R.choice([x for x in (8,16,32) if x!=a[1]]); r=nv()
      if nw<a[1]: body.append(f'  {r} = trunc i{a[1]} {a[0]} to i{nw}'); ops.append((r,nw,('trunc',a[0])))
      else:
        e=R.choice(['zext','sext']); body.append(f'  {r} = {e} i{a[1]} {a[0]} to i{nw}'); ops.append((r,nw,(e,a[0],a[1])))
      vals.append((r,nw))
    else:
      a=R.choice(vals); w=a[1]; off=R.randrange(0,32,w//8); g=nv()
      body+=[f'  {g} = getelementptr i8, ptr addrspace(1) %pout, i32 {off}', f'  store i{w} {a[0]}, ptr addrspace(1) {g}, align {R.choice([w//8,1]) if w>8 else 1}']
      ops.append((None,w,('store',off,a[0])))
  L=['target triple = "agx"','declare i32 @llvm.agx.thread.position.in.grid.x()','define agx_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {']+body+['  ret void','}']
  return '\n'.join(L)+'\n', ops
def sx(v,w): return v-(1<<w) if v>>(w-1) else v
def evaluate(ops, inb, outb0):
  out=bytearray(outb0); e={}
  for r,w,op in ops:
    k=op[0]; m=W[w]
    if k=='load': v=int.from_bytes(inb[op[1]:op[1]+w//8],'little')
    elif k=='dload': o=((e[op[1]]>>op[2])&7)*4; v=int.from_bytes(inb[o:o+4],'little')
    elif k=='store': out[op[1]:op[1]+w//8]=(e[op[2]]&m).to_bytes(w//8,'little'); continue
    elif k=='trunc': v=e[op[1]]
    elif k=='zext': v=e[op[1]]
    elif k=='sext': v=sx(e[op[1]],op[2])
    else:
      a=e[op[1]]; c=e[op[2]] if isinstance(op[2],str) else op[2]
      v={'add':lambda:a+c,'xor':lambda:a^c,'or':lambda:a|c,'and':lambda:a&c,'shl':lambda:a<<c,'lshr':lambda:a>>c,'mul':lambda:a*c}[k]()
    e[r]=v&m
  return out


N = 256


def run_seed(seed, a, tmp):
    src, ops = gen(seed)
    ll, obj, code = (os.path.join(tmp, "k" + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(src)
    r = subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + a.mcpu, "-O3", "-filetype=obj", ll, "-o", obj],
                       capture_output=True, text=True)
    if r.returncode:
        return src, "compile failure: " + (r.stderr.strip().splitlines() or ["?"])[0][:200]
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    rng = np.random.default_rng(seed)
    inb = rng.integers(0, 256, N * 32, dtype=np.uint16).astype(np.uint8)
    outb = rng.integers(0, 256, N * 32, dtype=np.uint16).astype(np.uint8)
    fin, fout0, fout = (os.path.join(tmp, s) for s in ("in.bin", "out0.bin", "out.bin"))
    inb.tofile(fin)
    outb.tofile(fout0)
    subprocess.run([a.run, code, "--grid", "%d,1,1" % N, "--wg", "64,1,1", "--map", "0x1000000=" + fin,
                    "--map", "0x2000000=" + fout0, "--dump", "0x2000000=%d:%s" % (N * 32, fout),
                    "--uniform", "0=16777216", "--uniform", "1=0", "--uniform", "2=33554432", "--uniform", "3=0"],
                   check=True, capture_output=True, timeout=600)
    got = np.fromfile(fout, np.uint8)
    for th in range(N):
        exp = evaluate(ops, bytes(inb[th * 32:th * 32 + 32]), bytes(outb[th * 32:th * 32 + 32]))
        g = bytes(got[th * 32:th * 32 + 32])
        if g != bytes(exp):
            return src, "thread %d wrote %s, the model %s" % (th, g.hex(), bytes(exp).hex())
    return src, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--seeds", default="0:500")
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
