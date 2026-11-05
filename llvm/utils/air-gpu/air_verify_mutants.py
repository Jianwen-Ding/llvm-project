#!/usr/bin/env python3
"""Does llc's check of AIR miss what the device rejects? Breaks a valid kernel in many ways and compares.

Each mutant of a lowered valid kernel is written with -air-skip-device-checks and run on the device behind
the AIR service, one at a time under the device lock, and written normally. The exit status is 1 if the device rejects
a mutant llc accepts: a rule missing from AIRVerify.cpp. llc rejecting one the device takes is reported as stricter, which
may be right (the device is no complete judge). The mutants are straight-line kernels, and the run stops at any timeout.
Run from a scratch directory, where it writes them:

    python3 air_verify_mutants.py --llc build/bin/llc
"""
import argparse, base64, fcntl, json, os, re, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument('--llc', default='llc')
ap.add_argument('--queue', default=os.environ.get('AIR_QUEUE', '/air'))
ap.add_argument('--lock', default=os.environ.get('AIR_DEVICE_LOCK', '/scratch/air-device.lock'))
args = ap.parse_args()
sys.path.insert(0, os.environ.get('AIR_CLIENT', '/air/client'))
from client import submit  # noqa: E402
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([args.llc, '-mtriple=air', os.path.join(here, 'verify-cases', 'ok.ll'), '-o', 'ok.low.ll'], check=True)
base = re.sub(r'^; ModuleID.*\n|^source_filename.*\n', '', open('ok.low.ll').read(), flags=re.M)
M = {}
def rep(name, o, n, count=1):
    if o in base:
        M[name] = base.replace(o, n, count)
def rx(name, p, n):
    s, c = re.subn(p, n, base, count=1, flags=re.M)
    if c:
        M[name] = s
M['control']=base
rx('no-air-version',r'^!air\.version = .*\n','')
rx('no-lang-version',r'^!air\.language_version = .*\n','')
rx('no-compile-options',r'^!air\.compile_options = .*\n','')
rx('no-kernel-list',r'^!air\.kernel = .*\n','')
rx('bad-air-version',r'^(!\d+ = !\{i32 )2(, i32 \d+, i32 0\})',r'\g<1>9\2')
rx('bad-lang-name',r'!"Metal"',r'!"Vulkan"')
rx('triple-other',r'^target triple = .*$','target triple = "x86_64-unknown-linux-gnu"')
rx('datalayout-bad',r'^target datalayout = .*$','target datalayout = "e-p:32:32"')
rx('no-datalayout',r'^target datalayout = .*\n','')
rx('arg-role-unknown',r'!"air\.buffer"',r'!"air.nonsense"')
rx('arg-as-wrong',r'!"air\.address_space", i32 1',r'!"air.address_space", i32 7')
rx('arg-loc-negative',r'!"air\.location_index", i32 0',r'!"air.location_index", i32 -1')
rx('arg-loc-huge',r'!"air\.location_index", i32 0',r'!"air.location_index", i32 4000')
rx('arg-no-size',r', !"air\.arg_type_size", i32 \d+',r'')
rx('arg-no-align',r', !"air\.arg_type_align_size", i32 \d+',r'')
rx('arg-no-typename',r', !"air\.arg_type_name", !"[^"]*"',r'')
rx('arg-no-name',r', !"air\.arg_name", !"[^"]*"',r'')
rx('arg-index-wrong',r'^(!\d+ = !\{i32 )0(, !"air\.buffer")',r'\g<1>5\2')
rx('access-unknown',r'!"air\.read_write"',r'!"air.scribble"')
rx('fn-rename-in-md',r'!\{ptr @k,',r'!{ptr null,')
rx('call-unknown-intrinsic',r'^  ret void','  call void @llvm.donothing()\n  ret void')
rx('as5-load',r'^  ret void','  %mz = alloca i32, align 4, addrspace(5)\n  ret void')
rx('fp128',r'^  ret void','  %mz = fadd fp128 0xL0, 0xL0\n  ret void')
rx('x86-fp80',r'^  ret void','  %mz = fadd x86_fp80 0xK0, 0xK0\n  ret void')
rx('i7',r'^  ret void','  %mz = add i7 1, 2\n  ret void')
rx('v3i64',r'^  ret void','  %mz = add <3 x i64> zeroinitializer, zeroinitializer\n  ret void')
rx('v9f32',r'^  ret void','  %mz = fadd <9 x float> zeroinitializer, zeroinitializer\n  ret void')
rx('frem',r'^  ret void','  %mz = frem float 1.0, 2.0\n  ret void')
rx('fneg-half',r'^  ret void','  %mz = fneg half 1.0\n  ret void')
rx('switch',r'^  ret void','  switch i32 0, label %e [ i32 1, label %e ]\ne:\n  ret void')
rx('indirectbr',r'^  ret void','  indirectbr ptr blockaddress(@k, %e), [label %e]\ne:\n  ret void')
rx('unreachable-end',r'^  ret void','  unreachable')
rx('invoke-less-landing',r'^  ret void','  %mz = freeze i32 1\n  ret void')
rx('vaarg-call',r'^  ret void','  %mz = extractvalue { i32, i32 } { i32 1, i32 2 }, 0\n  ret void')
rx('atomicrmw',r'^  ret void','  %mz = atomicrmw add ptr addrspace(1) null, i32 1 monotonic\n  ret void')
rx('cmpxchg',r'^  ret void','  %mz = cmpxchg ptr addrspace(1) null, i32 0, i32 1 monotonic monotonic\n  ret void')
rx('fence',r'^  ret void','  fence seq_cst\n  ret void')
rx('addrspacecast13',r'^  ret void','  %mz = addrspacecast ptr addrspace(1) null to ptr addrspace(3)\n  ret void')
rx('global-tls',r'^(define)','@tv = thread_local addrspace(1) global i32 0\n\\1')
rx('global-device-write',r'^(define)','@gw = addrspace(1) global i32 0\n\\1')
rx('alias',r'^(define)','@aa = internal addrspace(3) global i32 undef\n@ab = alias i32, ptr addrspace(3) @aa\n\\1')
rx('recursive',r'^(define)','define i32 @rr(i32 %x) { %y = call i32 @rr(i32 %x)\n ret i32 %y }\n\\1')
R=r'^  ret void'
def body(n,txt,decl=''):
    t=re.sub(R,'  '+txt.replace('\n','\n  ')+'\n  ret void',base,count=1,flags=re.M)
    M[n]=(decl+'\n' if decl else '')+t if not decl else re.sub(r'^(define)',decl+'\n\\1',t,count=1,flags=re.M)
# metadata
rx('kernel-md-2ops',r'!\{ptr @k, !2, !3\}',r'!{ptr @k, !2}')
rx('kernel-md-noargs',r'!\{ptr @k, !2, !3\}',r'!{ptr @k, !2, !2}')
rx('md-sysval-wrong-name',r'!"air\.thread_position_in_threadgroup"',r'!"air.threads_per_grid"')
rx('md-sysval-typename',r'!"uint3"',r'!"float"')
rx('md-buffer-size-0',r'!"air\.arg_type_size", i32 4',r'!"air.arg_type_size", i32 0')
rx('md-buffer-align-3',r'!"air\.arg_type_align_size", i32 4',r'!"air.arg_type_align_size", i32 3')
rx('md-loc-noncount',r'i32 0, i32 1, !"air\.read_write"',r'i32 0, i32 0, !"air.read_write"')
rx('md-as-const-mismatch',r'!"air\.address_space", i32 1',r'!"air.address_space", i32 2')
rx('md-read-on-write',r'!"air\.read_write"',r'!"air.read"')
rx('lang-version-bad',r'!\{!"Metal", i32 4, i32 0, i32 0\}',r'!{!"Metal", i32 99, i32 0, i32 0}')
rx('air-version-minor-99',r'!\{i32 2, i32 8, i32 0\}',r'!{i32 2, i32 99, i32 0}')
rx('air-version-short',r'!\{i32 2, i32 8, i32 0\}',r'!{i32 2}')
rx('compile-opt-unknown',r'!"air\.compile\.denorms_disable"',r'!"air.compile.nonsense"')
rx('flag-buffers-1',r'!"air\.max_device_buffers", i32 31',r'!"air.max_device_buffers", i32 1')
# signature against metadata
rx('sysval-type-i32',r'<3 x i32> %tidtg\) #0 \{\n  %tidtg_x = extractelement <3 x i32> %tidtg, i32 0',r'i32 %tidtg) #0 {\n  %tidtg_x = add i32 %tidtg, 0')
rx('buffer-as-i64',r'ptr addrspace\(1\) "air-buffer-no-alias" %o,',r'i64 %oi,')
rx('kernel-returns',r'define void @k\((.*)\) #0 \{',r'define i32 @k(\1) #0 {')
# calls of AIR functions
body('barrier-1arg','call void @air.wg.barrier(i32 2)','declare void @air.wg.barrier(i32)')
body('barrier-badflags','call void @air.wg.barrier(i32 99, i32 99)','declare void @air.wg.barrier(i32, i32)')
body('fence-badargs','call void @air.atomic.fence(i32 99, i32 99, i32 99)','declare void @air.atomic.fence(i32, i32, i32)')
body('atomic-badorder','%mz = call i32 @air.atomic.global.load.i32(ptr addrspace(1) null, i32 99, i32 99, i1 true)','declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)')
body('atomic-wrongas','%mz = call i32 @air.atomic.global.load.i32(ptr addrspace(3) null, i32 0, i32 2, i1 true)','declare i32 @air.atomic.global.load.i32(ptr addrspace(3), i32, i32, i1)')
body('sin-2args','%mz = call float @air.sin.f32(float 1.0, float 2.0)','declare float @air.sin.f32(float, float)')
body('sin-noret','call void @air.sin.f32(float 1.0)','declare void @air.sin.f32(float)')
body('sin-suffix-f64','%mz = call double @air.sin.f64(double 1.0)','declare double @air.sin.f64(double)')
body('sin-suffix-x','%mz = call float @air.sin.xyz(float 1.0)','declare float @air.sin.xyz(float)')
body('simd-shuffle-noargs','%mz = call i32 @air.simd_shuffle.u.i32()','declare i32 @air.simd_shuffle.u.i32()')
body('simd-unknown','%mz = call i32 @air.simd_nonsense.i32(i32 1)','declare i32 @air.simd_nonsense.i32(i32)')
body('atomic-unknown','%mz = call i32 @air.atomic.global.nonsense.i32(ptr addrspace(1) null)','declare i32 @air.atomic.global.nonsense.i32(ptr addrspace(1))')
# types
body('i128','%mz = add i128 1, 2')
body('i256','%mz = add i256 1, 2')
body('v128i8','%mz = add <128 x i8> zeroinitializer, zeroinitializer')
body('v2i1store','store <2 x i1> zeroinitializer, ptr addrspace(1) %o')
body('scalable','%mz = alloca <vscale x 4 x i32>')
body('token','%mz = call token @llvm.experimental.convergence.anchor()','declare token @llvm.experimental.convergence.anchor()')
body('struct-load','%mz = load {i32,float}, ptr addrspace(1) %o')
body('array-load','%mz = load [4 x i32], ptr addrspace(1) %o')
body('ptr-store','store ptr addrspace(1) %o, ptr addrspace(1) %o')
body('as9-store','store i32 1, ptr addrspace(9) null')
body('bitcast-vec','%mz = bitcast <2 x i32> zeroinitializer to i64')
body('va-function','%mz = call i32 (i32, ...) @vf(i32 1, i32 2)','declare i32 @vf(i32, ...)')
body('float-as-ptr','%mz = inttoptr i64 1 to ptr addrspace(1)\n%my = load i32, ptr addrspace(1) %mz')
body('call-self','call void @k(ptr addrspace(1) %o, <3 x i32> %tidtg)')
def top(n,decl): M[n]=re.sub(r'^(define)',decl+'\n\\1',base,count=1,flags=re.M)
# globals
top('tg-init','@g = addrspace(3) global i32 7')
top('tg-noinit-external','@g = external addrspace(3) global i32')
top('const-noinit','@g = external addrspace(2) constant i32')
top('const-zero-size','@g = addrspace(2) constant [0 x i32] zeroinitializer')
top('device-global','@g = addrspace(1) global i32 1')
top('thread-global','@g = global i32 1')
top('as6-global','@g = addrspace(6) global i32 1')
top('as9-global','@g = addrspace(9) global i32 1')
top('global-section','@g = addrspace(2) constant i32 1, section "x"')
top('global-huge-align','@g = addrspace(2) constant i32 1, align 1073741824')
top('global-fp128','@g = addrspace(2) constant fp128 0xL0')
top('global-ifunc','@r = internal addrspace(2) constant i32 1\n@p = addrspace(2) constant ptr addrspace(2) @r')
top('comdat','$c = comdat any\n@g = addrspace(2) constant i32 1, comdat($c)')
top('tg-huge','@g = internal addrspace(3) global [100000 x i32] undef')
top('function-section','define void @h() section "x" { ret void }')
top('function-gc','define void @h() gc "statepoint-example" { ret void }')
top('function-personality','declare i32 @pp(...)\ndefine void @h() personality ptr @pp { ret void }')
top('function-naked','define void @h() naked { unreachable }')
top('function-cc-fast','define fastcc void @h() { ret void }')
top('function-cc-agx','define cc 128 void @h() { ret void }')
# instructions
body('load-as9-ptr','%mz = load i32, ptr addrspace(9) null')
body('gep-as9','%mz = getelementptr i8, ptr addrspace(9) null, i64 1')
body('cast-to-as9','%mz = addrspacecast ptr addrspace(1) %o to ptr addrspace(9)')
body('ptrtoint-as9','%mz = ptrtoint ptr addrspace(9) null to i64')
body('select-vector-i1','%mz = select <2 x i1> zeroinitializer, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer')
body('load-v2i1','%mz = load <2 x i1>, ptr addrspace(1) %o')
body('load-i128','%mz = load i128, ptr addrspace(1) %o')
body('store-i128','store i128 1, ptr addrspace(1) %o')
body('atomic-load-i64','%mz = load atomic i64, ptr addrspace(1) %o monotonic, align 8')
body('atomic-store-i8','store atomic i8 1, ptr addrspace(1) %o monotonic, align 1')
body('udiv-i128','%mz = udiv i128 7, 3')
body('shl-i128','%mz = shl i128 7, 3')
body('frem-double','%mz = frem double 1.0, 2.0')
body('fadd-double','%mz = fadd double 1.0, 2.0')
body('alloca-dyn','%mz = alloca i32, i32 %tidtg_x')
body('alloca-huge','%mz = alloca [10000000 x i32]')
body('memcpy-tg','call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) null, ptr addrspace(1) %o, i64 4, i1 false)','declare void @llvm.memcpy.p3.p1.i64(ptr addrspace(3), ptr addrspace(1), i64, i1)')
body('intrinsic-x86','%mz = call i32 @llvm.x86.rdtsc()','declare i32 @llvm.x86.rdtsc()')
body('intrinsic-readcycle','%mz = call i64 @llvm.readcyclecounter()','declare i64 @llvm.readcyclecounter()')
body('intrinsic-stacksave','%mz = call ptr @llvm.stacksave.p0()','declare ptr @llvm.stacksave.p0()')
body('intrinsic-trap-ubsan','call void @llvm.ubsantrap(i8 1)','declare void @llvm.ubsantrap(i8)')
body('intrinsic-vector-reduce','%mz = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> zeroinitializer)','declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)')
body('intrinsic-masked','%mz = call <2 x i32> @llvm.masked.load.v2i32.p1(ptr addrspace(1) %o, i32 4, <2 x i1> zeroinitializer, <2 x i32> zeroinitializer)','declare <2 x i32> @llvm.masked.load.v2i32.p1(ptr addrspace(1), i32, <2 x i1>, <2 x i32>)')
body('intrinsic-ctlz-i128','%mz = call i128 @llvm.ctlz.i128(i128 1, i1 false)','declare i128 @llvm.ctlz.i128(i128, i1)')
body('indirect-call','%fp = inttoptr i64 8 to ptr\ncall void %fp()')
body('call-undef-args','%mz = call float @air.sin.f32(float undef)','declare float @air.sin.f32(float)')
body('poison-store','store i32 poison, ptr addrspace(1) %o')
body('freeze-ptr','%mz = freeze ptr addrspace(1) %o')
body('insertvalue','%mz = insertvalue {i32,i32} undef, i32 1, 0')
body('landing','%mz = fcmp false float 1.0, 2.0')
# metadata on a kernel
rx('role-texture-on-ptr',r'!"air\.buffer"',r'!"air.texture"')
rx('role-sampler-on-ptr',r'!"air\.buffer"',r'!"air.sampler"')
rx('sysval-grid-on-buffer',r'!"air\.buffer"',r'!"air.thread_position_in_grid"')
rx('buffer-on-sysval',r'!"air\.thread_position_in_threadgroup"',r'!"air.buffer"')
rx('fragment-role',r'!"air\.thread_position_in_threadgroup"',r'!"air.position"')
rx('sysval-int-name',r'!"air\.arg_name", !"tidtg"',r'!"air.arg_name", i32 7')
rx('md-loc-string',r'!"air\.location_index", i32 0',r'!"air.location_index", !"zero"')
rx('md-arg-idx-string',r'!\{i32 0, !"air\.buffer"',r'!{!"zero", !"air.buffer"')
rx('kernel-twice',r'^!air\.kernel = !\{!4\}',r'!air.kernel = !{!4, !4}')
rx('kernel-outputs-nonempty',r'!\{ptr @k, !2, !3\}',r'!{ptr @k, !3, !3}')
rx('fragment-list',r'^!air\.kernel = ',r'!air.fragment = ')
rx('vertex-list',r'^!air\.kernel = ',r'!air.vertex = ')
rx('lang-name-int',r'!\{!"Metal", i32 4',r'!{i32 7, i32 4')
rx('lang-short',r'!\{!"Metal", i32 4, i32 0, i32 0\}',r'!{!"Metal"}')
rx('version-strings',r'!\{i32 2, i32 8, i32 0\}',r'!{!"2", !"8", !"0"}')
rx('alias-scope-bad',r'!alias\.scope !18',r'!alias.scope !7')
# Systematic, on a kernel with a constant buffer, threadgroup memory, an atomic and a barrier: without each
# metadata entry and key, and with each integer changed.
rich = """target triple = "air"
@tg = internal addrspace(3) global [4 x i32] undef, align 16
@c = internal addrspace(2) constant [2 x i32] [i32 3, i32 5], align 4
declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)
define void @k(ptr addrspace(1) %o, ptr addrspace(2) byref(i64) %p) "air-kernel" {
  %v = load i32, ptr addrspace(2) @c
  store i32 %v, ptr addrspace(3) @tg
  call void @air.wg.barrier(i32 2, i32 1)
  %w = load i32, ptr addrspace(3) @tg
  %a = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %o, i32 %w, i32 0, i32 2, i1 false)
  %q = load i64, ptr addrspace(2) %p
  store i32 %a, ptr addrspace(1) %o
  ret void
}
"""
open('rich.ll', 'w').write(rich)
subprocess.run([args.llc, '-mtriple=air', 'rich.ll', '-o', 'rich.low.ll'], check=True)
base = re.sub(r'^; ModuleID.*\n|^source_filename.*\n', '', open('rich.low.ll').read(), flags=re.M)
M['rich-control'] = base
lines=base.split('\n') 
for i,l in enumerate(lines):
    if re.match(r'^!(air\.[a-z_]+|llvm\.module\.flags) = ',l):
        M['del-named-'+l.split()[0][1:]]='\n'.join(lines[:i]+lines[i+1:])
    m=re.match(r'^(!\d+) = (distinct )?!\{(.*)\}$',l)
    if not m or 'distinct' in l: continue
    tag=m.group(1)[1:]
    toks=m.group(3)
    for k in re.finditer(r'!"(air\.[a-z_.]+)"(?:, (i32 -?\d+|!"[^"]*"))?',toks):
        new=toks[:k.start()]+toks[k.end():]
        new=re.sub(r'^, |, $','',new).replace(', , ',', ')
        M['del-%s-%s'%(tag,k.group(1))]='\n'.join(lines[:i]+['%s = !{%s}'%(m.group(1),new)]+lines[i+1:])
    for k in re.finditer(r'i32 (-?\d+)',toks):
        n=int(k.group(1))
        for v,lab in ((n+100,'p100'),(-1,'neg')):
            new=toks[:k.start()]+'i32 %d'%v+toks[k.end():]
            M['int-%s-%d-%s'%(tag,k.start(),lab)]='\n'.join(lines[:i]+['%s = !{%s}'%(m.group(1),new)]+lines[i+1:])
res=[]
for n,src in M.items():
    open(n+'.ll','w').write(src)
    w=subprocess.run([args.llc,'-mtriple=air','-filetype=obj','-air-output=air','-air-skip-device-checks',n+'.ll','-o',n+'.air'],capture_output=True,text=True)
    if w.returncode:
        print(f'{n:24} llc-stops-anyway {w.stderr.strip()[-90:]!r}',flush=True); res.append((n,'llc-stops','',w.stderr[-200:])); continue
    c=subprocess.run([args.llc,'-mtriple=air','-filetype=obj','-air-output=air',n+'.ll','-o',os.devnull],capture_output=True,text=True)
    verdict='ok' if c.returncode==0 else 'error'
    job={'op':'execute','air':base64.b64encode(open(n+'.air','rb').read()).decode(),'functionality':'AIR mutant','gpu_timeout_ms':2000,'compile_timeout_ms':20000,'buffers':[{'size':256},{'size':256}],'commands':[{'kernel':'k','bindings':[{'index':0,'buffer':0},{'index':1,'buffer':1}],'grid':[1,1,1],'group':[1,1,1]}],'readback':[0]}
    with open(args.lock,'a') as l:
        fcntl.flock(l,fcntl.LOCK_EX); rep_=submit(job,args.queue,120)
    dev='ok' if rep_.get('ok') else 'REJECT'
    err=str(rep_.get('error',''))
    if 'exceeded' in err or 'Refusing GPU work' in err: print('STOP timeout',n,err[:100]); break
    if n=='control' and dev!='ok': print('CONTROL REJECTED',err[:200]); break
    tag='' if (dev=='REJECT')==(verdict=='error') else ' <<< '+('GAP' if dev=='REJECT' else 'stricter')
    print(f'{n:24} device={dev:6} verify={verdict:5}{tag} {err[:75]!r}',flush=True)
    res.append((n,dev,verdict,err[:200]))
gaps=[r for r in res if r[1]=='REJECT' and r[2]=='ok']
print('%d mutants, %d the device rejects and llc accepts' % (len(res), len(gaps)))
sys.exit(1 if gaps else 0)
