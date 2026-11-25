; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-speculate-threadgroup-loads=0 < %s | FileCheck %s --check-prefix=OFF

; v = l < nw ? sm[l] : 0 with l = tid & 31 within sm's 32 floats: the load
; runs for every thread and v is a select (L15-E416). At an index that may
; leave the global the if stays.

@sm = internal addrspace(3) global [32 x float] poison, align 4

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i32 @llvm.agx.threads.per.threadgroup.x()

; CHECK-LABEL: @bounded(
; CHECK: load float, ptr addrspace(3)
; CHECK-NEXT: select i1
; CHECK-NOT: br i1
; OFF-LABEL: @bounded(
; OFF: br i1
define agx_kernel void @bounded(ptr addrspace(1) %p) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %n = call i32 @llvm.agx.threads.per.threadgroup.x()
  %nw = lshr i32 %n, 5
  %l = and i32 %t, 31
  %c = icmp ult i32 %l, %nw
  br i1 %c, label %ld, label %j
ld:
  %a = getelementptr float, ptr addrspace(3) @sm, i32 %l
  %v = load float, ptr addrspace(3) %a, align 4
  br label %j
j:
  %r = phi float [ 0.0, %entry ], [ %v, %ld ]
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: @unbounded(
; CHECK: br i1
define agx_kernel void @unbounded(ptr addrspace(1) %p, i32 %i) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %c = icmp ult i32 %i, 32
  br i1 %c, label %ld, label %j
ld:
  %a = getelementptr float, ptr addrspace(3) @sm, i32 %i
  %v = load float, ptr addrspace(3) %a, align 4
  br label %j
j:
  %r = phi float [ 0.0, %entry ], [ %v, %ld ]
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}
