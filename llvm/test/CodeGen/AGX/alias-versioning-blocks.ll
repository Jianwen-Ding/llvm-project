; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-alias-version -agx-alias-versioning-blocks=0 < %s | FileCheck %s --check-prefix=OFF

; A loop of several blocks around conditional stores (an unrolled
; y[j] = x[j] where x[j] > t) is versioned too (L15-E573), and in its copy
; for ranges not overlapping the second copy's load moves up to the header,
; ahead of the first copy's store: the two loads issue together.

; CHECK-LABEL: define agx_kernel void @cond2(
; CHECK: .lver.check:
; CHECK: {{^  }}loop:
; CHECK: %v0 = load float, {{.*}} !alias.scope
; CHECK-NOT: store
; CHECK: %v1 = load float, {{.*}} !alias.scope
; CHECK-NEXT: br i1 %g0, label %then0, label %mid
; OFF-LABEL: define agx_kernel void @cond2(
; OFF-NOT: lver
; OFF: ret void

define agx_kernel void @cond2(ptr addrspace(1) %x, ptr addrspace(1) %y, float %t) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j2, %latch ]
  %i0 = add i32 %row, %j
  %p0 = getelementptr float, ptr addrspace(1) %x, i32 %i0
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %g0 = fcmp ogt float %v0, %t
  br i1 %g0, label %then0, label %mid
then0:
  %q0 = getelementptr float, ptr addrspace(1) %y, i32 %i0
  store float %v0, ptr addrspace(1) %q0, align 4
  br label %mid
mid:
  %i1 = add i32 %i0, 1
  %p1 = getelementptr float, ptr addrspace(1) %x, i32 %i1
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %g1 = fcmp ogt float %v1, %t
  br i1 %g1, label %then1, label %latch
then1:
  %q1 = getelementptr float, ptr addrspace(1) %y, i32 %i1
  store float %v1, ptr addrspace(1) %q1, align 4
  br label %latch
latch:
  %j2 = add nuw nsw i32 %j, 2
  %c = icmp ult i32 %j2, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()

; A loop of several blocks with a break (two exiting blocks) is left:
; LoopVersioning takes one exiting block (the control-flow fuzzer's seeds
; 126 and 248 crashed in it).
; CHECK-LABEL: define agx_kernel void @brk(
; CHECK-NOT: lver
; CHECK: ret void
define agx_kernel void @brk(ptr addrspace(1) %x, ptr addrspace(1) %y, float %t) {
entry:
  %r = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %r, 256
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %latch ]
  %i = add i32 %row, %j
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p, align 4
  %g = fcmp ogt float %v, %t
  br i1 %g, label %exit, label %latch
latch:
  %q = getelementptr float, ptr addrspace(1) %y, i32 %i
  store float %v, ptr addrspace(1) %q, align 4
  %j1 = add nuw nsw i32 %j, 1
  %c = icmp ult i32 %j1, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
