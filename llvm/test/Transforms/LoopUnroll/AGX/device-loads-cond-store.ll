; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-blocks=0 -S < %s | FileCheck %s --check-prefix=OFF

; A row loop around a conditional store (y[j] = x[j] where x[j] > t: three
; blocks) takes the device-load factors too (L15-E572); it was left a loop
; of one load an iteration.

; CHECK-LABEL: @cond(
; CHECK-COUNT-4: load float
; OFF-LABEL: @cond(
; OFF: load float
; OFF-NOT: load float
; OFF: ret void
define void @cond(ptr addrspace(1) %x, ptr addrspace(1) %y, float %t) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %latch ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %g = fcmp ogt float %v, %t
  br i1 %g, label %then, label %latch
then:
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %v, ptr addrspace(1) %py, align 4
  br label %latch
latch:
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
