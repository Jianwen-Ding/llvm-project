; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-runtime-stores=0 -S < %s | FileCheck %s --check-prefix=OFF

; A row loop of a runtime length that stores (L15-E565): its contiguous word
; loads and stores, light arithmetic, unrolled 4 times with a remainder loop,
; so that alias versioning's copy has iterations' loads to issue together.

; CHECK-LABEL: @scale(
; CHECK-COUNT-4: store float
; CHECK: .epil
; OFF-LABEL: @scale(
; OFF: store float
; OFF-NOT: store float
; OFF: ret void
define void @scale(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i64 %n) {
entry:
  %e = icmp eq i64 %n, 0
  br i1 %e, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i64 %i
  %v = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr inbounds float, ptr addrspace(1) %w, i64 %i
  %u = load float, ptr addrspace(1) %pw, align 4
  %o = fmul float %v, %u
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i64 %i
  store float %o, ptr addrspace(1) %py, align 4
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
