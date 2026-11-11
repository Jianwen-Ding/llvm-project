; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll=1 -S < %s | FileCheck %s --check-prefix=OFF

; A single-block loop that loads device memory, with a constant trip count 4
; divides, is unrolled 4 times (its loads in flight together); one of a trip
; count 4 does not divide, or of a runtime trip count, is not (no remainder
; loop).

define void @sum512(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @sum512(
; CHECK-COUNT-4: load float
; CHECK-NOT: load float
; CHECK: br i1
; OFF-LABEL: @sum512(
; OFF-COUNT-1: load float
; OFF-NOT: load float
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

define void @sum510(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @sum510(
; OFF-LABEL: @sum510(
; CHECK-COUNT-1: load float
; CHECK-NOT: load float
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 510
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

define void @sumn(ptr addrspace(1) %p, ptr addrspace(1) %o, i64 %n) {
; CHECK-LABEL: @sumn(
; CHECK-COUNT-1: load float
; CHECK-NOT: load float
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
