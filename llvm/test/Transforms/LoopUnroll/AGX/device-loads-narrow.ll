; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-narrow=0 -S < %s | FileCheck %s --check-prefix=FOUR

; A device-load loop (device-loads.ll) whose loads are all of one or two
; bytes (half, int8 rows) is unrolled 16 times where 16 divides its trip
; count, 4 otherwise; one with a wider load keeps 4 (L15-E234).

define void @sumh512(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @sumh512(
; CHECK-COUNT-16: load half
; CHECK-NOT: load half
; CHECK: br i1
; FOUR-LABEL: @sumh512(
; FOUR-COUNT-4: load half
; FOUR-NOT: load half
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds half, ptr addrspace(1) %p, i64 %i
  %v = load half, ptr addrspace(1) %g, align 2
  %e = fpext half %v to float
  %s1 = fadd float %s, %e
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

; 520 iterations: 16 does not divide them, so 4.
define void @sumb520(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @sumb520(
; FOUR-LABEL: @sumb520(
; CHECK-COUNT-4: load i8
; CHECK-NOT: load i8
; CHECK: br i1
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %i
  %v = load i8, ptr addrspace(1) %g, align 1
  %e = sext i8 %v to i32
  %s1 = add i32 %s, %e
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 520
  br i1 %c, label %loop, label %exit
exit:
  store i32 %s1, ptr addrspace(1) %o, align 4
  ret void
}

; A half and a float load: 4.
define void @mixed512(ptr addrspace(1) %p, ptr addrspace(1) %q, ptr addrspace(1) %o) {
; CHECK-LABEL: @mixed512(
; CHECK-COUNT-4: load float
; CHECK-NOT: load float
; CHECK: br i1
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds half, ptr addrspace(1) %p, i64 %i
  %v = load half, ptr addrspace(1) %g, align 2
  %e = fpext half %v to float
  %h = getelementptr inbounds float, ptr addrspace(1) %q, i64 %i
  %w = load float, ptr addrspace(1) %h, align 4
  %m = fmul float %e, %w
  %s1 = fadd float %s, %m
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
