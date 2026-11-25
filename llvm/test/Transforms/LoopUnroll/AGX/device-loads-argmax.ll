; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-argmax=0 -S < %s | FileCheck %s --check-prefix=FOUR
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-carried-select=0 -S < %s | FileCheck %s --check-prefix=NOSEL

; An arg-max over a thread's row (a running maximum chosen by a select on a
; compare against it, its index by the same compare) is unrolled 32 times
; (L15-E552's 16, 32 since L15-E752), 4 before. A running sum keeps 4.

; CHECK-LABEL: @argmax(
; CHECK-COUNT-32: fcmp ogt float
; CHECK-NOT: fcmp ogt float
; CHECK: br i1
; FOUR-LABEL: @argmax(
; FOUR-COUNT-4: fcmp ogt float
; FOUR-NOT: fcmp ogt float
; FOUR: br i1
; A conditional count and sum (the values kept by selects on one compare
; against a threshold) likewise (L15-E555).
; CHECK-LABEL: @count_above(
; CHECK-COUNT-32: fcmp ogt float
; CHECK-NOT: fcmp ogt float
; CHECK: br i1
; NOSEL-LABEL: @count_above(
; NOSEL-COUNT-4: fcmp ogt float
; NOSEL-NOT: fcmp ogt float
; NOSEL: br i1
; CHECK-LABEL: @rowsum(
; CHECK-COUNT-4: fadd float
; CHECK-NOT: fadd float
; CHECK: br i1
define void @argmax(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %loop ]
  %b = phi i32 [ 0, %entry ], [ %b1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %c = fcmp ogt float %v, %m
  %m1 = select i1 %c, float %v, float %m
  %b1 = select i1 %c, i32 %i, i32 %b
  %i1 = add nuw nsw i32 %i, 1
  %e = icmp ult i32 %i1, 256
  br i1 %e, label %loop, label %exit
exit:
  store i32 %b1, ptr addrspace(1) %o, align 4
  ret void
}

define void @count_above(ptr addrspace(1) %p, ptr addrspace(1) %o, float %t) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %n = phi i32 [ 0, %entry ], [ %n2, %loop ]
  %a = phi float [ 0.0, %entry ], [ %a2, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %c = fcmp ogt float %v, %t
  %n1 = add i32 %n, 1
  %a1 = fadd float %a, %v
  %n2 = select i1 %c, i32 %n1, i32 %n
  %a2 = select i1 %c, float %a1, float %a
  %i1 = add nuw nsw i32 %i, 1
  %e = icmp ult i32 %i1, 256
  br i1 %e, label %loop, label %exit
exit:
  store i32 %n2, ptr addrspace(1) %o, align 4
  ret void
}

define void @rowsum(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i
  %v = load float, ptr addrspace(1) %g, align 4
  %s1 = fadd float %s, %v
  %i1 = add nuw nsw i32 %i, 1
  %e = icmp ult i32 %i1, 256
  br i1 %e, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
