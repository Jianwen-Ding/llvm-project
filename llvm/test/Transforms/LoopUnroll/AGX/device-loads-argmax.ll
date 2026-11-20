; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-argmax=0 -S < %s | FileCheck %s --check-prefix=FOUR

; An arg-max over a thread's row (a running maximum chosen by a select on a
; compare against it, its index by the same compare) is unrolled 16 times
; (L15-E552), 4 before. A running sum keeps 4.

; CHECK-LABEL: @argmax(
; CHECK-COUNT-16: fcmp ogt float
; CHECK-NOT: fcmp ogt float
; CHECK: br i1
; FOUR-LABEL: @argmax(
; FOUR-COUNT-4: fcmp ogt float
; FOUR-NOT: fcmp ogt float
; FOUR: br i1
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
