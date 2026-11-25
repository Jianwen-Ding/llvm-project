; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-conditional-sums=0 < %s | FileCheck %s --check-prefix=OFF
;
; A running value's select(c, p + v, p) as p + select(c, v, -0), and
; p * select(c, v, 1) for a product (either side): p is a phi of the selects
; and constants, so p + -0 and p 1 are p itself (arithmetic flushes its
; denormals and gives the default NaN), and the select reads v, not p and
; p + v (L15-E700).

; CHECK-LABEL: @sum(
; CHECK:      %s = phi float [ 0.000000e+00, %entry ], [ %s2, %loop ]
; CHECK-NEXT: %p = phi float [ 1.000000e+00, %entry ], [ %p2, %loop ]
; CHECK:      [[V:%[0-9]+]] = select i1 %g, float %v, float -0.000000e+00
; CHECK-NEXT: %s2 = fadd float %s, [[V]]
; CHECK-NEXT: [[W:%[0-9]+]] = select i1 %g, float 1.000000e+00, float %v
; CHECK-NEXT: %p2 = fmul float %p, [[W]]
; OFF-LABEL: @sum(
; OFF:      %s2 = select i1 %g, float %s1, float %s
; OFF:      %p2 = select i1 %g, float %p, float %p1
define agx_kernel void @sum(ptr addrspace(1) %x, ptr addrspace(1) %out, float %t) {
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s2, %loop ]
  %p = phi float [ 1.0, %entry ], [ %p2, %loop ]
  %a = getelementptr float, ptr addrspace(1) %x, i32 %j
  %v = load float, ptr addrspace(1) %a
  %g = fcmp ogt float %v, %t
  %s1 = fadd float %s, %v
  %s2 = select i1 %g, float %s1, float %s
  %p1 = fmul float %v, %p
  %p2 = select i1 %g, float %p, float %p1
  %j1 = add i32 %j, 1
  %e = icmp eq i32 %j1, 64
  br i1 %e, label %done, label %loop
done:
  store float %s2, ptr addrspace(1) %out
  %o1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %p2, ptr addrspace(1) %o1
  ret void
}

; A start value that may be a NaN (an argument): kept.
; CHECK-LABEL: @nan_start(
; CHECK:      %s2 = select i1 %g, float %s1, float %s
define agx_kernel void @nan_start(ptr addrspace(1) %x, ptr addrspace(1) %out, float %t, float %init) {
entry:
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %s = phi float [ %init, %entry ], [ %s2, %loop ]
  %a = getelementptr float, ptr addrspace(1) %x, i32 %j
  %v = load float, ptr addrspace(1) %a
  %g = fcmp ogt float %v, %t
  %s1 = fadd float %s, %v
  %s2 = select i1 %g, float %s1, float %s
  %j1 = add i32 %j, 1
  %e = icmp eq i32 %j1, 64
  br i1 %e, label %done, label %loop
done:
  store float %s2, ptr addrspace(1) %out
  ret void
}
