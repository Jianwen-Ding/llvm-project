; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-guard-versioning-min=0 < %s | FileCheck %s --check-prefix=OFF

; A loop whose loads are each guarded by a loop-invariant test (a stencil's
; taps inside the image) gets a copy without the tests, run by SIMD-groups
; whose every active thread passes them all (a ballot): its loads issue
; together; the others run the loop as it was (L15-E247). The copy's blocks
; are merged and its accesses' inttoptrs of one address made one, so the
; load/store vectorizer joins neighbouring taps (L15-E268): here one load of
; the four.

; CHECK-LABEL: taps:
; CHECK: icmp_ballot
; CHECK: icmp_ballot
; The original loop: a test per load.
; CHECK: Loop Header
; CHECK: if_icmp
; CHECK: jmp_exec_any
; The copy (inside.preheader): one load of the four taps, no test.
; CHECK: else_fcmp
; CHECK: inside.preheader
; CHECK: Loop Header
; CHECK-NOT: if_
; CHECK: device_load 0, i32, xyzw,
; CHECK-NOT: if_
; CHECK-NOT: device_load
; CHECK: jmp_exec_any
; OFF-LABEL: taps:
; OFF-NOT: icmp_ballot
; OFF: stop
define agx_kernel void @taps(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n, i1 %g0, i1 %g1, i1 %g2, i1 %g3) {
entry:
  br label %h
h:
  %i = phi i32 [ 0, %entry ], [ %i1, %m3 ]
  %s = phi float [ 0.0, %entry ], [ %s3, %m3 ]
  %b = shl i32 %i, 2
  br i1 %g0, label %t0, label %m0
t0:
  %p0 = getelementptr float, ptr addrspace(1) %in, i32 %b
  %v0 = load float, ptr addrspace(1) %p0, align 4
  %a0 = fadd float %s, %v0
  br label %m0
m0:
  %s0 = phi float [ %s, %h ], [ %a0, %t0 ]
  br i1 %g1, label %t1, label %m1
t1:
  %b1 = add i32 %b, 1
  %p1 = getelementptr float, ptr addrspace(1) %in, i32 %b1
  %v1 = load float, ptr addrspace(1) %p1, align 4
  %a1 = fadd float %s0, %v1
  br label %m1
m1:
  %s1 = phi float [ %s0, %m0 ], [ %a1, %t1 ]
  br i1 %g2, label %t2, label %m2
t2:
  %b2 = add i32 %b, 2
  %p2 = getelementptr float, ptr addrspace(1) %in, i32 %b2
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %a2 = fadd float %s1, %v2
  br label %m2
m2:
  %s2 = phi float [ %s1, %m1 ], [ %a2, %t2 ]
  br i1 %g3, label %t3, label %m3
t3:
  %b3 = add i32 %b, 3
  %p3 = getelementptr float, ptr addrspace(1) %in, i32 %b3
  %v3 = load float, ptr addrspace(1) %p3, align 4
  %a3 = fadd float %s2, %v3
  br label %m3
m3:
  %s3 = phi float [ %s2, %m2 ], [ %a3, %t3 ]
  %i1 = add i32 %i, 1
  %c = icmp slt i32 %i1, %n
  br i1 %c, label %h, label %exit
exit:
  store float %s3, ptr addrspace(1) %out, align 4
  ret void
}
