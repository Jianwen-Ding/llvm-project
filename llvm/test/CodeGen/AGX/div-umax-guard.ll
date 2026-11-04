; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; A division by umax(d, 1) (ZLUDA's guard, with its own select for d = 0):
; the expansion's reciprocal is of d, with no maximum, and where the only use
; is that select, no select of its own: one compare-select for d = 0.

; CHECK-LABEL: ud:
; CHECK: convert u32_to_f, r{{[0-9]+}}, u3
; CHECK-NOT: icmpsel ugt
; CHECK-COUNT-1: icmpsel ueq
; CHECK-NOT: icmpsel ueq
; CHECK: stop
define agx_kernel void @ud(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %g = call i32 @llvm.umax.i32(i32 %b, i32 1)
  %q = udiv i32 %a, %g
  %z = icmp eq i32 %b, 0
  %r = select i1 %z, i32 -1, i32 %q
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; Another use: the result of dividing by one where d = 0 (x % 1 = 0).
; CHECK-LABEL: ur2:
; CHECK-NOT: icmpsel ugt
; CHECK: icmpsel ueq, r{{[0-9]+}}, u3, 0, 0,
; CHECK: stop
define agx_kernel void @ur2(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %g = call i32 @llvm.umax.i32(i32 %b, i32 1)
  %q = urem i32 %a, %g
  store i32 %q, ptr addrspace(1) %p
  ret void
}
