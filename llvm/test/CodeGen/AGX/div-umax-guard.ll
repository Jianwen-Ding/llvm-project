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

; The signed guard, d + 1 <u 2 ? 1 : d, with ZLUDA's selects for d = -1 and
; d = 0: the expansion of d with no select of its own (one compare-select
; each for d = -1 and d = 0, after the quotient's sign).
; CHECK-LABEL: sd:
; CHECK-NOT: icmpsel ult, r{{[0-9]+}}, r{{[0-9]+}}, 2
; CHECK: stop
define agx_kernel void @sd(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %z = icmp eq i32 %b, 0
  %m = icmp eq i32 %b, -1
  %b1 = add i32 %b, 1
  %c = icmp ult i32 %b1, 2
  %g = select i1 %c, i32 1, i32 %b
  %q = sdiv i32 %a, %g
  %n = sub i32 0, %a
  %r1 = select i1 %m, i32 %n, i32 %q
  %r = select i1 %z, i32 -1, i32 %r1
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; Without them, the result of dividing by one where the guard took 1.
; CHECK-LABEL: sr2:
; CHECK: icmpsel ult, r{{[0-9]+}}, r{{[0-9]+}}, 2, 0,
; CHECK: stop
define agx_kernel void @sr2(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %b1 = add i32 %b, 1
  %c = icmp ult i32 %b1, 2
  %g = select i1 %c, i32 1, i32 %b
  %q = srem i32 %a, %g
  store i32 %q, ptr addrspace(1) %p
  ret void
}
