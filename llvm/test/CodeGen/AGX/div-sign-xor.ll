; RUN: llc -mtriple=agx < %s | FileCheck %s
; RUN: llc -mtriple=agx -agx-div-sign-from-xor=0 < %s | FileCheck %s --check-prefix=OFF

; A signed 32-bit division's quotient takes its sign from (x ^ d) <s 0: one
; xor, and the compare is the negating select's (L15-E324). The remainder
; keeps x's sign.

; CHECK-LABEL: sdiv:
; CHECK: xor [[S:r[0-9]+]], u2, u3
; CHECK-NEXT: isub [[N:r[0-9]+]], 0, [[Q:r[0-9]+]]
; CHECK-NEXT: icmpsel slt, {{r[0-9]+}}, [[S]], 0, [[N]], [[Q]]
; OFF-LABEL: sdiv:
; OFF-DAG: icmpsel slt, {{r[0-9]+}}, u3, 0, 1, 0
; OFF-DAG: icmpsel slt, {{r[0-9]+}}, u2, 0, 1, 0
define void @sdiv(ptr addrspace(1) %p, i32 %x, i32 %y) {
  %r = sdiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: srem:
; CHECK-NOT: xor
; CHECK: stop
define void @srem(ptr addrspace(1) %p, i32 %x, i32 %y) {
  %r = srem i32 %x, %y
  store i32 %r, ptr addrspace(1) %p
  ret void
}
