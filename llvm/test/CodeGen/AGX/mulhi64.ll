; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The high half of a 64 x 64-bit product: four 32 x 32 -> 64-bit multiply-adds,
; the partial sums in their 64-bit addends (none overflows), each product
; computed once (a 32-bit low half is not taken from a second multiply); the
; signed one is the unsigned one less the operands that the other's sign
; selects.

; CHECK-LABEL: mulhu64:
; CHECK-COUNT-4: imadd r{{[0-9]+}}_r{{[0-9]+}},
; CHECK-NOT:     imadd
; CHECK-NOT:     icmpsel
; CHECK:         stop

; CHECK-LABEL: mulhs64:
; CHECK-COUNT-4: imadd r{{[0-9]+}}_r{{[0-9]+}},
; CHECK-NOT:     imadd
; CHECK-NOT:     icmpsel
; CHECK-COUNT-2: isub
; CHECK:         stop

; CHECK-LABEL: mul64:
; CHECK-COUNT-3: imadd
; CHECK:         stop
define agx_kernel void @mulhu64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %x = zext i64 %a to i128
  %y = zext i64 %b to i128
  %m = mul i128 %x, %y
  %h = lshr i128 %m, 64
  %r = trunc i128 %h to i64
  store i64 %r, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @mulhs64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %x = sext i64 %a to i128
  %y = sext i64 %b to i128
  %m = mul i128 %x, %y
  %h = lshr i128 %m, 64
  %r = trunc i128 %h to i64
  store i64 %r, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @mul64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %r = mul i64 %a, %b
  store i64 %r, ptr addrspace(1) %p
  ret void
}
