; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 8- and 16-bit division and remainder: the float reciprocal of the divisor,
; the truncated product and one correction either way, no bit loop.

; CHECK-LABEL: udiv16:
; CHECK: rcp
; CHECK-NOT: while_icmp
; CHECK: stop
define agx_kernel void @udiv16(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %q = udiv i16 %a, %b
  store i16 %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: srem8:
; CHECK: rcp
; CHECK-NOT: while_icmp
; CHECK: stop
define agx_kernel void @srem8(ptr addrspace(1) %p, i8 %a, i8 %b) {
  %r = srem i8 %a, %b
  store i8 %r, ptr addrspace(1) %p
  ret void
}
