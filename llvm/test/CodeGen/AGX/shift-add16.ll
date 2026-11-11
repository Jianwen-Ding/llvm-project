; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A 16-bit shift by up to 4, alone or into an addition / subtraction, is the
; 16-bit iadd / isub with its shift (as for 32 bits), not a bfi on the shared
; unit and an add. Checked with agx-tv and on the functional model.

; CHECK-LABEL: a:
; CHECK: iadd {{r[0-9]+l}}, {{u[0-9]+h}}, {{u[0-9]+l}}, lsl 3
define agx_kernel void @a(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %s = shl i16 %a, 3
  %r = add i16 %s, %b
  store i16 %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: b:
; CHECK: isub {{r[0-9]+l}}, {{u[0-9]+h}}, {{u[0-9]+l}}, lsl 4
define agx_kernel void @b(ptr addrspace(1) %p, i16 %a, i16 %b) {
  %s = shl i16 %a, 4
  %r = sub i16 %b, %s
  store i16 %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: c:
; CHECK: iadd {{r[0-9]+l}}, 0, {{u[0-9]+l}}, lsl 2
define agx_kernel void @c(ptr addrspace(1) %p, i16 %a) {
  %s = shl i16 %a, 2
  store i16 %s, ptr addrspace(1) %p
  ret void
}
