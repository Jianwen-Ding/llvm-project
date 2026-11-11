; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A byte to half is the byte convert with a 16-bit destination, not a mask
; and the 16-bit convert (as to float already). Checked with agx-tv.

; CHECK-LABEL: a:
; CHECK-NOT: and
; CHECK: convert u8_to_f, {{r[0-9]+l}}, {{u[0-9]+l}}, rte
define agx_kernel void @a(ptr addrspace(1) %p, i32 %a) {
  %b = trunc i32 %a to i8
  %r = uitofp i8 %b to half
  store half %r, ptr addrspace(1) %p
  ret void
}
