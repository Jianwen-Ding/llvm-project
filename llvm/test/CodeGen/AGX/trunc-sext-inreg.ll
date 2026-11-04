; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The low half of a 64-bit sign extension in register from 16 bits (zext of
; a sext i16 -> i32, as sext i16 -> i64 masked to 32 bits): the extension of
; the low word, not a 64-bit extension and its truncation.

; CHECK-LABEL: f:
; CHECK: convert f_to_s32, [[Q:r[0-9]+]],
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK: iadd r{{[0-9]+}}, [[Q]]l.sx, 0
; CHECK-NOT: iadd
; CHECK: device_store
define agx_kernel void @f(ptr addrspace(1) %p, float %x) {
  %q = fptosi float %x to i32
  %t = trunc i32 %q to i16
  %s = sext i16 %t to i64
  %z = and i64 %s, 4294967295
  store i64 %z, ptr addrspace(1) %p
  ret void
}
