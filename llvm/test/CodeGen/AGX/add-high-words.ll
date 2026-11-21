; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-add-high-words=0 < %s | FileCheck %s --check-prefix=OFF

; A 64-bit value plus or minus one whose low word is zero: the high words
; added or subtracted with a 32-bit iadd / isub, the low word kept (no
; carry), instead of the 64-bit iadd on the unit 64-bit adds and multiply-
; adds share (L15-E660).
; CHECK-LABEL: add_hi:
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK: iadd r3, u3, u4
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK: device_store
; OFF-LABEL: add_hi:
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}
define agx_kernel void @add_hi(ptr addrspace(1) %p, i64 %x, i32 %y) {
  %w = zext i32 %y to i64
  %h = shl i64 %w, 32
  %r = add i64 %x, %h
  store i64 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: sub_hi:
; CHECK-NOT: isub r{{[0-9]+}}_r{{[0-9]+}}
; CHECK: isub r3, u3, r{{[0-9]+}}
; CHECK: device_store
; OFF-LABEL: sub_hi:
; OFF: isub r{{[0-9]+}}_r{{[0-9]+}}
define agx_kernel void @sub_hi(ptr addrspace(1) %p, i64 %x, i32 %y) {
  %w = zext i32 %y to i64
  %h = shl i64 %w, 40
  %r = sub i64 %x, %h
  store i64 %r, ptr addrspace(1) %p
  ret void
}

; A 32 x 32 -> 64-bit product plus a zero-extended 32-bit value: the
; multiply-add's addend (a pair with a zero high word), not an iadd of the
; product after it (L15-E661).
; CHECK-LABEL: mad_zext:
; CHECK: mov_imm r3, 0
; CHECK-NEXT: imadd r2_r3, u2, u3, r2_r3
; CHECK-NOT: iadd
; CHECK: device_store
define agx_kernel void @mad_zext(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %c) {
  %a64 = zext i32 %a to i64
  %b64 = zext i32 %b to i64
  %c64 = zext i32 %c to i64
  %m = mul i64 %a64, %b64
  %r = add i64 %m, %c64
  store i64 %r, ptr addrspace(1) %p
  ret void
}
