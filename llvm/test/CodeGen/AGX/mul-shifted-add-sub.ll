; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-decompose-mul-more=0 < %s | FileCheck %s --check-prefix=OFF

; A 16-bit multiplication by 2^k + 1 (k <= 4) as one shifted iadd, and a
; multiplication by 1 - 2^k (k = 3, 4) as one shifted isub, on the ALU
; instead of a mov_imm and imadd (L15-E356).

; CHECK-LABEL: m16_9:
; CHECK: iadd {{r[0-9]+l}}, [[X:[ur][0-9]+l]], [[X]], lsl 3
; OFF-LABEL: m16_9:
; OFF: imadd
define void @m16_9(ptr addrspace(1) %p, i16 %x) {
  %r = mul i16 %x, 9
  store i16 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: m_n15:
; CHECK: isub {{r[0-9]+}}, [[X:[ur][0-9]+]], [[X]], lsl 4
; CHECK-NOT: imadd
; OFF-LABEL: m_n15:
; OFF: imadd
define void @m_n15(ptr addrspace(1) %p, i32 %x) {
  %r = mul i32 %x, -15
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; A bit test by a register amount, (x >> n) & 1: one bfeil with the field's
; width (L15-E357; the register-offset masked form is recorded on hardware).
; CHECK-LABEL: bittest:
; CHECK: bfeil {{r[0-9]+}}, 0, {{[ur][0-9]+}}, {{[ur][0-9]+}}, mask 0x1
; CHECK-NOT: and
; CHECK: stop
define void @bittest(ptr addrspace(1) %p, i32 %x, i32 %n) {
  %s = lshr i32 %x, %n
  %r = and i32 %s, 1
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; (x & 255) << n: one bfi of the field into 0 (L15-E358).
; CHECK-LABEL: andshl:
; CHECK: bfi {{r[0-9]+}}, 0, {{[ur][0-9]+}}, {{[ur][0-9]+}}, mask 0xFF
; CHECK-NOT: and
; CHECK: stop
define void @andshl(ptr addrspace(1) %p, i32 %x, i32 %n) {
  %a = and i32 %x, 255
  %r = shl i32 %a, %n
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; trunc(x >> n) & 255 of a 64-bit x: one extr with the field's width
; (L15-E359).
; CHECK-LABEL: field64:
; CHECK: extr {{r[0-9]+}}, {{[ur][0-9]+}}, {{[ur][0-9]+}}, {{[ur][0-9]+}}, mask 0xFF
; CHECK-NOT: and
; CHECK: stop
define void @field64(ptr addrspace(1) %p, i64 %x, i32 %n) {
  %nz = zext i32 %n to i64
  %s = lshr i64 %x, %nz
  %t = trunc i64 %s to i32
  %r = and i32 %t, 255
  store i32 %r, ptr addrspace(1) %p
  ret void
}
