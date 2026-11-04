; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; Two 16-bit values packed into a 32-bit one are its halves: no extension,
; shift or or (results of f16x2 / bf16x2 / 16-bit SIMD operations).

; CHECK-LABEL: pk:
; CHECK-NEXT: // uniform
; CHECK: bfi r1l,
; CHECK-NEXT: asr r1h,
; CHECK-NEXT: device_store 0, i32, x, r1,
define agx_kernel void @pk(ptr addrspace(1) %p, i32 %x, i32 %z) {
  %a = trunc i32 %x to i16
  %s = lshr i32 %x, 16
  %b = trunc i32 %s to i16
  %zz = trunc i32 %z to i16
  %h = shl i16 %a, %zz
  %i = ashr i16 %b, %zz
  %v0 = insertelement <2 x i16> poison, i16 %h, i64 0
  %v = insertelement <2 x i16> %v0, i16 %i, i64 1
  %w = bitcast <2 x i16> %v to i32
  store i32 %w, ptr addrspace(1) %p
  ret void
}

; The halves of 32-bit values: x >> 16 under y & 0xffff0000 is one 16-bit
; move of x's high half into y's low half.
; CHECK-LABEL: a:
; CHECK: iadd r2, u3, 9
; CHECK-NEXT: bitop 0101, r2l, r1h, 0
; CHECK-NEXT: device_store 0, i32, x, r2,
define agx_kernel void @a(ptr addrspace(1) %p, i32 %x, i32 %y) {
  %x1 = add i32 %x, 7
  %y1 = add i32 %y, 9
  %l = lshr i32 %x1, 16
  %h = and i32 %y1, -65536
  %r = or i32 %l, %h
  store i32 %r, ptr addrspace(1) %p
  ret void
}
