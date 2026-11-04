; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; Bytes packed into a word (cvt.pack.sat, byte permutes): a field inserted at
; a byte offset into a value known zero there is one bfi, and a 16-bit value
; under an assembled one is the register's high half.

; CHECK-LABEL: pk:
; CHECK: bfi r1l, u3l, u2l, 8, mask 0xFF
; CHECK-NEXT: bitop 0101, r1h, u4l, 0
; CHECK-NEXT: device_store
define agx_kernel void @pk(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %c) {
  %a1 = and i32 %a, 255
  %a2 = shl i32 %a1, 8
  %b1 = and i32 %b, 255
  %c1 = shl i32 %c, 16
  %t = or i32 %a2, %b1
  %r = or i32 %t, %c1
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; A funnel shift stays one extr.
; CHECK-LABEL: funnel:
; CHECK: extr r2, r2, r1, 10
; CHECK-NOT: bfi
; CHECK: stop
define agx_kernel void @funnel(ptr addrspace(1) %p, i64 %x, i64 %y) {
  %a = mul i64 %x, %y
  %r = lshr i64 %a, 10
  store i64 %r, ptr addrspace(1) %p
  ret void
}
