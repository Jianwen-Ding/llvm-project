; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; A 128-bit shift by a variable amount on the 32-bit words: the words moved
; by 64 places where n >= 64 and by 32 where bit 5 is set (icmpsel), then
; each result word a funnel shift of two neighbours by n mod 32 (extr): 20
; instructions against the 64-bit halves' 26. agx-tv proves shl, lshr and
; ashr.

; CHECK-LABEL: shl128:
; CHECK-NOT: isub {{.*}}, 64,
; CHECK: icmpsel ugt, {{.*}}, 63,
; CHECK: extr
; CHECK: extr
; CHECK: extr
; CHECK-NOT: isub {{.*}}, 64,
; CHECK: stop
define agx_kernel void @shl128(ptr addrspace(1) %p, i128 %x, i32 %n) {
  %m = and i32 %n, 127
  %z = zext i32 %m to i128
  %r = shl i128 %x, %z
  store i128 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: ashr128:
; CHECK: asr {{.*}}, 31
; CHECK: icmpsel ugt, {{.*}}, 63,
; CHECK: extr
; CHECK: extr
; CHECK: extr
; CHECK: stop
define agx_kernel void @ashr128(ptr addrspace(1) %p, i128 %x, i128 %z) {
  %r = ashr i128 %x, %z
  store i128 %r, ptr addrspace(1) %p
  ret void
}
