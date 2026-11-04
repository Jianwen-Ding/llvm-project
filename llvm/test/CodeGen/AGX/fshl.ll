; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; fshl / rotl as one extr by 32 - (s mod 32): an amount of 32 gives the high
; word, which s = 0 needs.

; CHECK-LABEL: f:
; CHECK: and r1, u4, 31
; CHECK: extr r3, u3, u2, 25
; CHECK: isub r1, 32, r1
; CHECK: extr r2, u3, u2, r1
; CHECK: extr r1, r2, r2, r1
define agx_kernel void @f(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %c) {
  %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %q = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 7)
  %s = call i32 @llvm.fshl.i32(i32 %r, i32 %r, i32 %c)
  %t = xor i32 %r, %q
  %u = xor i32 %t, %s
  store i32 %u, ptr addrspace(1) %p
  ret void
}
