; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
;
; x ^ ((x >> 31) & c), the order-preserving key of a float's bits (c =
; 0x7fffffff), as (x < 0) ? x ^ c : x: a bitop and an icmpsel instead of an
; asr, an and and the xor (L15-E459).

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: key:
; CHECK:       mov_imm [[C:r[0-9]+]], 2147483647
; CHECK:       xor [[X:r[0-9]+]], [[V:r[0-9]+]], [[C]]
; CHECK-NEXT:  icmpsel slt, {{r[0-9]+}}, [[V]], 0, [[X]], [[V]]
; CHECK-NOT:   asr
define agx_kernel void @key(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  %x = load i32, ptr addrspace(1) %q
  %s = ashr i32 %x, 31
  %m = and i32 %s, 2147483647
  %r = xor i32 %x, %m
  store i32 %r, ptr addrspace(1) %q
  ret void
}
