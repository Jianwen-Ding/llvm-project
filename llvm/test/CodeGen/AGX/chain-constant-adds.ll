; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-chain-constant-adds=0 < %s | FileCheck %s --check-prefix=OFF

; Wide constants added to one base, k + c, k + 2c, k + 3c (philox's key
; schedule): each sum after the first is the previous plus c's register, and
; the other constants' mov_imms go (L15-E420).

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: keys:
; CHECK: mov_imm [[C:r[0-9]+]], -1640531527
; CHECK-NEXT: iadd [[K1:r[0-9]+]], u2, [[C]]
; CHECK-NEXT: iadd [[K2:r[0-9]+]], [[K1]], [[C]]
; CHECK-NOT: mov_imm
; CHECK: iadd {{r[0-9]+}}, [[K2]], [[C]]
; CHECK-NOT: mov_imm
; CHECK: stop
; OFF-LABEL: keys:
; OFF: mov_imm {{r[0-9]+}}, 1013904242
; OFF: mov_imm {{r[0-9]+}}, -626627285
define agx_kernel void @keys(ptr addrspace(1) %p, i32 %k) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %k1 = add i32 %k, -1640531527
  %k2 = add i32 %k, 1013904242
  %k3 = add i32 %k, -626627285
  %x = xor i32 %k1, %t
  %y = mul i32 %x, %k2
  %z = xor i32 %y, %k3
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  store i32 %z, ptr addrspace(1) %q, align 4
  ret void
}

; A difference that is an 8-bit immediate needs no register.
; CHECK-LABEL: near:
; CHECK: mov_imm [[D:r[0-9]+]], 100000
; CHECK-NEXT: iadd [[A:r[0-9]+]], r{{[0-9]+}}, [[D]]
; CHECK-NEXT: iadd {{r[0-9]+}}, [[A]], 16
; CHECK-NOT: mov_imm
define agx_kernel void @near(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = add i32 %t, 100000
  %b = add i32 %t, 100016
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %a
  store i32 %b, ptr addrspace(1) %q, align 4
  ret void
}
