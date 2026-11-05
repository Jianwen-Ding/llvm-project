; A 64-bit unsigned multiplication with overflow takes the high word from 32-bit pieces
; instead of dividing: Metal's compiler crashes on the division for some operands
; (XPC_ERROR_CONNECTION_INTERRUPTED on an M2 Max, PTX rcp.approx.f64).
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define {{.*}}@k(
; CHECK-NOT: udiv
; CHECK: lshr i64 {{.*}}, 32
; CHECK-NOT: udiv
; CHECK: ret void

target triple = "air"
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i64, ptr addrspace(1) %o
  %t = trunc i64 %v to i32
  %x = or i32 %t, 1
  %z = zext i32 %x to i64
  %r = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %z, i64 %v)
  %b = extractvalue { i64, i1 } %r, 1
  %y = zext i1 %b to i64
  store i64 %y, ptr addrspace(1) %o
  ret void
}
