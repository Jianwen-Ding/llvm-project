; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s

; An i64 smax of two values with the same sign-extended high word: type
; legalization makes the and / or of the high and low halves' compares, which
; E369 narrows to a 16-bit truth select; the high compare then folds to a
; constant, and the truth select goes back to an ordinary select (it failed
; to select; found by a differential fuzz, L15-E376).

; CHECK-LABEL: k:
; CHECK: device_store
; CHECK: stop
define agx_kernel void @k(ptr addrspace(1) %p) {
  %v1 = load i32, ptr addrspace(1) %p, align 4
  %v10 = sext i32 %v1 to i64
  %v11 = xor i64 %v10, 1
  %v15 = call i64 @llvm.smax.i64(i64 %v11, i64 %v10)
  %v26 = trunc i64 %v15 to i32
  store i32 %v26, ptr addrspace(1) %p, align 4
  ret void
}

declare i64 @llvm.smax.i64(i64, i64)
