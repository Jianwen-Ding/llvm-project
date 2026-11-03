; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A 64-bit bit reversal is the 32-bit instruction on each half, the halves
; swapped, not the generic shift-and-mask expansion.

; CHECK-LABEL: brev64:
; CHECK-COUNT-2: bitrev
; CHECK-NOT:     bfeil
; CHECK:         stop
define agx_kernel void @brev64(ptr addrspace(1) %p, i64 %a) {
  %r = call i64 @llvm.bitreverse.i64(i64 %a)
  store i64 %r, ptr addrspace(1) %p
  ret void
}
