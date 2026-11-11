; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; ldexp of half is computed in float, where scaling half values is exact
; (and what underflows float is far below half's denormals): the rounding to
; half is the only one. The generic expansion in half scaled by at most 2^-6
; before clamping n to half's exponent range, so ldexp(91.6, -37) came out
; 8.7e-5 (not 0) and ldexp(0x6ec0, -32) 0x1ec0 (not the denormal 0x1b).

; CHECK-LABEL: ldexp_half:
; CHECK-NOT: fmul16
; CHECK: fmul32
; CHECK-NOT: fmul16
; CHECK: fmul32 r{{[0-9]+}}l,
; CHECK-NOT: fmul16
; CHECK: stop
define agx_kernel void @ldexp_half(ptr addrspace(1) %p, half %x, i32 %n) {
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 %n)
  store half %r, ptr addrspace(1) %p
  ret void
}
