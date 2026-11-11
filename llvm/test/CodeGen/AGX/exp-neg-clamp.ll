; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; e^-x (sigmoid, silu) clamps x to the mirrored bounds and negates the
; clamp: the negation folds into the multiplication by log2 e and the fma's
; addend modifier, instead of an xor before the clamp's selects (bit-identical
; on the simulator's functional model).

; CHECK-LABEL: nexp:
; CHECK-NOT: xor
; CHECK: fcmpsel lt,
; CHECK: fcmpsel gt,
; CHECK-NOT: xor
; CHECK: stop
define agx_kernel void @nexp(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  store float %e, ptr addrspace(1) %p
  ret void
}
declare float @llvm.exp.f32(float)
