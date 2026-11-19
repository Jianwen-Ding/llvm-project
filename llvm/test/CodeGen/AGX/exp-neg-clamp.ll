; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; e^-x (sigmoid, silu) clamps x to the mirrored bounds and negates the
; clamp: the negation folds into the multiplication by log2 e and the fma's
; addend modifier, instead of an xor before the clamp's selects (bit-identical
; on the simulator's functional model).

; Since L15-E474 exp has no clamps: its +inf and +0 selects compare the
; argument, here -x, x itself against the negated bounds.
; CHECK-LABEL: nexp:
; CHECK-NOT: xor
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, u2.neg
; CHECK-NOT: fcmpsel lt,
; CHECK-NOT: fcmpsel gt,
; CHECK: fcmpsel lte, {{r[0-9]+}}, u2,
; CHECK: fcmpsel gte, {{r[0-9]+}}, u2, {{r[0-9]+}}, 0,
; CHECK-NOT: xor
; CHECK: stop
define agx_kernel void @nexp(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  store float %e, ptr addrspace(1) %p
  ret void
}
declare float @llvm.exp.f32(float)

; 2^-x likewise (its clamp of x itself).
; CHECK-LABEL: nexp2:
; CHECK-NOT: xor
; CHECK: stop
define agx_kernel void @nexp2(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp2.f32(float %n)
  store float %e, ptr addrspace(1) %p
  ret void
}
declare float @llvm.exp2.f32(float)
