; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; maximumnum / minimumnum with a nonzero constant need no fix for zeros of
; two signs (equal operands then have the same bits): a saturating
; conversion to s8 clamps with one compare-select per bound.

; CHECK-LABEL: s8:
; CHECK: fcmpsel gtn
; CHECK-NEXT: fcmpsel ltn
; CHECK-NEXT: convert f_to_s16
; CHECK: stop
define agx_kernel void @s8(ptr addrspace(1) %p, float %a) {
  %r = call float @llvm.roundeven.f32(float %a)
  %c = call i8 @llvm.fptosi.sat.i8.f32(float %r)
  %f = sitofp i8 %c to float
  store float %f, ptr addrspace(1) %p
  ret void
}

; A zero bound keeps the fix: maximumnum(-0, +0) is +0.
; CHECK-LABEL: z:
; CHECK: fcmpsel gtn
; CHECK: fcmpsel eq
; CHECK: stop
define agx_kernel void @z(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.maximumnum.f32(float %a, float 0.0)
  store float %m, ptr addrspace(1) %p
  ret void
}
