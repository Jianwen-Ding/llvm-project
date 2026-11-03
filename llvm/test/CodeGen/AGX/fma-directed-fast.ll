; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -o /dev/null -print-after=agx-strict-fp < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -verify-machineinstrs < %s -o /dev/null

; A float fma rounded upward: the hardware fma rounded to nearest, corrected
; by the sign of its exact error (Boldo and Muller's ErrFma: TwoProd, two
; TwoSums), with the integer computation in a block for the operands where
; that is not exact.

; CHECK-LABEL: @fma_rp(
; CHECK: call float @llvm.fma.f32(
; CHECK: fmul float
; CHECK: call float @llvm.fma.f32(
; CHECK: br i1 %{{.*}}, label %[[SLOW:[^,]*]], label %[[JOIN:[^,]*]], !prof
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi float
define agx_kernel void @fma_rp(ptr addrspace(1) %out, float %a, float %b, float %c) #0 {
  %r = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.upward", metadata !"fpexcept.ignore") #0
  store float %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { strictfp "denormal-fp-math-f32"="ieee,ieee" }
