; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; fptrunc to half of one float operation of halves is the half operation
; (float's 24 bits make the second rounding harmless), and a min / max, fneg,
; fabs or select of such commutes with the rounding: MLIR's mixed precision
; truncf(maximumf(addf(extf a, extf b), 0)) (L15-E440). @two_ops's float sum
; is not a half: its product stays in float.

; CHECK-LABEL: @relu(
; CHECK:       [[S:%.*]] = fadd half %a, %b
; CHECK:       call half @llvm.maximum.f16(half [[S]], half 0.000000e+00)
; CHECK-NOT:   float
; CHECK:       ret void
define agx_kernel void @relu(ptr addrspace(1) %o, half %a, half %b) {
  %x = fpext half %a to float
  %y = fpext half %b to float
  %s = fadd float %x, %y
  %m = call float @llvm.maximum.f32(float %s, float 0.0)
  %h = fptrunc float %m to half
  store half %h, ptr addrspace(1) %o, align 2
  ret void
}

; CHECK-LABEL: @two_ops(
; CHECK:       fadd float
; CHECK:       fmul float
; CHECK:       fptrunc float
define agx_kernel void @two_ops(ptr addrspace(1) %o, half %a, half %b, half %c) {
  %x = fpext half %a to float
  %y = fpext half %b to float
  %z = fpext half %c to float
  %s = fadd float %x, %y
  %p = fmul float %s, %z
  %h = fptrunc float %p to half
  store half %h, ptr addrspace(1) %o, align 2
  ret void
}

declare float @llvm.maximum.f32(float, float)
