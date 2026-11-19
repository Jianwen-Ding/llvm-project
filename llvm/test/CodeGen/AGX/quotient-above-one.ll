; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; tanh's (1 - e) / (1 + e), e = exp2 of a value that is not positive (-2|x| k):
; the divisor in [1, 2], the numerator 0 or at least 2^-24 in magnitude, so
; Markstein's quotient step from the correctly rounded 1 / d is the division,
; without its scaling and NaN select (L15-E409; agx_run: every e from 0 to
; 1.125 and 5 million infinities and NaNs, bit for bit). @maybe_positive's exponent may be positive, and
; @small_c's numerator constant is below 1: both keep the division.

declare float @llvm.agx.exp2.f32(float)
declare float @llvm.fabs.f32(float)

; CHECK-LABEL: @tanh_ratio(
; CHECK-NOT:   fdiv
; CHECK:       call float @llvm.agx.rcp.f32
; CHECK:       ret void
define void @tanh_ratio(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %a = call float @llvm.fabs.f32(float %x)
  %s = fmul float %a, -2.885390e+00
  %e = call float @llvm.agx.exp2.f32(float %s)
  %n = fsub float 1.0, %e
  %d = fadd float %e, 1.0
  %q = fdiv float %n, %d
  store float %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @maybe_positive(
; CHECK:       fdiv
define void @maybe_positive(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %e = call float @llvm.agx.exp2.f32(float %x)
  %n = fsub float 1.0, %e
  %d = fadd float %e, 1.0
  %q = fdiv float %n, %d
  store float %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @small_c(
; CHECK:       fdiv
define void @small_c(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %a = call float @llvm.fabs.f32(float %x)
  %s = fneg float %a
  %e = call float @llvm.agx.exp2.f32(float %s)
  %n = fsub float 0.5, %e
  %d = fadd float %e, 1.0
  %q = fdiv float %n, %d
  store float %q, ptr addrspace(1) %p
  ret void
}
