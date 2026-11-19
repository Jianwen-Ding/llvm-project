; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-silu-quotient=0 < %s | FileCheck %s --check-prefix=OFF
;
; silu's x / (e^-x + 1): d in [1, +inf], scaled with x by 2^-64 past 2^64,
; E282's reciprocal step and Markstein's correction (its residual negated,
; for x = -0), y as 0 at d = +inf and q0 where the residual is a NaN
; (x = +inf), instead of the division's scalings (L15-E483).

; CHECK-LABEL: @silu(
; CHECK: fcmp ogt float %d, f0x5F800000
; CHECK: call float @llvm.agx.rcp.f32
; CHECK-NOT: fdiv
; CHECK: fcmp uno
; CHECK: ret void
; OFF-LABEL: @silu(
; OFF: fdiv float %x, %d
define agx_kernel void @silu(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  %d = fadd float %e, 1.0
  %r = fdiv float %x, %d
  store float %r, ptr addrspace(1) %p
  ret void
}

; Another numerator: the general division.
; CHECK-LABEL: @other(
; CHECK: fdiv float %y, %d
define agx_kernel void @other(ptr addrspace(1) %p, float %x, float %y) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  %d = fadd float %e, 1.0
  %r = fdiv float %y, %d
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.exp.f32(float)
