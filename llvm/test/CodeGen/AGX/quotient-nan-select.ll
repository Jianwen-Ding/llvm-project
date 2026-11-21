; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-quotient-nan-select=0 < %s | FileCheck %s --check-prefix=OFF
;
; silu's quotient (L15-E483) and 1 / sqrt's reciprocal (L15-E283) select
; their first step where the result of the correction is a NaN, not its
; residual: the same values (a finite residual's correction is finite
; there), and the compare-select reads two registers, not three (L15-E699).

; CHECK-LABEL: @silu(
; CHECK:      fcmp oeq float %d, +inf
; CHECK:      [[Q0:%[0-9]+]] = fmul float
; CHECK:      [[R:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[Q0]], float %{{[0-9]+}})
; CHECK-NEXT: [[N:%[0-9]+]] = fneg float [[R]]
; CHECK-NEXT: [[Q:%[0-9]+]] = call float @llvm.fma.f32(float [[N]], float %{{[0-9]+}}, float [[Q0]])
; CHECK-NEXT: [[C:%[0-9]+]] = fcmp uno float [[Q]], [[Q]]
; CHECK-NEXT: %r = select i1 [[C]], float [[Q0]], float [[Q]]
; OFF-LABEL: @silu(
; OFF:      fcmp oeq float %d, +inf
; OFF:      [[Q0:%[0-9]+]] = fmul float
; OFF:      [[R:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[Q0]], float %{{[0-9]+}})
; OFF-NEXT: [[N:%[0-9]+]] = fneg float [[R]]
; OFF-NEXT: [[Q:%[0-9]+]] = call float @llvm.fma.f32(float [[N]], float %{{[0-9]+}}, float [[Q0]])
; OFF-NEXT: [[C:%[0-9]+]] = fcmp uno float [[R]], [[R]]
; OFF-NEXT: %r = select i1 [[C]], float [[Q0]], float [[Q]]
define agx_kernel void @silu(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  %d = fadd float %e, 1.0
  %r = fdiv float %x, %d
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @rsqrt(
; CHECK:      [[Y0:%[0-9]+]] = call float @llvm.agx.rcp.f32(float %s)
; CHECK:      [[E:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[Y0]], float 1.000000e+00)
; CHECK-NEXT: [[Y:%[0-9]+]] = call float @llvm.fma.f32(float [[E]], float [[Y0]], float [[Y0]])
; CHECK-NEXT: [[C:%[0-9]+]] = fcmp uno float [[Y]], [[Y]]
; CHECK-NEXT: %r = select i1 [[C]], float [[Y0]], float [[Y]]
; OFF-LABEL: @rsqrt(
; OFF:      [[E:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[Y0:%[0-9]+]], float 1.000000e+00)
; OFF-NEXT: [[Y:%[0-9]+]] = call float @llvm.fma.f32(float [[E]], float [[Y0]], float [[Y0]])
; OFF-NEXT: [[C:%[0-9]+]] = fcmp uno float [[E]], [[E]]
; OFF-NEXT: %r = select i1 [[C]], float [[Y0]], float [[Y]]
define agx_kernel void @rsqrt(ptr addrspace(1) %p, float %x) {
  %s = call float @llvm.sqrt.f32(float %x)
  %r = fdiv float 1.0, %s
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.exp.f32(float)
declare float @llvm.sqrt.f32(float)
