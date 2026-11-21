; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-exp-core-product=0 -agx-cos-horner=0 < %s | FileCheck %s --check-prefix=OFF
;
; expf's r + r^2 p as fma(p r, r, r) (L15-E704) and cosf's 1 - z / 2 + z^2 p
; as fma(z, fma(z, p, -1/2), 1) (L15-E705): fmas of two registers and a
; constant, where fma(p, r r, r) and fma(z z, p, 1 - z / 2) read three.

; CHECK-LABEL: @e(
; CHECK:      [[P:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[R:%[0-9]+]], float 5.000000e-01)
; CHECK-NEXT: [[PR:%[0-9]+]] = fmul float [[P]], [[R]]
; CHECK-NEXT: [[E:%[0-9]+]] = call float @llvm.fma.f32(float [[PR]], float [[R]], float [[R]])
; CHECK-NEXT: fadd float [[E]], 1.000000e+00
; OFF-LABEL: @e(
; OFF:      [[P:%[0-9]+]] = call float @llvm.fma.f32(float %{{[0-9]+}}, float [[R:%[0-9]+]], float 5.000000e-01)
; OFF-NEXT: [[R2:%[0-9]+]] = fmul float [[R]], [[R]]
; OFF-NEXT: [[E:%[0-9]+]] = call float @llvm.fma.f32(float [[P]], float [[R2]], float [[R]])
define agx_kernel void @e(ptr addrspace(1) %p, float %x) {
  %y = call float @llvm.exp.f32(float %x)
  store float %y, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @c(
; CHECK:      [[Q:%[0-9]+]] = call float @llvm.fma.f32(float [[Z:%[0-9]+]], float %{{[0-9]+}}, float -5.000000e-01)
; CHECK-NEXT: call float @llvm.fma.f32(float [[Z]], float [[Q]], float 1.000000e+00)
; OFF-LABEL: @c(
; OFF:      call float @llvm.fma.f32(float -5.000000e-01, float [[Z:%[0-9]+]], float 1.000000e+00)
define agx_kernel void @c(ptr addrspace(1) %p, float %x) {
  %y = call float @llvm.cos.f32(float %x)
  store float %y, ptr addrspace(1) %p
  ret void
}

declare float @llvm.exp.f32(float)
declare float @llvm.cos.f32(float)
