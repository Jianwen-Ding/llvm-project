; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-tanh-plus-one=0 -stop-after=agx-codegenprepare < %s | FileCheck %s --check-prefix=OFF
;
; 1 + tanh(t) (GELU's tanh form) as 2 / (1 + e^-2t) and 1 - tanh(t) as
; 2 / (1 + e^2t): one exponential and the reciprocal of e^x + 1, and no
; cancellation where tanh(t) is near -1 (L15-E788). The tanh's other uses,
; or a fast-math add, keep it.

; CHECK-LABEL: @plus(
; CHECK-NOT: call float @llvm.tanh.f32
; CHECK: fmul float %t, -2.000000e+00
; CHECK: fadd float {{%[0-9.a-z]+}}, 1.000000e+00
; CHECK: fmul float {{%[0-9.a-z]+}}, 2.000000e+00
; OFF-LABEL: @plus(
; OFF: fadd float {{%[0-9.a-z]+}}, 1.000000e+00
; OFF-NOT: fmul float %t, -2.000000e+00
define agx_kernel void @plus(ptr addrspace(1) %p) {
  %t = load float, ptr addrspace(1) %p
  %h = call float @llvm.tanh.f32(float %t)
  %u = fadd float %h, 1.0
  store float %u, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @minus(
; CHECK: fmul float %t, 2.000000e+00
; CHECK: fmul float {{%[0-9.a-z]+}}, 2.000000e+00
define agx_kernel void @minus(ptr addrspace(1) %p) {
  %t = load float, ptr addrspace(1) %p
  %h = call float @llvm.tanh.f32(float %t)
  %u = fsub float 1.0, %h
  store float %u, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @other_use(
; CHECK-NOT: fmul float %t, -2.000000e+00
; CHECK: store float %u
define agx_kernel void @other_use(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = load float, ptr addrspace(1) %p
  %h = call float @llvm.tanh.f32(float %t)
  %u = fadd float %h, 1.0
  store float %u, ptr addrspace(1) %p
  store float %h, ptr addrspace(1) %q
  ret void
}

declare float @llvm.tanh.f32(float)
