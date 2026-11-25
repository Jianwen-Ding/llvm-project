; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-math-fused-exact-scales=0 < %s | FileCheck %s --check-prefix=OFF
;
; c - 2 y as one fma where the doubling is exact (L15-E638): tanh's
; 1 - 2 / (e^2a + 1) from the reciprocal y in [1.5e-8, 0.5], asin's
; pi / 2 - 2 p, and asin's and acos's (1 - a) / 2 as 0.5 - 0.5 a (1 - a is
; exact where the square root's result is read, a in [0.5, 1]). Every
; float's result the same (the f32 sweep of E638).

; CHECK-LABEL: {{^}}t:
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}, -2.0, 1.0
; CHECK: stop
; OFF-LABEL: {{^}}t:
; OFF: fadd32 [[T:r[0-9]+]], [[Y:r[0-9]+]], [[Y]]
; OFF: fadd32 {{r[0-9]+}}, [[T]].neg, 1.0

; CHECK-LABEL: {{^}}s:
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}.abs, -0.5, 0.5
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}, -2.0, {{r[0-9]+}}
; CHECK: stop
; OFF-LABEL: {{^}}s:
; OFF: fadd32 [[D:r[0-9]+]], {{r[0-9]+}}.abs.neg, 1.0
; OFF-NEXT: fmul32 {{r[0-9]+}}, [[D]], 0.5

; CHECK-LABEL: {{^}}c:
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}.abs, -0.5, 0.5
; CHECK: stop
; OFF-LABEL: {{^}}c:
; OFF: fadd32 [[E:r[0-9]+]], {{r[0-9]+}}.abs.neg, 1.0
; OFF-NEXT: fmul32 {{r[0-9]+}}, [[E]], 0.5

define agx_kernel void @t(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.tanh.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @s(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.asin.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

define agx_kernel void @c(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.acos.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.tanh.f32(float)
declare float @llvm.asin.f32(float)
declare float @llvm.acos.f32(float)
