; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-tanh-doubled-clamp=0 < %s | FileCheck %s --check-prefix=OFF
;
; tanh's exponential of 2 min(|x|, 9) takes min(|x| + |x|, 18): the doubling
; reads x with the abs modifier, where the select of |x| needed |x| in a
; register (an and with 0x7fffffff) and the 9 in another (L15-E471).

; CHECK-LABEL: t:
; CHECK: device_load 0, i32, x, [[X:r[0-9]+]]
; CHECK-NOT: and
; CHECK: fadd32 [[D:r[0-9]+]], [[X]].abs, [[X]].abs
; CHECK-NEXT: fcmpsel gt, {{r[0-9]+}}, [[X]].abs, 9.0, {{r[0-9]+}}, [[D]]
; CHECK: stop
; OFF-LABEL: t:
; OFF: and
; OFF: fcmpsel gt
define agx_kernel void @t(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.tanh.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.tanh.f32(float)
