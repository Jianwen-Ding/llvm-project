; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-exp-selected-non-negative=0 < %s | FileCheck %s --check-prefix=OFF

; v > t ? e^(v - t) : v: where the select takes the exponential its argument
; is at least 0, so no underflow select (+0 below -87.3), only the overflow
; one (L15-E732, cond_compute -8.5% tput).
; CHECK-LABEL: gt_runtime:
; CHECK: fcmpsel gte
; CHECK-NOT: fcmpsel lte
; CHECK: fcmpsel gt
; CHECK: stop
; OFF-LABEL: gt_runtime:
; OFF: fcmpsel lte
; OFF: stop
define agx_kernel void @gt_runtime(ptr addrspace(1) %x, ptr addrspace(1) %y, float %t) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %g = fcmp ogt float %v, %t
  %d = fsub float %v, %t
  %e = call float @llvm.exp.f32(float %d)
  %r = select i1 %g, float %e, float %v
  %q = getelementptr inbounds float, ptr addrspace(1) %y, i64 %z
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; v + k, instcombine's v - (-k), under v >= -k (on the select's false side
; of v < -k).
; CHECK-LABEL: ge_constant:
; CHECK-NOT: fcmpsel lte
; CHECK: stop
define agx_kernel void @ge_constant(ptr addrspace(1) %x, ptr addrspace(1) %y) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %g = fcmp ult float %v, 3.0
  %d = fadd float %v, -3.0
  %e = call float @llvm.exp.f32(float %d)
  %r = select i1 %g, float %v, float %e
  %q = getelementptr inbounds float, ptr addrspace(1) %y, i64 %z
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; Not where the select takes it below t.
; CHECK-LABEL: lt_runtime:
; CHECK: fcmpsel lte
; CHECK: stop
define agx_kernel void @lt_runtime(ptr addrspace(1) %x, ptr addrspace(1) %y, float %t) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %z = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %g = fcmp olt float %v, %t
  %d = fsub float %v, %t
  %e = call float @llvm.exp.f32(float %d)
  %r = select i1 %g, float %e, float %v
  %q = getelementptr inbounds float, ptr addrspace(1) %y, i64 %z
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)
