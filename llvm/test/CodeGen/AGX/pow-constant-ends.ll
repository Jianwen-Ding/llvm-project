; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-pow-constant-ends=0 < %s | FileCheck %s --check-prefix=OFF
;
; pow(x, y) of a finite nonzero constant y (|y| <= 2^20): y log|x| is
; finite but for |x| = 0 and inf, whose results y's sign gives (here 0 and
; inf), two selects on |x| (L15-E648), not log's three selects of its
; special values, their product with y and its test for an infinity.
; CHECK-LABEL: {{^}}pw:
; CHECK-NOT: fcmpsel gte, {{r[0-9]+}}, {{r[0-9]+}}.abs, 0.0
; CHECK: fcmpsel eq, {{r[0-9]+}}, [[X:r[0-9]+]].abs, 0.0, 0,
; CHECK-NOT: fcmpsel gte, {{r[0-9]+}}, {{r[0-9]+}}.abs, 0.0
; CHECK: stop
; OFF-LABEL: {{^}}pw:
; OFF: fcmpsel gte, {{r[0-9]+}}, {{r[0-9]+}}.abs, 0.0
define agx_kernel void @pw(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a, align 4
  %y = call float @llvm.pow.f32(float %x, float 0.75)
  store float %y, ptr addrspace(1) %a, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.pow.f32(float, float)
