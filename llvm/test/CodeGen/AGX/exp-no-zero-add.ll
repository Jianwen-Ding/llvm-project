; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The precise e^x reduces x to r = x - n ln 2 and evaluates e^r; with no
; low part to add (exp, unlike exp2 and exp10) there is no r + 0.0, which
; would only turn r = -0 into +0, where e^r is 1 either way; and no +inf / +0
; selects beyond the clamps of x to [-104, 89], where the result overflows
; and rounds to +0 by itself (both checked for every float, flushing and
; not). Three selects stay: the two clamps and the NaN one.

; CHECK-LABEL: e:
; CHECK: fcmpsel lt,
; CHECK: fcmpsel gt,
; CHECK: rint
; CHECK-NOT: fadd32 {{r[0-9]+}}, {{r[0-9]+}}, 0.0
; CHECK-NOT: fcmpsel {{lt|gt}},
; CHECK: fcmpsel eq, [[X:r[0-9]+]], [[X]], [[X]],
; CHECK-NOT: fcmpsel
; CHECK: stop
define agx_kernel void @e(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp.f32(float)
