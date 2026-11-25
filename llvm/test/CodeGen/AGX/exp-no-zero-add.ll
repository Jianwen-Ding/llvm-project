; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The precise e^x reduces x to r = x - n ln 2 and evaluates e^r; with no
; low part to add (exp, unlike exp2 and exp10) there is no r + 0.0, which
; would only turn r = -0 into +0, where e^r is 1 either way; and no +inf / +0
; selects beyond the clamps of x to [-104, 89], where the result overflows
; and rounds to +0 by itself (both checked for every float, flushing and
; not) (until L15-E474, below). n = round(x / ln 2) is t + 1.5 2^23 - 1.5 2^23, the constant read
; negated the second time (L15-E469). The two clamps stay; the scale 2^n, an
; exponent addition where f32 flushes, ends in its +inf select, a compare of
; the argument with 88.7228394 that a NaN fails, and no NaN select (a NaN's
; sum adds 0 to the exponent: L15-E473).

; Since L15-E474 the +0 select is a compare of the argument too (with
; -87.3365479), and the clamps go: beyond them the two selects give the
; +inf and +0 the clamped arguments gave.

; CHECK-LABEL: e:
; CHECK-NOT: fcmpsel lt,
; CHECK-NOT: fcmpsel gt,
; CHECK: fadd32 [[M:r[0-9]+]], {{r[0-9]+}}, [[C:r[0-9]+]]
; CHECK-NEXT: fadd32 {{r[0-9]+}}, [[M]], [[C]].neg
; CHECK-NOT: fadd32 {{r[0-9]+}}, {{r[0-9]+}}, 0.0
; CHECK-NOT: fcmpsel eq
; CHECK-NOT: icmpsel
; CHECK: fcmpsel gte
; CHECK: fcmpsel lte
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
