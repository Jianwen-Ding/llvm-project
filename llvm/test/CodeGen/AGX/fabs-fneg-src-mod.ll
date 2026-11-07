; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s

; |-x| is |x|: the negation inside the absolute value is not a modifier.
; (neg.ftz.f16's flush test compares |-x| with the smallest normal; folding it
; as x.abs.neg flushed every negated half to a signed zero.)

; CHECK-LABEL: flush_test:
; CHECK-NOT: .abs.neg
; CHECK: fcmpsel lt, {{.*}}.abs,
define agx_kernel void @flush_test(ptr addrspace(1) %p, half %x, half %zero) {
  %n = fneg half %x
  %a = call half @llvm.fabs.f16(half %n)
  %small = fcmp olt half %a, 0xH0400
  %r = select i1 %small, half %zero, half %n
  store half %r, ptr addrspace(1) %p
  ret void
}

; -|x| < c: LLVM compares |x| > -c instead (one modifier).
; CHECK-LABEL: neg_abs:
; CHECK: fcmpsel gt, {{.*}}.abs,
define agx_kernel void @neg_abs(ptr addrspace(1) %p, half %x, half %zero) {
  %a = call half @llvm.fabs.f16(half %x)
  %n = fneg half %a
  %small = fcmp olt half %n, 0xH0400
  %r = select i1 %small, half %zero, half %x
  store half %r, ptr addrspace(1) %p
  ret void
}

declare half @llvm.fabs.f16(half)
