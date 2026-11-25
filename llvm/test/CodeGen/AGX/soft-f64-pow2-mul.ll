; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 -agx-soft-f64-pow2-mul=0 < %s 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; A double times a constant power of two: the short path adds the constant's
; exponent to x's exponent field (x normal and so is the product, or x zero),
; the general multiplication in the slow path (L15-E327).

; CHECK-LABEL: @half(
; CHECK: [[B:%.*]] = bitcast double %x to i64
; CHECK: add i64 [[B]], -4503599627370496
; CHECK: br i1
; OFF-LABEL: @half(
; OFF-NOT: add i64 {{%.*}}, -4503599627370496
; OFF: ret void
define agx_kernel void @half(ptr addrspace(1) %p, double %x) {
  %r = fmul double %x, 0.5
  store double %r, ptr addrspace(1) %p
  ret void
}

; a negative power of two on the left: its exponent added, the sign flipped
; CHECK-LABEL: @neg4(
; CHECK: [[B:%.*]] = bitcast double %x to i64
; CHECK: [[S:%.*]] = add i64 [[B]], 9007199254740992
; CHECK: xor i64 [[S]], -9223372036854775808
define agx_kernel void @neg4(ptr addrspace(1) %p, double %x) {
  %r = fmul double -4.0, %x
  store double %r, ptr addrspace(1) %p
  ret void
}

; a factor whose significand field is known zero (bits built as e << 52, as
; exp's 2^k): its exponent field less the bias added, the signs combined
; CHECK-LABEL: @pow2bits(
; CHECK: [[Y:%.*]] = bitcast double %y to i64
; CHECK: [[SY:%.*]] = and i64 [[Y]], -9223372036854775808
; CHECK: [[MY:%.*]] = xor i64 [[Y]], [[SY]]
; CHECK: [[D:%.*]] = sub i64 [[MY]], 4607182418800017408
; CHECK: add i64 {{%.*}}, [[D]]
define agx_kernel void @pow2bits(ptr addrspace(1) %p, double %x, i64 %e) {
  %yb = shl i64 %e, 52
  %y = bitcast i64 %yb to double
  %r = fmul double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}
