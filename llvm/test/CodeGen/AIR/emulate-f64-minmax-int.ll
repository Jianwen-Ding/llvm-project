; air-emulate-f64: double min / max on the bits (x ^ ((x >> 63) & 0x7fff...)
; as a signed integer orders doubles, -0 below +0) and the conversions between
; double and integers of at most 32 bits in 32-bit integer operations, with no
; soft-f64 library calls.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s

target triple = "air"

; CHECK-LABEL: @min(
; CHECK-NOT: call i32 @__air_f64_
; CHECK: ashr i64 %x, 63
; CHECK: icmp slt i64
; CHECK: icmp ugt i64 {{.*}}, 9218868437227405312
; CHECK: ret i64
define double @min(double %x, double %y) {
  %r = call double @llvm.minimumnum.f64(double %x, double %y)
  ret double %r
}

; a NaN operand returned, quiet
; CHECK-LABEL: @maximum(
; CHECK-NOT: call i32 @__air_f64_
; CHECK: icmp sgt i64
; CHECK: or i64 %y, 2251799813685248
; CHECK: or i64 %x, 2251799813685248
define double @maximum(double %x, double %y) {
  %r = call double @llvm.maximum.f64(double %x, double %y)
  ret double %r
}

; CHECK-LABEL: @to_u16(
; CHECK-NOT: __air_f64_to
; CHECK: lshr i32
; CHECK: call i32 @llvm.umin.i32({{.*}}, i32 65535)
; CHECK: trunc i32 {{.*}} to i16
define i16 @to_u16(double %x) {
  %r = call i16 @llvm.fptoui.sat.i16.f64(double %x)
  ret i16 %r
}

; CHECK-LABEL: @from_u32(
; CHECK-NOT: __air_f64_from
; CHECK: call i32 @llvm.ctlz.i32
; CHECK: shl i64 {{.*}}, 32
define double @from_u32(i32 %x) {
  %r = uitofp i32 %x to double
  ret double %r
}

declare double @llvm.minimumnum.f64(double, double)
declare double @llvm.maximum.f64(double, double)
declare i16 @llvm.fptoui.sat.i16.f64(double)
