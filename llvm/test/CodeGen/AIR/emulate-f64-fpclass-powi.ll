; air-emulate-f64: llvm.is.fpclass of a double as integer tests of its bits
; (a test of classes the same for both signs and adjacent in magnitude, one
; range test of |x|; any other test the classes ORed), and llvm.powi of a
; double as compiler-rt's __powidf2 over the soft-f64 multiply and divide.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s

target triple = "air"

declare i1 @llvm.is.fpclass.f64(double, i32)
declare double @llvm.powi.f64.i32(double, i32)

; finite: |x| < inf
; CHECK-LABEL: define {{.*}} @finite(
; CHECK: [[A:%.*]] = and i64 %x, 9223372036854775807
; CHECK-NEXT: [[R:%.*]] = icmp ult i64 [[A]], 9218868437227405312
; CHECK-NOT: icmp
; CHECK: ret
define i1 @finite(double %x) {
  %r = call i1 @llvm.is.fpclass.f64(double %x, i32 504)
  ret i1 %r
}

; NaN: |x| >= inf + 1
; CHECK-LABEL: define {{.*}} @nan(
; CHECK: icmp uge i64 {{.*}}, 9218868437227405313
; CHECK-NOT: icmp
; CHECK: ret
define i1 @nan(double %x) {
  %r = call i1 @llvm.is.fpclass.f64(double %x, i32 3)
  ret i1 %r
}

; infinite: |x| == inf
; CHECK-LABEL: define {{.*}} @inf(
; CHECK: icmp eq i64 {{.*}}, 9218868437227405312
; CHECK-NOT: icmp
; CHECK: ret
define i1 @inf(double %x) {
  %r = call i1 @llvm.is.fpclass.f64(double %x, i32 516)
  ret i1 %r
}

; subnormal or normal: |x| - 1 < inf - 1
; CHECK-LABEL: define {{.*}} @nonzero_finite(
; CHECK: [[S:%.*]] = sub i64 {{.*}}, 1
; CHECK-NEXT: icmp ult i64 [[S]], 9218868437227405311
; CHECK: ret
define i1 @nonzero_finite(double %x) {
  %r = call i1 @llvm.is.fpclass.f64(double %x, i32 408)
  ret i1 %r
}

; -inf, +normal or sNaN: the classes, each tested
; CHECK-LABEL: define {{.*}} @mixed(
; CHECK: icmp slt i64 %x, 0
; CHECK: icmp uge i64 {{.*}}, 9221120237041090560
; CHECK: ret
define i1 @mixed(double %x) {
  %r = call i1 @llvm.is.fpclass.f64(double %x, i32 261)
  ret i1 %r
}

; CHECK-LABEL: define {{.*}} @powi(
; CHECK: powi.loop{{.*}}:
; CHECK: br i1 {{.*}}, label %[[MUL:powi.mul.*]], label %[[STEP:powi.step.*]]
; CHECK: [[MUL]]:
; CHECK-NEXT: call i64 @__air_f64_mul(
; CHECK: [[STEP]]:
; CHECK: sdiv i32 {{.*}}, 2
; CHECK: powi.next{{.*}}:
; CHECK-NEXT: call i64 @__air_f64_mul(
; CHECK: call i64 @__air_f64_div(i64 4607182418800017408,
; CHECK: icmp slt i32 %n, 0
define double @powi(double %x, i32 %n) {
  %r = call double @llvm.powi.f64.i32(double %x, i32 %n)
  ret double %r
}
