; air-emulate-f64: llvm.round of a double on the bits (half way cases away
; from zero: the fraction bits cleared and one unit added where they are at
; least half of it), not floor(|x|) and soft-f64 subtraction, comparison and
; addition. alive-tv proves it for every double.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s

target triple = "air"

declare double @llvm.round.f64(double)

; CHECK-LABEL: define {{.*}} @round(
; CHECK-NOT: call {{.*}}@__air_f64_
; CHECK: call i64 @llvm.umax.i64(i64 {{.*}}, i64 1023)
; CHECK: call i64 @llvm.umin.i64(i64 {{.*}}, i64 1074)
; CHECK: shl i64 1,
; CHECK: icmp uge i64 {{.*}}, 4602678819172646912
; CHECK-NOT: call {{.*}}@__air_f64_
; CHECK: ret i64
define double @round(double %x) {
  %r = call double @llvm.round.f64(double %x)
  ret double %r
}
