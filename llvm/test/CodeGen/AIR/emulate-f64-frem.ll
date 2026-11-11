; air-emulate-f64: frem of doubles, the exact fmod on the bits (one bit of
; the quotient an iteration, then normalized at the divisor's exponent), with
; no soft-f64 library calls.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s

target triple = "air"

; CHECK-LABEL: define void @rem(
; CHECK-NOT: call {{.*}}@__air_f64_
; CHECK: icmp uge i64 {{.*}}, 9218868437227405312
; CHECK: call i64 @llvm.ctlz.i64({{.*}}, i1 true)
; CHECK: br i1 {{.*}}, label %[[DONE:.*]], label %[[LOOP:fmod.loop.*]]
; CHECK: [[LOOP]]:
; CHECK: [[D:%.*]] = sub i64 {{.*}}
; CHECK: icmp sge i64 [[D]], 0
; CHECK: shl i64 {{.*}}, 1
; CHECK: br i1 {{.*}}, label %[[TAIL:fmod.tail.*]], label %[[LOOP]]
; CHECK: [[TAIL]]:
; CHECK: call i64 @llvm.ctlz.i64({{.*}}, i1 false)
; CHECK: [[DONE]]:
; CHECK: phi i64 [
; CHECK-NOT: call {{.*}}@__air_f64_
; CHECK: ret void
define void @rem(ptr addrspace(1) %p, double %x, double %y) "air-kernel" {
  %r = frem double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}
