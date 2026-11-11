; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; frem of doubles is the exact fmod on the bits: special cases (a NaN, an
; infinite a or a zero b; |a| <= |b|) without the loop, else one bit of the
; quotient an iteration, then the remainder normalized at b's exponent.

; CHECK-LABEL: @rem(
; CHECK-NOT: frem
; CHECK: icmp uge i64 {{.*}}, 9218868437227405312
; CHECK: call i64 @llvm.ctlz.i64({{.*}}, i1 true)
; CHECK: br i1 {{.*}}, label %[[PRE:.*]], label %[[JOIN:.*]]
; CHECK: [[PRE]]:
; CHECK-NEXT: br label %[[LOOP:fmod.loop]]
; CHECK: [[LOOP]]:
; CHECK: [[D:%.*]] = sub i64 {{.*}}
; CHECK: icmp sge i64 [[D]], 0
; CHECK: shl i64 {{.*}}, 1
; CHECK: br i1 {{.*}}, label %fmod.tail, label %[[LOOP]]
; CHECK: fmod.tail:
; CHECK: call i64 @llvm.ctlz.i64({{.*}}, i1 false)
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
; CHECK: bitcast i64 {{.*}} to double
; ASM-LABEL: rem:
; ASM-NOT: error
define agx_kernel void @rem(ptr addrspace(1) %out, double %a, double %b) {
  %r = frem double %a, %b
  store double %r, ptr addrspace(1) %out
  ret void
}
