; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; Double precision addition, multiplication, fused multiply-add and division
; rounded to nearest: normal operands with a normal result take a short path
; (no normalization of the significands, a rounding without the denormal,
; overflow and special cases), and the general computation is in a block
; that only lanes with other operands enter.

; CHECK-LABEL: @mul(
; The product's top bit is at bit 104 or 105: a select of one shift, no
; count of leading zeros before the branch.
; CHECK: mul i128
; CHECK-NOT: ctlz
; CHECK: br i1 %{{.*}}, label %[[SLOW:.*]], label %[[JOIN:.*]]
; CHECK: [[SLOW]]:
; CHECK: ctlz
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
define agx_kernel void @mul(ptr addrspace(1) %out, double %a, double %b) {
  %r = fmul double %a, %b
  store double %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @add(
; CHECK: call i64 @llvm.ctlz.i64
; CHECK: br i1 %{{.*}}, label %[[SLOW:.*]], label %[[JOIN:.*]]
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
define agx_kernel void @add(ptr addrspace(1) %out, double %a, double %b) {
  %r = fsub double %a, %b
  store double %r, ptr addrspace(1) %out
  ret void
}

; One 128-bit alignment (of the operand with the lower exponent) and one
; normalization.
; CHECK-LABEL: @fma(
; CHECK: mul i128
; CHECK: call i128 @llvm.ctlz.i128
; CHECK-NOT: call i128 @llvm.ctlz.i128
; CHECK: br i1 %{{.*}}, label %[[SLOW:.*]], label %[[JOIN:.*]]
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
define agx_kernel void @fma(ptr addrspace(1) %out, double %a, double %b, double %c) {
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  store double %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @div(
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: br i1 %{{.*}}, label %[[SLOW:.*]], label %[[JOIN:.*]]
; CHECK: [[SLOW]]:
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
define agx_kernel void @div(ptr addrspace(1) %out, double %a, double %b) {
  %r = fdiv double %a, %b
  store double %r, ptr addrspace(1) %out
  ret void
}

; Directed rounding keeps the general computation alone.
; CHECK-LABEL: @mul_rz(
; CHECK-NOT: br i1
; CHECK: ret void
define agx_kernel void @mul_rz(ptr addrspace(1) %out, double %a, double %b) #0 {
  %r = call double @llvm.experimental.constrained.fmul.f64(double %a, double %b, metadata !"round.towardzero", metadata !"fpexcept.ignore") #0
  store double %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { strictfp }
