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
; CHECK: br i1 %{{.*}}, label %[[SLOW:[^,]*]], label %[[JOIN:[^,]*]], !prof
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
; CHECK: br i1 %{{.*}}, label %[[SLOW:[^,]*]], label %[[JOIN:[^,]*]], !prof
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
; CHECK: br i1 %{{.*}}, label %[[SLOW:[^,]*]], label %[[JOIN:[^,]*]], !prof
; CHECK: [[JOIN]]:
; CHECK-NEXT: phi i64
define agx_kernel void @fma(ptr addrspace(1) %out, double %a, double %b, double %c) {
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  store double %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @div(
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: br i1 %{{.*}}, label %[[SLOW:[^,]*]], label %[[JOIN:[^,]*]], !prof
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

; Conversions: a float (normal or zero) widens without a slow path condition
; on the exponent's range beyond its being normal; a double narrows to a
; normal float with one rounding; an integer below 2^53 converts exactly
; (a 32-bit one always: no branch).
; CHECK-LABEL: @conv(
; CHECK: br i1 %{{.*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof
; CHECK: br i1 %{{.*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof
; CHECK: call i64 @llvm.ctlz.i64
; CHECK-NOT: br i1
; CHECK: ret void
define agx_kernel void @conv(ptr addrspace(1) %out, float %f, double %d, i32 %i) {
  %e = fpext float %f to double
  %t = fptrunc double %d to float
  %c = sitofp i32 %i to double
  store volatile double %e, ptr addrspace(1) %out
  store volatile float %t, ptr addrspace(1) %out
  store volatile double %c, ptr addrspace(1) %out
  ret void
}

; Rounding to integral values by masks on the bits, the general code only
; for NaNs.
; CHECK-LABEL: @floor(
; CHECK-NOT: ctlz
; CHECK: br i1 %{{.*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof
; CHECK: phi i64
define agx_kernel void @floor(ptr addrspace(1) %out, double %d) {
  %r = call double @llvm.floor.f64(double %d)
  store double %r, ptr addrspace(1) %out
  ret void
}
