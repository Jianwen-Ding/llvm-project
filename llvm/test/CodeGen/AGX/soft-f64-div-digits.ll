; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-soft-f64 -agx-soft-f64-div-digits=0 < %s 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; Double division's significand quotient in two 28-bit digits: one Newton
; step to a 32-bit reciprocal (Y0 * Eh >> 49), then each digit from a 32-bit
; product with it and an exact remainder, N 2^26 - q1 Dv and R1 2^28 - q2 Dv,
; with one correction (L15-E658). Without it, the reciprocal to 2^-60 from
; 64-bit products.

; CHECK-LABEL: @quot(
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: lshr i32 {{%.*}}, 17
; CHECK: shl i64 {{%.*}}, 26
; CHECK: shl i64 {{%.*}}, 28
; CHECK: icmp uge i64
; OFF-LABEL: @quot(
; OFF: call float @llvm.agx.rcp.f32
; OFF-NOT: shl i64 {{%.*}}, 28
; OFF: ret void
define agx_kernel void @quot(ptr addrspace(1) %p, double %x, double %y) {
  %r = fdiv double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}
