; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s --check-prefix=MULADD
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 -agx-soft-f64-polynomial-fma < %s 2>&1 | FileCheck %s --check-prefix=PFMA

; exp and log of a double ("no libcall available" before): fdlibm's e_exp.c
; and e_log.c in double operations, which become integer code (within 1 ulp
; on the functional model). Their polynomials are fdlibm's multiplies and
; adds (L15-E816): no fma's 128-bit normalization (its ctlz.i128) is left;
; with -agx-soft-f64-polynomial-fma, fmas.

; MULADD-LABEL: @dexp(
; MULADD-NOT: @llvm.ctlz.i128
; MULADD: ret void
; MULADD-LABEL: @dlog(
; MULADD-NOT: @llvm.ctlz.i128
; MULADD: ret void
; PFMA-LABEL: @dexp(
; PFMA: @llvm.ctlz.i128
; PFMA: ret void
; PFMA-LABEL: @dlog(
; PFMA: @llvm.ctlz.i128
; PFMA: ret void

; CHECK-LABEL: @dexp(
; CHECK-NOT: call double @llvm.exp
; y 2^k1 as k1 added to y's exponent field (L15-E326): an i64 add and a
; select of y where it is a NaN
; CHECK: [[KS:%.*]] = shl i64 {{.*}}, 52
; CHECK-NEXT: add i64 [[KS]], {{%.*}}
; CHECK-NOT: {{fmul|fdiv|fadd|fsub}} double
; CHECK: ret void
define agx_kernel void @dexp(ptr addrspace(1) %p, double %x) {
  %r = call double @llvm.exp.f64(double %x)
  store double %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @dlog(
; CHECK-NOT: call double @llvm.log
; CHECK-NOT: {{fmul|fdiv|fadd|fsub}} double
; CHECK: ret void
define agx_kernel void @dlog(ptr addrspace(1) %p, double %x) {
  %r = call double @llvm.log.f64(double %x)
  store double %r, ptr addrspace(1) %p
  ret void
}
