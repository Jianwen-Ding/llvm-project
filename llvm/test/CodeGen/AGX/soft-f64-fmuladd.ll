; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 -agx-soft-f64-fmuladd-fused < %s 2>&1 | FileCheck %s --check-prefix=FUSED
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; llvm.fmuladd of doubles, fused or not as the implementation chooses: a
; multiply and an add (L15-E833), which the emulation does for less than its
; fma; no fma's 128-bit normalization (its ctlz.i128) is left. With
; -agx-soft-f64-fmuladd-fused, the fma.

; CHECK-LABEL: @dmuladd(
; CHECK-NOT: @llvm.ctlz.i128
; CHECK: ret void
; FUSED-LABEL: @dmuladd(
; FUSED: @llvm.ctlz.i128
; FUSED: ret void
define agx_kernel void @dmuladd(ptr addrspace(1) %p, double %x, double %y, double %z) {
  %r = call double @llvm.fmuladd.f64(double %x, double %y, double %z)
  store double %r, ptr addrspace(1) %p
  ret void
}

; llvm.fma stays fused.
; CHECK-LABEL: @dfma(
; CHECK: @llvm.ctlz.i128
; CHECK: ret void
define agx_kernel void @dfma(ptr addrspace(1) %p, double %x, double %y, double %z) {
  %r = call double @llvm.fma.f64(double %x, double %y, double %z)
  store double %r, ptr addrspace(1) %p
  ret void
}
