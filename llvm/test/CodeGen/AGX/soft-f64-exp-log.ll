; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; exp and log of a double ("no libcall available" before): fdlibm's e_exp.c
; and e_log.c in double operations, which become integer code (within 1 ulp
; on the functional model).

; CHECK-LABEL: @dexp(
; CHECK-NOT: call double @llvm.exp
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
