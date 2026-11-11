; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-soft-f64 < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s -o /dev/null

; modf of a double (SelectionDAG cannot soften it: "do not know how to
; soften fmodf"): trunc(x), and x - trunc(x) with the sign of x (zero for an
; infinite x), as integer code.

; CHECK-LABEL: @modf(
; CHECK-NOT: call {{.*}} @llvm.modf
; CHECK-NOT: {{fsub|fcmp}} double
; CHECK: insertvalue { double, double }
; CHECK: ret void
define agx_kernel void @modf(ptr addrspace(1) %p, double %x) {
  %m = call { double, double } @llvm.modf.f64(double %x)
  %f = extractvalue { double, double } %m, 0
  %i = extractvalue { double, double } %m, 1
  store double %f, ptr addrspace(1) %p
  %q = getelementptr double, ptr addrspace(1) %p, i32 1
  store double %i, ptr addrspace(1) %q
  ret void
}
