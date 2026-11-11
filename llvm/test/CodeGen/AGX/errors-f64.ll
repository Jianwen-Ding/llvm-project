; RUN: not llc -mtriple=agx -mcpu=agx-g13g < %s 2>&1 | FileCheck %s

; Double precision arithmetic is computed in software (see soft-f64.ll and
; soft-f64-frem.ll), except for the transcendental functions.
; CHECK: error: {{.*}}in function doubles {{.*}}: double precision floating point arithmetic (AGX has no double precision hardware)
; CHECK-NOT: error

define agx_kernel void @doubles(ptr addrspace(1) %out, double %a) {
  %r = call double @llvm.exp2.f64(double %a)
  store double %r, ptr addrspace(1) %out
  ret void
}
