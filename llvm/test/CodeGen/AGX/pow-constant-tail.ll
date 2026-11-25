; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-pow-constant-tail=0 < %s | FileCheck %s --check-prefix=OFF
;
; pow by a finite nonzero constant: x = 1 needs no select (log(1) is 0 and
; e^(+-0) 1), and a non-integer y's NaN for a negative finite x and a NaN
; x's NaN are one select of ult(x, 0) and x != -inf (L15-E718).

; CHECK-LABEL: @frac(
; CHECK-NOT:  fcmp oeq float %x, 1.000000e+00
; CHECK:      [[L:%[0-9]+]] = fcmp ult float %x, 0.000000e+00
; CHECK-NEXT: [[M:%[0-9]+]] = fcmp une float %x, -inf
; CHECK-NEXT: [[A:%[0-9]+]] = and i1 [[L]], [[M]]
; CHECK-NEXT: select i1 [[A]], float +qnan
; CHECK-NOT:  fcmp uno float %x
; OFF-LABEL: @frac(
; OFF:       fcmp uno float %x
; OFF:       fcmp oeq float %x, 1.000000e+00
define agx_kernel void @frac(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.pow.f32(float %x, float 0.75)
  store float %r, ptr addrspace(1) %p
  ret void
}

; An odd integer: the sign of a negative x, and the NaN select of a NaN x.
; CHECK-LABEL: @odd(
; CHECK-NOT:  fcmp oeq float %x, 1.000000e+00
; CHECK:      fneg float
; CHECK:      fcmp uno float %x, %x
; CHECK-NOT:  fcmp oeq float %x, 1.000000e+00
define agx_kernel void @odd(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.pow.f32(float %x, float 3.0)
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.pow.f32(float, float)
