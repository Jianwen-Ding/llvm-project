; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; half division without arcp / afn is correctly rounded: in float, the
; rounded reciprocal and one correction, then rounded to half (the half rcp
; times a was an ulp off for a fifth of random pairs). With arcp it stays
; the rcp and the product.

; CHECK-LABEL: hdiv:
; CHECK: rcp [[Y0:r[0-9]+]], [[B:r[0-9]+]]
; CHECK: fmadd32 {{r[0-9]+}}, [[B]].neg, [[Y0]], 1.0
; CHECK: fcmpsel eq
; CHECK: fcmpsel eq
; CHECK: fadd32 {{r[0-9]+l}}, {{r[0-9]+}}, -0.0
define agx_kernel void @hdiv(ptr addrspace(1) %p, half %a, half %b) {
  %r = fdiv half %a, %b
  store half %r, ptr addrspace(1) %p
  ret void
}

; A constant numerator +-2^k: a y, exact, is the correctly rounded float
; quotient; no correction step and no zero test (all 65,536 halves the same
; for 1, -1, 2, 0.5, +-2^-14 and 2^15 on the simulator) (L15-E284).
; CHECK-LABEL: hrcp:
; CHECK: rcp [[Y0:r[0-9]+]], [[B:r[0-9]+]]
; CHECK-NEXT: fmadd32 [[E:r[0-9]+]], [[B]].neg, [[Y0]], 1.0
; CHECK-NEXT: fmadd32 [[Y:r[0-9]+]], [[E]], [[Y0]], [[Y0]]
; CHECK-NEXT: fcmpsel eq, {{r[0-9]+}}, [[Y]], [[Y]], [[Y]], [[Y0]]
; CHECK-NEXT: fadd32 {{r[0-9]+l}}, {{r[0-9]+}}, -0.0
define agx_kernel void @hrcp(ptr addrspace(1) %p, half %b) {
  %r = fdiv half 1.0, %b
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: hdiv_arcp:
; CHECK: rcp [[Y:r[0-9]+l]], {{u[0-9]+h}}
; CHECK-NEXT: fmul16 {{r[0-9]+l}}, {{u[0-9]+l}}, [[Y]]
define agx_kernel void @hdiv_arcp(ptr addrspace(1) %p, half %a, half %b) {
  %r = fdiv arcp half %a, %b
  store half %r, ptr addrspace(1) %p
  ret void
}

; half sqrt without afn likewise: the float steps (x y, the remainder's
; correction, y = rsqrt_special(x)) rounded to half; x rsqrt_special(x) in
; half was an ulp off for 8,520 of the 65,536 halves.
; CHECK-LABEL: hsqrt:
; CHECK: rsqrt_special [[Y:r[0-9]+]], [[X:r[0-9]+]]
; CHECK: fmadd32
; CHECK: fadd32 {{r[0-9]+l}}, {{r[0-9]+}}, -0.0
define agx_kernel void @hsqrt(ptr addrspace(1) %p, half %a) {
  %r = call half @llvm.sqrt.f16(half %a)
  store half %r, ptr addrspace(1) %p
  ret void
}
declare half @llvm.sqrt.f16(half)
