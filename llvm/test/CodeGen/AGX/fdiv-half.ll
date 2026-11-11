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

; CHECK-LABEL: hdiv_arcp:
; CHECK: rcp [[Y:r[0-9]+l]], {{u[0-9]+h}}
; CHECK-NEXT: fmul16 {{r[0-9]+l}}, {{u[0-9]+l}}, [[Y]]
define agx_kernel void @hdiv_arcp(ptr addrspace(1) %p, half %a, half %b) {
  %r = fdiv arcp half %a, %b
  store half %r, ptr addrspace(1) %p
  ret void
}
