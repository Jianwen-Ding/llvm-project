; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+canonical-nan < %s | FileCheck %s --check-prefix=CANON

; Rounding a float to bfloat: a result of the float arithmetic, whose NaN is
; 0x7fc00000 (still a NaN once rounded), needs no NaN test; any other float
; does, and so does arithmetic whose NaNs are PTX's canonical 0x7fffffff.

; CHECK-LABEL: add:
; CHECK: fadd32
; CHECK-NOT: icmpsel
; CHECK: stop
; CANON-LABEL: add:
; CANON: icmpsel
define agx_kernel void @add(ptr addrspace(1) %p, float %a, float %b) {
  %s = fadd float %a, %b
  %r = fptrunc float %s to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: cvt:
; CHECK: icmpsel ugt
define agx_kernel void @cvt(ptr addrspace(1) %p, float %a) {
  %r = fptrunc float %a to bfloat
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
