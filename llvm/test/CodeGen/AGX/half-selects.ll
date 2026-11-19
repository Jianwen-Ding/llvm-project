; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-half-selects=0 < %s | FileCheck %s --check-prefix=OFF
;
; Selects and compares of halves extended to float (and of float constants
; exact in half) are done on the halves, the extension folding into the float
; operations that read them (L15-E449).

target triple = "agx"

declare float @llvm.exp.f32(float)

; e^-x of an extended half: the clamps of the expansion's argument are
; selects of halves, the conversion folds into the multiplication and fma
define agx_kernel void @exp_neg(ptr addrspace(1) %p, ptr addrspace(1) %q) {
; CHECK-LABEL: exp_neg:
; CHECK-NOT: fadd32 {{r[0-9]+}}, r1l, -0.0
; CHECK: fcmpsel gt, r1h, r1l, r1h, r1h, r1l
; CHECK-NEXT: fcmpsel lt, r1l, r1l, r2l, r2l, r1h
; CHECK: fmul32 r2, r1l.neg, r2
; CHECK: fmadd32 r1, r2, r3, r1l.neg
; OFF-LABEL: exp_neg:
; OFF: fadd32 r1, r1l, -0.0
; OFF-NEXT: fcmpsel gt, r2, r1, r2, r2, r1
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  store float %e, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: clamp:
; CHECK: fcmpsel gt, r1h, r1l, 4.0, r1h, r1l
; CHECK-NEXT: fcmpsel lt, r1l, r1l, -4.0, r2l, r1h
; CHECK-NEXT: fmul32 r1, r1l, u4
; a clamp of an extended half to [-4, 4] read by a multiplication
define agx_kernel void @clamp(ptr addrspace(1) %p, ptr addrspace(1) %q, float %s) {
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %c1 = fcmp ogt float %x, 4.0
  %a = select i1 %c1, float 4.0, float %x
  %c2 = fcmp olt float %x, -4.0
  %b = select i1 %c2, float -4.0, float %a
  %m = fmul float %b, %s
  store float %m, ptr addrspace(1) %q
  ret void
; CHECK-LABEL: inexact:
; CHECK: fadd32 r2, r1l, -0.0
; CHECK-NEXT: fcmpsel gt, r1, r1l, 0.5, r3, r2
}

; 0.1 is not a half: the select stays a float one
define agx_kernel void @inexact(ptr addrspace(1) %p, ptr addrspace(1) %q, float %s) {
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %c1 = fcmp ogt float %x, 0.5
  %a = select i1 %c1, float 0x3FB99999A0000000, float %x
  %m = fmul float %a, %s
  store float %m, ptr addrspace(1) %q
  ret void
; CHECK-LABEL: stored:
; CHECK: fadd32 r2, r1l, -0.0
; CHECK-NEXT: fcmpsel gt, r1, r1l, 4.0, r3, r2
}

; the select's value is stored: it stays a float one
define agx_kernel void @stored(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %c1 = fcmp ogt float %x, 4.0
  %a = select i1 %c1, float 4.0, float %x
  store float %a, ptr addrspace(1) %q
  ret void
}
