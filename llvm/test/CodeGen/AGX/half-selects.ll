; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-half-selects=0 < %s | FileCheck %s --check-prefix=OFF
;
; Selects and compares of halves extended to float (and of float constants
; exact in half) are done on the halves, the extension folding into the float
; operations that read them (L15-E449).

target triple = "agx"

declare float @llvm.exp.f32(float)

; e^-x of an extended half: the compares of the expansion's argument (since
; L15-E474 its +inf and +0 selects', with no clamps) are of the half, the
; conversion folds into the multiplication and fma
define agx_kernel void @exp_neg(ptr addrspace(1) %p, ptr addrspace(1) %q) {
; CHECK-LABEL: exp_neg:
; CHECK-NOT: fadd32 {{r[0-9]+}}, r1l, -0.0
; CHECK: fmul32 r2, r1l.neg, r2
; CHECK: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, r1l.neg
; CHECK: fcmpsel lt, {{r[0-9]+}}, r1l, r1h,
; CHECK: fcmpsel gt, {{r[0-9]+}}, r1l, r1h,
; OFF-LABEL: exp_neg:
; OFF: fadd32 r1, r1l, -0.0
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  store float %e, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: clamp:
; CHECK: fcmpsel gt, r1h, r1l, 4.0, r1h, r1l
; CHECK-NEXT: fcmpsel lt, r2l, r1l, -4.0, r2l, r1h
; CHECK-NEXT: fmul32 r1, r2l, u4
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

; A compare with a constant inexact in half is the half's with the half next
; to it: x >= 0.1 is h > 0.0999755859375 (0x2e66), -x <= -0.1 likewise
; (L15-E473).
; CHECK-LABEL: inexact_cmp:
; CHECK-NOT: fadd32 {{r[0-9]+}}, r1l, -0.0
; CHECK: mov_imm [[K:r[0-9]+[lh]]], 11878
; CHECK: fcmpsel gt, {{r[0-9]+}}, r1l, [[K]],
; CHECK: fcmpsel gt, {{r[0-9]+}}, r1l, [[K]],
; OFF-LABEL: inexact_cmp:
; OFF: fadd32 {{r[0-9]+}}, r1l, -0.0
define agx_kernel void @inexact_cmp(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %h = load half, ptr addrspace(1) %p
  %x = fpext half %h to float
  %c = fcmp oge float %x, 0x3FB99999A0000000
  %n = fneg float %x
  %d = fcmp ole float %n, 0xBFB99999A0000000
  %s = select i1 %c, float 1.0, float 2.0
  %t = select i1 %d, float 3.0, float %s
  store float %t, ptr addrspace(1) %q
  ret void
}
