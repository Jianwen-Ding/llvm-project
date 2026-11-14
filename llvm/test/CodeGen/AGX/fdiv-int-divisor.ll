; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-fdiv-int-divisor=0 -agx-fdiv-bounded-operands=0 < %s | FileCheck %s --check-prefix=OFF

; A float division by an integer converted to float (a mean's sum / n): the
; divisor is +-0 or of magnitude in [1, 2^64], so its scale 2^(128 - eb) is
; applied first and the dividend's exponent after, with no test of the
; dividend's size (2^-124, 0x1800000) and its selects (L15-E343).

; CHECK-LABEL: mean:
; CHECK: rcp
; CHECK-NOT: 25165824
; CHECK: stop
; OFF-LABEL: mean:
; OFF: mov_imm {{r[0-9]+}}, 25165824
define void @mean(ptr addrspace(1) %p, float %s, i32 %n) {
  %f = sitofp i32 %n to float
  %r = fdiv float %s, %f
  store float %r, ptr addrspace(1) %p
  ret void
}

; More generally (L15-E344): a divisor bounded by how it is computed
; (1 + sqrt(v) is below 2^65) and a dividend that is +-0 or not below 2^-124
; (a half extended): no size test either.
; CHECK-LABEL: by_sqrt:
; CHECK-NOT: 25165824
; CHECK: stop
; OFF-LABEL: by_sqrt:
; OFF: 25165824
define void @by_sqrt(ptr addrspace(1) %p, float %a, float %v) {
  %s = call float @llvm.sqrt.f32(float %v)
  %d = fadd float %s, 1.0
  %r = fdiv float %a, %d
  store float %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: half_num:
; CHECK-NOT: 25165824
; CHECK: stop
define void @half_num(ptr addrspace(1) %p, half %h, float %b) {
  %a = fpext half %h to float
  %r = fdiv float %a, %b
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.sqrt.f32(float)

; No NaN select for an operand known not to be a NaN (the converted integer
; divisor): one for the dividend only (L15-E345).
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s --check-prefix=NAN
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-fdiv-known-not-nan=0 < %s | FileCheck %s --check-prefix=NANOFF
; NAN-LABEL: mean:
; NAN-COUNT-1: fcmpsel eq
; NAN-NOT: fcmpsel eq
; NAN: stop
; NANOFF-LABEL: mean:
; NANOFF-COUNT-2: fcmpsel eq

; (1 - e) / (1 + e) with e = 2^(-c |x|) (a tanh-form GELU): e is at most 1,
; so the divisor is below 2^2: no size test (L15-E346).
; CHECK-LABEL: tanh_form:
; CHECK-NOT: 25165824
; CHECK: stop
define void @tanh_form(ptr addrspace(1) %p, float %x) {
  %ax = call float @llvm.fabs.f32(float %x)
  %s = fmul float %ax, -2.885390e+00
  %e = call float @llvm.agx.exp2.f32(float %s)
  %num = fsub float 1.0, %e
  %den = fadd float %e, 1.0
  %r = fdiv float %num, %den
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.fabs.f32(float)
declare float @llvm.agx.exp2.f32(float)
