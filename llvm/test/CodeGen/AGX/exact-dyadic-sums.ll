; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; Float sums of converted bytes times dyadic constants (a blur's weights 1/16,
; 1/8, 1/4) are exact: every partial sum is an integer times 2^-4 below 2^24,
; so the float operations round nothing. Converted back to an integer they
; are the integer sum, shifted: (a + 2 b + 4 c + 8) >> 4 (L15-E401), in 16
; bits where every scaled value fits (L15-E402; @wide's 16-bit inputs take
; i32).
; @weight03's 0.3 is not dyadic, @used's sum is stored as a float as well,
; @negative's sum of signed bytes may be negative at a negative exponent:
; each keeps its float arithmetic.

declare float @llvm.fma.f32(float, float, float)
declare i16 @llvm.fptosi.sat.i16.f32(float)

; CHECK-LABEL: @blur(
; CHECK-NOT:   fma
; CHECK-NOT:   uitofp
; CHECK:       lshr i16 {{.*}}, 4
; CHECK:       ret void
define void @blur(ptr addrspace(1) %p) {
  %a = load i8, ptr addrspace(1) %p
  %q1 = getelementptr i8, ptr addrspace(1) %p, i32 1
  %b = load i8, ptr addrspace(1) %q1
  %q2 = getelementptr i8, ptr addrspace(1) %p, i32 2
  %c = load i8, ptr addrspace(1) %q2
  %fa = uitofp i8 %a to float
  %fb = uitofp i8 %b to float
  %fc = uitofp i8 %c to float
  %s0 = call float @llvm.fma.f32(float %fa, float 6.250000e-02, float 0.0)
  %s1 = call float @llvm.fma.f32(float %fb, float 1.250000e-01, float %s0)
  %s2 = call float @llvm.fma.f32(float %fc, float 2.500000e-01, float %s1)
  %r = fadd float %s2, 5.000000e-01
  %i = call i16 @llvm.fptosi.sat.i16.f32(float %r)
  store i16 %i, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @weight03(
; CHECK:       fma
define void @weight03(ptr addrspace(1) %p) {
  %a = load i8, ptr addrspace(1) %p
  %fa = uitofp i8 %a to float
  %s0 = call float @llvm.fma.f32(float %fa, float 3.000000e-01, float 0.0)
  %r = fadd float %s0, 5.000000e-01
  %i = call i16 @llvm.fptosi.sat.i16.f32(float %r)
  store i16 %i, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @used(
; CHECK:       fma
define void @used(ptr addrspace(1) %p, ptr addrspace(1) %f) {
  %a = load i8, ptr addrspace(1) %p
  %fa = uitofp i8 %a to float
  %s0 = call float @llvm.fma.f32(float %fa, float 6.250000e-02, float 0.0)
  store float %s0, ptr addrspace(1) %f
  %r = fadd float %s0, 5.000000e-01
  %i = call i16 @llvm.fptosi.sat.i16.f32(float %r)
  store i16 %i, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @negative(
; CHECK:       sitofp
define void @negative(ptr addrspace(1) %p) {
  %a = load i8, ptr addrspace(1) %p
  %fa = sitofp i8 %a to float
  %s0 = fmul float %fa, 6.250000e-02
  %i = fptosi float %s0 to i32
  store i32 %i, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @wide(
; CHECK-NOT:   fma
; CHECK:       lshr i32 {{.*}}, 1
; CHECK:       ret void
define void @wide(ptr addrspace(1) %p) {
  %a = load i16, ptr addrspace(1) %p
  %q1 = getelementptr i16, ptr addrspace(1) %p, i32 1
  %b = load i16, ptr addrspace(1) %q1
  %fa = uitofp i16 %a to float
  %fb = uitofp i16 %b to float
  %s0 = call float @llvm.fma.f32(float %fa, float 5.000000e-01, float 0.0)
  %s1 = call float @llvm.fma.f32(float %fb, float 5.000000e-01, float %s0)
  %i = fptoui float %s1 to i32
  store i32 %i, ptr addrspace(1) %p
  ret void
}

; An exact tree whose value stays a float (a box filter's sum, then x 1/3):
; the sum in integers, converted once (L15-E407). @floatsub's fsub could make
; -0 where the integer sum's conversion gives +0: it keeps its float sum.
; CHECK-LABEL: @box(
; CHECK:       add i16
; CHECK:       uitofp i16
; CHECK-NOT:   fadd
; CHECK:       fmul float
; CHECK:       ret void
define void @box(ptr addrspace(1) %p, ptr addrspace(1) %f) {
  %a = load i8, ptr addrspace(1) %p
  %q1 = getelementptr i8, ptr addrspace(1) %p, i32 1
  %b = load i8, ptr addrspace(1) %q1
  %q2 = getelementptr i8, ptr addrspace(1) %p, i32 2
  %c = load i8, ptr addrspace(1) %q2
  %fa = uitofp i8 %a to float
  %fb = uitofp i8 %b to float
  %fc = uitofp i8 %c to float
  %s1 = fadd float %fa, %fb
  %s = fadd float %s1, %fc
  %m = fmul float %s, 0x3FD5555560000000
  store float %m, ptr addrspace(1) %f
  ret void
}

; CHECK-LABEL: @floatsub(
; CHECK:       fsub float
define void @floatsub(ptr addrspace(1) %p, ptr addrspace(1) %f) {
  %a = load i8, ptr addrspace(1) %p
  %q1 = getelementptr i8, ptr addrspace(1) %p, i32 1
  %b = load i8, ptr addrspace(1) %q1
  %fa = uitofp i8 %a to float
  %fb = uitofp i8 %b to float
  %s = fsub float %fa, %fb
  store float %s, ptr addrspace(1) %f
  ret void
}
