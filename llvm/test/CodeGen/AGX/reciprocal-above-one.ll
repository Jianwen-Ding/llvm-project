; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-reciprocal-above-one=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-reciprocal-sqrt=0 < %s | FileCheck %s --check-prefix=OFF

; 1 / (e^x + c), c >= 1: the divisor is at least 1 or a NaN, so the
; reciprocal step of the division without its scaling of the divisor's
; exponent (bfi) and its NaN select, and +0 past 2^126 (2122317824), where the
; quotient flushes (bit-identical to the division for every divisor from 1 up
; and NaNs on the simulator's functional model) (L15-E282).

; CHECK-LABEL: sigmoid:
; CHECK: fadd32 [[D:r[0-9]+]], {{r[0-9]+}}, 1.0
; CHECK-NOT: bfi
; CHECK: rcp {{r[0-9]+}}, [[D]]
; CHECK-NOT: bfi
; CHECK: mov_imm [[K:r[0-9]+]], 2122317824
; CHECK: fcmpsel gt, {{r[0-9]+}}, [[D]], [[K]], 0,
; CHECK-NOT: fcmpsel
; CHECK: stop
; OFF-LABEL: sigmoid:
; OFF: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
define agx_kernel void @sigmoid(ptr addrspace(1) %p, float %x) {
  %n = fneg float %x
  %e = call float @llvm.exp.f32(float %n)
  %d = fadd float %e, 1.0
  %r = fdiv float 1.0, %d
  store float %r, ptr addrspace(1) %p
  ret void
}

; Not below 1: e^x + 0.5.
; CHECK-LABEL: half:
; CHECK: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
define agx_kernel void @half(ptr addrspace(1) %p, float %x) {
  %e = call float @llvm.exp.f32(float %x)
  %d = fadd float %e, 0.5
  %r = fdiv float 1.0, %d
  store float %r, ptr addrspace(1) %p
  ret void
}

; 1 / sqrt(v): the square root is +-0, +inf, a NaN or a normal float whose
; reciprocal is normal; where the residual is a NaN (d +-0 or +inf, or a
; NaN), rcp's own result (L15-E283).
; CHECK-LABEL: rsqrt:
; CHECK: rcp [[Y0:r[0-9]+]], [[D:r[0-9]+]]
; CHECK-NEXT: fmadd32 [[E:r[0-9]+]], [[D]].neg, [[Y0]], 1.0
; CHECK-NEXT: fmadd32 [[Y:r[0-9]+]], [[E]], [[Y0]], [[Y0]]
; CHECK-NEXT: fcmpsel eq, {{r[0-9]+}}, [[E]], [[E]], [[Y]], [[Y0]]
; CHECK-NOT: fcmpsel
; CHECK: stop
; OFF-LABEL: rsqrt:
; OFF: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
define agx_kernel void @rsqrt(ptr addrspace(1) %p, float %x) {
  %s = call float @llvm.sqrt.f32(float %x)
  %r = fdiv float 1.0, %s
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.exp.f32(float)
declare float @llvm.sqrt.f32(float)
