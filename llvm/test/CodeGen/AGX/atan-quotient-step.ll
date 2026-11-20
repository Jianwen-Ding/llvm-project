; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-atan-quotient-step=0 < %s | FileCheck %s --check-prefix=OFF
;
; atan's -1 / a (a > tan(3 pi / 8)) or (a - 1) / (a + 1): the divisor at
; least 1 (+inf, a NaN), the numerator 0, -1 or at least 2^-24 in magnitude,
; so E282's reciprocal step (rcp, two fmas, +0 past 2^126: 2122317824) and
; Markstein's correction q0 + y (n - d q0), without the division's exponent
; scaling (bfi ..., 128, 23) and special cases; every float's atan the same
; (L15-E501).

; CHECK-LABEL: atan_f32:
; CHECK-NOT: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
; CHECK: rcp
; CHECK: mov_imm {{r[0-9]+}}, 2122317824
; CHECK-NOT: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
; CHECK: stop
; OFF-LABEL: atan_f32:
; OFF: bfi {{r[0-9]+}}, {{r[0-9]+}}, 128, 23
define agx_kernel void @atan_f32(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.atan.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.atan.f32(float)
