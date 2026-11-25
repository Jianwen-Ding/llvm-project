; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; minimumnum / maximumnum: the hardware's NaN-ignoring min / max, and for
; operands that compare equal the bits ORed / ANDed (-0 below +0).

; CHECK-LABEL: max:
; CHECK-DAG:   fcmpsel gtn, [[M:r[0-9]+]], u2, u3, u2, u3
; CHECK-DAG:   and [[Z:r[0-9]+]], u2, u3
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], u2, u3, [[Z]], [[M]]
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @max(ptr addrspace(1) %p, float %a, float %b) {
  %m = call float @llvm.maximumnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: min_half:
; CHECK-DAG:   fcmpsel ltn, [[M:r[0-9]+[lh]]],
; CHECK-DAG:   or [[Z:r[0-9]+[lh]]],
; CHECK:       fcmpsel eq, {{r[0-9]+[lh]}}, {{.*}}, [[Z]], [[M]]
define agx_kernel void @min_half(ptr addrspace(1) %p, half %a, half %b) {
  %m = call half @llvm.minimumnum.f16(half %a, half %b)
  store half %m, ptr addrspace(1) %p
  ret void
}

; Without signed zeros, the hardware min / max alone.
; CHECK-LABEL: max_nsz:
; CHECK:       fcmpsel gtn, [[R:r[0-9]+]], u2, u3, u2, u3
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @max_nsz(ptr addrspace(1) %p, float %a, float %b) {
  %m = call nsz float @llvm.maximumnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}

; minimum / maximum: ltn(b, a) ? a : b, the larger, a NaN operand the result
; (b where both are: the "n" condition makes a number lose to a NaN, L15-E452),
; and equal operands the and of their bits (canonicalized: two denormals
; compare equal, and the hardware flushes f32 ones).
; CHECK-LABEL: maximum:
; CHECK-DAG:   and [[B:r[0-9]+]], u2, u3
; CHECK-DAG:   fcmpsel ltn, [[M:r[0-9]+]], u3, u2, u2, u3
; CHECK:       fmul32 [[C:r[0-9]+]], [[B]], 1.0
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], u2, u3, [[C]], [[M]]
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
; CHECK-NOT:   mov_imm
define agx_kernel void @maximum(ptr addrspace(1) %p, float %a, float %b) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}

; relu: maximum(x, +0) is x ugt 0 ? x : +0, one fcmpsel with the zero as an
; immediate; maximum(x, -0) gives x's zero (x & sign) where x compares equal
; to it (a denormal x too).
; CHECK-LABEL: relu:
; CHECK:       fcmpsel lte, [[R:r[0-9]+]], u2, 0.0, 0, u2
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @relu(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.maximum.f32(float %a, float 0.0)
  store float %m, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: max_neg_zero:
; CHECK:       fcmpsel lte, [[M:r[0-9]+]], u2, -0.0, {{r[0-9]+}}, u2
; CHECK:       and [[Z:r[0-9]+]], u2,
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], u2, -0.0, [[Z]], [[M]]
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @max_neg_zero(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.maximum.f32(float %a, float -0.0)
  store float %m, ptr addrspace(1) %p
  ret void
}

; nsz minimum / maximum: one fcmpsel, ltn (gtn) on (b, a) selecting a or b: the
; larger (smaller), a NaN operand the result, b where both are (L15-E452).
; CHECK-LABEL: maximum_nsz:
; CHECK:       fcmpsel ltn, [[R:r[0-9]+]], u3, u2, u2, u3
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @maximum_nsz(ptr addrspace(1) %p, float %a, float %b) {
  %m = call nsz float @llvm.maximum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: minimum_nsz:
; CHECK:       fcmpsel gtn, [[R:r[0-9]+]], u3, u2, u2, u3
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @minimum_nsz(ptr addrspace(1) %p, float %a, float %b) {
  %m = call nsz float @llvm.minimum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: maximum_nsz_f16:
; CHECK:       fcmpsel ltn, [[R:r[0-9]+]]l, u2h, u2l, u2l, u2h
define agx_kernel void @maximum_nsz_f16(ptr addrspace(1) %p, half %a, half %b) {
  %m = call nsz half @llvm.maximum.f16(half %a, half %b)
  store half %m, ptr addrspace(1) %p
  ret void
}
declare float @llvm.minimum.f32(float, float)
declare half @llvm.maximum.f16(half, half)
