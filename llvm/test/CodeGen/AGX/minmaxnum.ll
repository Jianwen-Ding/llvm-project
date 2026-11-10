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

; minimum / maximum: a ugt b ? a : b (a NaN a is taken by the unordered
; comparison), equal operands the and of their bits (canonicalized: two
; denormals compare equal, and the hardware flushes f32 ones), and b where b
; is a NaN (a NaN operand is a NaN result, no default NaN constant).
; CHECK-LABEL: maximum:
; CHECK-DAG:   and [[B:r[0-9]+]], u2, u3
; CHECK-DAG:   fcmpsel lte, [[M:r[0-9]+]], u2, u3, u3, u2
; CHECK:       fmul32 [[C:r[0-9]+]], [[B]], 1.0
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], u2, u3, [[C]], [[M]]
; CHECK:       fcmpsel eq, [[R2:r[0-9]+]], u3, u3, [[R]], u3
; CHECK-NEXT:  device_store 0, i32, x, [[R2]],
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
