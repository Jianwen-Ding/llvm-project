; AIR has no minimum / maximum: the portable lowering is fmin / fmax with a
; zero check on the operands' bits and a NaN check, which agx-lower-air
; recovers as llvm.minimum / llvm.maximum. With a constant operand (MLIR's
; relu6, maximumf(x, 0) then minimumf(.., 6)) the constant's bits are folded
; into the or / and of the check; they are recognized too (L15-E400).
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.air.ll | FileCheck %s

; CHECK-LABEL: define agx_kernel void @relu6(
; CHECK: [[M:%.*]] = call <4 x float> @llvm.maximum.v4f32(<4 x float> %x, <4 x float> zeroinitializer)
; CHECK: call <4 x float> @llvm.minimum.v4f32(<4 x float> [[M]], <4 x float> splat (float 6.000000e+00))
; CHECK-NOT: air.fmax
; CHECK: ret void
define void @relu6(ptr addrspace(1) %out, ptr addrspace(1) %in) "air-kernel" {
  %x = load <4 x float>, ptr addrspace(1) %in, align 4
  %m = call <4 x float> @llvm.maximum.v4f32(<4 x float> %x, <4 x float> zeroinitializer)
  %r = call <4 x float> @llvm.minimum.v4f32(<4 x float> %m, <4 x float> splat (float 6.0))
  store <4 x float> %r, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: define agx_kernel void @relu(
; CHECK: call float @llvm.maximum.f32(float %x, float 0.000000e+00)
; CHECK: ret void
define void @relu(ptr addrspace(1) %out, ptr addrspace(1) %in) "air-kernel" {
  %x = load float, ptr addrspace(1) %in, align 4
  %m = call float @llvm.maximum.f32(float %x, float 0.0)
  store float %m, ptr addrspace(1) %out, align 4
  ret void
}

declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare float @llvm.maximum.f32(float, float)
