; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 < %s | FileCheck %s --check-prefix=IEEE

; minimumnum(maximumnum(x, +0), 1) is the hardware's .sat clamp (a NaN and -0
; become +0), folded into the operation that produced x. A float keeps its
; denormals under ieee-fp32, which .sat flushes: there only halves fold. With
; -0 as the lower bound, -0 stays -0: not a .sat.

; CHECK-LABEL: fs:
; CHECK: fmadd16.sat
; CHECK: fmadd16.sat
; CHECK-NOT: fcmpsel
; CHECK: stop
; IEEE-LABEL: fs:
; IEEE: fmadd16.sat
define agx_kernel void @fs(ptr addrspace(1) %p, <2 x half> %a, <2 x half> %b, <2 x half> %c) {
  %f = call <2 x half> @llvm.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  %m = call <2 x half> @llvm.maximumnum.v2f16(<2 x half> %f, <2 x half> zeroinitializer)
  %r = call <2 x half> @llvm.minimumnum.v2f16(<2 x half> %m, <2 x half> splat (half 1.0))
  store <2 x half> %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: s32:
; CHECK: fmul32.sat
; CHECK-NOT: fcmpsel
; CHECK: stop
; IEEE-LABEL: s32:
; IEEE-NOT: .sat
; IEEE: stop
define agx_kernel void @s32(ptr addrspace(1) %p, float %a, float %b) {
  %f = fmul float %a, %b
  %m = call float @llvm.maximumnum.f32(float %f, float 0.0)
  %r = call float @llvm.minimumnum.f32(float %m, float 1.0)
  store float %r, ptr addrspace(1) %p
  ret void
}
; CHECK-LABEL: neg:
; CHECK-NOT: .sat
; CHECK: stop
define agx_kernel void @neg(ptr addrspace(1) %p, float %a) {
  %m = call float @llvm.maximumnum.f32(float %a, float -0.0)
  %r = call float @llvm.minimumnum.f32(float %m, float 1.0)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>)
