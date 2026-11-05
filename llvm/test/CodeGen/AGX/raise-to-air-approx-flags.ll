; AGX's approximate operations written as AIR allow approximation and promise
; nothing else: with nnan, a strict floating point module read back from AIR
; (AGXStrictFP) would leave their NaNs as the hardware makes them.
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-raise-to-air -S %s | FileCheck %s

; CHECK-LABEL: define {{.*}}@log(
; CHECK-NOT: nnan
; CHECK-NOT: fast
; CHECK: call afn float @llvm.log2.f32(

declare float @llvm.agx.log2.f32(float)

define agx_kernel void @log(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %y = call float @llvm.agx.log2.f32(float %x)
  store float %y, ptr addrspace(1) %p
  ret void
}
