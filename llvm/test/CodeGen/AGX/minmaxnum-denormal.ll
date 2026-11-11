; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -O3 < %s | FileCheck %s --check-prefix=IEEE

; minimumnum / maximumnum of two f32 variables: where compares flush, two
; different denormals compare equal, so the zeros fix's or / and of the bits
; is canonicalized (flushed to the zero of its sign) instead of being a third
; denormal. Under ieee-fp32 compares keep denormals and the bits stay.

; CHECK-LABEL: mn:
; CHECK: or [[B:r[0-9]+]],
; CHECK: fmul32 [[B]], [[B]], 1.0
; CHECK: fcmpsel eq, {{r[0-9]+}}, {{[ru][0-9]+}}, {{[ru][0-9]+}}, [[B]],
; IEEE-LABEL: mn:
; IEEE-NOT: fmul32 {{r[0-9]+}}, {{r[0-9]+}}, 1.0
; IEEE: stop
define agx_kernel void @mn(ptr addrspace(1) %p, float %a, float %b) {
  %r = call float @llvm.minimumnum.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: mx:
; CHECK: and [[B:r[0-9]+]],
; CHECK: fmul32 [[B]], [[B]], 1.0
; CHECK: fcmpsel eq, {{r[0-9]+}}, {{[ru][0-9]+}}, {{[ru][0-9]+}}, [[B]],
define agx_kernel void @mx(ptr addrspace(1) %p, float %a, float %b) {
  %r = call float @llvm.maximumnum.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %p
  ret void
}

declare float @llvm.minimumnum.f32(float, float)
declare float @llvm.maximumnum.f32(float, float)
