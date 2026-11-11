; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs \
; RUN:   -agx-copysign-bfi=0 < %s | FileCheck %s --check-prefix=OFF

; copysign of a variable magnitude is one bfi (the magnitude's low bits into
; the sign's value); a constant magnitude keeps the masks on the ALUs.

; CHECK-LABEL: cs32:
; CHECK:       bfi [[R:r[0-9]+]], u3, u2, 0, mask 0x7FFFFFFF
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
; OFF-LABEL:   cs32:
; OFF:         or
; CHECK-LABEL: cs16:
; CHECK:       bfi [[R:r[0-9]+l]], u2h, u2l, 0, mask 0x7FFF
; CHECK-NEXT:  device_store 0, i16, x, [[R]],
; CHECK-LABEL: one:
; CHECK-NOT:   bfi
; CHECK:       or
define agx_kernel void @cs32(ptr addrspace(1) %p, float %a, float %b) {
  %r = call float @llvm.copysign.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @cs16(ptr addrspace(1) %p, half %a, half %b) {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  store half %r, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @one(ptr addrspace(1) %p, float %b) {
  %r = call float @llvm.copysign.f32(float 1.0, float %b)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.copysign.f32(float, float)
declare half @llvm.copysign.f16(half, half)
