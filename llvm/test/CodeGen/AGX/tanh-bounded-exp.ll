; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; tanh's exponential of 2 min(|x|, 9), in [0, 18] or a NaN, needs no clamp
; of its argument to [-104, 89] (bit-identical on the simulator's
; functional model).

; CHECK-LABEL: th:
; CHECK-NOT: mov_imm {{r[0-9]+}}, -1026555904
; CHECK-NOT: mov_imm {{r[0-9]+}}, 1118961664
; CHECK: stop
define agx_kernel void @th(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.tanh.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.tanh.f32(float)

; Nor the scale's +inf and +0 selects (its n is in [0, 26]): the one select
; left after the exponential is its NaN select (fcmpsel eq), then the small
; argument's polynomial (L15-E274).
; CHECK-LABEL: th2:
; CHECK-NOT: icmpsel
; CHECK: fcmpsel eq
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @th2(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.tanh.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}

; cosh's (and sinh's) e^|x| / 2 has a non-negative argument: no clamp at -104
; (-1026555904), no +0 select, and of the +inf selects only the scale's
; (icmpsel sgt), which covers |x| past 89 as well.
; CHECK-LABEL: ch:
; CHECK-NOT: mov_imm {{r[0-9]+}}, -1026555904
; CHECK-NOT: icmpsel slt
; CHECK: icmpsel sgt
; CHECK-NOT: icmpsel
; CHECK: stop
define agx_kernel void @ch(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.cosh.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.cosh.f32(float)
