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
