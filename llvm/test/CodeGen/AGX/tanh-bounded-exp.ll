; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+canonical-nan -O3 < %s | FileCheck %s --check-prefix=CNAN

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

; Nor the scale's +inf and +0 selects (its n is in [0, 26]) (L15-E274), nor
; its NaN select: a NaN argument's 1.5 2^23 sum adds 0 to the scale's
; exponent (L15-E472; with PTX's canonical NaN the select stays).
; CHECK-LABEL: th2:
; CHECK-NOT: icmpsel
; CHECK-NOT: fcmpsel eq
; CHECK: stop
; CNAN-LABEL: th2:
; CNAN: fcmpsel eq
define agx_kernel void @th2(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.tanh.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}

; cosh's (and sinh's) e^|x| / 2 has a non-negative argument: no clamp at -104
; (-1026555904), no +0 select, and of the +inf selects only the scale's
; (icmpsel sgt), which covers |x| past 89 as well. Now (L15-E486) that one
; by a compare of |x| with 89.4159927 (0x42b2d4fd: 1119016189), from which
; it is +inf, without the clamp at 89 and the NaN select (fcmpsel eq);
; with canonical NaNs as before.
; CHECK-LABEL: ch:
; CHECK-NOT: mov_imm {{r[0-9]+}}, -1026555904
; CHECK-NOT: icmpsel
; CHECK-NOT: fcmpsel eq
; CHECK: mov_imm [[T:r[0-9]+]], 1119016189
; CHECK: fcmpsel gte, {{r[0-9]+}}, {{.*}}.abs, [[T]]
; CHECK-NOT: {{icmpsel|fcmpsel eq}}
; CHECK: stop
; CNAN-LABEL: ch:
; CNAN: icmpsel sgt
; CNAN: fcmpsel eq
define agx_kernel void @ch(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.cosh.f32(float %x)
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.cosh.f32(float)
