; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; half sin / cos reduce to turns exactly: x times a 13-bit part of 1 / 2pi
; (exact for a half), its fraction, plus x times the next part by an fma, the
; fraction again. One rounded x / 2pi left them up to 1,517 / 4,800 ulp off
; for large halves; now at most 6.5 / 4.6 (simulator, every half). afn keeps
; the short reduction.

; CHECK-LABEL: s:
; CHECK: mov_imm [[CHI:r[0-9]+]], 1042479104
; CHECK: floor
; CHECK: mov_imm [[CMID:r[0-9]+]], 918664999
; CHECK: fmadd32 {{r[0-9]+}}, {{u[0-9]+l}}, [[CMID]],
; CHECK: floor
; CHECK: sin_pt_1
define agx_kernel void @s(ptr addrspace(1) %p, half %x) {
  %r = call half @llvm.sin.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: s_afn:
; CHECK-NOT: fmadd32
; CHECK: sin_pt_1
define agx_kernel void @s_afn(ptr addrspace(1) %p, half %x) {
  %r = call afn half @llvm.sin.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}
declare half @llvm.sin.f16(half)
