; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-half-fast-hyperbolic=0 -agx-half-fast-division=0 < %s | FileCheck %s --check-prefix=OFF

; Half tanh, sinh and cosh with the hardware's exponential and reciprocal
; (no Newton step), and half atan's and tan's division by the hardware's
; reciprocal: every half input's result as before (tan: closer to the exact
; value for 4 of them), checked on the functional model (L15-E329). bfloat
; keeps the float expansion (its range reaches the float overflow).


; CHECK-LABEL: tanh_h:
; CHECK: exp2
; CHECK: rcp [[Y:r[0-9]+]], {{r[0-9]+}}
; CHECK-NOT: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}.neg, [[Y]], 1.0
; CHECK: stop
; OFF-LABEL: tanh_h:
; OFF: rint
; OFF: fmadd32 {{r[0-9]+}}, {{r[0-9]+}}.neg, {{r[0-9]+}}, 1.0
define void @tanh_h(ptr addrspace(1) %p) {
  %x = load half, ptr addrspace(1) %p
  %r = call half @llvm.tanh.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: cosh_h:
; CHECK: exp2
; CHECK: rcp
; CHECK-NOT: rint
; CHECK: stop
; OFF-LABEL: cosh_h:
; OFF: rint
define void @cosh_h(ptr addrspace(1) %p) {
  %x = load half, ptr addrspace(1) %p
  %r = call half @llvm.cosh.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: atan_h:
; CHECK: rcp [[R:r[0-9]+]], {{r[0-9]+}}
; CHECK-NOT: bitop
; CHECK: fmul32 {{r[0-9]+}}, {{r[0-9]+}}, [[R]]
; CHECK: stop
; OFF-LABEL: atan_h:
; OFF: rcp
; OFF: bitop
define void @atan_h(ptr addrspace(1) %p) {
  %x = load half, ptr addrspace(1) %p
  %r = call half @llvm.atan.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; bfloat: the float expansion
; CHECK-LABEL: tanh_bf:
; CHECK: rint
define void @tanh_bf(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.tanh.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
