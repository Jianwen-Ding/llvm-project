; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-half-fast-hyperbolic=0 -agx-half-fast-division=0 -agx-narrow-fast-sqrt=0 -agx-narrow-sincos-reduction=0 -agx-bfloat-short-polys=0 -agx-atan-quotient-step=0 < %s | FileCheck %s --check-prefix=OFF

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
; OFF: mov_imm {{r[0-9]+}}, 1262485504
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
; CHECK-NOT: mov_imm {{r[0-9]+}}, 1262485504
; CHECK: stop
; OFF-LABEL: cosh_h:
; OFF: mov_imm {{r[0-9]+}}, 1262485504
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
; CHECK: mov_imm {{r[0-9]+}}, 1262485504
define void @tanh_bf(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.tanh.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; bfloat atan: the division by the reciprocal too (L15-E331)
; CHECK-LABEL: atan_bf:
; CHECK: rcp
; CHECK-NOT: bitop
; CHECK: stop
define void @atan_bf(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.atan.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; half asin: the hardware's square root (L15-E333)
; CHECK-LABEL: asin_h:
; CHECK: rsqrt_special [[R:r[0-9]+]], [[X:r[0-9]+]]
; CHECK-NEXT: fmul32 {{r[0-9]+}}, [[X]], [[R]]
; CHECK-NOT: rsqrt_special
; CHECK: stop
define void @asin_h(ptr addrspace(1) %p) {
  %x = load half, ptr addrspace(1) %p
  %r = call half @llvm.asin.f16(half %x)
  store half %r, ptr addrspace(1) %p
  ret void
}

; bfloat sin: pi / 2 in two parts (L15-E350)
; CHECK-LABEL: sin_bf:
; CHECK-NOT: mov_imm {{r[0-9]+}}, 653733101
; CHECK: stop
; OFF-LABEL: sin_bf:
; OFF: mov_imm {{r[0-9]+}}, 653733101
define void @sin_bf(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.sin.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}

; bfloat cosh: its exponential's polynomial two terms shorter (L15-E351)
; CHECK-LABEL: cosh_bf:
; CHECK-NOT: mov_imm {{r[0-9]+}}, 961571175
; CHECK: stop
; OFF-LABEL: cosh_bf:
; OFF: mov_imm {{r[0-9]+}}, 961571175
define void @cosh_bf(ptr addrspace(1) %p) {
  %x = load bfloat, ptr addrspace(1) %p
  %r = call bfloat @llvm.cosh.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %p
  ret void
}
