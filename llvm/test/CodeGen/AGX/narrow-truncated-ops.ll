; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-narrow-truncated-ops=0 -agx-fold-constant-address-chains=0 < %s | FileCheck %s --check-prefix=OFF

; A called function stores through a generic pointer that its only caller
; makes from a private array (Cycles' SVM stack): the private address is the
; low half of (tag | offset) + 4 * index, a 32-bit add, not a 64-bit one.
; CHECK-LABEL: store_stack:
; CHECK-NOT:   iadd r{{[0-9]+}}_r{{[0-9]+}},
; CHECK:       stack_store
; OFF-LABEL:   store_stack:
; OFF:         iadd r{{[0-9]+}}_r{{[0-9]+}},
; OFF:         stack_store
define void @store_stack(ptr %stack, i64 %i, float %v) noinline {
  %p = getelementptr inbounds float, ptr %stack, i64 %i
  store float %v, ptr %p, align 4
  ret void
}

; Two device loads at constant offsets from a base plus a constant that both
; use: each takes the base and the summed offset as its immediate, without a
; 64-bit add of the shared constant.
; CHECK-LABEL: shared_offset:
; CHECK-NOT:   iadd r{{[0-9]+}}_r{{[0-9]+}},
; CHECK-DAG:   device_load {{.*}}, 428, signed
; CHECK-DAG:   device_load {{.*}}, 429, signed
define void @shared_offset(ptr addrspace(1) %base, ptr addrspace(1) %out) noinline {
  %b = getelementptr inbounds i8, ptr addrspace(1) %base, i64 368
  %p = getelementptr inbounds i32, ptr addrspace(1) %b, i64 336
  %q = getelementptr inbounds i32, ptr addrspace(1) %b, i64 337
  %x = load volatile i32, ptr addrspace(1) %p, align 4
  %y = load volatile i32, ptr addrspace(1) %q, align 4
  %s = add i32 %x, %y
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}

define agx_kernel void @k(ptr addrspace(1) %base, ptr addrspace(1) %out, i64 %i, float %v) {
  %a = alloca [16 x float], align 4, addrspace(5)
  %g = addrspacecast ptr addrspace(5) %a to ptr
  call void @store_stack(ptr %g, i64 %i, float %v)
  %r = load float, ptr addrspace(5) %a, align 4
  %ro = getelementptr inbounds i8, ptr addrspace(1) %out, i64 64
  store float %r, ptr addrspace(1) %ro, align 4
  call void @shared_offset(ptr addrspace(1) %base, ptr addrspace(1) %out)
  ret void
}
