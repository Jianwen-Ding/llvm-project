; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32,+canonical-nan -stop-after=agx-strict-fp < %s | FileCheck %s

; Strict f32 addition: the hardware's result is right unless both operands
; are below 2^-101, never so with a constant operand at least that large: no
; software path. A variable addition keeps it.

; CHECK-LABEL: @c(
; CHECK: fadd float %x, 1.638400e+04
; CHECK-NOT: br i1
; CHECK: ret void
define void @c(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %s = fadd float %x, 16384.0
  store float %s, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @v(
; CHECK: br i1
define void @v(ptr addrspace(1) %p, float %y) {
  %x = load float, ptr addrspace(1) %p
  %s = fadd float %x, %y
  store float %s, ptr addrspace(1) %p
  ret void
}

; nnan: no canonical NaN for the result.
; CHECK-LABEL: @n(
; CHECK-NOT: fcmp uno
; CHECK: ret void
define void @n(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %s = fadd nnan float %x, 16384.0
  store float %s, ptr addrspace(1) %p
  ret void
}
