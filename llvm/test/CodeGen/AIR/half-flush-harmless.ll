; An operation a front end marks air.flush.harmless (its result is the same
; whether subnormal operands or results are flushed) is not flushed in a
; flushing function.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare half @llvm.fma.f16(half, half, half)

; CHECK-LABEL: define void @h
; CHECK-NOT: fcmp
; CHECK: call half @air.fma.f16(half %x, half 2.000000e+00, half -1.000000e+00), !air.flush.harmless
; CHECK-NOT: fcmp
; CHECK: ret void
define void @h(ptr addrspace(1) %a) #0 {
  %x = load half, ptr addrspace(1) %a
  %r = call half @llvm.fma.f16(half %x, half 2.0, half -1.0), !air.flush.harmless !0
  store half %r, ptr addrspace(1) %a
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) }

!0 = !{}
