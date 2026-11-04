; agx-lower-air keeps a front end's air.flush.harmless on the operation it
; makes of an AIR math function (AGXCodeGenPrepare then flushes no half
; operand or result of it).
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

; CHECK-LABEL: define {{.*}} @k(
; CHECK: call half @llvm.fma.f16({{.*}}), !air.flush.harmless
declare half @air.fma.f16(half, half, half)

define void @k(ptr addrspace(1) %p) #0 {
  %x = load half, ptr addrspace(1) %p
  %r = call half @air.fma.f16(half %x, half 2.0, half -1.0), !air.flush.harmless !3
  store half %r, ptr addrspace(1) %p
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) }

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
!3 = !{}
