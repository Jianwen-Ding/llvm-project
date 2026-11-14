; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

; AIR's half and bfloat atanh and tanpi, computed in float: the division by
; the reciprocal only where every input's result is unchanged (L15-E339):
; half atanh and bfloat tanpi; half tanpi and bfloat atanh keep the
; correctly rounded division.

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: fptrunc float {{.*}} to half
; CHECK: fdiv float
; CHECK: fptrunc float {{.*}} to bfloat
; CHECK: fdiv float
; CHECK: fptrunc float {{.*}} to half
; CHECK: call float @llvm.agx.rcp.f32
; CHECK: fptrunc float {{.*}} to bfloat
define void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %h = load half, ptr addrspace(1) %p
  %b = load bfloat, ptr addrspace(1) %q
  %a1 = call half @air.atanh.f16(half %h)
  store volatile half %a1, ptr addrspace(1) %p
  %a2 = call bfloat @air.atanh.bf16(bfloat %b)
  store volatile bfloat %a2, ptr addrspace(1) %q
  %t1 = call half @air.tanpi.f16(half %h)
  store volatile half %t1, ptr addrspace(1) %p
  %t2 = call bfloat @air.tanpi.bf16(bfloat %b)
  store volatile bfloat %t2, ptr addrspace(1) %q
  ret void
}
declare half @air.atanh.f16(half)
declare bfloat @air.atanh.bf16(bfloat)
declare half @air.tanpi.f16(half)
declare bfloat @air.tanpi.bf16(bfloat)

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1}
