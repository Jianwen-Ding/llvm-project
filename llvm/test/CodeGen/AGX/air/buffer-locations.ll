; MSL binds buffers by [[buffer(N)]] in any order and with gaps, as Xcode's
; AIR gives them (air.location_index). The AGX kernel takes each buffer at its
; location's position, in location order; a location no parameter has is an
; unused argument, so the later buffers keep their locations.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: define agx_kernel void @k(
; CHECK-SAME: ptr addrspace(1) %out,
; CHECK-SAME: ptr addrspace(1) %unused.buffer1,
; CHECK-SAME: ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 dereferenceable(4) %scale,
; CHECK-SAME: ptr addrspace(1) %in)
; CHECK: %v = load float, ptr addrspace(1) %in.i
; CHECK: %s = load float, ptr addrspace(4) %scale
; CHECK: store float %r, ptr addrspace(1) %out.i

; ASM-LABEL: k:
; ASM: device_load {{.*}}, u6_u7,
; ASM: fmul32 {{.*}}, u4,
; ASM: device_store {{.*}}, u0_u1,

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) noalias nocapture readonly %in, ptr addrspace(2) noalias nocapture readonly align 4 dereferenceable(4) %scale, ptr addrspace(1) noalias nocapture %out, i32 %gid) {
  %in.i = getelementptr inbounds float, ptr addrspace(1) %in, i32 %gid
  %v = load float, ptr addrspace(1) %in.i, align 4
  %s = load float, ptr addrspace(2) %scale, align 4
  %r = fmul float %v, %s
  %out.i = getelementptr inbounds float, ptr addrspace(1) %out, i32 %gid
  store float %r, ptr addrspace(1) %out.i, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2, !3, !4}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scale"}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
