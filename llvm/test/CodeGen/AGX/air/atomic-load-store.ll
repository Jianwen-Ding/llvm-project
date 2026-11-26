; AIR's atomic loads and stores (what the AIR writer makes of LLVM's atomic
; loads and stores) become atomic loads and stores again.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: load atomic volatile i32, ptr addrspace(1) %p monotonic, align 4
; CHECK: load atomic volatile float, ptr addrspace(1) %q monotonic, align 4
; CHECK: store atomic volatile i32 %v, ptr addrspace(1) %p monotonic, align 4
; CHECK: store atomic volatile float %f, ptr addrspace(1) %q monotonic, align 4
define void @k(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p, i32 0, i32 2, i1 true)
  %f = call float @air.atomic.global.load.f32(ptr addrspace(1) %q, i32 0, i32 2, i1 true)
  call void @air.atomic.global.store.i32(ptr addrspace(1) %p, i32 %v, i32 0, i32 2, i1 true)
  call void @air.atomic.global.store.f32(ptr addrspace(1) %q, float %f, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
declare float @air.atomic.global.load.f32(ptr addrspace(1), i32, i32, i1)
declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i1)
declare void @air.atomic.global.store.f32(ptr addrspace(1), float, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"q"}
