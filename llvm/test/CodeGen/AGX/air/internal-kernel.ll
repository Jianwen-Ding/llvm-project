; An AIR module may describe an internal function as a kernel (OpenCL front
; ends leave such copies): it stays a kernel after the calls are inlined
; instead of being erased as dead.
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx26.0.0"

; CHECK-LABEL: k:
; CHECK-LABEL: inner:
define void @k(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p
  ret void
}

define internal void @inner(ptr addrspace(1) %p) {
  store i32 2, ptr addrspace(1) %p
  ret void
}

!air.kernel = !{!0, !2}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}
!2 = !{ptr @inner, !{}, !{!1}}
