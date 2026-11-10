; A kernel whose buffers air-argument-buffer packs (the generic-GPU path, no
; pre-baked !air.kernel) takes its scalars in constant buffers after the
; argument buffer, so that !air.kernel describes every parameter (the AIR
; writer rejected the scalars, undescribed, before).
; RUN: llc -mtriple=air64_v28-apple-macosx26.0.0 %s -o - | FileCheck %s

; CHECK-LABEL: define void @k(ptr addrspace(2) %argbuf, ptr addrspace(2) {{.*}}dereferenceable(4) %a, ptr addrspace(2) {{.*}}dereferenceable(8) %n,
; CHECK: !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0,
; CHECK: !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, {{.*}}!"float"
; CHECK: !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, {{.*}}!"long"
; RUN: llc -mtriple=air64_v28-apple-macosx26.0.0 -filetype=obj %s -o /dev/null
target triple = "air64_v28-apple-macosx26.0.0"

declare i64 @_Z13get_global_idj(i32)

define spir_kernel void @k(float %a, ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %n) {
  %i = call i64 @_Z13get_global_idj(i32 0)
  %ok = icmp slt i64 %i, %n
  br i1 %ok, label %body, label %done
body:
  %px = getelementptr float, ptr addrspace(1) %x, i64 %i
  %py = getelementptr float, ptr addrspace(1) %y, i64 %i
  %vx = load float, ptr addrspace(1) %px
  %r = fmul float %a, %vx
  store float %r, ptr addrspace(1) %py
  br label %done
done:
  ret void
}
