; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-tan-quotient-signs=0 < %s | FileCheck %s --check-prefix=OFF

; tan (L15-E503): sin / cos by quadrant, the quotient's sign that of an odd
; quadrant (bit 1 of q and of q + 1 differ exactly there): -(c / s) as
; (-c) / s, s / c otherwise, without bit 1 of q or q + 1 or the two halves'
; sign flips.

; CHECK-LABEL: tan_f32:
; CHECK-NOT: bfeil
; CHECK: and {{r[0-9]+}}, {{r[0-9]+}}, 1{{$}}
; CHECK-NOT: bfeil
; CHECK: rcp
; CHECK-NOT: bfeil
; CHECK: device_store

; OFF-LABEL: tan_f32:
; OFF: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 1, mask 0x1
; OFF: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 1, mask 0x1
; OFF: device_store

; CHECK-LABEL: tan_bf16:
; CHECK-NOT: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 1, mask 0x1
; CHECK: device_store

; OFF-LABEL: tan_bf16:
; OFF: bfeil {{r[0-9]+}}, 0, {{r[0-9]+}}, 1, mask 0x1
; OFF: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.tan.f32(float)
declare bfloat @llvm.tan.bf16(bfloat)

define agx_kernel void @tan_f32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.tan.f32(float %x)
  store float %r, ptr addrspace(1) %a
  ret void
}

define agx_kernel void @tan_bf16(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr bfloat, ptr addrspace(1) %p, i32 %i
  %x = load bfloat, ptr addrspace(1) %a
  %r = call bfloat @llvm.tan.bf16(bfloat %x)
  store bfloat %r, ptr addrspace(1) %a
  ret void
}
