; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rare-hardware-branch=0 < %s | FileCheck %s --check-prefix=OFF

; sin and cos past 2^22 take the hardware's reduction (L15-E504): sincos's
; two and tan's run in a block of their own, skipped by a SIMD-group none of
; whose threads has such an argument. Without it, every thread computed them
; and a select chose. One alone (8 instructions) would run with an empty
; mask anyway: left as the select.

; CHECK-LABEL: sin_f32:
; CHECK-NOT: if_fcmp
; CHECK: sin_pt_1
; CHECK-NOT: if_fcmp
; CHECK: fcmpsel lt
; CHECK: device_store

; CHECK-LABEL: sincos_f32:
; CHECK: if_fcmp [[E:r[0-9]+l]], nlt, {{r[0-9]+}}.abs, {{r[0-9]+}}, 1
; CHECK-NEXT: jmp_exec_none [[J:.LBB[0-9_]+]]
; CHECK-COUNT-2: sin_pt_1
; CHECK: [[J]]:
; CHECK-NEXT: pop_exec [[E]], 1
; CHECK-NOT: sin_pt_1
; CHECK: device_store

; CHECK-LABEL: tan_f32:
; CHECK: if_fcmp {{r[0-9]+l}}, nlt, {{r[0-9]+}}.abs, {{r[0-9]+}}, 1
; CHECK-NEXT: jmp_exec_none
; CHECK-COUNT-2: sin_pt_1
; CHECK: pop_exec
; CHECK: rcp
; CHECK: device_store

; OFF-LABEL: sincos_f32:
; OFF-NOT: jmp_exec_none
; OFF: sin_pt_1
; OFF-NOT: jmp_exec_none
; OFF: fcmpsel lt
; OFF: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.sin.f32(float)
declare { float, float } @llvm.sincos.f32(float)
declare float @llvm.tan.f32(float)

define agx_kernel void @sin_f32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.sin.f32(float %x)
  store float %r, ptr addrspace(1) %a
  ret void
}

define agx_kernel void @sincos_f32(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %sc = call { float, float } @llvm.sincos.f32(float %x)
  %s = extractvalue { float, float } %sc, 0
  %c = extractvalue { float, float } %sc, 1
  store float %s, ptr addrspace(1) %a
  %b = getelementptr float, ptr addrspace(1) %q, i32 %i
  store float %c, ptr addrspace(1) %b
  ret void
}

define agx_kernel void @tan_f32(ptr addrspace(1) %p) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr float, ptr addrspace(1) %p, i32 %i
  %x = load float, ptr addrspace(1) %a
  %r = call float @llvm.tan.f32(float %x)
  store float %r, ptr addrspace(1) %a
  ret void
}
