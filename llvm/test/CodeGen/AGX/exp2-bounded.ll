; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 2^x and 10^x clamp x itself (to [-151, 130] and [-46, 40]), so that no
; infinity makes the low part of x ln b a NaN: two selects, instead of a
; clamp of x ln b and +inf / +0 selects after (the same result for every
; float, checked exhaustively, flushing and not).

; CHECK-LABEL: e2:
; CHECK-COUNT-2: fcmpsel
; CHECK-NOT: fcmpsel
; CHECK: stop
define agx_kernel void @e2(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp2.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: e10:
; CHECK-COUNT-2: fcmpsel
; CHECK-NOT: fcmpsel
; CHECK: stop
define agx_kernel void @e10(ptr addrspace(1) %p) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %q = getelementptr float, ptr addrspace(1) %p, i32 %t
  %x = load float, ptr addrspace(1) %q, align 4
  %y = call float @llvm.exp10.f32(float %x)
  store float %y, ptr addrspace(1) %q, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.exp2.f32(float)
declare float @llvm.exp10.f32(float)
