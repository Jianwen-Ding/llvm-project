; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; An integer converted to bfloat goes through float: one of at most 8
; significant bits (an i8) is a bfloat exactly, the float's top half without
; the rounding; any converted integer is never a NaN, so the rounding needs
; no NaN test (L15-E312).

; CHECK-LABEL: from_i8:
; CHECK: convert s8_to_f
; CHECK-NOT: iadd
; CHECK-NOT: fcmpsel
; CHECK: device_store 0, i16
define agx_kernel void @from_i8(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i8, ptr addrspace(1) %p, i32 %t
  %x = load i8, ptr addrspace(1) %a
  %f = sitofp i8 %x to bfloat
  %b = getelementptr bfloat, ptr addrspace(1) %o, i32 %t
  store bfloat %f, ptr addrspace(1) %b
  ret void
}

; CHECK-LABEL: from_i16:
; CHECK: convert u16_to_f
; CHECK: iadd
; CHECK-NOT: fcmpsel
; CHECK: device_store 0, i16
define agx_kernel void @from_i16(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr i16, ptr addrspace(1) %p, i32 %t
  %x = load i16, ptr addrspace(1) %a
  %f = uitofp i16 %x to bfloat
  %b = getelementptr bfloat, ptr addrspace(1) %o, i32 %t
  store bfloat %f, ptr addrspace(1) %b
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
