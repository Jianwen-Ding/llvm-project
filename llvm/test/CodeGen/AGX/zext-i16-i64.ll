; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A 16-bit value widened to 64 bits is the widening copy as the low half and
; a zero as the high one, which does not wait for the 16-bit value (it was
; a 64-bit addition of zero to the widened value).

; CHECK-LABEL: store64:
; CHECK-DAG: bitop 0101, {{r[0-9]+}}, r{{[0-9]+[lh]}}, 0
; CHECK-DAG: mov_imm {{r[0-9]+}}, 0
; CHECK-NOT: iadd
; CHECK: device_store 0, i32, xy
define agx_kernel void @store64(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr i16, ptr addrspace(1) %p, i32 %t
  %h = load i16, ptr addrspace(1) %pa, align 2
  %w = zext i16 %h to i64
  %qa = getelementptr i64, ptr addrspace(1) %q, i32 %t
  store i64 %w, ptr addrspace(1) %qa, align 8
  ret void
}

; CHECK-LABEL: index64:
; CHECK-NOT: iadd {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}, 0
; CHECK: stop
define agx_kernel void @index64(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr i16, ptr addrspace(1) %p, i32 %t
  %h = load i16, ptr addrspace(1) %pa, align 2
  %w = zext i16 %h to i64
  %qa = getelementptr i32, ptr addrspace(1) %q, i64 %w
  store i32 %t, ptr addrspace(1) %qa, align 4
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
