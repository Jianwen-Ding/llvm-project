; A memcpy, memset or memmove of a run-time length is a loop for Metal: with lengths
; that differ between threads, Metal's lowering followed by a barrier hangs the GPU
; (an M2 Max). A constant length keeps the intrinsic.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

target triple = "air"

declare void @llvm.memset.p3.i32(ptr addrspace(3), i8, i32, i1)
declare void @llvm.memcpy.p3.p1.i32(ptr addrspace(3), ptr addrspace(1), i32, i1)
@t = internal addrspace(3) global [64 x i8] undef, align 16

; CHECK-LABEL: define void @variable(
; CHECK-NOT: call void @llvm.mem
; CHECK: ret void
define void @variable(ptr addrspace(1) %o) "air-kernel" {
  %n = load i32, ptr addrspace(1) %o
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) @t, ptr addrspace(1) %o, i32 %n, i1 false)
  call void @llvm.memset.p3.i32(ptr addrspace(3) @t, i8 0, i32 %n, i1 false)
  ret void
}

; CHECK-LABEL: define void @constant(
; CHECK: call void @llvm.memcpy
; CHECK: call void @llvm.memset
define void @constant(ptr addrspace(1) %o) "air-kernel" {
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) @t, ptr addrspace(1) %o, i32 16, i1 false)
  call void @llvm.memset.p3.i32(ptr addrspace(3) @t, i8 0, i32 16, i1 false)
  ret void
}
