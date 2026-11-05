; A memset of a threadgroup byte array that is retyped for its other accesses: as a loop of accesses,
; or its i8 pointer no longer matches and the writer rejects the module.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; CHECK-NOT: call void @llvm.memset
target triple = "air"
@s = internal addrspace(3) global [32 x i8] undef, align 16
declare void @llvm.memset.p3.i32(ptr addrspace(3), i8, i32, i1)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  call void @llvm.memset.p3.i32(ptr addrspace(3) align 16 @s, i8 0, i32 32, i1 false)
  store i32 7, ptr addrspace(3) @s, align 4
  %p = getelementptr inbounds nuw i8, ptr addrspace(3) @s, i64 2
  %v = load i16, ptr addrspace(3) %p, align 2
  store i16 %v, ptr addrspace(1) %o
  ret void
}
