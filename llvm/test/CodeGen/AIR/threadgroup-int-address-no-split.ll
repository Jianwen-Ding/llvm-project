; A threadgroup variable whose address is held as an integer can be reached anywhere by arithmetic: it stays
; one global. Split, a read at a computed address reads another variable (PTX mov.u32 %r, sym, then
; cp.reduce.async.bulk from it, on an M2 Max).
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; CHECK-NOT: __off
target triple = "air"
@s = internal addrspace(3) global [144 x i8] undef, align 128
define void @k(ptr addrspace(1) %o, i32 %i) "air-kernel" {
  store i32 1065353216, ptr addrspace(3) @s, align 4
  store i32 1065353216, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 4), align 4
  store i32 1065353216, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 8), align 4
  store i32 1065353216, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 12), align 4
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 128), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 130), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 132), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 134), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 136), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 138), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 140), align 2
  store i16 1, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @s, i64 142), align 2
  fence syncscope("workgroup") seq_cst
  %z = zext i32 %i to i64
  %a = add i64 %z, ptrtoint (ptr addrspace(3) @s to i64)
  %p = inttoptr i64 %a to ptr addrspace(3)
  %v = load i8, ptr addrspace(3) %p, align 1
  store i8 %v, ptr addrspace(1) %o
  ret void
}
