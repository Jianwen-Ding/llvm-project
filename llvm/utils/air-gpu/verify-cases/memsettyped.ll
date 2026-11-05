; a memset of a pointer to another type than i8: repaired by the writer
target triple = "air"
declare void @llvm.memset.p3.i32(ptr addrspace(3), i8, i32, i1)
@t = internal addrspace(3) global [8 x i32] undef, align 16
define void @k(ptr addrspace(1) %o) "air-kernel" {
  call void @llvm.memset.p3.i32(ptr addrspace(3) align 16 @t, i8 0, i32 32, i1 false)
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr addrspace(3) @t, align 16
  %v = load i32, ptr addrspace(3) @t
  store i32 %v, ptr addrspace(1) %o
  ret void
}
