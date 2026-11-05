; ?: a getelementptr on a null threadgroup pointer with a run-time offset (an address LLVM
; folded from integers)
target triple = "air"
@b = internal addrspace(3) global [16 x i32] undef, align 16
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %i = ptrtoint ptr addrspace(3) @b to i64
  %j = load i64, ptr addrspace(1) %o
  %a = add i64 %i, %j
  %p = getelementptr inbounds nuw i8, ptr addrspace(3) null, i64 %a
  store i32 1, ptr addrspace(3) %p, align 4
  ret void
}
