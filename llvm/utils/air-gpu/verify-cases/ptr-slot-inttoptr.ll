; ?: a pointer from an integer in private memory
target triple = "air"
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
define void @k(ptr addrspace(1) %o, i64 %a) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  %i = inttoptr i64 %a to ptr addrspace(1)
  store ptr addrspace(1) %i, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %o, align 4
  ret void
}
