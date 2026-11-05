; ?: a device pointer in private memory, used at two types
target triple = "air"
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %o, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %p, align 4
  %q = load ptr addrspace(1), ptr %s, align 8
  store i8 1, ptr addrspace(1) %q, align 1
  ret void
}
