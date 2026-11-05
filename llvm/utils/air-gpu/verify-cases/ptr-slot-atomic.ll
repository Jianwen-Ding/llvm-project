; ?: a device pointer in private memory, then an atomic load
target triple = "air"
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %o, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  %v = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p, i32 0, i32 2, i1 true)
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}
