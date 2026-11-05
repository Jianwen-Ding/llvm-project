; ?: a GEP of a device pointer stored in private memory
target triple = "air"
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %s = alloca ptr addrspace(1), align 8
  %g = getelementptr inbounds i8, ptr addrspace(1) %o, i64 4
  store ptr addrspace(1) %g, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  store i32 7, ptr addrspace(1) %p, align 4
  ret void
}
