; ?: a pointer in an address space AIR does not have
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %a = inttoptr i64 64 to ptr addrspace(5)
  %v = load i32, ptr addrspace(5) %a
  store i32 %v, ptr addrspace(1) %o
  ret void
}
