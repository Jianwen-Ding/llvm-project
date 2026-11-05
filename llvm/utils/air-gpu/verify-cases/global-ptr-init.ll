; ?: a constant global initialized with the address of another
target triple = "air"
@r = internal addrspace(2) constant i32 1
@p = addrspace(2) constant ptr addrspace(2) @r
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %q = load ptr addrspace(2), ptr addrspace(2) @p
  %v = load i32, ptr addrspace(2) %q
  store i32 %v, ptr addrspace(1) %o
  ret void
}
