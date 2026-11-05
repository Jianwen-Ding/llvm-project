; ?: an address space cast between device and constant
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %c = addrspacecast ptr addrspace(1) %o to ptr addrspace(2)
  %v = load i32, ptr addrspace(2) %c
  store i32 %v, ptr addrspace(1) %o
  ret void
}
