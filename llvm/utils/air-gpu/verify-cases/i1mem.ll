; ?: an i1 in device memory
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i1 true, ptr addrspace(1) %o
  ret void
}
