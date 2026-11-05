; accept: a plain kernel
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i32 7, ptr addrspace(1) %o
  ret void
}
