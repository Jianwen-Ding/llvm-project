; ?: a call with another function type than its callee's declaration
target triple = "air"
declare void @air.wg.barrier(i32)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  call void @air.wg.barrier(i32 2, i32 1)
  store i32 1, ptr addrspace(1) %o
  ret void
}
