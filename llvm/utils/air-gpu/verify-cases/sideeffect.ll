; ?: a kernel that calls llvm.sideeffect
target triple = "air"
declare void @llvm.sideeffect()
define void @k(ptr addrspace(1) %o) "air-kernel" {
  call void @llvm.sideeffect()
  store i32 1, ptr addrspace(1) %o
  ret void
}
