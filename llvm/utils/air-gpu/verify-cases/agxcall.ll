; ?: a call of an AGX intrinsic
target triple = "air"
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  store i32 %v, ptr addrspace(1) %o
  ret void
}
