; ?: a kernel that calls llvm.trap on a path that does not run here
target triple = "air"
declare void @llvm.trap()
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i32, ptr addrspace(1) %o
  %c = icmp eq i32 %v, 12345
  br i1 %c, label %t, label %e
t:
  call void @llvm.trap()
  unreachable
e:
  store i32 1, ptr addrspace(1) %o
  ret void
}
