; ?: a store through a null threadgroup pointer, on a path that does not run here
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i32, ptr addrspace(1) %o
  %c = icmp eq i32 %v, 12345
  br i1 %c, label %t, label %e
t:
  store i32 0, ptr addrspace(3) null, align 4
  br label %e
e:
  ret void
}
