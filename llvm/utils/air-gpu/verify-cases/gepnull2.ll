; ?: a getelementptr on a null threadgroup pointer, the same base used for two pointee types
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %j = load i64, ptr addrspace(1) %o
  %c = icmp eq i64 %j, 12345
  br i1 %c, label %t, label %e
t:
  %p = getelementptr inbounds nuw i8, ptr addrspace(3) null, i64 %j
  %q = getelementptr inbounds nuw i8, ptr addrspace(3) %p, i64 16
  store i32 1, ptr addrspace(3) %p, align 4
  store float 1.0, ptr addrspace(3) %q, align 4
  br label %e
e:
  ret void
}
