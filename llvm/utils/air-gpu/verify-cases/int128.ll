; ?: i128 arithmetic
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i64, ptr addrspace(1) %o
  %w = zext i64 %v to i128
  %m = mul i128 %w, %w
  %s = lshr i128 %m, 64
  %t = trunc i128 %s to i64
  store i64 %t, ptr addrspace(1) %o
  ret void
}
