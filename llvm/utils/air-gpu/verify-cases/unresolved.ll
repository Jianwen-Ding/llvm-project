; ?: a call of a function nothing defines
target triple = "air"
declare i32 @nothing_defines_this(i32)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = call i32 @nothing_defines_this(i32 1)
  store i32 %v, ptr addrspace(1) %o
  ret void
}
