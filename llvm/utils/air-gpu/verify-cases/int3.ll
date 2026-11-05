; accept?: an i3 value in the module
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load i32, ptr addrspace(1) %o
  %t = trunc i32 %v to i3
  %a = add i3 %t, 1
  %z = zext i3 %a to i32
  store i32 %z, ptr addrspace(1) %o
  ret void
}
