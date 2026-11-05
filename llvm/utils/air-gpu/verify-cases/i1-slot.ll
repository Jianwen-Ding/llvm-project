; ?: an i1 in private memory
target triple = "air"
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
define void @k(ptr addrspace(1) %o, i32 %n) "air-kernel" {
  %s = alloca i1, align 1
  %c = icmp eq i32 %n, 0
  store volatile i1 %c, ptr %s, align 1
  %l = load volatile i1, ptr %s, align 1
  %z = zext i1 %l to i32
  store i32 %z, ptr addrspace(1) %o
  ret void
}
