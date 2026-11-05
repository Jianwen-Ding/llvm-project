; stricter: a math function declared with another type than its name says (float suffix, i32
; values). Metal accepts it and what it computes is undefined; llc reports it all the same.
target triple = "air"
declare i32 @air.sin.f32(i32)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %x = load i32, ptr addrspace(1) %o
  %y = call i32 @air.sin.f32(i32 %x)
  store i32 %y, ptr addrspace(1) %o
  ret void
}
