; ?: a call of a function AIR does not have (llvm.copysign stays an LLVM intrinsic)
target triple = "air"
declare float @air.copysign.f32(float, float)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %x = load float, ptr addrspace(1) %o
  %y = call float @air.copysign.f32(float %x, float %x)
  store float %y, ptr addrspace(1) %o
  ret void
}
