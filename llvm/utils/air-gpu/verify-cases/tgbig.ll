; ?: 64 KiB of threadgroup memory used by one kernel
target triple = "air"
@t = internal addrspace(3) global [65536 x i8] undef, align 16
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i8 1, ptr addrspace(3) @t
  %v = load i8, ptr addrspace(3) @t
  store i8 %v, ptr addrspace(1) %o
  ret void
}
