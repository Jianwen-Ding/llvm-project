; ?: double arithmetic (the target demotes it)
target triple = "air"
define void @k(ptr addrspace(1) %o) "air-kernel" {
  %v = load float, ptr addrspace(1) %o
  %d = fpext float %v to double
  %a = fadd double %d, 1.0
  %t = fptrunc double %a to float
  store float %t, ptr addrspace(1) %o
  ret void
}
