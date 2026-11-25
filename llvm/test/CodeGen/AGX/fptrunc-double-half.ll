; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A double rounded to half or bfloat: a short path for a zero or normal
; result (the exponent rebiased, the fraction rounded to nearest even), the
; general conversion behind it in a conditional block for subnormal results,
; overflow, infinities and NaNs (L15-E314).

; CHECK-LABEL: to_half:
; CHECK: if_icmp
; CHECK: jmp_exec_any
; CHECK: device_store 0, i16
define agx_kernel void @to_half(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr double, ptr addrspace(1) %p, i32 %t
  %x = load double, ptr addrspace(1) %a
  %h = fptrunc double %x to half
  %b = getelementptr half, ptr addrspace(1) %o, i32 %t
  store half %h, ptr addrspace(1) %b
  ret void
}

; CHECK-LABEL: to_bfloat:
; CHECK: if_icmp
; CHECK: jmp_exec_any
; CHECK: device_store 0, i16
define agx_kernel void @to_bfloat(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr double, ptr addrspace(1) %p, i32 %t
  %x = load double, ptr addrspace(1) %a
  %h = fptrunc double %x to bfloat
  %b = getelementptr bfloat, ptr addrspace(1) %o, i32 %t
  store bfloat %h, ptr addrspace(1) %b
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
