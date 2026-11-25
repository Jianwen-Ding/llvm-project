; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; A half widened to double: the hardware's float conversion (exact for every
; finite half, subnormals included) and the float's short path, the general
; conversion of the half's bits behind it for infinities and NaNs
; (L15-E313).

; CHECK-LABEL: ext:
; CHECK: fadd32 [[F:r[0-9]+]], {{r[0-9]+}}l, -0.0
; CHECK: if_icmp
; CHECK: jmp_exec_any
define agx_kernel void @ext(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %a = getelementptr half, ptr addrspace(1) %p, i32 %t
  %x = load half, ptr addrspace(1) %a
  %d = fpext half %x to double
  %b = getelementptr double, ptr addrspace(1) %o, i32 %t
  store double %d, ptr addrspace(1) %b
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
