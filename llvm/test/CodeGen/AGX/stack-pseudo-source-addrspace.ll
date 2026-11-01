; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s

; Two zero <2 x float> stores into one alloca, 8 bytes apart: the DAG rebuilds the
; first's memory operand from the frame index, whose pseudo source value must be in
; the private address space for the stack store patterns to select it.

; CHECK-LABEL: two_zero_stores:
; CHECK: stack_store
; CHECK: stack_store
define agx_kernel void @two_zero_stores() {
entry:
  %0 = alloca [16 x i8], align 8, addrspace(5)
  store <2 x float> zeroinitializer, ptr addrspace(5) %0, align 8
  %1 = ptrtoint ptr addrspace(5) %0 to i32
  %2 = add i32 %1, 8
  %3 = inttoptr i32 %2 to ptr addrspace(5)
  store <2 x float> zeroinitializer, ptr addrspace(5) %3, align 8
  ret void
}
