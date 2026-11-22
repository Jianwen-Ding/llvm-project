; A kernel (ptx_kernel) with a non-void return type is emitted as a .entry
; with no return parameter, but the return value is still stored to
; func_retval0, which is never declared: the PTX refers to an undefined
; symbol.  (A kernel's return value is not observable, so the store should
; be dropped, or such kernels rejected.)
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 | FileCheck %s
;
; CHECK: .visible .entry k(
; CHECK-NOT: func_retval0[
; CHECK-NOT: .param .b32 func_retval0
; CHECK: st.param.b32 [func_retval0],

define ptx_kernel i32 @k(i32 %x) {
  %y = add i32 %x, 1
  ret i32 %y
}
