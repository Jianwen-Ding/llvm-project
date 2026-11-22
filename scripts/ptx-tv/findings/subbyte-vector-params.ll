; Vectors with sub-byte elements are passed/returned through a .param that is
; declared with the LLVM store size, but accessed with one or more bytes per
; element, i.e. past the end of the declared parameter.
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 | FileCheck %s
;
; CHECK: .param .align 1 .b8 func_retval0[1]) ret_v4i1(
; CHECK: st.param.b8 [func_retval0+3]
; CHECK: .param .align 1 .b8 arg_v4i1_param_0[1]
; CHECK: ld.param.b8 %r{{[0-9]+}}, [arg_v4i1_param_0+3]
; CHECK: .param .align 2 .b8 pass_v8i2_param_0[2]
; CHECK: ld.param.b16 %rs{{[0-9]+}}, [pass_v8i2_param_0+14]

define <4 x i1> @ret_v4i1(i8 %x) {
  %v = bitcast i8 %x to <8 x i1>
  %r = shufflevector <8 x i1> %v, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i1> %r
}

define i1 @arg_v4i1(<4 x i1> %v) {
  %e = extractelement <4 x i1> %v, i32 3
  ret i1 %e
}

declare <8 x i2> @ext8(<8 x i2>)
define <8 x i2> @pass_v8i2(<8 x i2> %x) {
  %r = call <8 x i2> @ext8(<8 x i2> %x)
  ret <8 x i2> %r
}
