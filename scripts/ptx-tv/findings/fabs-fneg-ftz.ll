; fneg/fabs are not floating-point math operations: LangRef says they "act
; directly on the underlying bit representation and never change anything
; except possibly for the sign bit".  Under denormal_fpenv(float:
; preservesign) the backend selects abs.ftz.f32 / neg.ftz.f32, which flush
; a subnormal input to zero: fabs(0x00000001) returns +0.0 instead of
; 0x00000001.  copysign stays bitwise (copysign.f32) and is correct;
; and.b32/xor.b32 with the sign mask would be exact.
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 | FileCheck %s
;
; CHECK-LABEL: fabs_f(
; CHECK: abs.ftz.f32
; CHECK-LABEL: fneg_f(
; CHECK: neg.ftz.f32

define float @fabs_f(float %a) #0 {
  %r = call float @llvm.fabs.f32(float %a)
  ret float %r
}
define float @fneg_f(float %a) #0 {
  %r = fneg float %a
  ret float %r
}
attributes #0 = { denormal_fpenv(float: preservesign) }
