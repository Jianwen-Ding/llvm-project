; llvm.minimum/llvm.maximum under denormal_fpenv(float: preservesign) can
; return an unflushed subnormal operand.  The lowering handles the
; "operands equal" case (to order -0.0 and +0.0) with
;   setp.eq.ftz.f32  p, a, b      ; a subnormal compares equal to 0
;   selp.f32         r, a_or_b, min.ftz(a, b), p
; and selects one of the *original* operands, which is not flushed.
; Example: minimum(+0.0, 0x00000001) returns 0x00000001 (1e-45); with the
; input flushed the result must be +0.0 (or -0.0).  The ieee version of the
; same function is correct.
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 | FileCheck %s
;
; CHECK-LABEL: min_ftz(
; CHECK: min.ftz.f32
; CHECK: setp.eq.ftz.f32 [[P:%p[0-9]+]]
; CHECK: selp.f32 {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, [[P]];

define float @min_ftz(float %a, float %b) #0 {
  %r = call float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}
define float @max_ftz(float %a, float %b) #0 {
  %r = call float @llvm.maximum.f32(float %a, float %b)
  ret float %r
}
attributes #0 = { denormal_fpenv(float: preservesign) }
