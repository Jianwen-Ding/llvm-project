; [su]itofp i32/i64 -> bfloat on targets without cvt.rn.bf16.{s,u}{32,64}
; (before sm_90) is lowered as int -> f32 (rounded) -> bf16 (rounded again).
; Double rounding gives a result one ulp off for some inputs, e.g.
;   uitofp i64 538968071 (0x20200007)        to bfloat:
;     correct 0x4e01 (1.0000001b x 2^29), llc result 0x4e00
;   uitofp i64 0xC37FFF8C80000000            to bfloat:
;     correct 0x5f43, llc result 0x5f44
; fptrunc double -> bfloat on the same targets uses round-to-odd and is
; correct; the integer paths do not.
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s
;
; CHECK-LABEL: u64(
; CHECK: cvt.rn.f32.u64
; CHECK: cvt.rn.bf16.f32
; CHECK-LABEL: s32(
; CHECK: cvt.rn.f32.s32
; CHECK: cvt.rn.bf16.f32

define bfloat @u64(i64 %a) {
  %r = uitofp i64 %a to bfloat
  ret bfloat %r
}
define bfloat @s64(i64 %a) {
  %r = sitofp i64 %a to bfloat
  ret bfloat %r
}
define bfloat @u32(i32 %a) {
  %r = uitofp i32 %a to bfloat
  ret bfloat %r
}
define bfloat @s32(i32 %a) {
  %r = sitofp i32 %a to bfloat
  ret bfloat %r
}
