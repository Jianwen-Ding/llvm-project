; llvm.fabs must only clear the sign bit (LangRef: "if the input is a NaN,
; then the quiet/signaling bit and payload are perfectly preserved").
; Without native abs.f16 (PTX < 6.5) / abs.bf16 (< sm_80), the backend
; computes |x| in f32 and converts back; the bf16 path explicitly sets the
; quiet bit (or 0x400000) for NaN results, so a signaling NaN input is not
; preserved.  A bitwise and.b16 with 0x7fff is exact on every target (it is
; what the <2 x half> path already does).
;
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | FileCheck %s
;
; CHECK-LABEL: fabs_h(
; CHECK: cvt.f32.f16
; CHECK: abs.f32
; CHECK: cvt.rn.f16.f32
; CHECK-LABEL: fabs_bf(
; CHECK: or.b32 {{%r[0-9]+}}, {{%r[0-9]+}}, 4194304;
; CHECK-LABEL: fabs_v2h(
; CHECK: and.b32 {{%r[0-9]+}}, {{%r[0-9]+}}, 2147450879;

define half @fabs_h(half %a) {
  %r = call half @llvm.fabs.f16(half %a)
  ret half %r
}
define bfloat @fabs_bf(bfloat %a) {
  %r = call bfloat @llvm.fabs.bf16(bfloat %a)
  ret bfloat %r
}
define <2 x half> @fabs_v2h(<2 x half> %a) {
  %r = call <2 x half> @llvm.fabs.v2f16(<2 x half> %a)
  ret <2 x half> %r
}
