; Reproducer for ptx-tv finding 6: llvm.convert.from.arbitrary.fp quiets
; signaling NaNs (LangRef: "quiet NaNs remain quiet, signaling NaNs remain
; signaling"), both when constant folded and in the dynamic expansion.
; RUN: llc < %s -mtriple=nvptx64-unknown-unknown | FileCheck %s

; Float8E5M2 0x7D = S.11111.01: a signaling NaN.  Result must be a signaling
; f32 NaN (quiet bit clear), but is the quiet NaN 0x7FC00000.
; CHECK-LABEL: snan_const(
; CHECK: st.param{{.*}}.b32 [func_retval0], 2143289344;
define float @snan_const() {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 125, metadata !"Float8E5M2")
  ret float %r
}

; Dynamic: every NaN input yields the constant 0x7FC00000 (sign and quiet
; bit dropped).
; CHECK-LABEL: snan_dynamic(
; CHECK: selp.b32 {{%r[0-9]+}}, 2143289344,
define float @snan_dynamic(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E5M2")
  ret float %r
}

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
