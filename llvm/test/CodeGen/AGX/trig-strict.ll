; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s --check-prefix=FAST
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 < %s | FileCheck %s --check-prefix=STRICT

; cos(x) is sin of a quadrant ahead. By default (fast) the quarter turn is
; added to the turns; with ieee-fp32 (strict floating point, which takes what
; costs time) to the quadrants, q + 1 or q - 3 past 3, within PTX's 2^-20.4 on
; [-2pi, 2pi].
; FAST-LABEL: cos_k:
; FAST:       fadd32 {{r[0-9]+}}, {{r[0-9]+}}, 0.25
; FAST-NOT:   3.0
; FAST:       sin_pt_1
; STRICT-LABEL: cos_k:
; STRICT-NOT:   0.25
; STRICT:       fcmpsel gte, {{r[0-9]+}}, {{r[0-9]+}}, 3.0
; STRICT:       sin_pt_1
define agx_kernel void @cos_k(ptr addrspace(1) %out, float %x) {
  %c = call afn float @llvm.cos.f32(float %x)
  store float %c, ptr addrspace(1) %out
  ret void
}
declare float @llvm.cos.f32(float)
