; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -o /dev/null -print-after=agx-codegenprepare < %s 2>&1 | FileCheck %s

; atan2 reduces to atan of min(|x|, |y|) / max(|x|, |y|) (0 / 0 and inf / inf
; replaced), which is at most 1 or NaN: the range around pi / 2 (above
; tan(3 pi / 8)) never applies there, so its test and selects go; atan keeps
; it (L15-E310).

; CHECK-LABEL: @a2(
; CHECK: fcmp ogt float {{.*}}, f0x3ED413CD
; CHECK-NOT: f0x401A827A
; CHECK-LABEL: @a1(
; CHECK: fcmp ogt float {{.*}}, f0x401A827A
define agx_kernel void @a2(ptr addrspace(1) %p, float %x, float %y) {
  %e = call float @llvm.atan2.f32(float %y, float %x)
  store float %e, ptr addrspace(1) %p
  ret void
}
define agx_kernel void @a1(ptr addrspace(1) %p, float %x) {
  %e = call float @llvm.atan.f32(float %x)
  store float %e, ptr addrspace(1) %p
  ret void
}
