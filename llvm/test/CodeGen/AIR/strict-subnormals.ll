; Under air.ieee-fp32 the Metal pipeline tells subnormals apart on the bits, where AIR's
; functions read them as zero. Without the flag, and in portable AIR, the calls stay alone.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE

; CHECK-LABEL: define float @c(
; CHECK: call float @air.ceil.f32(float %x)
; CHECK: bitcast float %x to i32
; CHECK: select i1 %{{.*}}, float -0.000000e+00, float 1.000000e+00
; CHECK: ret float
; PORTABLE-LABEL: define float @c(
; PORTABLE: call float @air.ceil.f32(
; PORTABLE-NOT: bitcast
; PORTABLE: ret float
define float @c(float %x) {
  %r = call float @llvm.ceil.f32(float %x)
  ret float %r
}

; CHECK-LABEL: define float @l(
; CHECK: call {{.*}}float @air.log2.f32(float %x)
; CHECK: uitofp i32 %{{.*}} to float
; CHECK: call {{.*}}float @air.log2.f32(float %{{.*}})
; CHECK: fadd float %{{.*}}, -1.490000e+02
define float @l(float %x) {
  %r = call afn float @air.fast_log2.f32(float %x)
  ret float %r
}

; CHECK-LABEL: define float @e(
; CHECK: call {{.*}}float @air.exp2.f32(float %x)
; CHECK: fptosi float %{{.*}} to i32
; CHECK: shl i32 %{{.*}}, 23
define float @e(float %x) {
  %r = call afn float @air.fast_exp2.f32(float %x)
  ret float %r
}

declare float @air.fast_exp2.f32(float)
declare float @llvm.ceil.f32(float)
declare float @air.fast_log2.f32(float)
!llvm.module.flags = !{!0}
!0 = !{i32 4, !"air.ieee-fp32", i32 1}
