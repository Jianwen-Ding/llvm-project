; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 -agx-other-base-exps=0 -agx-exp2-below-max=0 -agx-exp2-bounds-by-arg=0 < %s | FileCheck %s --check-prefix=OFF
;
; 2^x and 10^x as exp in the online softmax rewrite (one exponential of the
; pair) and below a maximum: x <= 0 (or a NaN), so no clamp at 128 / 38.6,
; and n <= 0, so no +inf select nor NaN select after the scaling (L15-E484).
; (+0 by a compare of x and no clamp at all: L15-E485.)

declare float @llvm.exp.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.exp10.f32(float)
declare float @llvm.maximum.f32(float, float)

; CHECK-LABEL: @pair_exp2(
; CHECK:       [[C:%.*]] = fcmp ogt float %b, %a
; CHECK:       select i1 [[C]], float %da, float %db
; CHECK-NOT:   {{1.280000e\+02|-1.510000e\+02}}
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   +inf
; CHECK-NOT:   fcmp uno
; CHECK:       [[Z:%.*]] = fcmp ole float [[X:%.*]], f0xC2FC0001
; CHECK:       %exp.one = select i1 [[Z]], float 0.000000e+00
; CHECK:       ret void
; OFF-LABEL:   @pair_exp2(
; OFF:         fcmp ogt float {{.*}}, 1.280000e+02
; OFF:         fadd float {{.*}}, f0x4B400000
; OFF:         float +inf
; OFF:         fadd float {{.*}}, f0x4B400000
; OFF:         ret void
define agx_kernel void @pair_exp2(ptr addrspace(1) %p, float %a, float %b) {
  %u = call float @llvm.maximum.f32(float %a, float %b)
  %da = fsub float %a, %u
  %db = fsub float %b, %u
  %ea = call float @llvm.exp2.f32(float %da)
  %eb = call float @llvm.exp2.f32(float %db)
  %s = fadd float %ea, %eb
  store float %s, ptr addrspace(1) %p
  ret void
}

; Different bases are not a pair.
; CHECK-LABEL: @mixed(
; CHECK-NOT:   %exp.one
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK:       ret void
define agx_kernel void @mixed(ptr addrspace(1) %p, float %a, float %b) {
  %u = call float @llvm.maximum.f32(float %a, float %b)
  %da = fsub float %a, %u
  %db = fsub float %b, %u
  %ea = call float @llvm.exp.f32(float %da)
  %eb = call float @llvm.exp2.f32(float %db)
  %s = fadd float %ea, %eb
  store float %s, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @below_exp10(
; CHECK-NOT:   3.860000e+01
; CHECK:       fcmp ole float %da, f0xC217B819
; CHECK-NOT:   +inf
; CHECK-NOT:   fcmp uno
; CHECK:       ret void
; OFF-LABEL:   @below_exp10(
; OFF:         3.860000e+01
; OFF:         float +inf
; OFF:         ret void
define agx_kernel void @below_exp10(ptr addrspace(1) %p, float %a, float %b) {
  %u = call float @llvm.maximum.f32(float %a, float %b)
  %da = fsub float %a, %u
  %ea = call float @llvm.exp10.f32(float %da)
  store float %ea, ptr addrspace(1) %p
  ret void
}

; Anywhere else: the +inf and +0 selects (E485's compares of x).
; CHECK-LABEL: @any_exp2(
; CHECK:       fcmp oge float %a, 1.280000e+02
; CHECK:       float +inf
; CHECK:       fcmp ole float %a, f0xC2FC0001
; CHECK-NOT:   fcmp uno
; CHECK:       ret void
define agx_kernel void @any_exp2(ptr addrspace(1) %p, float %a) {
  %ea = call float @llvm.exp2.f32(float %a)
  store float %ea, ptr addrspace(1) %p
  ret void
}
