; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 < %s | FileCheck %s
;
; e^(a - u) and e^(b - u), u = maximum(a, b): one of the two arguments is
; x - x (+0, or a NaN for an infinite x), whose exponential is 1 + (x - x);
; one exponential of the other (L15-E464).

declare float @llvm.exp.f32(float)
declare float @llvm.maximum.f32(float, float)

; CHECK-LABEL: @pair(
; CHECK:       [[DA:%.*]] = fsub float %a, %u
; CHECK:       [[DB:%.*]] = fsub float %b, %u
; CHECK:       [[C:%.*]] = fcmp ogt float %b, %a
; CHECK:       select i1 [[C]], float [[DA]], float [[DB]]
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   fadd float {{.*}}, f0x4B400000
; CHECK:       %exp.one = select
; CHECK:       [[OA:%.*]] = fadd float 1.000000e+00, [[DA]]
; CHECK:       select i1 [[C]], float %exp.one, float [[OA]]
; CHECK:       [[OB:%.*]] = fadd float 1.000000e+00, [[DB]]
; CHECK:       select i1 [[C]], float [[OB]], float %exp.one
; CHECK:       ret void
define agx_kernel void @pair(ptr addrspace(1) %p, float %a, float %b) {
  %u = call float @llvm.maximum.f32(float %a, float %b)
  %da = fsub float %a, %u
  %db = fsub float %b, %u
  %ea = call float @llvm.exp.f32(float %da)
  %eb = call float @llvm.exp.f32(float %db)
  %s = fadd float %ea, %eb
  store float %s, ptr addrspace(1) %p
  ret void
}

; maxnum (C's fmaxf) likewise: it returns an operand and drops one NaN
; (L15-E466).
declare float @llvm.maxnum.f32(float, float)
; CHECK-LABEL: @pair_maxnum(
; CHECK:       fcmp ogt float %b, %a
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   fadd float {{.*}}, f0x4B400000
; CHECK:       ret void
define agx_kernel void @pair_maxnum(ptr addrspace(1) %p, float %a, float %b) {
  %u = call float @llvm.maxnum.f32(float %a, float %b)
  %da = fsub float %a, %u
  %db = fsub float %b, %u
  %ea = call float @llvm.exp.f32(float %da)
  %eb = call float @llvm.exp.f32(float %db)
  %s = fadd float %ea, %eb
  store float %s, ptr addrspace(1) %p
  ret void
}
