; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 -agx-online-softmax-shared-one=0 < %s | FileCheck %s --check-prefix=OFF
;
; e^(a - u) and e^(b - u), u = maximum(a, b): one of the two arguments is
; x - x (+0, or a NaN for an infinite x), whose exponential is 1 + (x - x);
; one exponential of the other (L15-E464). For maximum that 1 + (x - x) is
; u 0 + 1, one fma for both sides (L15-E513).

declare float @llvm.exp.f32(float)
declare float @llvm.maximum.f32(float, float)

; CHECK-LABEL: @pair(
; CHECK:       [[DA:%.*]] = fsub float %a, %u
; CHECK:       [[DB:%.*]] = fsub float %b, %u
; CHECK:       [[C:%.*]] = fcmp ogt float %b, %a
; CHECK:       select i1 [[C]], float [[DA]], float [[DB]]
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   fadd float {{.*}}, f0x4B400000
; CHECK:       %exp.one = {{select|bitcast}}
; CHECK:       [[ONE:%.*]] = call float @llvm.fma.f32(float %u, float 0.000000e+00, float 1.000000e+00)
; CHECK:       select i1 [[C]], float %exp.one, float [[ONE]]
; CHECK:       select i1 [[C]], float [[ONE]], float %exp.one
; CHECK-NOT:   fadd float 1.000000e+00
; CHECK:       ret void
; OFF-LABEL: @pair(
; OFF:       [[DA:%.*]] = fsub float %a, %u
; OFF:       [[DB:%.*]] = fsub float %b, %u
; OFF:       [[C:%.*]] = fcmp ogt float %b, %a
; OFF:       %exp.one = {{select|bitcast}}
; OFF:       [[OA:%.*]] = fadd float 1.000000e+00, [[DA]]
; OFF:       select i1 [[C]], float %exp.one, float [[OA]]
; OFF:       [[OB:%.*]] = fadd float 1.000000e+00, [[DB]]
; OFF:       select i1 [[C]], float [[OB]], float %exp.one
; OFF:       ret void
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
; (L15-E466). Its trivial sides stay 1 + (x - u): a NaN a gives u = b, and
; u 0 + 1 would be 1.
declare float @llvm.maxnum.f32(float, float)
; CHECK-LABEL: @pair_maxnum(
; CHECK:       fcmp ogt float %b, %a
; CHECK:       fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   fadd float {{.*}}, f0x4B400000
; CHECK-NOT:   @llvm.fma.f32(float %u, float 0.000000e+00
; CHECK:       fadd float 1.000000e+00
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
