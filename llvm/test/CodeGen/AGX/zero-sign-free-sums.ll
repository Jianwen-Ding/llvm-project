; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 -agx-zero-sign-free-sums=0 < %s | FileCheck %s --check-prefix=OFF
;
; A maximum's zero sign does not matter in a + b where b is never -0 (+-0 + b
; is the same then), in c - a where c is never -0, nor in a - c where c is
; never +0: the maximum is nsz (no fix ordering -0 below +0). A log-sum-exp's
; m + log(s), s >= 1 (L15-E475).

declare float @llvm.maximum.f32(float, float)
declare float @llvm.log.f32(float)
declare float @llvm.fabs.f32(float)

; m + log(1 + |s|): log of at least 1 is never -0
; CHECK-LABEL: @lse(
; CHECK: call nsz float @llvm.maximum.f32
; OFF-LABEL: @lse(
; OFF-NOT: call nsz float @llvm.maximum.f32
define agx_kernel void @lse(ptr addrspace(1) %p, float %a, float %b, float %s) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  %as = call float @llvm.fabs.f32(float %s)
  %s1 = fadd float %as, 1.0
  %l = call float @llvm.log.f32(float %s1)
  %r = fadd float %m, %l
  store float %r, ptr addrspace(1) %p
  ret void
}

; |c| - m: |c| is never -0
; CHECK-LABEL: @sub_from(
; CHECK: call nsz float @llvm.maximum.f32
define agx_kernel void @sub_from(ptr addrspace(1) %p, float %a, float %b, float %c) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  %ac = call float @llvm.fabs.f32(float %c)
  %r = fsub float %ac, %m
  store float %r, ptr addrspace(1) %p
  ret void
}

; m - |c|: |c| may be +0, and -0 - +0 is -0, +0 - +0 is +0: the sign stays
; CHECK-LABEL: @sub_of(
; CHECK-NOT: call nsz float @llvm.maximum.f32
define agx_kernel void @sub_of(ptr addrspace(1) %p, float %a, float %b, float %c) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  %ac = call float @llvm.fabs.f32(float %c)
  %r = fsub float %m, %ac
  store float %r, ptr addrspace(1) %p
  ret void
}
