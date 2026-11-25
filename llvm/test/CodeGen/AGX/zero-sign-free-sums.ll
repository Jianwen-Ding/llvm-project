; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 -agx-zero-sign-free-sums=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare -agx-deferred-minmax-reductions=0 -agx-zero-sign-free-ordered-max=0 < %s | FileCheck %s --check-prefix=ORD
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

; MLIR's maximumf subgroup reduction: m's bits in an order key (the sign
; mask's xor, NaN as 0x7fc00000), the SIMD-group's signed maximum, decoded:
; m's zero sign only orders -0 below +0 there, so m is free of it when the
; decoded maximum is: in e^(m - M) yes (L15-E477), stored negated no.
declare i32 @llvm.agx.simd.reduce.i32(i32, i32)
declare float @llvm.exp.f32(float)

define internal float @ordered_max(float %m) {
  %u = fcmp uno float %m, 0.0
  %b = bitcast float %m to i32
  %n = icmp slt i32 %b, 0
  %s = select i1 %n, i32 2147483647, i32 0
  %k = xor i32 %s, %b
  %c = select i1 %u, i32 2143289344, i32 %k
  %r = call i32 @llvm.agx.simd.reduce.i32(i32 %c, i32 22)
  %rn = icmp slt i32 %r, 0
  %rs = select i1 %rn, i32 2147483647, i32 0
  %d = xor i32 %rs, %r
  %f = bitcast i32 %d to float
  ret float %f
}

; CHECK-LABEL: @ordered_exp(
; CHECK: call nsz float @llvm.maximum.f32
; OFF-LABEL: @ordered_exp(
; OFF: call nsz float @llvm.maximum.f32
; ORD-LABEL: @ordered_exp(
; ORD-NOT: call nsz float @llvm.maximum.f32
; ORD: ret void
define agx_kernel void @ordered_exp(ptr addrspace(1) %p, float %a, float %b) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  %M = call float @ordered_max(float %m)
  %d = fsub float %m, %M
  %e = call float @llvm.exp.f32(float %d)
  store float %e, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @ordered_neg(
; CHECK-NOT: call nsz float @llvm.maximum.f32
define agx_kernel void @ordered_neg(ptr addrspace(1) %p, float %a, float %b) {
  %m = call float @llvm.maximum.f32(float %a, float %b)
  %M = call float @ordered_max(float %m)
  %n = fneg float %M
  store float %n, ptr addrspace(1) %p
  ret void
}
