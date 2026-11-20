; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-full-trips=0 -S < %s | FileCheck %s --check-prefix=OFF

; A device-load loop of at most 32 trips is unrolled whole (L15-E545): a row
; pass of a SIMD-group's lane (32 trips, a 128-byte stride) that a later pass,
; also whole, reads again takes its loads' and exponentials' values. Before,
; the strided factor (16) left it a loop.

; CHECK-LABEL: @sum_exp(
; CHECK-NOT: br i1
; CHECK-COUNT-32: call float @llvm.exp.f32
; CHECK-NOT: call float @llvm.exp.f32
; OFF-LABEL: @sum_exp(
; OFF-COUNT-16: call float @llvm.exp.f32
; OFF-NOT: call float @llvm.exp.f32
; OFF: br i1

; A running maxnum stays a loop (online_softmax_exp2_maxnum lost whole).
; CHECK-LABEL: @online_max(
; CHECK: br i1
; CHECK: ret void
define void @sum_exp(ptr addrspace(1) %p, ptr addrspace(1) %o, i64 %lane, float %m) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %k = mul nuw nsw i64 %i, 32
  %j = add nuw nsw i64 %k, %lane
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %j
  %v = load float, ptr addrspace(1) %g, align 4
  %d = fsub float %v, %m
  %e = call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %e
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 32
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

define void @online_max(ptr addrspace(1) %p, ptr addrspace(1) %o, i64 %lane) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %k = mul nuw nsw i64 %i, 32
  %j = add nuw nsw i64 %k, %lane
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %j
  %v = load float, ptr addrspace(1) %g, align 4
  %m1 = call float @llvm.maxnum.f32(float %m, float %v)
  %d = fsub float %m, %m1
  %sc = call float @llvm.exp2.f32(float %d)
  %e0 = fsub float %v, %m1
  %ev = call float @llvm.exp2.f32(float %e0)
  %s1 = call float @llvm.fma.f32(float %s, float %sc, float %ev)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 32
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}
declare float @llvm.exp.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.fma.f32(float, float, float)
