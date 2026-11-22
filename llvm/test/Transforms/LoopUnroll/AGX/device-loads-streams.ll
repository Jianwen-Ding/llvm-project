; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-streams=0 -agx-device-unroll-streams-word=0 -S < %s | FileCheck %s --check-prefix=FOUR
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-streams-word=0 -S < %s | FileCheck %s --check-prefix=EIGHT
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-streams-vectors=0 -S < %s | FileCheck %s --check-prefix=SCALARS
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-streams-dword=0 -S < %s | FileCheck %s --check-prefix=DW8

; A device-load loop (device-loads.ll) whose loads step through two or more
; pointer bases, without stores (a dot product's row and vector), is
; unrolled 8 times where 8 divides its trip count; a loop of one base (a row
; sum) keeps 4 (L15-E506). Where every load is of 4 bytes at most, 16
; (L15-E507), or of 8 at most (L15-E750). Only where every load is a vector
; (L15-E508): a scalar dot product (newk gemv, rowdot) lost with 8 and 16.

define void @dot256(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {
; CHECK-LABEL: @dot256(
; CHECK-COUNT-16: load <4 x float>
; CHECK-NOT: load <4 x float>
; CHECK: br i1
; EIGHT-LABEL: @dot256(
; EIGHT-COUNT-16: load <4 x float>
; EIGHT-NOT: load <4 x float>
; FOUR-LABEL: @dot256(
; FOUR-COUNT-8: load <4 x float>
; FOUR-NOT: load <4 x float>
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi <4 x float> [ zeroinitializer, %entry ], [ %s1, %loop ]
  %ga = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i64 %i
  %gb = getelementptr inbounds <4 x float>, ptr addrspace(1) %b, i64 %i
  %u = load <4 x float>, ptr addrspace(1) %ga, align 16
  %v = load <4 x float>, ptr addrspace(1) %gb, align 16
  %s1 = call <4 x float> @llvm.fma.v4f32(<4 x float> %u, <4 x float> %v, <4 x float> %s)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  %r = extractelement <4 x float> %s1, i64 0
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @sum256(ptr addrspace(1) %a, ptr addrspace(1) %o) {
; CHECK-LABEL: @sum256(
; EIGHT-LABEL: @sum256(
; EIGHT-COUNT-4: load <4 x float>
; FOUR-LABEL: @sum256(
; FOUR-COUNT-4: load <4 x float>
; CHECK-COUNT-4: load <4 x float>
; CHECK-NOT: load <4 x float>
; CHECK: br i1
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi <4 x float> [ zeroinitializer, %entry ], [ %s1, %loop ]
  %ga = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i64 %i
  %u = load <4 x float>, ptr addrspace(1) %ga, align 16
  %s1 = fadd <4 x float> %s, %u
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  %r = extractelement <4 x float> %s1, i64 0
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

define void @dotf512(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {
; CHECK-LABEL: @dotf512(
; CHECK-COUNT-8: load float
; CHECK-NOT: load float
; CHECK: br i1
; SCALARS-LABEL: @dotf512(
; SCALARS-COUNT-32: load float
; SCALARS-NOT: load float
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s1, %loop ]
  %ga = getelementptr inbounds float, ptr addrspace(1) %a, i64 %i
  %gb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %i
  %u = load float, ptr addrspace(1) %ga, align 4
  %v = load float, ptr addrspace(1) %gb, align 4
  %s1 = call float @llvm.fma.f32(float %u, float %v, float %s)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 512
  br i1 %c, label %loop, label %exit
exit:
  store float %s1, ptr addrspace(1) %o, align 4
  ret void
}

define void @doti8x4(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {
; CHECK-LABEL: @doti8x4(
; CHECK-COUNT-32: load <4 x i8>
; CHECK-NOT: load <4 x i8>
; CHECK: br i1
; EIGHT-LABEL: @doti8x4(
; EIGHT-COUNT-16: load <4 x i8>
; EIGHT-NOT: load <4 x i8>
; FOUR-LABEL: @doti8x4(
; FOUR-COUNT-8: load <4 x i8>
; FOUR-NOT: load <4 x i8>
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi <4 x i32> [ zeroinitializer, %entry ], [ %s1, %loop ]
  %ga = getelementptr inbounds <4 x i8>, ptr addrspace(1) %a, i64 %i
  %gb = getelementptr inbounds <4 x i8>, ptr addrspace(1) %b, i64 %i
  %u = load <4 x i8>, ptr addrspace(1) %ga, align 4
  %v = load <4 x i8>, ptr addrspace(1) %gb, align 4
  %ue = sext <4 x i8> %u to <4 x i32>
  %ve = sext <4 x i8> %v to <4 x i32>
  %m = mul <4 x i32> %ue, %ve
  %s1 = add <4 x i32> %s, %m
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  %r = extractelement <4 x i32> %s1, i64 0
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}

; Of 8 bytes a load (four halves), 16 as well (L15-E750: dot_f16x4 -7.3%).
define void @doth4(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) {
; CHECK-LABEL: @doth4(
; CHECK-COUNT-32: load <4 x half>
; CHECK-NOT: load <4 x half>
; CHECK: br i1
; DW8-LABEL: @doth4(
; DW8-COUNT-16: load <4 x half>
; DW8-NOT: load <4 x half>
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %s = phi <4 x float> [ zeroinitializer, %entry ], [ %s1, %loop ]
  %ga = getelementptr inbounds <4 x half>, ptr addrspace(1) %a, i64 %i
  %gb = getelementptr inbounds <4 x half>, ptr addrspace(1) %b, i64 %i
  %u = load <4 x half>, ptr addrspace(1) %ga, align 8
  %v = load <4 x half>, ptr addrspace(1) %gb, align 8
  %ue = fpext <4 x half> %u to <4 x float>
  %ve = fpext <4 x half> %v to <4 x float>
  %s1 = call <4 x float> @llvm.fma.v4f32(<4 x float> %ue, <4 x float> %ve, <4 x float> %s)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %exit
exit:
  %r = extractelement <4 x float> %s1, i64 0
  store float %r, ptr addrspace(1) %o, align 4
  ret void
}

declare float @llvm.fma.f32(float, float, float)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
