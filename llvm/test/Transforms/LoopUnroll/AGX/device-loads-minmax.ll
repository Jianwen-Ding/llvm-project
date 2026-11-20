; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes='loop-unroll' -agx-device-unroll-keep-loop=0 -S < %s | FileCheck %s --check-prefix=WHOLE

; A device-load loop of a loop-carried llvm.maximum and more than 16 trips
; stays a loop (L15-E511): its factor (the narrow 32 here) at most half the
; trip count and no full unrolling, as AGXCodeGenPrepare rewrites the
; maximum's selects only in a loop.

define void @maxh32(ptr addrspace(1) %p, ptr addrspace(1) %o) {
; CHECK-LABEL: @maxh32(
; CHECK-COUNT-16: load half
; CHECK-NOT: load half
; CHECK: br i1
; WHOLE-LABEL: @maxh32(
; WHOLE-COUNT-32: load half
; WHOLE-NOT: br i1
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %loop ]
  %k = mul nuw nsw i64 %i, 32
  %g = getelementptr inbounds half, ptr addrspace(1) %p, i64 %k
  %v = load half, ptr addrspace(1) %g, align 2
  %e = fpext half %v to float
  %m1 = call float @llvm.maximum.f32(float %m, float %e)
  %i1 = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i1, 32
  br i1 %c, label %loop, label %exit
exit:
  store float %m1, ptr addrspace(1) %o, align 4
  ret void
}

declare float @llvm.maximum.f32(float, float)
