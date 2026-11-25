; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-mark-exps-below-max -S < %s | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=agx-mark-exps-below-max -agx-mark-exps-below-max-early=0 -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -O3 -S < %s | FileCheck %s --check-prefix=O3

; A row's softmax sum: e^(x_j - m), m the maximum another loop took of the
; same loads, is at most 0 or a NaN (AGXCodeGenPrepare's expBelowLoopMax:
; its overflow clamp needless). Marked in opt before the loop unroller: llc's
; own match takes one maximum a header phi updates, not an unrolled loop's
; chain of them and its remainder loop (softmax_f16_dyn -4.9%, softmax_f32v
; -12.3% tput, L15-E731). Every unrolled copy keeps the mark.
; CHECK-LABEL: define agx_kernel void @softmax_sum(
; CHECK: call float @llvm.exp.f32(float %d), !agx.exp.below.max
; OFF-LABEL: define agx_kernel void @softmax_sum(
; OFF-NOT: agx.exp.below.max
; O3-LABEL: define agx_kernel void @softmax_sum(
; O3-NOT: call float @llvm.exp.f32({{.*}}){{$}}
; O3: call float @llvm.exp.f32({{.*}}), !agx.exp.below.max
; O3-NOT: call float @llvm.exp.f32({{.*}}){{$}}
; O3: ret void
; Not the exps of another row than the maximum's.
; CHECK-LABEL: define agx_kernel void @other_row(
; CHECK-NOT: agx.exp.below.max
; CHECK: ret void

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.exp.f32(float)

define agx_kernel void @softmax_sum(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %max, label %out
max:
  %i = phi i32 [ 0, %entry ], [ %i1, %max ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %max ]
  %k = add nuw i32 %row, %i
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = call float @llvm.maxnum.f32(float %m, float %v)
  %i1 = add nuw nsw i32 %i, 1
  %c1 = icmp slt i32 %i1, %n
  br i1 %c1, label %max, label %sum.pre
sum.pre:
  br label %sum
sum:
  %j = phi i32 [ 0, %sum.pre ], [ %j1, %sum ]
  %s = phi float [ 0.0, %sum.pre ], [ %s1, %sum ]
  %k2 = add nuw i32 %row, %j
  %z2 = zext i32 %k2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z2
  %w = load float, ptr addrspace(1) %p2, align 4
  %d = fsub float %w, %m1
  %e = call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %e
  %j1 = add nuw nsw i32 %j, 1
  %c2 = icmp slt i32 %j1, %n
  br i1 %c2, label %sum, label %done
done:
  %tz = zext i32 %t to i64
  %q = getelementptr inbounds float, ptr addrspace(1) %o, i64 %tz
  store float %s1, ptr addrspace(1) %q, align 4
  br label %out
out:
  ret void
}

define agx_kernel void @other_row(ptr addrspace(1) %x, ptr addrspace(1) %o, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %max, label %out
max:
  %i = phi i32 [ 0, %entry ], [ %i1, %max ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m1, %max ]
  %k = add nuw i32 %row, %i
  %z = zext i32 %k to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %m1 = call float @llvm.maxnum.f32(float %m, float %v)
  %i1 = add nuw nsw i32 %i, 1
  %c1 = icmp slt i32 %i1, %n
  br i1 %c1, label %max, label %sum.pre
sum.pre:
  br label %sum
sum:
  %j = phi i32 [ 0, %sum.pre ], [ %j1, %sum ]
  %s = phi float [ 0.0, %sum.pre ], [ %s1, %sum ]
  %r2 = add nuw i32 %row, 512
  %k2 = add nuw i32 %r2, %j
  %z2 = zext i32 %k2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %z2
  %w = load float, ptr addrspace(1) %p2, align 4
  %d = fsub float %w, %m1
  %e = call float @llvm.exp.f32(float %d)
  %s1 = fadd float %s, %e
  %j1 = add nuw nsw i32 %j, 1
  %c2 = icmp slt i32 %j1, %n
  br i1 %c2, label %sum, label %done
done:
  %tz = zext i32 %t to i64
  %q = getelementptr inbounds float, ptr addrspace(1) %o, i64 %tz
  store float %s1, ptr addrspace(1) %q, align 4
  br label %out
out:
  ret void
}
