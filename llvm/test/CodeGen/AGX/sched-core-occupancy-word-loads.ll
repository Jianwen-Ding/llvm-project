; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sched-core-occupancy-many-cut=0 < %s | FileCheck %s --check-prefix=FREE
;
; MLIR's matmul_naive (a thread per element of C, A's row and B's column
; read a word at a time; mlir-opt, opt -O3): its unrolled K loop is 20 word
; loads in 41 instructions, which the free order issues all ahead of their
; fmas, the core's SIMD-groups 96 -> 40 (FREE). A region of at most 3
; instructions a word load keeps its original order at a cut of 2 times or
; more (L15-E764): 29 registers, 55 resident groups in agx-bench's count,
; tput 345.7 -> 293.4.

; CHECK: general purpose registers: 29 x 32-bit
; FREE: general purpose registers: 41 x 32-bit

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @matmul_naive(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2) local_unnamed_addr #0 {
entry:
  %3 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %4 = tail call i32 @llvm.agx.thread.position.in.grid.y()
  %5 = shl i32 %4, 8
  %6 = zext i32 %5 to i64
  %7 = zext i32 %3 to i64
  %invariant.gep = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %6
  %invariant.gep1 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %7
  br label %8

8:                                                ; preds = %8, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.15, %8 ]
  %9 = phi float [ 0.000000e+00, %entry ], [ %57, %8 ]
  %gep = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv
  %10 = load float, ptr addrspace(1) %gep, align 4
  %.idx = shl nuw nsw i64 %indvars.iv, 10
  %gep2 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx
  %11 = load float, ptr addrspace(1) %gep2, align 4
  %12 = tail call float @llvm.fma.f32(float %10, float %11, float %9)
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %gep.1 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next
  %13 = load float, ptr addrspace(1) %gep.1, align 4
  %.idx.1 = shl nuw nsw i64 %indvars.iv.next, 10
  %gep2.1 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.1
  %14 = load float, ptr addrspace(1) %gep2.1, align 4
  %15 = tail call float @llvm.fma.f32(float %13, float %14, float %12)
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %gep.2 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.1
  %16 = load float, ptr addrspace(1) %gep.2, align 4
  %.idx.2 = shl nuw nsw i64 %indvars.iv.next.1, 10
  %gep2.2 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.2
  %17 = load float, ptr addrspace(1) %gep2.2, align 4
  %18 = tail call float @llvm.fma.f32(float %16, float %17, float %15)
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %gep.3 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.2
  %19 = load float, ptr addrspace(1) %gep.3, align 4
  %.idx.3 = shl nuw nsw i64 %indvars.iv.next.2, 10
  %gep2.3 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.3
  %20 = load float, ptr addrspace(1) %gep2.3, align 4
  %21 = tail call float @llvm.fma.f32(float %19, float %20, float %18)
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %gep.4 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.3
  %22 = load float, ptr addrspace(1) %gep.4, align 4
  %.idx.4 = shl nuw nsw i64 %indvars.iv.next.3, 10
  %gep2.4 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.4
  %23 = load float, ptr addrspace(1) %gep2.4, align 4
  %24 = tail call float @llvm.fma.f32(float %22, float %23, float %21)
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %gep.5 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.4
  %25 = load float, ptr addrspace(1) %gep.5, align 4
  %.idx.5 = shl nuw nsw i64 %indvars.iv.next.4, 10
  %gep2.5 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.5
  %26 = load float, ptr addrspace(1) %gep2.5, align 4
  %27 = tail call float @llvm.fma.f32(float %25, float %26, float %24)
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %gep.6 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.5
  %28 = load float, ptr addrspace(1) %gep.6, align 4
  %.idx.6 = shl nuw nsw i64 %indvars.iv.next.5, 10
  %gep2.6 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.6
  %29 = load float, ptr addrspace(1) %gep2.6, align 4
  %30 = tail call float @llvm.fma.f32(float %28, float %29, float %27)
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %gep.7 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.6
  %31 = load float, ptr addrspace(1) %gep.7, align 4
  %.idx.7 = shl nuw nsw i64 %indvars.iv.next.6, 10
  %gep2.7 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.7
  %32 = load float, ptr addrspace(1) %gep2.7, align 4
  %33 = tail call float @llvm.fma.f32(float %31, float %32, float %30)
  %indvars.iv.next.7 = or disjoint i64 %indvars.iv, 8
  %gep.8 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.7
  %34 = load float, ptr addrspace(1) %gep.8, align 4
  %.idx.8 = shl nuw nsw i64 %indvars.iv.next.7, 10
  %gep2.8 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.8
  %35 = load float, ptr addrspace(1) %gep2.8, align 4
  %36 = tail call float @llvm.fma.f32(float %34, float %35, float %33)
  %indvars.iv.next.8 = or disjoint i64 %indvars.iv, 9
  %gep.9 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.8
  %37 = load float, ptr addrspace(1) %gep.9, align 4
  %.idx.9 = shl nuw nsw i64 %indvars.iv.next.8, 10
  %gep2.9 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.9
  %38 = load float, ptr addrspace(1) %gep2.9, align 4
  %39 = tail call float @llvm.fma.f32(float %37, float %38, float %36)
  %indvars.iv.next.9 = or disjoint i64 %indvars.iv, 10
  %gep.10 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.9
  %40 = load float, ptr addrspace(1) %gep.10, align 4
  %.idx.10 = shl nuw nsw i64 %indvars.iv.next.9, 10
  %gep2.10 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.10
  %41 = load float, ptr addrspace(1) %gep2.10, align 4
  %42 = tail call float @llvm.fma.f32(float %40, float %41, float %39)
  %indvars.iv.next.10 = or disjoint i64 %indvars.iv, 11
  %gep.11 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.10
  %43 = load float, ptr addrspace(1) %gep.11, align 4
  %.idx.11 = shl nuw nsw i64 %indvars.iv.next.10, 10
  %gep2.11 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.11
  %44 = load float, ptr addrspace(1) %gep2.11, align 4
  %45 = tail call float @llvm.fma.f32(float %43, float %44, float %42)
  %indvars.iv.next.11 = or disjoint i64 %indvars.iv, 12
  %gep.12 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.11
  %46 = load float, ptr addrspace(1) %gep.12, align 4
  %.idx.12 = shl nuw nsw i64 %indvars.iv.next.11, 10
  %gep2.12 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.12
  %47 = load float, ptr addrspace(1) %gep2.12, align 4
  %48 = tail call float @llvm.fma.f32(float %46, float %47, float %45)
  %indvars.iv.next.12 = or disjoint i64 %indvars.iv, 13
  %gep.13 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.12
  %49 = load float, ptr addrspace(1) %gep.13, align 4
  %.idx.13 = shl nuw nsw i64 %indvars.iv.next.12, 10
  %gep2.13 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.13
  %50 = load float, ptr addrspace(1) %gep2.13, align 4
  %51 = tail call float @llvm.fma.f32(float %49, float %50, float %48)
  %indvars.iv.next.13 = or disjoint i64 %indvars.iv, 14
  %gep.14 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.13
  %52 = load float, ptr addrspace(1) %gep.14, align 4
  %.idx.14 = shl nuw nsw i64 %indvars.iv.next.13, 10
  %gep2.14 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.14
  %53 = load float, ptr addrspace(1) %gep2.14, align 4
  %54 = tail call float @llvm.fma.f32(float %52, float %53, float %51)
  %indvars.iv.next.14 = or disjoint i64 %indvars.iv, 15
  %gep.15 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.14
  %55 = load float, ptr addrspace(1) %gep.15, align 4
  %.idx.15 = shl nuw nsw i64 %indvars.iv.next.14, 10
  %gep2.15 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep1, i64 %.idx.15
  %56 = load float, ptr addrspace(1) %gep2.15, align 4
  %57 = tail call float @llvm.fma.f32(float %55, float %56, float %54)
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %exitcond.not.15 = icmp eq i64 %indvars.iv.next.15, 256
  br i1 %exitcond.not.15, label %58, label %8

58:                                               ; preds = %8
  %59 = add i32 %5, %3
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %60
  store float %57, ptr addrspace(1) %61, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.y() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threadgroups-per-grid"="16,16,1" "agx-threads-per-grid"="256,256,1" "agx-threads-per-threadgroup"="16,16,1" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
