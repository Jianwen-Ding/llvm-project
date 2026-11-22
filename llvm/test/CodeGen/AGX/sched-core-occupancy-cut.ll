; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sched-core-occupancy-cut=18 < %s | FileCheck %s --check-prefix=OLD
;
; MLIR's row_exp (y = exp(x - s) over rows of 512, a thread per row; mlir-opt,
; opt -O3): its unrolled loop of 16 exponentials, four 16-byte loads, is a
; region of few device loads for each instruction (agx-sched-core-occupancy),
; whose free order took 48 -> 74 halves, the core's SIMD-groups 68 -> 40 (a
; cut of 1.7 times): kept until L15-E761, the original order since (a cut of
; 1.2 times or more), 37 -> 29 registers, 44 -> 55 resident groups in
; agx-bench's count, tput 2608.8 -> 2472.3.

; CHECK: general purpose registers: 29 x 32-bit
; OLD: general purpose registers: 37 x 32-bit

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @row_exp(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readnone captures(none) %1, ptr addrspace(1) nofree readnone captures(none) %2, ptr addrspace(1) nofree writeonly captures(none) %3, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %4) local_unnamed_addr #0 {
entry:
  %.val = load float, ptr addrspace(4) %4, align 4
  %5 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %6 = shl i32 %5, 9
  %7 = zext i32 %6 to i64
  br label %8

8:                                                ; preds = %8, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.15, %8 ]
  %9 = add nuw nsw i64 %indvars.iv, %7
  %10 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %9
  %11 = load float, ptr addrspace(1) %10, align 4
  %12 = fsub float %11, %.val
  %13 = tail call float @llvm.exp.f32(float %12)
  %14 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %9
  store float %13, ptr addrspace(1) %14, align 4
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %15 = add nuw nsw i64 %indvars.iv.next, %7
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %15
  %17 = load float, ptr addrspace(1) %16, align 4
  %18 = fsub float %17, %.val
  %19 = tail call float @llvm.exp.f32(float %18)
  %20 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %15
  store float %19, ptr addrspace(1) %20, align 4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %21 = add nuw nsw i64 %indvars.iv.next.1, %7
  %22 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %21
  %23 = load float, ptr addrspace(1) %22, align 4
  %24 = fsub float %23, %.val
  %25 = tail call float @llvm.exp.f32(float %24)
  %26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %21
  store float %25, ptr addrspace(1) %26, align 4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %27 = add nuw nsw i64 %indvars.iv.next.2, %7
  %28 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4
  %30 = fsub float %29, %.val
  %31 = tail call float @llvm.exp.f32(float %30)
  %32 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %27
  store float %31, ptr addrspace(1) %32, align 4
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %33 = add nuw nsw i64 %indvars.iv.next.3, %7
  %34 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %33
  %35 = load float, ptr addrspace(1) %34, align 4
  %36 = fsub float %35, %.val
  %37 = tail call float @llvm.exp.f32(float %36)
  %38 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %33
  store float %37, ptr addrspace(1) %38, align 4
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %39 = add nuw nsw i64 %indvars.iv.next.4, %7
  %40 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %39
  %41 = load float, ptr addrspace(1) %40, align 4
  %42 = fsub float %41, %.val
  %43 = tail call float @llvm.exp.f32(float %42)
  %44 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %39
  store float %43, ptr addrspace(1) %44, align 4
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %45 = add nuw nsw i64 %indvars.iv.next.5, %7
  %46 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %45
  %47 = load float, ptr addrspace(1) %46, align 4
  %48 = fsub float %47, %.val
  %49 = tail call float @llvm.exp.f32(float %48)
  %50 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %45
  store float %49, ptr addrspace(1) %50, align 4
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %51 = add nuw nsw i64 %indvars.iv.next.6, %7
  %52 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %51
  %53 = load float, ptr addrspace(1) %52, align 4
  %54 = fsub float %53, %.val
  %55 = tail call float @llvm.exp.f32(float %54)
  %56 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %51
  store float %55, ptr addrspace(1) %56, align 4
  %indvars.iv.next.7 = or disjoint i64 %indvars.iv, 8
  %57 = add nuw nsw i64 %indvars.iv.next.7, %7
  %58 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %57
  %59 = load float, ptr addrspace(1) %58, align 4
  %60 = fsub float %59, %.val
  %61 = tail call float @llvm.exp.f32(float %60)
  %62 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %57
  store float %61, ptr addrspace(1) %62, align 4
  %indvars.iv.next.8 = or disjoint i64 %indvars.iv, 9
  %63 = add nuw nsw i64 %indvars.iv.next.8, %7
  %64 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %63
  %65 = load float, ptr addrspace(1) %64, align 4
  %66 = fsub float %65, %.val
  %67 = tail call float @llvm.exp.f32(float %66)
  %68 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %63
  store float %67, ptr addrspace(1) %68, align 4
  %indvars.iv.next.9 = or disjoint i64 %indvars.iv, 10
  %69 = add nuw nsw i64 %indvars.iv.next.9, %7
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %69
  %71 = load float, ptr addrspace(1) %70, align 4
  %72 = fsub float %71, %.val
  %73 = tail call float @llvm.exp.f32(float %72)
  %74 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %69
  store float %73, ptr addrspace(1) %74, align 4
  %indvars.iv.next.10 = or disjoint i64 %indvars.iv, 11
  %75 = add nuw nsw i64 %indvars.iv.next.10, %7
  %76 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %75
  %77 = load float, ptr addrspace(1) %76, align 4
  %78 = fsub float %77, %.val
  %79 = tail call float @llvm.exp.f32(float %78)
  %80 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %75
  store float %79, ptr addrspace(1) %80, align 4
  %indvars.iv.next.11 = or disjoint i64 %indvars.iv, 12
  %81 = add nuw nsw i64 %indvars.iv.next.11, %7
  %82 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %81
  %83 = load float, ptr addrspace(1) %82, align 4
  %84 = fsub float %83, %.val
  %85 = tail call float @llvm.exp.f32(float %84)
  %86 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %81
  store float %85, ptr addrspace(1) %86, align 4
  %indvars.iv.next.12 = or disjoint i64 %indvars.iv, 13
  %87 = add nuw nsw i64 %indvars.iv.next.12, %7
  %88 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %87
  %89 = load float, ptr addrspace(1) %88, align 4
  %90 = fsub float %89, %.val
  %91 = tail call float @llvm.exp.f32(float %90)
  %92 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %87
  store float %91, ptr addrspace(1) %92, align 4
  %indvars.iv.next.13 = or disjoint i64 %indvars.iv, 14
  %93 = add nuw nsw i64 %indvars.iv.next.13, %7
  %94 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %93
  %95 = load float, ptr addrspace(1) %94, align 4
  %96 = fsub float %95, %.val
  %97 = tail call float @llvm.exp.f32(float %96)
  %98 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %93
  store float %97, ptr addrspace(1) %98, align 4
  %indvars.iv.next.14 = or disjoint i64 %indvars.iv, 15
  %99 = add nuw nsw i64 %indvars.iv.next.14, %7
  %100 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %99
  %101 = load float, ptr addrspace(1) %100, align 4
  %102 = fsub float %101, %.val
  %103 = tail call float @llvm.exp.f32(float %102)
  %104 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %99
  store float %103, ptr addrspace(1) %104, align 4
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %exitcond.not.15 = icmp eq i64 %indvars.iv.next.15, 512
  br i1 %exitcond.not.15, label %105, label %8

105:                                              ; preds = %8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
