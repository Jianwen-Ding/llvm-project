; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-core-occupancy=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's tiled matrix product (mlir/utils/agx-kernels/bench/gemm_smem_tile):
; the machine scheduler hoists the inner loop's threadgroup loads ahead of
; their fmas (clustered since L15-E255), 28 -> 60 or more register halves for
; the region, which still lets 1024 threads in a threadgroup but halves the
; SIMD-groups resident on a core. A
; region of fewer than 4 device loads whose schedule cuts those by 1.8 times
; keeps its original order (L15-E239).

; CHECK: general purpose registers: 19 x 32-bit
; OFF: general purpose registers: 42 x 32-bit

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

@__wg_gemm_smem_tile_0 = internal unnamed_addr addrspace(3) global [256 x float] undef
@__wg_gemm_smem_tile_1 = internal unnamed_addr addrspace(3) global [256 x float] undef

; Function Attrs: norecurse nounwind
define agx_kernel void @gemm_smem_tile(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i64) align 8 captures(none) dereferenceable(8) %3) local_unnamed_addr #0 {
entry:
  %.val = load i64, ptr addrspace(4) %3, align 8
  %4 = tail call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %5 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %6 = tail call i32 @llvm.agx.thread.position.in.grid.y()
  %7 = tail call i64 @llvm.umin.i64(i64 %.val, i64 1024)
  %.not = icmp slt i64 %.val, 1
  br i1 %.not, label %entry.._crit_edge_crit_edge, label %.lr.ph

entry.._crit_edge_crit_edge:                      ; preds = %entry
  %.pre = shl i32 %6, 10
  br label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %8 = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %9 = shl i32 %6, 10
  %invariant.op = add i32 %8, %9
  %10 = shl i32 %4, 4
  %11 = add i32 %10, %8
  %12 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__wg_gemm_smem_tile_0, i32 %11
  %13 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__wg_gemm_smem_tile_1, i32 %11
  %14 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__wg_gemm_smem_tile_0, i32 %10
  %invariant.gep = getelementptr [4 x i8], ptr addrspace(3) @__wg_gemm_smem_tile_1, i32 %8
  %15 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 4
  %gep.1 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 64
  %16 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 8
  %gep.2 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 128
  %17 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 12
  %gep.3 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 192
  %18 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 16
  %gep.4 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 256
  %19 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 20
  %gep.5 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 320
  %20 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 24
  %gep.6 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 384
  %21 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 28
  %gep.7 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 448
  %22 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 32
  %gep.8 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 512
  %23 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 36
  %gep.9 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 576
  %24 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 40
  %gep.10 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 640
  %25 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 44
  %gep.11 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 704
  %26 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 48
  %gep.12 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 768
  %27 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 52
  %gep.13 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 832
  %28 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 56
  %gep.14 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 896
  %29 = getelementptr inbounds nuw i8, ptr addrspace(3) %14, i32 60
  %gep.15 = getelementptr i8, ptr addrspace(3) %invariant.gep, i32 960
  br label %30

30:                                               ; preds = %.lr.ph, %30
  %indvars.iv = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next, %30 ]
  %31 = phi float [ 0.000000e+00, %.lr.ph ], [ %89, %30 ]
  %32 = trunc nuw nsw i64 %indvars.iv to i32
  %.reass = add i32 %invariant.op, %32
  %33 = zext i32 %.reass to i64
  %34 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %33
  %35 = load float, ptr addrspace(1) %34, align 4
  store float %35, ptr addrspace(3) %12, align 4
  %36 = add i32 %4, %32
  %37 = shl i32 %36, 10
  %38 = add i32 %37, %5
  %39 = zext i32 %38 to i64
  %40 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %39
  %41 = load float, ptr addrspace(1) %40, align 4
  store float %41, ptr addrspace(3) %13, align 4
  fence syncscope("workgroup") release
  tail call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %42 = load float, ptr addrspace(3) %14, align 16
  %43 = load float, ptr addrspace(3) %invariant.gep, align 4
  %44 = tail call float @llvm.fma.f32(float %42, float %43, float %31)
  %45 = load float, ptr addrspace(3) %15, align 4
  %46 = load float, ptr addrspace(3) %gep.1, align 4
  %47 = tail call float @llvm.fma.f32(float %45, float %46, float %44)
  %48 = load float, ptr addrspace(3) %16, align 8
  %49 = load float, ptr addrspace(3) %gep.2, align 4
  %50 = tail call float @llvm.fma.f32(float %48, float %49, float %47)
  %51 = load float, ptr addrspace(3) %17, align 4
  %52 = load float, ptr addrspace(3) %gep.3, align 4
  %53 = tail call float @llvm.fma.f32(float %51, float %52, float %50)
  %54 = load float, ptr addrspace(3) %18, align 16
  %55 = load float, ptr addrspace(3) %gep.4, align 4
  %56 = tail call float @llvm.fma.f32(float %54, float %55, float %53)
  %57 = load float, ptr addrspace(3) %19, align 4
  %58 = load float, ptr addrspace(3) %gep.5, align 4
  %59 = tail call float @llvm.fma.f32(float %57, float %58, float %56)
  %60 = load float, ptr addrspace(3) %20, align 8
  %61 = load float, ptr addrspace(3) %gep.6, align 4
  %62 = tail call float @llvm.fma.f32(float %60, float %61, float %59)
  %63 = load float, ptr addrspace(3) %21, align 4
  %64 = load float, ptr addrspace(3) %gep.7, align 4
  %65 = tail call float @llvm.fma.f32(float %63, float %64, float %62)
  %66 = load float, ptr addrspace(3) %22, align 16
  %67 = load float, ptr addrspace(3) %gep.8, align 4
  %68 = tail call float @llvm.fma.f32(float %66, float %67, float %65)
  %69 = load float, ptr addrspace(3) %23, align 4
  %70 = load float, ptr addrspace(3) %gep.9, align 4
  %71 = tail call float @llvm.fma.f32(float %69, float %70, float %68)
  %72 = load float, ptr addrspace(3) %24, align 8
  %73 = load float, ptr addrspace(3) %gep.10, align 4
  %74 = tail call float @llvm.fma.f32(float %72, float %73, float %71)
  %75 = load float, ptr addrspace(3) %25, align 4
  %76 = load float, ptr addrspace(3) %gep.11, align 4
  %77 = tail call float @llvm.fma.f32(float %75, float %76, float %74)
  %78 = load float, ptr addrspace(3) %26, align 16
  %79 = load float, ptr addrspace(3) %gep.12, align 4
  %80 = tail call float @llvm.fma.f32(float %78, float %79, float %77)
  %81 = load float, ptr addrspace(3) %27, align 4
  %82 = load float, ptr addrspace(3) %gep.13, align 4
  %83 = tail call float @llvm.fma.f32(float %81, float %82, float %80)
  %84 = load float, ptr addrspace(3) %28, align 8
  %85 = load float, ptr addrspace(3) %gep.14, align 4
  %86 = tail call float @llvm.fma.f32(float %84, float %85, float %83)
  %87 = load float, ptr addrspace(3) %29, align 4
  %88 = load float, ptr addrspace(3) %gep.15, align 4
  %89 = tail call float @llvm.fma.f32(float %87, float %88, float %86)
  fence syncscope("workgroup") release
  tail call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 16
  %90 = icmp samesign ult i64 %indvars.iv.next, %7
  br i1 %90, label %30, label %._crit_edge

._crit_edge:                                      ; preds = %30, %entry.._crit_edge_crit_edge
  %.pre-phi = phi i32 [ %.pre, %entry.._crit_edge_crit_edge ], [ %9, %30 ]
  %.lcssa2 = phi float [ 0.000000e+00, %entry.._crit_edge_crit_edge ], [ %89, %30 ]
  %91 = add i32 %.pre-phi, %5
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %92
  store float %.lcssa2, ptr addrspace(1) %93, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.y() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.agx.threadgroup.barrier() #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #3

attributes #0 = { norecurse nounwind "agx-max-threads-per-threadgroup"="256" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }


