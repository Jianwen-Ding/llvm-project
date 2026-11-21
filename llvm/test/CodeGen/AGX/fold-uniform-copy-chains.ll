; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs -agx-fold-uniform-copy-chains=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's row_axpy_dyn: the scale, a kernel argument, reaches the versioned
; loop through copies across blocks; its fmas read the argument's uniform
; register itself, not a general register copied from it, two register
; reads each, not three (L15-E687: tput -25.7%).

; CHECK-LABEL: row_axpy_dyn:
; CHECK: .LBB0_4:
; CHECK-COUNT-8: fmadd32 r{{[0-9]+}}, u7, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: jmp_exec_any .LBB0_4
; OFF-LABEL: row_axpy_dyn:
; OFF: .LBB0_4:
; OFF: fmadd32 r{{[0-9]+}}, r1, r{{[0-9]+}}, r{{[0-9]+}}

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @row_axpy_dyn(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 captures(none) dereferenceable(4) %3, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %4) local_unnamed_addr #0 {
entry:
  %.val = load i32, ptr addrspace(4) %3, align 4
  %.val1 = load float, ptr addrspace(4) %4, align 4
  %5 = icmp sgt i32 %.val, 0
  br i1 %5, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %6 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %7 = shl i32 %6, 9
  %wide.trip.count = zext nneg i32 %.val to i64
  %xtraiter = and i64 %wide.trip.count, 7
  %8 = icmp ult i32 %.val, 8
  br i1 %8, label %.epil.preheader, label %.lr.ph.new

.lr.ph.new:                                       ; preds = %.lr.ph
  %unroll_iter = and i64 %wide.trip.count, 2147483640
  br label %9

9:                                                ; preds = %9, %.lr.ph.new
  %indvars.iv = phi i64 [ 0, %.lr.ph.new ], [ %indvars.iv.next.7, %9 ]
  %niter = phi i64 [ 0, %.lr.ph.new ], [ %niter.next.7, %9 ]
  %10 = trunc nuw nsw i64 %indvars.iv to i32
  %11 = add i32 %7, %10
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %12
  %14 = load float, ptr addrspace(1) %13, align 4
  %15 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %12
  %16 = load float, ptr addrspace(1) %15, align 4
  %17 = tail call float @llvm.fma.f32(float %.val1, float %14, float %16)
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %12
  store float %17, ptr addrspace(1) %18, align 4
  %19 = trunc i64 %indvars.iv to i32
  %20 = or disjoint i32 %19, 1
  %21 = add i32 %7, %20
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %22
  %24 = load float, ptr addrspace(1) %23, align 4
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %22
  %26 = load float, ptr addrspace(1) %25, align 4
  %27 = tail call float @llvm.fma.f32(float %.val1, float %24, float %26)
  %28 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %22
  store float %27, ptr addrspace(1) %28, align 4
  %29 = trunc i64 %indvars.iv to i32
  %30 = or disjoint i32 %29, 2
  %31 = add i32 %7, %30
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %32
  %34 = load float, ptr addrspace(1) %33, align 4
  %35 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %32
  %36 = load float, ptr addrspace(1) %35, align 4
  %37 = tail call float @llvm.fma.f32(float %.val1, float %34, float %36)
  %38 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %32
  store float %37, ptr addrspace(1) %38, align 4
  %39 = trunc i64 %indvars.iv to i32
  %40 = or disjoint i32 %39, 3
  %41 = add i32 %7, %40
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %42
  %44 = load float, ptr addrspace(1) %43, align 4
  %45 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %42
  %46 = load float, ptr addrspace(1) %45, align 4
  %47 = tail call float @llvm.fma.f32(float %.val1, float %44, float %46)
  %48 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %42
  store float %47, ptr addrspace(1) %48, align 4
  %49 = trunc i64 %indvars.iv to i32
  %50 = or disjoint i32 %49, 4
  %51 = add i32 %7, %50
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %52
  %54 = load float, ptr addrspace(1) %53, align 4
  %55 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %52
  %56 = load float, ptr addrspace(1) %55, align 4
  %57 = tail call float @llvm.fma.f32(float %.val1, float %54, float %56)
  %58 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %52
  store float %57, ptr addrspace(1) %58, align 4
  %59 = trunc i64 %indvars.iv to i32
  %60 = or disjoint i32 %59, 5
  %61 = add i32 %7, %60
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %62
  %64 = load float, ptr addrspace(1) %63, align 4
  %65 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %62
  %66 = load float, ptr addrspace(1) %65, align 4
  %67 = tail call float @llvm.fma.f32(float %.val1, float %64, float %66)
  %68 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %62
  store float %67, ptr addrspace(1) %68, align 4
  %69 = trunc i64 %indvars.iv to i32
  %70 = or disjoint i32 %69, 6
  %71 = add i32 %7, %70
  %72 = zext i32 %71 to i64
  %73 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %72
  %74 = load float, ptr addrspace(1) %73, align 4
  %75 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %72
  %76 = load float, ptr addrspace(1) %75, align 4
  %77 = tail call float @llvm.fma.f32(float %.val1, float %74, float %76)
  %78 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %72
  store float %77, ptr addrspace(1) %78, align 4
  %79 = trunc i64 %indvars.iv to i32
  %80 = or disjoint i32 %79, 7
  %81 = add i32 %7, %80
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %82
  %84 = load float, ptr addrspace(1) %83, align 4
  %85 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %82
  %86 = load float, ptr addrspace(1) %85, align 4
  %87 = tail call float @llvm.fma.f32(float %.val1, float %84, float %86)
  %88 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %82
  store float %87, ptr addrspace(1) %88, align 4
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %._crit_edge.loopexit.unr-lcssa, label %9

._crit_edge.loopexit.unr-lcssa:                   ; preds = %9
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %._crit_edge, label %.epil.preheader

.epil.preheader:                                  ; preds = %._crit_edge.loopexit.unr-lcssa, %.lr.ph
  %indvars.iv.epil.init = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next.7, %._crit_edge.loopexit.unr-lcssa ]
  %lcmp.mod1 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod1)
  br label %89

89:                                               ; preds = %89, %.epil.preheader
  %indvars.iv.epil = phi i64 [ %indvars.iv.epil.init, %.epil.preheader ], [ %indvars.iv.next.epil, %89 ]
  %epil.iter = phi i64 [ 0, %.epil.preheader ], [ %epil.iter.next, %89 ]
  %90 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %91 = add i32 %7, %90
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %92
  %94 = load float, ptr addrspace(1) %93, align 4
  %95 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %92
  %96 = load float, ptr addrspace(1) %95, align 4
  %97 = tail call float @llvm.fma.f32(float %.val1, float %94, float %96)
  %98 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %92
  store float %97, ptr addrspace(1) %98, align 4
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %._crit_edge, label %89, !llvm.loop !1

._crit_edge:                                      ; preds = %._crit_edge.loopexit.unr-lcssa, %89, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}
