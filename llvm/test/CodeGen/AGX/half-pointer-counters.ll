; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs -agx-half-pointer-counters=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's hdot_dyn: LSR steps both half rows by pointers, 16 bytes an
; iteration beside the counter of halves j (by 8). Each pointer is its base
; plus j halves, the 8-byte loads at a half index of the 16-bit format, the
; 16-byte loads two from bases 8 bytes apart: no 64-bit add in the loop
; (L15-E688).

; CHECK-LABEL: hdot_dyn:
; CHECK: .LBB0_8:
; CHECK-COUNT-4: device_load {{[01]}}, i16, xyzw, r{{[0-9]+}}l_r{{[0-9]+}}h_r{{[0-9]+}}l_r{{[0-9]+}}h, {{[ru][0-9]+}}_{{[ru][0-9]+}}, r[[J:[0-9]+]], unsigned
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 16
; CHECK: iadd r[[J]], r[[J]], 8
; CHECK: jmp_exec_any .LBB0_8
; OFF-LABEL: hdot_dyn:
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 16

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @hdot_dyn(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 captures(none) dereferenceable(4) %3) local_unnamed_addr #0 {
entry:
  %.val = load i32, ptr addrspace(4) %3, align 4
  %4 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %5 = icmp sgt i32 %.val, 0
  br i1 %5, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %6 = shl i32 %4, 9
  %wide.trip.count = zext nneg i32 %.val to i64
  %xtraiter = and i64 %wide.trip.count, 7
  %7 = icmp ult i32 %.val, 8
  br i1 %7, label %.epil.preheader, label %.lr.ph.new

.lr.ph.new:                                       ; preds = %.lr.ph
  %unroll_iter = and i64 %wide.trip.count, 2147483640
  br label %8

8:                                                ; preds = %8, %.lr.ph.new
  %indvars.iv = phi i64 [ 0, %.lr.ph.new ], [ %indvars.iv.next.7, %8 ]
  %9 = phi float [ 0.000000e+00, %.lr.ph.new ], [ %89, %8 ]
  %niter = phi i64 [ 0, %.lr.ph.new ], [ %niter.next.7, %8 ]
  %10 = trunc nuw nsw i64 %indvars.iv to i32
  %11 = add i32 %6, %10
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %12
  %14 = load half, ptr addrspace(1) %13, align 2
  %15 = fpext half %14 to float
  %16 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv
  %17 = load half, ptr addrspace(1) %16, align 2
  %18 = fpext half %17 to float
  %19 = tail call float @llvm.fma.f32(float %15, float %18, float %9)
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %20 = trunc nuw nsw i64 %indvars.iv.next to i32
  %21 = add i32 %6, %20
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %22
  %24 = load half, ptr addrspace(1) %23, align 2
  %25 = fpext half %24 to float
  %26 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next
  %27 = load half, ptr addrspace(1) %26, align 2
  %28 = fpext half %27 to float
  %29 = tail call float @llvm.fma.f32(float %25, float %28, float %19)
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %30 = trunc nuw nsw i64 %indvars.iv.next.1 to i32
  %31 = add i32 %6, %30
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %32
  %34 = load half, ptr addrspace(1) %33, align 2
  %35 = fpext half %34 to float
  %36 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.1
  %37 = load half, ptr addrspace(1) %36, align 2
  %38 = fpext half %37 to float
  %39 = tail call float @llvm.fma.f32(float %35, float %38, float %29)
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %40 = trunc nuw nsw i64 %indvars.iv.next.2 to i32
  %41 = add i32 %6, %40
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %42
  %44 = load half, ptr addrspace(1) %43, align 2
  %45 = fpext half %44 to float
  %46 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.2
  %47 = load half, ptr addrspace(1) %46, align 2
  %48 = fpext half %47 to float
  %49 = tail call float @llvm.fma.f32(float %45, float %48, float %39)
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %50 = trunc nuw nsw i64 %indvars.iv.next.3 to i32
  %51 = add i32 %6, %50
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %52
  %54 = load half, ptr addrspace(1) %53, align 2
  %55 = fpext half %54 to float
  %56 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.3
  %57 = load half, ptr addrspace(1) %56, align 2
  %58 = fpext half %57 to float
  %59 = tail call float @llvm.fma.f32(float %55, float %58, float %49)
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %60 = trunc nuw nsw i64 %indvars.iv.next.4 to i32
  %61 = add i32 %6, %60
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %62
  %64 = load half, ptr addrspace(1) %63, align 2
  %65 = fpext half %64 to float
  %66 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.4
  %67 = load half, ptr addrspace(1) %66, align 2
  %68 = fpext half %67 to float
  %69 = tail call float @llvm.fma.f32(float %65, float %68, float %59)
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %70 = trunc nuw nsw i64 %indvars.iv.next.5 to i32
  %71 = add i32 %6, %70
  %72 = zext i32 %71 to i64
  %73 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %72
  %74 = load half, ptr addrspace(1) %73, align 2
  %75 = fpext half %74 to float
  %76 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.5
  %77 = load half, ptr addrspace(1) %76, align 2
  %78 = fpext half %77 to float
  %79 = tail call float @llvm.fma.f32(float %75, float %78, float %69)
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %80 = trunc nuw nsw i64 %indvars.iv.next.6 to i32
  %81 = add i32 %6, %80
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %82
  %84 = load half, ptr addrspace(1) %83, align 2
  %85 = fpext half %84 to float
  %86 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.6
  %87 = load half, ptr addrspace(1) %86, align 2
  %88 = fpext half %87 to float
  %89 = tail call float @llvm.fma.f32(float %85, float %88, float %79)
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %._crit_edge.loopexit.unr-lcssa, label %8

._crit_edge.loopexit.unr-lcssa:                   ; preds = %8
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %._crit_edge, label %.epil.preheader

.epil.preheader:                                  ; preds = %._crit_edge.loopexit.unr-lcssa, %.lr.ph
  %indvars.iv.epil.init = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next.7, %._crit_edge.loopexit.unr-lcssa ]
  %.epil.init = phi float [ 0.000000e+00, %.lr.ph ], [ %89, %._crit_edge.loopexit.unr-lcssa ]
  %lcmp.mod6 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod6)
  br label %90

90:                                               ; preds = %90, %.epil.preheader
  %indvars.iv.epil = phi i64 [ %indvars.iv.epil.init, %.epil.preheader ], [ %indvars.iv.next.epil, %90 ]
  %91 = phi float [ %.epil.init, %.epil.preheader ], [ %101, %90 ]
  %epil.iter = phi i64 [ 0, %.epil.preheader ], [ %epil.iter.next, %90 ]
  %92 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %93 = add i32 %6, %92
  %94 = zext i32 %93 to i64
  %95 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %94
  %96 = load half, ptr addrspace(1) %95, align 2
  %97 = fpext half %96 to float
  %98 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1, i64 %indvars.iv.epil
  %99 = load half, ptr addrspace(1) %98, align 2
  %100 = fpext half %99 to float
  %101 = tail call float @llvm.fma.f32(float %97, float %100, float %91)
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %._crit_edge, label %90, !llvm.loop !1

._crit_edge:                                      ; preds = %._crit_edge.loopexit.unr-lcssa, %90, %entry
  %.lcssa = phi float [ 0.000000e+00, %entry ], [ %89, %._crit_edge.loopexit.unr-lcssa ], [ %101, %90 ]
  %102 = zext i32 %4 to i64
  %103 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %102
  store float %.lcssa, ptr addrspace(1) %103, align 4
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
