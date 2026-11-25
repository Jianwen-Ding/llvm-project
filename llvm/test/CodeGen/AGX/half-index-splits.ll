; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -verify-machineinstrs -agx-half-index-splits=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's widen_dyn: its aligned copy loads 8 halves at a time at the row's
; base plus the counter of halves, in a block that stores (no 16-byte half
; loads there). Two 8-byte loads at the same half index of the 16-bit
; format, from the base and the base plus 8 bytes (computed before the
; loop), not the second half's address in a 64-bit add (L15-E690).

; CHECK-LABEL: widen_dyn:
; CHECK: %.ph.unaligned.aligned
; CHECK: iadd r[[B8:[0-9]+]]_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 8
; CHECK: .LBB0_9:
; CHECK-NEXT: device_load 0, i16, xyzw, {{.*}}, u0_u1, r[[J:[0-9]+]], unsigned
; CHECK-NEXT: device_load 1, i16, xyzw, {{.*}}, r[[B8]]_r{{[0-9]+}}, r[[J]], unsigned
; CHECK-NOT: lsl 1
; CHECK: jmp_exec_any .LBB0_9
; OFF-LABEL: widen_dyn:
; OFF: %.ph.unaligned.aligned
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}, lsl 1
; OFF: device_load 1, i32, xy

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @widen_dyn(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 captures(none) dereferenceable(4) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %3) local_unnamed_addr #0 {
entry:
  %.val = load i32, ptr addrspace(4) %2, align 4
  %.val1 = load float, ptr addrspace(4) %3, align 4
  %4 = icmp sgt i32 %.val, 0
  br i1 %4, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %5 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %6 = shl i32 %5, 8
  %wide.trip.count = zext nneg i32 %.val to i64
  %xtraiter = and i64 %wide.trip.count, 15
  %7 = icmp ult i32 %.val, 16
  br i1 %7, label %.epil.preheader, label %.lr.ph.new

.lr.ph.new:                                       ; preds = %.lr.ph
  %unroll_iter = and i64 %wide.trip.count, 2147483632
  br label %8

8:                                                ; preds = %8, %.lr.ph.new
  %indvars.iv = phi i64 [ 0, %.lr.ph.new ], [ %indvars.iv.next.15, %8 ]
  %niter = phi i64 [ 0, %.lr.ph.new ], [ %niter.next.15, %8 ]
  %9 = trunc nuw nsw i64 %indvars.iv to i32
  %10 = add i32 %6, %9
  %11 = zext i32 %10 to i64
  %12 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %11
  %13 = load half, ptr addrspace(1) %12, align 2
  %14 = fpext half %13 to float
  %15 = fmul float %.val1, %14
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %11
  store float %15, ptr addrspace(1) %16, align 4
  %17 = trunc i64 %indvars.iv to i32
  %18 = or disjoint i32 %17, 1
  %19 = add i32 %6, %18
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %20
  %22 = load half, ptr addrspace(1) %21, align 2
  %23 = fpext half %22 to float
  %24 = fmul float %.val1, %23
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %20
  store float %24, ptr addrspace(1) %25, align 4
  %26 = trunc i64 %indvars.iv to i32
  %27 = or disjoint i32 %26, 2
  %28 = add i32 %6, %27
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %29
  %31 = load half, ptr addrspace(1) %30, align 2
  %32 = fpext half %31 to float
  %33 = fmul float %.val1, %32
  %34 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %29
  store float %33, ptr addrspace(1) %34, align 4
  %35 = trunc i64 %indvars.iv to i32
  %36 = or disjoint i32 %35, 3
  %37 = add i32 %6, %36
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %38
  %40 = load half, ptr addrspace(1) %39, align 2
  %41 = fpext half %40 to float
  %42 = fmul float %.val1, %41
  %43 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %38
  store float %42, ptr addrspace(1) %43, align 4
  %44 = trunc i64 %indvars.iv to i32
  %45 = or disjoint i32 %44, 4
  %46 = add i32 %6, %45
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %47
  %49 = load half, ptr addrspace(1) %48, align 2
  %50 = fpext half %49 to float
  %51 = fmul float %.val1, %50
  %52 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %47
  store float %51, ptr addrspace(1) %52, align 4
  %53 = trunc i64 %indvars.iv to i32
  %54 = or disjoint i32 %53, 5
  %55 = add i32 %6, %54
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %56
  %58 = load half, ptr addrspace(1) %57, align 2
  %59 = fpext half %58 to float
  %60 = fmul float %.val1, %59
  %61 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %56
  store float %60, ptr addrspace(1) %61, align 4
  %62 = trunc i64 %indvars.iv to i32
  %63 = or disjoint i32 %62, 6
  %64 = add i32 %6, %63
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %65
  %67 = load half, ptr addrspace(1) %66, align 2
  %68 = fpext half %67 to float
  %69 = fmul float %.val1, %68
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %65
  store float %69, ptr addrspace(1) %70, align 4
  %71 = trunc i64 %indvars.iv to i32
  %72 = or disjoint i32 %71, 7
  %73 = add i32 %6, %72
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %74
  %76 = load half, ptr addrspace(1) %75, align 2
  %77 = fpext half %76 to float
  %78 = fmul float %.val1, %77
  %79 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %74
  store float %78, ptr addrspace(1) %79, align 4
  %80 = trunc i64 %indvars.iv to i32
  %81 = or disjoint i32 %80, 8
  %82 = add i32 %6, %81
  %83 = zext i32 %82 to i64
  %84 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %83
  %85 = load half, ptr addrspace(1) %84, align 2
  %86 = fpext half %85 to float
  %87 = fmul float %.val1, %86
  %88 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %83
  store float %87, ptr addrspace(1) %88, align 4
  %89 = trunc i64 %indvars.iv to i32
  %90 = or disjoint i32 %89, 9
  %91 = add i32 %6, %90
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %92
  %94 = load half, ptr addrspace(1) %93, align 2
  %95 = fpext half %94 to float
  %96 = fmul float %.val1, %95
  %97 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %92
  store float %96, ptr addrspace(1) %97, align 4
  %98 = trunc i64 %indvars.iv to i32
  %99 = or disjoint i32 %98, 10
  %100 = add i32 %6, %99
  %101 = zext i32 %100 to i64
  %102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %101
  %103 = load half, ptr addrspace(1) %102, align 2
  %104 = fpext half %103 to float
  %105 = fmul float %.val1, %104
  %106 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %101
  store float %105, ptr addrspace(1) %106, align 4
  %107 = trunc i64 %indvars.iv to i32
  %108 = or disjoint i32 %107, 11
  %109 = add i32 %6, %108
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %110
  %112 = load half, ptr addrspace(1) %111, align 2
  %113 = fpext half %112 to float
  %114 = fmul float %.val1, %113
  %115 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %110
  store float %114, ptr addrspace(1) %115, align 4
  %116 = trunc i64 %indvars.iv to i32
  %117 = or disjoint i32 %116, 12
  %118 = add i32 %6, %117
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %119
  %121 = load half, ptr addrspace(1) %120, align 2
  %122 = fpext half %121 to float
  %123 = fmul float %.val1, %122
  %124 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %119
  store float %123, ptr addrspace(1) %124, align 4
  %125 = trunc i64 %indvars.iv to i32
  %126 = or disjoint i32 %125, 13
  %127 = add i32 %6, %126
  %128 = zext i32 %127 to i64
  %129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %128
  %130 = load half, ptr addrspace(1) %129, align 2
  %131 = fpext half %130 to float
  %132 = fmul float %.val1, %131
  %133 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %128
  store float %132, ptr addrspace(1) %133, align 4
  %134 = trunc i64 %indvars.iv to i32
  %135 = or disjoint i32 %134, 14
  %136 = add i32 %6, %135
  %137 = zext i32 %136 to i64
  %138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %137
  %139 = load half, ptr addrspace(1) %138, align 2
  %140 = fpext half %139 to float
  %141 = fmul float %.val1, %140
  %142 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %137
  store float %141, ptr addrspace(1) %142, align 4
  %143 = trunc i64 %indvars.iv to i32
  %144 = or disjoint i32 %143, 15
  %145 = add i32 %6, %144
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %146
  %148 = load half, ptr addrspace(1) %147, align 2
  %149 = fpext half %148 to float
  %150 = fmul float %.val1, %149
  %151 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %146
  store float %150, ptr addrspace(1) %151, align 4
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %niter.next.15 = add i64 %niter, 16
  %niter.ncmp.15 = icmp eq i64 %niter.next.15, %unroll_iter
  br i1 %niter.ncmp.15, label %._crit_edge.loopexit.unr-lcssa, label %8

._crit_edge.loopexit.unr-lcssa:                   ; preds = %8
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %._crit_edge, label %.epil.preheader

.epil.preheader:                                  ; preds = %._crit_edge.loopexit.unr-lcssa, %.lr.ph
  %indvars.iv.epil.init = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next.15, %._crit_edge.loopexit.unr-lcssa ]
  %lcmp.mod1 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod1)
  br label %152

152:                                              ; preds = %152, %.epil.preheader
  %indvars.iv.epil = phi i64 [ %indvars.iv.epil.init, %.epil.preheader ], [ %indvars.iv.next.epil, %152 ]
  %epil.iter = phi i64 [ 0, %.epil.preheader ], [ %epil.iter.next, %152 ]
  %153 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %154 = add i32 %6, %153
  %155 = zext i32 %154 to i64
  %156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %0, i64 %155
  %157 = load half, ptr addrspace(1) %156, align 2
  %158 = fpext half %157 to float
  %159 = fmul float %.val1, %158
  %160 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %155
  store float %159, ptr addrspace(1) %160, align 4
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %._crit_edge, label %152, !llvm.loop !1

._crit_edge:                                      ; preds = %._crit_edge.loopexit.unr-lcssa, %152, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}
