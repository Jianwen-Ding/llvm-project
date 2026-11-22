; REQUIRES: asserts
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -debug-only=agx-target-machine -o /dev/null < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -debug-only=agx-target-machine -agx-sched-core-occupancy-final=0 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=OFF

; where_rows (MLIR corpus): its first region's free order would leave the core
; 82 SIMD-groups of 96, so E799 put it back in its original order; another
; block's pressure leaves the function 46 anyway. At the function's end the
; region is scheduled freely again: no fewer groups, its latency hidden
; (L15-E804).
; CHECK: agx-occupancy reverted
; CHECK: agx-occupancy final: free again, halves {{[0-9]+}} beside {{[0-9]+}}
; OFF: agx-occupancy reverted
; OFF-NOT: agx-occupancy final
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @where_rows(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree readonly captures(none) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #0 {
entry:
  %4 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %5 = shl i32 %4, 8
  %6 = zext i32 %5 to i64
  br label %7

7:                                                ; preds = %7, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.15, %7 ]
  %8 = add nuw nsw i64 %indvars.iv, %6
  %9 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %8
  %10 = load i8, ptr addrspace(1) %9, align 1
  %11 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %8
  %12 = load float, ptr addrspace(1) %11, align 4
  %13 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %8
  %14 = load float, ptr addrspace(1) %13, align 4
  %.not = icmp eq i8 %10, 0
  %15 = select i1 %.not, float %14, float %12
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %8
  store float %15, ptr addrspace(1) %16, align 4
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %17 = add nuw nsw i64 %indvars.iv.next, %6
  %18 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %17
  %19 = load i8, ptr addrspace(1) %18, align 1
  %20 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %17
  %21 = load float, ptr addrspace(1) %20, align 4
  %22 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %17
  %23 = load float, ptr addrspace(1) %22, align 4
  %.not.1 = icmp eq i8 %19, 0
  %24 = select i1 %.not.1, float %23, float %21
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %17
  store float %24, ptr addrspace(1) %25, align 4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %26 = add nuw nsw i64 %indvars.iv.next.1, %6
  %27 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %26
  %28 = load i8, ptr addrspace(1) %27, align 1
  %29 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %26
  %30 = load float, ptr addrspace(1) %29, align 4
  %31 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %26
  %32 = load float, ptr addrspace(1) %31, align 4
  %.not.2 = icmp eq i8 %28, 0
  %33 = select i1 %.not.2, float %32, float %30
  %34 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %26
  store float %33, ptr addrspace(1) %34, align 4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %35 = add nuw nsw i64 %indvars.iv.next.2, %6
  %36 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %35
  %37 = load i8, ptr addrspace(1) %36, align 1
  %38 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %35
  %39 = load float, ptr addrspace(1) %38, align 4
  %40 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %35
  %41 = load float, ptr addrspace(1) %40, align 4
  %.not.3 = icmp eq i8 %37, 0
  %42 = select i1 %.not.3, float %41, float %39
  %43 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %35
  store float %42, ptr addrspace(1) %43, align 4
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %44 = add nuw nsw i64 %indvars.iv.next.3, %6
  %45 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %44
  %46 = load i8, ptr addrspace(1) %45, align 1
  %47 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %44
  %48 = load float, ptr addrspace(1) %47, align 4
  %49 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %44
  %50 = load float, ptr addrspace(1) %49, align 4
  %.not.4 = icmp eq i8 %46, 0
  %51 = select i1 %.not.4, float %50, float %48
  %52 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %44
  store float %51, ptr addrspace(1) %52, align 4
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %53 = add nuw nsw i64 %indvars.iv.next.4, %6
  %54 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %53
  %55 = load i8, ptr addrspace(1) %54, align 1
  %56 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %53
  %57 = load float, ptr addrspace(1) %56, align 4
  %58 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %53
  %59 = load float, ptr addrspace(1) %58, align 4
  %.not.5 = icmp eq i8 %55, 0
  %60 = select i1 %.not.5, float %59, float %57
  %61 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %53
  store float %60, ptr addrspace(1) %61, align 4
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %62 = add nuw nsw i64 %indvars.iv.next.5, %6
  %63 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %62
  %64 = load i8, ptr addrspace(1) %63, align 1
  %65 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %62
  %66 = load float, ptr addrspace(1) %65, align 4
  %67 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %62
  %68 = load float, ptr addrspace(1) %67, align 4
  %.not.6 = icmp eq i8 %64, 0
  %69 = select i1 %.not.6, float %68, float %66
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %62
  store float %69, ptr addrspace(1) %70, align 4
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %71 = add nuw nsw i64 %indvars.iv.next.6, %6
  %72 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %71
  %73 = load i8, ptr addrspace(1) %72, align 1
  %74 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %71
  %75 = load float, ptr addrspace(1) %74, align 4
  %76 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %71
  %77 = load float, ptr addrspace(1) %76, align 4
  %.not.7 = icmp eq i8 %73, 0
  %78 = select i1 %.not.7, float %77, float %75
  %79 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %71
  store float %78, ptr addrspace(1) %79, align 4
  %indvars.iv.next.7 = or disjoint i64 %indvars.iv, 8
  %80 = add nuw nsw i64 %indvars.iv.next.7, %6
  %81 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %80
  %82 = load i8, ptr addrspace(1) %81, align 1
  %83 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %80
  %84 = load float, ptr addrspace(1) %83, align 4
  %85 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %80
  %86 = load float, ptr addrspace(1) %85, align 4
  %.not.8 = icmp eq i8 %82, 0
  %87 = select i1 %.not.8, float %86, float %84
  %88 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %80
  store float %87, ptr addrspace(1) %88, align 4
  %indvars.iv.next.8 = or disjoint i64 %indvars.iv, 9
  %89 = add nuw nsw i64 %indvars.iv.next.8, %6
  %90 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %89
  %91 = load i8, ptr addrspace(1) %90, align 1
  %92 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %89
  %93 = load float, ptr addrspace(1) %92, align 4
  %94 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %89
  %95 = load float, ptr addrspace(1) %94, align 4
  %.not.9 = icmp eq i8 %91, 0
  %96 = select i1 %.not.9, float %95, float %93
  %97 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %89
  store float %96, ptr addrspace(1) %97, align 4
  %indvars.iv.next.9 = or disjoint i64 %indvars.iv, 10
  %98 = add nuw nsw i64 %indvars.iv.next.9, %6
  %99 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %98
  %100 = load i8, ptr addrspace(1) %99, align 1
  %101 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %98
  %102 = load float, ptr addrspace(1) %101, align 4
  %103 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %98
  %104 = load float, ptr addrspace(1) %103, align 4
  %.not.10 = icmp eq i8 %100, 0
  %105 = select i1 %.not.10, float %104, float %102
  %106 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %98
  store float %105, ptr addrspace(1) %106, align 4
  %indvars.iv.next.10 = or disjoint i64 %indvars.iv, 11
  %107 = add nuw nsw i64 %indvars.iv.next.10, %6
  %108 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %107
  %109 = load i8, ptr addrspace(1) %108, align 1
  %110 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %107
  %111 = load float, ptr addrspace(1) %110, align 4
  %112 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %107
  %113 = load float, ptr addrspace(1) %112, align 4
  %.not.11 = icmp eq i8 %109, 0
  %114 = select i1 %.not.11, float %113, float %111
  %115 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %107
  store float %114, ptr addrspace(1) %115, align 4
  %indvars.iv.next.11 = or disjoint i64 %indvars.iv, 12
  %116 = add nuw nsw i64 %indvars.iv.next.11, %6
  %117 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %116
  %118 = load i8, ptr addrspace(1) %117, align 1
  %119 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %116
  %120 = load float, ptr addrspace(1) %119, align 4
  %121 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %116
  %122 = load float, ptr addrspace(1) %121, align 4
  %.not.12 = icmp eq i8 %118, 0
  %123 = select i1 %.not.12, float %122, float %120
  %124 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %116
  store float %123, ptr addrspace(1) %124, align 4
  %indvars.iv.next.12 = or disjoint i64 %indvars.iv, 13
  %125 = add nuw nsw i64 %indvars.iv.next.12, %6
  %126 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %125
  %127 = load i8, ptr addrspace(1) %126, align 1
  %128 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %125
  %129 = load float, ptr addrspace(1) %128, align 4
  %130 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %125
  %131 = load float, ptr addrspace(1) %130, align 4
  %.not.13 = icmp eq i8 %127, 0
  %132 = select i1 %.not.13, float %131, float %129
  %133 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %125
  store float %132, ptr addrspace(1) %133, align 4
  %indvars.iv.next.13 = or disjoint i64 %indvars.iv, 14
  %134 = add nuw nsw i64 %indvars.iv.next.13, %6
  %135 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %134
  %136 = load i8, ptr addrspace(1) %135, align 1
  %137 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %134
  %138 = load float, ptr addrspace(1) %137, align 4
  %139 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %134
  %140 = load float, ptr addrspace(1) %139, align 4
  %.not.14 = icmp eq i8 %136, 0
  %141 = select i1 %.not.14, float %140, float %138
  %142 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %134
  store float %141, ptr addrspace(1) %142, align 4
  %indvars.iv.next.14 = or disjoint i64 %indvars.iv, 15
  %143 = add nuw nsw i64 %indvars.iv.next.14, %6
  %144 = getelementptr inbounds nuw i8, ptr addrspace(1) %0, i64 %143
  %145 = load i8, ptr addrspace(1) %144, align 1
  %146 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %143
  %147 = load float, ptr addrspace(1) %146, align 4
  %148 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %143
  %149 = load float, ptr addrspace(1) %148, align 4
  %.not.15 = icmp eq i8 %145, 0
  %150 = select i1 %.not.15, float %149, float %147
  %151 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %143
  store float %150, ptr addrspace(1) %151, align 4
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %exitcond.not.15 = icmp eq i64 %indvars.iv.next.15, 256
  br i1 %exitcond.not.15, label %152, label %7

152:                                              ; preds = %7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

