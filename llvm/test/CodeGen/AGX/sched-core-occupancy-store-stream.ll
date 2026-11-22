; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sched-core-occupancy-store-cut=0 < %s | FileCheck %s --check-prefix=FREE
;
; MLIR's col_softmax_norm (a thread per column of 256: its sum, then each value
; times the sum's reciprocal stored back; mlir-opt, opt -O3): the second
; loop, 16 word loads down the column and 16 stores, is a stream of a store
; for every load. Its free order needs registers for every product before
; the clustered stores (the core's SIMD-groups 96 -> 68; the products, then
; the stores, without the rule); since L15-E765 such a region keeps its
; original order at E761's cut, each product stored as it is ready (agx_cost
; tput 510.5 -> 458.7).

; CHECK-LABEL: .LBB0_4:
; CHECK: wait 0
; CHECK-NEXT: fmul32
; CHECK-NEXT: device_store
; FREE-LABEL: .LBB0_4:
; FREE: wait 0
; FREE-NEXT: fmul32
; FREE-NEXT: wait 1

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @col_softmax_norm(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1) local_unnamed_addr #0 {
entry:
  %2 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %3 = zext i32 %2 to i64
  %invariant.gep = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %3
  br label %4

4:                                                ; preds = %4, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.15, %4 ]
  %5 = phi float [ 0.000000e+00, %entry ], [ %52, %4 ]
  %.idx = shl nuw nsw i64 %indvars.iv, 12
  %gep = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %.idx
  %6 = load float, ptr addrspace(1) %gep, align 4
  %7 = fadd float %5, %6
  %indvars.iv.next = shl i64 %indvars.iv, 12
  %8 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(1) %8, i64 4096
  %9 = load float, ptr addrspace(1) %gep.1, align 4
  %10 = fadd float %7, %9
  %indvars.iv.next.1 = shl i64 %indvars.iv, 12
  %11 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(1) %11, i64 8192
  %12 = load float, ptr addrspace(1) %gep.2, align 4
  %13 = fadd float %10, %12
  %indvars.iv.next.2 = shl i64 %indvars.iv, 12
  %14 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(1) %14, i64 12288
  %15 = load float, ptr addrspace(1) %gep.3, align 4
  %16 = fadd float %13, %15
  %indvars.iv.next.3 = shl i64 %indvars.iv, 12
  %17 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(1) %17, i64 16384
  %18 = load float, ptr addrspace(1) %gep.4, align 4
  %19 = fadd float %16, %18
  %indvars.iv.next.4 = shl i64 %indvars.iv, 12
  %20 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.4
  %gep.5 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 20480
  %21 = load float, ptr addrspace(1) %gep.5, align 4
  %22 = fadd float %19, %21
  %indvars.iv.next.5 = shl i64 %indvars.iv, 12
  %23 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.5
  %gep.6 = getelementptr inbounds nuw i8, ptr addrspace(1) %23, i64 24576
  %24 = load float, ptr addrspace(1) %gep.6, align 4
  %25 = fadd float %22, %24
  %indvars.iv.next.6 = shl i64 %indvars.iv, 12
  %26 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.6
  %gep.7 = getelementptr inbounds nuw i8, ptr addrspace(1) %26, i64 28672
  %27 = load float, ptr addrspace(1) %gep.7, align 4
  %28 = fadd float %25, %27
  %indvars.iv.next.7 = shl i64 %indvars.iv, 12
  %29 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.7
  %gep.8 = getelementptr inbounds nuw i8, ptr addrspace(1) %29, i64 32768
  %30 = load float, ptr addrspace(1) %gep.8, align 4
  %31 = fadd float %28, %30
  %indvars.iv.next.8 = shl i64 %indvars.iv, 12
  %32 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.8
  %gep.9 = getelementptr inbounds nuw i8, ptr addrspace(1) %32, i64 36864
  %33 = load float, ptr addrspace(1) %gep.9, align 4
  %34 = fadd float %31, %33
  %indvars.iv.next.9 = shl i64 %indvars.iv, 12
  %35 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.9
  %gep.10 = getelementptr inbounds nuw i8, ptr addrspace(1) %35, i64 40960
  %36 = load float, ptr addrspace(1) %gep.10, align 4
  %37 = fadd float %34, %36
  %indvars.iv.next.10 = shl i64 %indvars.iv, 12
  %38 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.10
  %gep.11 = getelementptr inbounds nuw i8, ptr addrspace(1) %38, i64 45056
  %39 = load float, ptr addrspace(1) %gep.11, align 4
  %40 = fadd float %37, %39
  %indvars.iv.next.11 = shl i64 %indvars.iv, 12
  %41 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.11
  %gep.12 = getelementptr inbounds nuw i8, ptr addrspace(1) %41, i64 49152
  %42 = load float, ptr addrspace(1) %gep.12, align 4
  %43 = fadd float %40, %42
  %indvars.iv.next.12 = shl i64 %indvars.iv, 12
  %44 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.12
  %gep.13 = getelementptr inbounds nuw i8, ptr addrspace(1) %44, i64 53248
  %45 = load float, ptr addrspace(1) %gep.13, align 4
  %46 = fadd float %43, %45
  %indvars.iv.next.13 = shl i64 %indvars.iv, 12
  %47 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.13
  %gep.14 = getelementptr inbounds nuw i8, ptr addrspace(1) %47, i64 57344
  %48 = load float, ptr addrspace(1) %gep.14, align 4
  %49 = fadd float %46, %48
  %indvars.iv.next.14 = shl i64 %indvars.iv, 12
  %50 = getelementptr inbounds nuw i8, ptr addrspace(1) %invariant.gep, i64 %indvars.iv.next.14
  %gep.15 = getelementptr inbounds nuw i8, ptr addrspace(1) %50, i64 61440
  %51 = load float, ptr addrspace(1) %gep.15, align 4
  %52 = fadd float %49, %51
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %exitcond.not.15 = icmp eq i64 %indvars.iv.next.15, 256
  br i1 %exitcond.not.15, label %53, label %4

53:                                               ; preds = %4
  %54 = fdiv float 1.000000e+00, %52
  br label %55

55:                                               ; preds = %55, %53
  %indvars.iv2 = phi i64 [ 0, %53 ], [ %indvars.iv.next3.15, %55 ]
  %indvars.iv2.tr = trunc nuw nsw i64 %indvars.iv2 to i32
  %56 = shl nuw nsw i32 %indvars.iv2.tr, 10
  %57 = add i32 %56, %2
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %58
  %60 = load float, ptr addrspace(1) %59, align 4
  %61 = fmul float %54, %60
  %62 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %58
  store float %61, ptr addrspace(1) %62, align 4
  %63 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.1 = shl i32 %63, 10
  %64 = or disjoint i32 %indvars.iv2.tr.1, 1024
  %65 = add i32 %64, %2
  %66 = zext i32 %65 to i64
  %67 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %66
  %68 = load float, ptr addrspace(1) %67, align 4
  %69 = fmul float %54, %68
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %66
  store float %69, ptr addrspace(1) %70, align 4
  %71 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.2 = shl i32 %71, 10
  %72 = or disjoint i32 %indvars.iv2.tr.2, 2048
  %73 = add i32 %72, %2
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %74
  %76 = load float, ptr addrspace(1) %75, align 4
  %77 = fmul float %54, %76
  %78 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %74
  store float %77, ptr addrspace(1) %78, align 4
  %79 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.3 = shl i32 %79, 10
  %80 = or disjoint i32 %indvars.iv2.tr.3, 3072
  %81 = add i32 %80, %2
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %82
  %84 = load float, ptr addrspace(1) %83, align 4
  %85 = fmul float %54, %84
  %86 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %82
  store float %85, ptr addrspace(1) %86, align 4
  %87 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.4 = shl i32 %87, 10
  %88 = or disjoint i32 %indvars.iv2.tr.4, 4096
  %89 = add i32 %88, %2
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %90
  %92 = load float, ptr addrspace(1) %91, align 4
  %93 = fmul float %54, %92
  %94 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %90
  store float %93, ptr addrspace(1) %94, align 4
  %95 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.5 = shl i32 %95, 10
  %96 = or disjoint i32 %indvars.iv2.tr.5, 5120
  %97 = add i32 %96, %2
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %98
  %100 = load float, ptr addrspace(1) %99, align 4
  %101 = fmul float %54, %100
  %102 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %98
  store float %101, ptr addrspace(1) %102, align 4
  %103 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.6 = shl i32 %103, 10
  %104 = or disjoint i32 %indvars.iv2.tr.6, 6144
  %105 = add i32 %104, %2
  %106 = zext i32 %105 to i64
  %107 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %106
  %108 = load float, ptr addrspace(1) %107, align 4
  %109 = fmul float %54, %108
  %110 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %106
  store float %109, ptr addrspace(1) %110, align 4
  %111 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.7 = shl i32 %111, 10
  %112 = or disjoint i32 %indvars.iv2.tr.7, 7168
  %113 = add i32 %112, %2
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %114
  %116 = load float, ptr addrspace(1) %115, align 4
  %117 = fmul float %54, %116
  %118 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %114
  store float %117, ptr addrspace(1) %118, align 4
  %119 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.8 = shl i32 %119, 10
  %120 = or disjoint i32 %indvars.iv2.tr.8, 8192
  %121 = add i32 %120, %2
  %122 = zext i32 %121 to i64
  %123 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %122
  %124 = load float, ptr addrspace(1) %123, align 4
  %125 = fmul float %54, %124
  %126 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %122
  store float %125, ptr addrspace(1) %126, align 4
  %127 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.9 = shl i32 %127, 10
  %128 = or disjoint i32 %indvars.iv2.tr.9, 9216
  %129 = add i32 %128, %2
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %130
  %132 = load float, ptr addrspace(1) %131, align 4
  %133 = fmul float %54, %132
  %134 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %130
  store float %133, ptr addrspace(1) %134, align 4
  %135 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.10 = shl i32 %135, 10
  %136 = or disjoint i32 %indvars.iv2.tr.10, 10240
  %137 = add i32 %136, %2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %138
  %140 = load float, ptr addrspace(1) %139, align 4
  %141 = fmul float %54, %140
  %142 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %138
  store float %141, ptr addrspace(1) %142, align 4
  %143 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.11 = shl i32 %143, 10
  %144 = or disjoint i32 %indvars.iv2.tr.11, 11264
  %145 = add i32 %144, %2
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %146
  %148 = load float, ptr addrspace(1) %147, align 4
  %149 = fmul float %54, %148
  %150 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %146
  store float %149, ptr addrspace(1) %150, align 4
  %151 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.12 = shl i32 %151, 10
  %152 = or disjoint i32 %indvars.iv2.tr.12, 12288
  %153 = add i32 %152, %2
  %154 = zext i32 %153 to i64
  %155 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %154
  %156 = load float, ptr addrspace(1) %155, align 4
  %157 = fmul float %54, %156
  %158 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %154
  store float %157, ptr addrspace(1) %158, align 4
  %159 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.13 = shl i32 %159, 10
  %160 = or disjoint i32 %indvars.iv2.tr.13, 13312
  %161 = add i32 %160, %2
  %162 = zext i32 %161 to i64
  %163 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %162
  %164 = load float, ptr addrspace(1) %163, align 4
  %165 = fmul float %54, %164
  %166 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %162
  store float %165, ptr addrspace(1) %166, align 4
  %167 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.14 = shl i32 %167, 10
  %168 = or disjoint i32 %indvars.iv2.tr.14, 14336
  %169 = add i32 %168, %2
  %170 = zext i32 %169 to i64
  %171 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %170
  %172 = load float, ptr addrspace(1) %171, align 4
  %173 = fmul float %54, %172
  %174 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %170
  store float %173, ptr addrspace(1) %174, align 4
  %175 = trunc i64 %indvars.iv2 to i32
  %indvars.iv2.tr.15 = shl i32 %175, 10
  %176 = or disjoint i32 %indvars.iv2.tr.15, 15360
  %177 = add i32 %176, %2
  %178 = zext i32 %177 to i64
  %179 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %178
  %180 = load float, ptr addrspace(1) %179, align 4
  %181 = fmul float %54, %180
  %182 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %178
  store float %181, ptr addrspace(1) %182, align 4
  %indvars.iv.next3.15 = add nuw nsw i64 %indvars.iv2, 16
  %exitcond4.not.15 = icmp eq i64 %indvars.iv.next3.15, 256
  br i1 %exitcond4.not.15, label %183, label %55

183:                                              ; preds = %55
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
