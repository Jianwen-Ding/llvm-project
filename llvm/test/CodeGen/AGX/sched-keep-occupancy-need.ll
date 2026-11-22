; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-uniform-constants -agx-sched-keep-occupancy-need=0 < %s | FileCheck %s --check-prefix=OFF
;
; MLIR's gemm_reg4 (each thread a 4x4 tile of C over K, A by rows, B by
; rows; mlir-opt, opt -O3): its unrolled K loop of 16 device loads, no stores,
; 147 instructions, scheduled freely takes 94 registers to issue each B row's
; load ahead of its fmas, threads 1024 -> 512 and the core's groups 34 -> 18:
; agx-sched-keep-occupancy kept its original order (56 registers), each B
; row waited for at its first fma (OFF). Its threadgroups of 256 still fit
; and fewer than 16 instructions a load leave the core's groups too little
; work to hide that latency behind: the free order since L15-E753 (agx_cost
; tput 6582.3 -> 6303.0).

; CHECK: general purpose registers: 94 x 32-bit
; CHECK-COUNT-2: wait
; CHECK-NOT: wait
; OFF: general purpose registers: 56 x 32-bit
; OFF-COUNT-10: wait
; OFF-NOT: wait

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @gemm_reg4(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree writeonly captures(none) %2) local_unnamed_addr #0 {
entry:
  %3 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %4 = zext i32 %3 to i64
  %5 = tail call i32 @llvm.agx.thread.position.in.grid.y()
  %6 = shl nuw nsw i64 %4, 2
  %7 = shl i32 %5, 11
  %8 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %6
  %9 = zext i32 %7 to i64
  br label %10

10:                                               ; preds = %10, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.7, %10 ]
  %11 = phi <4 x float> [ zeroinitializer, %entry ], [ %222, %10 ]
  %12 = phi <4 x float> [ zeroinitializer, %entry ], [ %221, %10 ]
  %13 = phi <4 x float> [ zeroinitializer, %entry ], [ %220, %10 ]
  %14 = phi <4 x float> [ zeroinitializer, %entry ], [ %219, %10 ]
  %.idx10 = shl nuw nsw i64 %indvars.iv, 11
  %15 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10
  %16 = load <4 x float>, ptr addrspace(1) %15, align 4
  %17 = add nuw nsw i64 %indvars.iv, %9
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %19 = load float, ptr addrspace(1) %18, align 4
  %20 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %21 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 2048
  %22 = load float, ptr addrspace(1) %21, align 4
  %23 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %24 = getelementptr inbounds nuw i8, ptr addrspace(1) %23, i64 4096
  %25 = load float, ptr addrspace(1) %24, align 4
  %26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %27 = getelementptr inbounds nuw i8, ptr addrspace(1) %26, i64 6144
  %28 = load float, ptr addrspace(1) %27, align 4
  %29 = insertelement <4 x float> poison, float %19, i64 0
  %30 = shufflevector <4 x float> %29, <4 x float> poison, <4 x i32> zeroinitializer
  %31 = insertelement <4 x float> poison, float %22, i64 0
  %32 = shufflevector <4 x float> %31, <4 x float> poison, <4 x i32> zeroinitializer
  %33 = insertelement <4 x float> poison, float %25, i64 0
  %34 = shufflevector <4 x float> %33, <4 x float> poison, <4 x i32> zeroinitializer
  %35 = insertelement <4 x float> poison, float %28, i64 0
  %36 = shufflevector <4 x float> %35, <4 x float> poison, <4 x i32> zeroinitializer
  %37 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %30, <4 x float> %16, <4 x float> %14)
  %38 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %32, <4 x float> %16, <4 x float> %13)
  %39 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %34, <4 x float> %16, <4 x float> %12)
  %40 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %36, <4 x float> %16, <4 x float> %11)
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %.idx10.1 = shl nuw nsw i64 %indvars.iv.next, 11
  %41 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.1
  %42 = load <4 x float>, ptr addrspace(1) %41, align 4
  %43 = add nuw nsw i64 %indvars.iv.next, %9
  %44 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %43
  %45 = load float, ptr addrspace(1) %44, align 4
  %46 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %43
  %47 = getelementptr inbounds nuw i8, ptr addrspace(1) %46, i64 2048
  %48 = load float, ptr addrspace(1) %47, align 4
  %49 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %43
  %50 = getelementptr inbounds nuw i8, ptr addrspace(1) %49, i64 4096
  %51 = load float, ptr addrspace(1) %50, align 4
  %52 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %43
  %53 = getelementptr inbounds nuw i8, ptr addrspace(1) %52, i64 6144
  %54 = load float, ptr addrspace(1) %53, align 4
  %55 = insertelement <4 x float> poison, float %45, i64 0
  %56 = shufflevector <4 x float> %55, <4 x float> poison, <4 x i32> zeroinitializer
  %57 = insertelement <4 x float> poison, float %48, i64 0
  %58 = shufflevector <4 x float> %57, <4 x float> poison, <4 x i32> zeroinitializer
  %59 = insertelement <4 x float> poison, float %51, i64 0
  %60 = shufflevector <4 x float> %59, <4 x float> poison, <4 x i32> zeroinitializer
  %61 = insertelement <4 x float> poison, float %54, i64 0
  %62 = shufflevector <4 x float> %61, <4 x float> poison, <4 x i32> zeroinitializer
  %63 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %56, <4 x float> %42, <4 x float> %37)
  %64 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %58, <4 x float> %42, <4 x float> %38)
  %65 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %60, <4 x float> %42, <4 x float> %39)
  %66 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %62, <4 x float> %42, <4 x float> %40)
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %.idx10.2 = shl nuw nsw i64 %indvars.iv.next.1, 11
  %67 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.2
  %68 = load <4 x float>, ptr addrspace(1) %67, align 4
  %69 = add nuw nsw i64 %indvars.iv.next.1, %9
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %69
  %71 = load float, ptr addrspace(1) %70, align 4
  %72 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %69
  %73 = getelementptr inbounds nuw i8, ptr addrspace(1) %72, i64 2048
  %74 = load float, ptr addrspace(1) %73, align 4
  %75 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %69
  %76 = getelementptr inbounds nuw i8, ptr addrspace(1) %75, i64 4096
  %77 = load float, ptr addrspace(1) %76, align 4
  %78 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %69
  %79 = getelementptr inbounds nuw i8, ptr addrspace(1) %78, i64 6144
  %80 = load float, ptr addrspace(1) %79, align 4
  %81 = insertelement <4 x float> poison, float %71, i64 0
  %82 = shufflevector <4 x float> %81, <4 x float> poison, <4 x i32> zeroinitializer
  %83 = insertelement <4 x float> poison, float %74, i64 0
  %84 = shufflevector <4 x float> %83, <4 x float> poison, <4 x i32> zeroinitializer
  %85 = insertelement <4 x float> poison, float %77, i64 0
  %86 = shufflevector <4 x float> %85, <4 x float> poison, <4 x i32> zeroinitializer
  %87 = insertelement <4 x float> poison, float %80, i64 0
  %88 = shufflevector <4 x float> %87, <4 x float> poison, <4 x i32> zeroinitializer
  %89 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %82, <4 x float> %68, <4 x float> %63)
  %90 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %84, <4 x float> %68, <4 x float> %64)
  %91 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %86, <4 x float> %68, <4 x float> %65)
  %92 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %88, <4 x float> %68, <4 x float> %66)
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %.idx10.3 = shl nuw nsw i64 %indvars.iv.next.2, 11
  %93 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.3
  %94 = load <4 x float>, ptr addrspace(1) %93, align 4
  %95 = add nuw nsw i64 %indvars.iv.next.2, %9
  %96 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %95
  %97 = load float, ptr addrspace(1) %96, align 4
  %98 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %95
  %99 = getelementptr inbounds nuw i8, ptr addrspace(1) %98, i64 2048
  %100 = load float, ptr addrspace(1) %99, align 4
  %101 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %95
  %102 = getelementptr inbounds nuw i8, ptr addrspace(1) %101, i64 4096
  %103 = load float, ptr addrspace(1) %102, align 4
  %104 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %95
  %105 = getelementptr inbounds nuw i8, ptr addrspace(1) %104, i64 6144
  %106 = load float, ptr addrspace(1) %105, align 4
  %107 = insertelement <4 x float> poison, float %97, i64 0
  %108 = shufflevector <4 x float> %107, <4 x float> poison, <4 x i32> zeroinitializer
  %109 = insertelement <4 x float> poison, float %100, i64 0
  %110 = shufflevector <4 x float> %109, <4 x float> poison, <4 x i32> zeroinitializer
  %111 = insertelement <4 x float> poison, float %103, i64 0
  %112 = shufflevector <4 x float> %111, <4 x float> poison, <4 x i32> zeroinitializer
  %113 = insertelement <4 x float> poison, float %106, i64 0
  %114 = shufflevector <4 x float> %113, <4 x float> poison, <4 x i32> zeroinitializer
  %115 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %108, <4 x float> %94, <4 x float> %89)
  %116 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %110, <4 x float> %94, <4 x float> %90)
  %117 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %112, <4 x float> %94, <4 x float> %91)
  %118 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %114, <4 x float> %94, <4 x float> %92)
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %.idx10.4 = shl nuw nsw i64 %indvars.iv.next.3, 11
  %119 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.4
  %120 = load <4 x float>, ptr addrspace(1) %119, align 4
  %121 = add nuw nsw i64 %indvars.iv.next.3, %9
  %122 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %121
  %123 = load float, ptr addrspace(1) %122, align 4
  %124 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %121
  %125 = getelementptr inbounds nuw i8, ptr addrspace(1) %124, i64 2048
  %126 = load float, ptr addrspace(1) %125, align 4
  %127 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %121
  %128 = getelementptr inbounds nuw i8, ptr addrspace(1) %127, i64 4096
  %129 = load float, ptr addrspace(1) %128, align 4
  %130 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %121
  %131 = getelementptr inbounds nuw i8, ptr addrspace(1) %130, i64 6144
  %132 = load float, ptr addrspace(1) %131, align 4
  %133 = insertelement <4 x float> poison, float %123, i64 0
  %134 = shufflevector <4 x float> %133, <4 x float> poison, <4 x i32> zeroinitializer
  %135 = insertelement <4 x float> poison, float %126, i64 0
  %136 = shufflevector <4 x float> %135, <4 x float> poison, <4 x i32> zeroinitializer
  %137 = insertelement <4 x float> poison, float %129, i64 0
  %138 = shufflevector <4 x float> %137, <4 x float> poison, <4 x i32> zeroinitializer
  %139 = insertelement <4 x float> poison, float %132, i64 0
  %140 = shufflevector <4 x float> %139, <4 x float> poison, <4 x i32> zeroinitializer
  %141 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %134, <4 x float> %120, <4 x float> %115)
  %142 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %136, <4 x float> %120, <4 x float> %116)
  %143 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %138, <4 x float> %120, <4 x float> %117)
  %144 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %140, <4 x float> %120, <4 x float> %118)
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %.idx10.5 = shl nuw nsw i64 %indvars.iv.next.4, 11
  %145 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.5
  %146 = load <4 x float>, ptr addrspace(1) %145, align 4
  %147 = add nuw nsw i64 %indvars.iv.next.4, %9
  %148 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %147
  %149 = load float, ptr addrspace(1) %148, align 4
  %150 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %147
  %151 = getelementptr inbounds nuw i8, ptr addrspace(1) %150, i64 2048
  %152 = load float, ptr addrspace(1) %151, align 4
  %153 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %147
  %154 = getelementptr inbounds nuw i8, ptr addrspace(1) %153, i64 4096
  %155 = load float, ptr addrspace(1) %154, align 4
  %156 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %147
  %157 = getelementptr inbounds nuw i8, ptr addrspace(1) %156, i64 6144
  %158 = load float, ptr addrspace(1) %157, align 4
  %159 = insertelement <4 x float> poison, float %149, i64 0
  %160 = shufflevector <4 x float> %159, <4 x float> poison, <4 x i32> zeroinitializer
  %161 = insertelement <4 x float> poison, float %152, i64 0
  %162 = shufflevector <4 x float> %161, <4 x float> poison, <4 x i32> zeroinitializer
  %163 = insertelement <4 x float> poison, float %155, i64 0
  %164 = shufflevector <4 x float> %163, <4 x float> poison, <4 x i32> zeroinitializer
  %165 = insertelement <4 x float> poison, float %158, i64 0
  %166 = shufflevector <4 x float> %165, <4 x float> poison, <4 x i32> zeroinitializer
  %167 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %160, <4 x float> %146, <4 x float> %141)
  %168 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %162, <4 x float> %146, <4 x float> %142)
  %169 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %164, <4 x float> %146, <4 x float> %143)
  %170 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %166, <4 x float> %146, <4 x float> %144)
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %.idx10.6 = shl nuw nsw i64 %indvars.iv.next.5, 11
  %171 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.6
  %172 = load <4 x float>, ptr addrspace(1) %171, align 4
  %173 = add nuw nsw i64 %indvars.iv.next.5, %9
  %174 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %173
  %175 = load float, ptr addrspace(1) %174, align 4
  %176 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %173
  %177 = getelementptr inbounds nuw i8, ptr addrspace(1) %176, i64 2048
  %178 = load float, ptr addrspace(1) %177, align 4
  %179 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %173
  %180 = getelementptr inbounds nuw i8, ptr addrspace(1) %179, i64 4096
  %181 = load float, ptr addrspace(1) %180, align 4
  %182 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %173
  %183 = getelementptr inbounds nuw i8, ptr addrspace(1) %182, i64 6144
  %184 = load float, ptr addrspace(1) %183, align 4
  %185 = insertelement <4 x float> poison, float %175, i64 0
  %186 = shufflevector <4 x float> %185, <4 x float> poison, <4 x i32> zeroinitializer
  %187 = insertelement <4 x float> poison, float %178, i64 0
  %188 = shufflevector <4 x float> %187, <4 x float> poison, <4 x i32> zeroinitializer
  %189 = insertelement <4 x float> poison, float %181, i64 0
  %190 = shufflevector <4 x float> %189, <4 x float> poison, <4 x i32> zeroinitializer
  %191 = insertelement <4 x float> poison, float %184, i64 0
  %192 = shufflevector <4 x float> %191, <4 x float> poison, <4 x i32> zeroinitializer
  %193 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %186, <4 x float> %172, <4 x float> %167)
  %194 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %188, <4 x float> %172, <4 x float> %168)
  %195 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %190, <4 x float> %172, <4 x float> %169)
  %196 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %192, <4 x float> %172, <4 x float> %170)
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %.idx10.7 = shl nuw nsw i64 %indvars.iv.next.6, 11
  %197 = getelementptr i8, ptr addrspace(1) %8, i64 %.idx10.7
  %198 = load <4 x float>, ptr addrspace(1) %197, align 4
  %199 = add nuw nsw i64 %indvars.iv.next.6, %9
  %200 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %199
  %201 = load float, ptr addrspace(1) %200, align 4
  %202 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %199
  %203 = getelementptr inbounds nuw i8, ptr addrspace(1) %202, i64 2048
  %204 = load float, ptr addrspace(1) %203, align 4
  %205 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %199
  %206 = getelementptr inbounds nuw i8, ptr addrspace(1) %205, i64 4096
  %207 = load float, ptr addrspace(1) %206, align 4
  %208 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %199
  %209 = getelementptr inbounds nuw i8, ptr addrspace(1) %208, i64 6144
  %210 = load float, ptr addrspace(1) %209, align 4
  %211 = insertelement <4 x float> poison, float %201, i64 0
  %212 = shufflevector <4 x float> %211, <4 x float> poison, <4 x i32> zeroinitializer
  %213 = insertelement <4 x float> poison, float %204, i64 0
  %214 = shufflevector <4 x float> %213, <4 x float> poison, <4 x i32> zeroinitializer
  %215 = insertelement <4 x float> poison, float %207, i64 0
  %216 = shufflevector <4 x float> %215, <4 x float> poison, <4 x i32> zeroinitializer
  %217 = insertelement <4 x float> poison, float %210, i64 0
  %218 = shufflevector <4 x float> %217, <4 x float> poison, <4 x i32> zeroinitializer
  %219 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %212, <4 x float> %198, <4 x float> %193)
  %220 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %214, <4 x float> %198, <4 x float> %194)
  %221 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %216, <4 x float> %198, <4 x float> %195)
  %222 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %218, <4 x float> %198, <4 x float> %196)
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %exitcond.not.7 = icmp eq i64 %indvars.iv.next.7, 512
  br i1 %exitcond.not.7, label %223, label %10

223:                                              ; preds = %10
  %224 = zext i32 %5 to i64
  %.idx = shl nuw nsw i64 %224, 13
  %225 = getelementptr i8, ptr addrspace(1) %2, i64 %.idx
  %226 = getelementptr [4 x i8], ptr addrspace(1) %225, i64 %6
  store <4 x float> %219, ptr addrspace(1) %226, align 4
  %227 = getelementptr i8, ptr addrspace(1) %225, i64 2048
  %228 = getelementptr [4 x i8], ptr addrspace(1) %227, i64 %6
  store <4 x float> %220, ptr addrspace(1) %228, align 4
  %229 = getelementptr i8, ptr addrspace(1) %225, i64 4096
  %230 = getelementptr [4 x i8], ptr addrspace(1) %229, i64 %6
  store <4 x float> %221, ptr addrspace(1) %230, align 4
  %231 = getelementptr i8, ptr addrspace(1) %225, i64 6144
  %232 = getelementptr [4 x i8], ptr addrspace(1) %231, i64 %6
  store <4 x float> %222, ptr addrspace(1) %232, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.y() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="16,16,1" "agx-upper-bounds:air.thread_position_in_grid"="128,128,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
