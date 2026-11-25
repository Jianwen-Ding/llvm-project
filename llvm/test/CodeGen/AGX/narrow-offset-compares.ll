; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-narrow-offset-compares=0 < %s | FileCheck %s --check-prefix=OFF
;
; MLIR's int8-weight GEMV (vector<4xi8> weights, unrolled by 4): loop
; strength reduction writes the exit test i | 12 < 1020 as
; zext(i') - 4 <u 1020 on the narrowed 32-bit counter i' = i + 16, a 64-bit
; subtraction and compare; as i' <u 1024 it is the branch's own 32-bit
; compare (L15-E496). Both copies of the alignment-versioned loop.

; CHECK-LABEL: gemv_q8f:
; CHECK-NOT: isub {{r[0-9]+}}_{{r[0-9]+}}
; CHECK: while_icmp r0l, ulte,
; CHECK-NOT: isub {{r[0-9]+}}_{{r[0-9]+}}
; CHECK: while_icmp r0l, ulte,
; CHECK: stop
; OFF-LABEL: gemv_q8f:
; OFF: isub {{r[0-9]+}}_{{r[0-9]+}}, {{r[0-9]+}}_{{r[0-9]+}}, {{4|8}}
; OFF: while_icmp r0l, ueq,

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @gemv_q8f(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree readonly captures(none) %1, ptr addrspace(1) nofree readonly captures(none) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #0 {
entry:
  %4 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %5 = zext i32 %4 to i64
  %6 = shl nuw nsw i64 %5, 10
  %7 = getelementptr i8, ptr addrspace(1) %0, i64 %6
  br label %8

8:                                                ; preds = %8, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.3, %8 ]
  %9 = phi <4 x float> [ zeroinitializer, %entry ], [ %33, %8 ]
  %10 = getelementptr i8, ptr addrspace(1) %7, i64 %indvars.iv
  %11 = load <4 x i8>, ptr addrspace(1) %10, align 1
  %12 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv
  %13 = load <4 x float>, ptr addrspace(1) %12, align 4
  %14 = sitofp <4 x i8> %11 to <4 x float>
  %15 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %14, <4 x float> %13, <4 x float> %9)
  %indvars.iv.next = or disjoint i64 %indvars.iv, 4
  %16 = getelementptr i8, ptr addrspace(1) %7, i64 %indvars.iv.next
  %17 = load <4 x i8>, ptr addrspace(1) %16, align 1
  %18 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next
  %19 = load <4 x float>, ptr addrspace(1) %18, align 4
  %20 = sitofp <4 x i8> %17 to <4 x float>
  %21 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %20, <4 x float> %19, <4 x float> %15)
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 8
  %22 = getelementptr i8, ptr addrspace(1) %7, i64 %indvars.iv.next.1
  %23 = load <4 x i8>, ptr addrspace(1) %22, align 1
  %24 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.1
  %25 = load <4 x float>, ptr addrspace(1) %24, align 4
  %26 = sitofp <4 x i8> %23 to <4 x float>
  %27 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %26, <4 x float> %25, <4 x float> %21)
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 12
  %28 = getelementptr i8, ptr addrspace(1) %7, i64 %indvars.iv.next.2
  %29 = load <4 x i8>, ptr addrspace(1) %28, align 1
  %30 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %indvars.iv.next.2
  %31 = load <4 x float>, ptr addrspace(1) %30, align 4
  %32 = sitofp <4 x i8> %29 to <4 x float>
  %33 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %32, <4 x float> %31, <4 x float> %27)
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 16
  %34 = icmp samesign ult i64 %indvars.iv.next.2, 1020
  br i1 %34, label %8, label %35

35:                                               ; preds = %8
  %36 = extractelement <4 x float> %33, i64 0
  %37 = extractelement <4 x float> %33, i64 1
  %38 = fadd float %36, %37
  %39 = extractelement <4 x float> %33, i64 2
  %40 = fadd float %39, %38
  %41 = extractelement <4 x float> %33, i64 3
  %42 = fadd float %41, %40
  %43 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %2, i64 %5
  %44 = load float, ptr addrspace(1) %43, align 4
  %45 = fmul float %42, %44
  %46 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %3, i64 %5
  store float %45, ptr addrspace(1) %46, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }


