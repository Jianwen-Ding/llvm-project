; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-core-occupancy-per-load=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's bswap_rows (a byte swap of each word of a row, the loop unrolled
; with four 16-byte loads): the machine scheduler hoists the loads ahead of
; the shifts, 30 -> 43 registers, which still lets 1024 threads in a
; threadgroup but cuts the SIMD-groups resident on a core 82 -> 36. With 91
; instructions for its 4 loads (16 or more each), the region keeps its
; original order, the resident groups hiding the loads' latency: tput -8.8%
; (L15-E681).

; CHECK: general purpose registers: 30 x 32-bit
; OFF: general purpose registers: 43 x 32-bit

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @bswap_rows(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1) local_unnamed_addr #0 {
entry:
  %2 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %3 = shl i32 %2, 8
  %4 = zext i32 %3 to i64
  br label %5

5:                                                ; preds = %5, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.15, %5 ]
  %6 = add nuw nsw i64 %indvars.iv, %4
  %7 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %6
  %8 = load i32, ptr addrspace(1) %7, align 4
  %9 = tail call i32 @llvm.bswap.i32(i32 %8)
  %10 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %6
  store i32 %9, ptr addrspace(1) %10, align 4
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %11 = add nuw nsw i64 %indvars.iv.next, %4
  %12 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %11
  %13 = load i32, ptr addrspace(1) %12, align 4
  %14 = tail call i32 @llvm.bswap.i32(i32 %13)
  %15 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %11
  store i32 %14, ptr addrspace(1) %15, align 4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %16 = add nuw nsw i64 %indvars.iv.next.1, %4
  %17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %16
  %18 = load i32, ptr addrspace(1) %17, align 4
  %19 = tail call i32 @llvm.bswap.i32(i32 %18)
  %20 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %16
  store i32 %19, ptr addrspace(1) %20, align 4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %21 = add nuw nsw i64 %indvars.iv.next.2, %4
  %22 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %21
  %23 = load i32, ptr addrspace(1) %22, align 4
  %24 = tail call i32 @llvm.bswap.i32(i32 %23)
  %25 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %21
  store i32 %24, ptr addrspace(1) %25, align 4
  %indvars.iv.next.3 = or disjoint i64 %indvars.iv, 4
  %26 = add nuw nsw i64 %indvars.iv.next.3, %4
  %27 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %26
  %28 = load i32, ptr addrspace(1) %27, align 4
  %29 = tail call i32 @llvm.bswap.i32(i32 %28)
  %30 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %26
  store i32 %29, ptr addrspace(1) %30, align 4
  %indvars.iv.next.4 = or disjoint i64 %indvars.iv, 5
  %31 = add nuw nsw i64 %indvars.iv.next.4, %4
  %32 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %31
  %33 = load i32, ptr addrspace(1) %32, align 4
  %34 = tail call i32 @llvm.bswap.i32(i32 %33)
  %35 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %31
  store i32 %34, ptr addrspace(1) %35, align 4
  %indvars.iv.next.5 = or disjoint i64 %indvars.iv, 6
  %36 = add nuw nsw i64 %indvars.iv.next.5, %4
  %37 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %36
  %38 = load i32, ptr addrspace(1) %37, align 4
  %39 = tail call i32 @llvm.bswap.i32(i32 %38)
  %40 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %36
  store i32 %39, ptr addrspace(1) %40, align 4
  %indvars.iv.next.6 = or disjoint i64 %indvars.iv, 7
  %41 = add nuw nsw i64 %indvars.iv.next.6, %4
  %42 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %41
  %43 = load i32, ptr addrspace(1) %42, align 4
  %44 = tail call i32 @llvm.bswap.i32(i32 %43)
  %45 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %41
  store i32 %44, ptr addrspace(1) %45, align 4
  %indvars.iv.next.7 = or disjoint i64 %indvars.iv, 8
  %46 = add nuw nsw i64 %indvars.iv.next.7, %4
  %47 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %46
  %48 = load i32, ptr addrspace(1) %47, align 4
  %49 = tail call i32 @llvm.bswap.i32(i32 %48)
  %50 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %46
  store i32 %49, ptr addrspace(1) %50, align 4
  %indvars.iv.next.8 = or disjoint i64 %indvars.iv, 9
  %51 = add nuw nsw i64 %indvars.iv.next.8, %4
  %52 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %51
  %53 = load i32, ptr addrspace(1) %52, align 4
  %54 = tail call i32 @llvm.bswap.i32(i32 %53)
  %55 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %51
  store i32 %54, ptr addrspace(1) %55, align 4
  %indvars.iv.next.9 = or disjoint i64 %indvars.iv, 10
  %56 = add nuw nsw i64 %indvars.iv.next.9, %4
  %57 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %56
  %58 = load i32, ptr addrspace(1) %57, align 4
  %59 = tail call i32 @llvm.bswap.i32(i32 %58)
  %60 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %56
  store i32 %59, ptr addrspace(1) %60, align 4
  %indvars.iv.next.10 = or disjoint i64 %indvars.iv, 11
  %61 = add nuw nsw i64 %indvars.iv.next.10, %4
  %62 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %61
  %63 = load i32, ptr addrspace(1) %62, align 4
  %64 = tail call i32 @llvm.bswap.i32(i32 %63)
  %65 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %61
  store i32 %64, ptr addrspace(1) %65, align 4
  %indvars.iv.next.11 = or disjoint i64 %indvars.iv, 12
  %66 = add nuw nsw i64 %indvars.iv.next.11, %4
  %67 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %66
  %68 = load i32, ptr addrspace(1) %67, align 4
  %69 = tail call i32 @llvm.bswap.i32(i32 %68)
  %70 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %66
  store i32 %69, ptr addrspace(1) %70, align 4
  %indvars.iv.next.12 = or disjoint i64 %indvars.iv, 13
  %71 = add nuw nsw i64 %indvars.iv.next.12, %4
  %72 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %71
  %73 = load i32, ptr addrspace(1) %72, align 4
  %74 = tail call i32 @llvm.bswap.i32(i32 %73)
  %75 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %71
  store i32 %74, ptr addrspace(1) %75, align 4
  %indvars.iv.next.13 = or disjoint i64 %indvars.iv, 14
  %76 = add nuw nsw i64 %indvars.iv.next.13, %4
  %77 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %76
  %78 = load i32, ptr addrspace(1) %77, align 4
  %79 = tail call i32 @llvm.bswap.i32(i32 %78)
  %80 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %76
  store i32 %79, ptr addrspace(1) %80, align 4
  %indvars.iv.next.14 = or disjoint i64 %indvars.iv, 15
  %81 = add nuw nsw i64 %indvars.iv.next.14, %4
  %82 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %81
  %83 = load i32, ptr addrspace(1) %82, align 4
  %84 = tail call i32 @llvm.bswap.i32(i32 %83)
  %85 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %81
  store i32 %84, ptr addrspace(1) %85, align 4
  %indvars.iv.next.15 = add nuw nsw i64 %indvars.iv, 16
  %exitcond.not.15 = icmp eq i64 %indvars.iv.next.15, 256
  br i1 %exitcond.not.15, label %86, label %5

86:                                               ; preds = %5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.bswap.i32(i32) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
