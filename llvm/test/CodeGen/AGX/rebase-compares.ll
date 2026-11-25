; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-masked-tail-exits=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rebase-compares=0 -agx-masked-tail-exits=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's causal_rowmax (convert-gpu-to-air, opt -O2): an attention row's
; mask j <= row over j = lane + c, c = 0, 32, ..., beside the load's index
; base + lane + c. The mask compares the index with row + base (neither
; passes 2^32 by the bounds), so j's add goes, and the counter c, then the
; exit test's only, merges into the index (L15-E447): the first loop is its
; load, the compare-select, the index's add, the maximum and the exit test
; (without L15-E476's bound for the mask, which adds to the preheader).

; CHECK-LABEL: causal_rowmax:
; CHECK:       if_fcmp
; CHECK-NEXT:  .LBB0_1:
; CHECK:       device_load
; CHECK-NEXT:  wait 0
; CHECK-NEXT:  icmpsel ugt
; CHECK-NEXT:  iadd
; CHECK-NEXT:  fcmpsel gtn
; CHECK-NEXT:  while_icmp
; OFF-LABEL:   causal_rowmax:
; OFF:         if_fcmp
; OFF-NEXT:    .LBB0_1:
; OFF:         device_load
; OFF-NEXT:    iadd
; OFF-NEXT:    iadd
; OFF-NEXT:    iadd

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @causal_rowmax(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1) local_unnamed_addr #0 {
entry:
  %2 = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %3 = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %4 = zext i32 %3 to i64
  %5 = zext i32 %2 to i64
  %6 = shl i32 %3, 10
  %7 = icmp ult i32 %2, 1024
  %8 = sub nsw i64 1024, %5
  %9 = tail call i64 @llvm.umin.i64(i64 %8, i64 1048576)
  %10 = trunc nuw nsw i64 %9 to i32
  %11 = select i1 %7, i32 %10, i32 0
  %invariant.op = add i32 %6, %2
  %.not11 = icmp eq i32 %11, 0
  br i1 %.not11, label %._crit_edge, label %.lr.ph.preheader

.lr.ph.preheader:                                 ; preds = %entry
  %12 = zext nneg i32 %11 to i64
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.lr.ph
  %indvars.iv = phi i64 [ 0, %.lr.ph.preheader ], [ %indvars.iv.next, %.lr.ph ]
  %13 = phi float [ -inf, %.lr.ph.preheader ], [ %20, %.lr.ph ]
  %14 = add nuw nsw i64 %indvars.iv, %5
  %15 = trunc nuw nsw i64 %indvars.iv to i32
  %.reass = add i32 %invariant.op, %15
  %16 = zext i32 %.reass to i64
  %17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %16
  %18 = load float, ptr addrspace(1) %17, align 4
  %.not3 = icmp samesign ugt i64 %14, %4
  %19 = select i1 %.not3, float -inf, float %18
  %20 = tail call float @llvm.maxnum.f32(float %13, float %19)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %21 = icmp samesign ult i64 %indvars.iv.next, %12
  br i1 %21, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %entry
  %.lcssa4 = phi float [ -inf, %entry ], [ %20, %.lr.ph ]
  %22 = tail call float @llvm.agx.simd.reduce.f32(float %.lcssa4, i32 7)
  br i1 %.not11, label %._crit_edge9, label %.lr.ph8.preheader

.lr.ph8.preheader:                                ; preds = %._crit_edge
  %23 = zext nneg i32 %11 to i64
  br label %.lr.ph8

.lr.ph8:                                          ; preds = %.lr.ph8.preheader, %.lr.ph8
  %indvars.iv15 = phi i64 [ 0, %.lr.ph8.preheader ], [ %indvars.iv.next16, %.lr.ph8 ]
  %24 = phi float [ 0.000000e+00, %.lr.ph8.preheader ], [ %33, %.lr.ph8 ]
  %25 = add nuw nsw i64 %indvars.iv15, %5
  %26 = trunc nuw nsw i64 %indvars.iv15 to i32
  %.reass6 = add i32 %invariant.op, %26
  %27 = zext i32 %.reass6 to i64
  %28 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %27
  %29 = load float, ptr addrspace(1) %28, align 4
  %.not = icmp samesign ugt i64 %25, %4
  %30 = fsub float %29, %22
  %31 = tail call float @llvm.exp.f32(float %30)
  %32 = select i1 %.not, float 0.000000e+00, float %31
  %33 = fadd float %24, %32
  %indvars.iv.next16 = add nuw nsw i64 %indvars.iv15, 32
  %34 = icmp samesign ult i64 %indvars.iv.next16, %23
  br i1 %34, label %.lr.ph8, label %._crit_edge9

._crit_edge9:                                     ; preds = %.lr.ph8, %._crit_edge
  %.lcssa = phi float [ 0.000000e+00, %._crit_edge ], [ %33, %.lr.ph8 ]
  %35 = tail call float @llvm.agx.simd.reduce.f32(float %.lcssa, i32 1)
  %36 = zext i32 %invariant.op to i64
  %37 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %36
  store float %35, ptr addrspace(1) %37, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.index.in.simdgroup() #1

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.agx.simd.reduce.f32(float, i32 immarg) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #3

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="32" "agx-threads-per-threadgroup"="32,1,1" "agx-upper-bounds:air.threadgroup_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
