; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s
;
; agx-mlir-fuzz.py's seed 491 (opt -O3 of a gpu.func: a guarded row loop of
; two stored statements and a running arith.maximumf from 1.0, stored after
; the loop). Its maximum reaches the guard's merge through the LCSSA phi of a
; dedicated exit, the latch its one predecessor: E444's deferred maximum put
; its result on that edge with SplitEdge, which for an edge that is not
; critical splits the exit after its phis, so the phi kept the latch's
; incoming value, the erased chain's: poison, a wrong maximum stored for
; every thread (L15-E773). The edge now gets a block of its own.

; CHECK: {{^ *}}[[EDGE:[._a-z0-9]+\.deferred]]:
; CHECK: [[R:%[0-9]+]] = select i1 {{%[0-9]+}}, float 0.000000e+00, float %.m.next
; CHECK: %.nan3.lcssa = phi float [ [[R]], %[[EDGE]] ]
; CHECK-NOT: [ poison,

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @k(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree captures(none) %1, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 captures(none) dereferenceable(4) %2) local_unnamed_addr #0 {
entry:
  %.val = load i32, ptr addrspace(4) %2, align 4
  %3 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %4 = icmp sgt i32 %.val, 0
  %5 = shl i32 %3, 8
  br i1 %4, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry
  %invariant.op = or disjoint i32 %5, 8
  %invariant.op1 = or disjoint i32 %5, 27
  %invariant.op3 = or disjoint i32 %5, 26
  %invariant.op5 = or disjoint i32 %5, 31
  %wide.trip.count = zext nneg i32 %.val to i64
  br label %6

6:                                                ; preds = %.lr.ph, %6
  %indvars.iv = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next, %6 ]
  %7 = phi float [ 1.000000e+00, %.lr.ph ], [ %.nan3, %6 ]
  %8 = trunc nuw nsw i64 %indvars.iv to i32
  %.reass = add i32 %invariant.op, %8
  %9 = zext i32 %.reass to i64
  %10 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %9
  %11 = load float, ptr addrspace(1) %10, align 4
  %12 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %9
  %13 = load float, ptr addrspace(1) %12, align 4
  %.nan = tail call float @llvm.maximum.f32(float %11, float %13)
  store float %.nan, ptr addrspace(1) %10, align 4
  %.reass2 = add i32 %invariant.op1, %8
  %14 = zext i32 %.reass2 to i64
  %15 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %14
  %16 = load float, ptr addrspace(1) %15, align 4
  %.reass4 = add i32 %invariant.op3, %8
  %17 = zext i32 %.reass4 to i64
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %17
  %19 = load float, ptr addrspace(1) %18, align 4
  %20 = fsub float %16, %19
  store float %20, ptr addrspace(1) %15, align 4
  %.reass6 = add i32 %invariant.op5, %8
  %21 = zext i32 %.reass6 to i64
  %22 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %21
  %23 = load float, ptr addrspace(1) %22, align 4
  %.nan3 = tail call float @llvm.maximum.f32(float %7, float %23)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %._crit_edge, label %6

._crit_edge:                                      ; preds = %6, %entry
  %.lcssa = phi float [ 1.000000e+00, %entry ], [ %.nan3, %6 ]
  %24 = or disjoint i32 %5, 255
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %25
  store float %.lcssa, ptr addrspace(1) %26, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
