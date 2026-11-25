; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-latch-bound-narrowing=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's grid-stride loop over a 2^20-element memref (the bench's
; grid_stride): its trip count is an i64 counter {0, +, umin(grid * 256,
; 2^20)} compared with umin(n - i, 2^20). Scalar evolution's range of it is
; its maximum trip count times the step's maximum, near 2^32; the latch's
; exit test bounds it by 2^20 - 1 and the increment by 2^21: a 32-bit
; counter and compare, not a 64-bit addition and three compare-selects
; (L15-E265).

; CHECK-LABEL: grid_stride:
; CHECK:       // %.lr.ph{{$}}
; CHECK-NOT:   iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK-NOT:   icmpsel
; CHECK:       while_icmp r0l, ult, r{{[0-9]+}}, r{{[0-9]+}}, 1
; OFF-LABEL:   grid_stride:
; OFF:         // %.lr.ph{{$}}
; OFF:         iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}

target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define agx_kernel void @grid_stride(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree captures(none) %1, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i64) align 8 captures(none) dereferenceable(8) %3) local_unnamed_addr #0 {
entry:
  %.val = load float, ptr addrspace(4) %2, align 4
  %.val1 = load i64, ptr addrspace(4) %3, align 8
  %4 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %5 = zext i32 %4 to i64
  %6 = tail call i32 @llvm.agx.threadgroups.per.grid.x()
  %7 = zext i32 %6 to i64
  %8 = shl nuw nsw i64 %7, 8
  %9 = icmp sgt i64 %.val1, %5
  %10 = sub nsw i64 %.val1, %5
  %11 = tail call i64 @llvm.umin.i64(i64 %10, i64 1048576)
  %12 = tail call i64 @llvm.umin.i64(i64 %8, i64 1048576)
  br i1 %9, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %entry, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %entry ]
  %13 = trunc nuw nsw i64 %indvars.iv to i32
  %14 = add i32 %4, %13
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %15
  %17 = load float, ptr addrspace(1) %16, align 4
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %15
  %19 = load float, ptr addrspace(1) %18, align 4
  %20 = tail call float @llvm.fma.f32(float %.val, float %17, float %19)
  store float %20, ptr addrspace(1) %18, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, %12
  %21 = icmp samesign ult i64 %indvars.iv.next, %11
  br i1 %21, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroups.per.grid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
