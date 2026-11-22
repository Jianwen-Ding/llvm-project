; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-narrow-inductions < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-narrow-inductions -agx-merge-lagging-counters=0 < %s | FileCheck %s --check-prefix=OFF

; find_first (an MLIR search loop with an early exit): loop strength
; reduction counts the exit value from -1 beside the index from 0, which
; narrowing makes 32-bit. The count is one step behind the index, so its
; increment is the index itself: one phi, the exit value the index
; (L15-E794).

; CHECK-LABEL: @find_first(
; CHECK: [[J:%[.a-z0-9]+]] = phi i32
; CHECK-NOT: phi i32
; CHECK: store i32 [[J]],
; OFF-LABEL: @find_first(
; OFF-COUNT-2: phi i32
target triple = "agx"

define agx_kernel void @find_first(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree writeonly captures(none) %1, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %2) local_unnamed_addr {
entry:
  %.val = load float, ptr addrspace(4) %2, align 4
  %3 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %4 = shl i32 %3, 8
  %5 = zext i32 %4 to i64
  %invariant.gep = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %0, i64 %5
  br label %6

6:                                                ; preds = %6, %entry
  %7 = phi i64 [ %13, %6 ], [ 0, %entry ]
  %8 = icmp samesign ult i64 %7, 256
  %9 = tail call i64 @llvm.umin.i64(i64 %7, i64 255)
  %gep = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %invariant.gep, i64 %9
  %10 = load float, ptr addrspace(1) %gep, align 4
  %11 = fcmp ole float %10, %.val
  %12 = select i1 %8, i1 %11, i1 false
  %13 = add nuw nsw i64 %7, 1
  br i1 %12, label %6, label %14

14:                                               ; preds = %6
  %15 = zext i32 %3 to i64
  %16 = trunc nuw nsw i64 %7 to i32
  %17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %1, i64 %15
  store i32 %16, ptr addrspace(1) %17, align 4
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i64 @llvm.umin.i64(i64, i64)
