; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-up-counters=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-up-counters-variable-step=0 < %s | FileCheck %s --check-prefix=VOFF

; An MLIR strided loop from the lane in steps of 32 after narrowing: a
; counter from zero against the trip n (min(n - lane, 1024)), beside the
; index (row * 1024 + lane) that addresses the loads, each stepped by an
; iadd. The counter only the exit test uses, the exit is the index reaching
; its start + t, t the first multiple of 32 at or above max(n, 1), exact mod
; 2^32 even where the start (a threadgroup position's row) is unbounded
; (L15-E386). Bounded starts keep <u against start + n; a step that is not
; a power of two with an unbounded start, or an unbounded n, keeps the
; counter.

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare float @llvm.maxnum.f32(float, float)
declare i64 @llvm.umin.i64(i64, i64)

; (the loop entered only for a trip != 0: no max(trip, 1), L15-E403)
; CHECK-LABEL: rowmax:
; CHECK-NOT: icmpsel ugt, r{{[0-9]+}}, r{{[0-9]+}}, 1, r{{[0-9]+}}, 1
; CHECK: and{{[[:space:]]}}r
; CHECK: Loop Header
; CHECK: iadd
; CHECK-NOT: iadd
; CHECK: while_icmp r0l, nueq, r{{[0-9]+}}, r{{[0-9]+}}, 1
; OFF-LABEL: rowmax:
; OFF: Loop Header
; OFF: iadd r{{[0-9]+}}, r{{[0-9]+}}, 32
; OFF: iadd r{{[0-9]+}}, r{{[0-9]+}}, 32
; OFF: while_icmp r0l, ult
define agx_kernel void @rowmax(ptr addrspace(1) %x, ptr addrspace(1) %out, i64 %n) {
entry:
  %lane = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %bx = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %lane64 = zext i32 %lane to i64
  %has = icmp sgt i64 %n, %lane64
  %rem = sub i64 %n, %lane64
  %cl = tail call i64 @llvm.umin.i64(i64 %rem, i64 1024)
  %cl32 = trunc nuw nsw i64 %cl to i32
  %trip = select i1 %has, i32 %cl32, i32 0
  %none = icmp eq i32 %trip, 0
  br i1 %none, label %exit, label %pre

pre:
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %loop ]
  %m = phi float [ 0xFFF0000000000000, %pre ], [ %m.next, %loop ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %loop, label %exit

exit:
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %loop ]
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; A bounded start (a thread position): <u start + n, no rounding.
; CHECK-LABEL: bounded:
; CHECK-NOT: and{{[[:space:]]}}r
; CHECK: Loop Header
; CHECK: iadd
; CHECK-NOT: iadd
; CHECK: while_icmp r0l, ult, r{{[0-9]+}}, r{{[0-9]+}}, 1
define agx_kernel void @bounded(ptr addrspace(1) %x, ptr addrspace(1) %out, i64 %n) {
entry:
  %lane = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %tx = tail call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %lane64 = zext i32 %lane to i64
  %has = icmp sgt i64 %n, %lane64
  %rem = sub i64 %n, %lane64
  %cl = tail call i64 @llvm.umin.i64(i64 %rem, i64 1024)
  %cl32 = trunc nuw nsw i64 %cl to i32
  %trip = select i1 %has, i32 %cl32, i32 0
  %none = icmp eq i32 %trip, 0
  br i1 %none, label %exit, label %pre

pre:
  %start = shl i32 %tx, 5
  %trip64 = zext nneg i32 %trip to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %loop ]
  %m = phi float [ 0xFFF0000000000000, %pre ], [ %m.next, %loop ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %loop, label %exit

exit:
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %loop ]
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; Step 24 from an unbounded start: the counter stays.
; CHECK-LABEL: step24:
; CHECK: Loop Header
; CHECK: iadd r{{[0-9]+}}, r{{[0-9]+}}, 24
; CHECK: iadd r{{[0-9]+}}, r{{[0-9]+}}, 24
; CHECK: while_icmp r0l, ult
define agx_kernel void @step24(ptr addrspace(1) %x, ptr addrspace(1) %out, i64 %n) {
entry:
  %lane = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %bx = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %lane64 = zext i32 %lane to i64
  %has = icmp sgt i64 %n, %lane64
  %rem = sub i64 %n, %lane64
  %cl = tail call i64 @llvm.umin.i64(i64 %rem, i64 1024)
  %cl32 = trunc nuw nsw i64 %cl to i32
  %trip = select i1 %has, i32 %cl32, i32 0
  %none = icmp eq i32 %trip, 0
  br i1 %none, label %exit, label %pre

pre:
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %loop ]
  %m = phi float [ 0xFFF0000000000000, %pre ], [ %m.next, %loop ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 24
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %loop, label %exit

exit:
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %loop ]
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

; The kernel's upper bound on the threadgroup position (MLIR's block_id
; upper_bound 256, L15-E391) bounds the start: <u start + n, no rounding.
; CHECK-LABEL: bounded_grid:
; CHECK-NOT: and{{[[:space:]]}}r
; CHECK: Loop Header
; CHECK: iadd
; CHECK-NOT: iadd
; CHECK: while_icmp r0l, ult, r{{[0-9]+}}, r{{[0-9]+}}, 1
define agx_kernel void @bounded_grid(ptr addrspace(1) %x, ptr addrspace(1) %out, i64 %n) #0 {
entry:
  %lane = tail call i32 @llvm.agx.thread.index.in.simdgroup()
  %bx = tail call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %lane64 = zext i32 %lane to i64
  %has = icmp sgt i64 %n, %lane64
  %rem = sub i64 %n, %lane64
  %cl = tail call i64 @llvm.umin.i64(i64 %rem, i64 1024)
  %cl32 = trunc nuw nsw i64 %cl to i32
  %trip = select i1 %has, i32 %cl32, i32 0
  %none = icmp eq i32 %trip, 0
  br i1 %none, label %exit, label %pre

pre:
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %loop ]
  %m = phi float [ 0xFFF0000000000000, %pre ], [ %m.next, %loop ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %loop, label %exit

exit:
  %r = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %loop ]
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %q, align 4
  ret void
}

attributes #0 = { "agx-upper-bounds:air.threadgroup_position_in_grid"="256,0,0" }

; A grid-stride loop (MLIR's from the grid position by block_dim * grid_dim,
; after narrowing): the step is invariant but not a constant; bounded start,
; trip and step keep <u against start + n (L15-E417): one iadd a trip.
; CHECK-LABEL: grid_stride:
; CHECK: .LBB4_2:
; CHECK-NOT: iadd
; CHECK: iadd r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK-NOT: iadd
; CHECK: while_icmp r0l, ult
; VOFF-LABEL: grid_stride:
; VOFF: .LBB4_2:
; VOFF: iadd
; VOFF: iadd
; VOFF: while_icmp r0l, ult
define agx_kernel void @grid_stride(ptr addrspace(1) nofree readonly captures(none) %0, ptr addrspace(1) nofree captures(none) %1, ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 captures(none) dereferenceable(4) %2, ptr addrspace(4) noalias noundef nonnull readonly byref(i64) align 8 captures(none) dereferenceable(8) %3) #9 {
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
declare noundef i32 @llvm.agx.thread.position.in.grid.x()
declare noundef i32 @llvm.agx.threadgroups.per.grid.x()
declare float @llvm.fma.f32(float, float, float)
attributes #9 = { "agx-max-threads-per-threadgroup"="256" "agx-threads-per-threadgroup"="256,1,1" "agx-upper-bounds:air.thread_position_in_grid"="1048576,0,0" }
