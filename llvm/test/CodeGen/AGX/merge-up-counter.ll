; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-merge-up-counters=0 < %s | FileCheck %s --check-prefix=OFF

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
