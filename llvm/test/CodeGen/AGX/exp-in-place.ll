; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-exp-in-place-stores=0 < %s | FileCheck %s --check-prefix=OFF

; e^(x - m), m an earlier loop's maximum of the values x's loop loads again
; (L15-E270), where x's loop also stores to the address it loaded x from: a
; softmax's in-place last pass (MLIR bench softmax_rows). Each iteration's
; store follows its load, and no other iteration's access overlaps it (the
; index steps by 32 elements over at most 32 iterations), so every x is
; still a value the maximum took in and e^(x - m) cannot overflow: no clamp
; above 89, no +inf select (L15-E387). @next stores to the next element,
; not the address it loaded (only its own is accepted); @short steps by 2
; bytes, less than the 4 it accesses, so iterations overlap: each keeps
; both.

; CHECK-LABEL: inplace:
; CHECK:       Loop Header
; CHECK:       Loop Header
; CHECK-NOT:   icmpsel sgt
; CHECK:       device_store
; CHECK:       jmp_exec_any
; OFF-LABEL:   inplace:
; OFF:         Loop Header
; OFF:         Loop Header
; OFF:         icmpsel sgt
; OFF:         device_store
; CHECK-LABEL: next:
; CHECK:       Loop Header
; CHECK:       Loop Header
; CHECK:       icmpsel sgt
; CHECK-LABEL: short:
; CHECK:       Loop Header
; CHECK:       Loop Header
; CHECK:       icmpsel sgt

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.exp.f32(float)
declare float @llvm.agx.simd.reduce.f32(float, i32)
declare i64 @llvm.umin.i64(i64, i64)

define agx_kernel void @inplace(ptr addrspace(1) %x, i64 %n) {
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
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br i1 %none, label %exit, label %l1

l1:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %l1 ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %l1 ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %l1, label %mid

mid:
  %mr = tail call float @llvm.agx.simd.reduce.f32(float %m.next, i32 7)
  br label %l2

l2:
  %jv = phi i64 [ 0, %mid ], [ %jv.next, %l2 ]
  %jv32 = trunc nuw nsw i64 %jv to i32
  %j = add i32 %start, %jv32
  %j64 = zext i32 %j to i64
  %q = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %j64
  %w = load float, ptr addrspace(1) %q, align 4
  %d = fsub float %w, %mr
  %e = tail call float @llvm.exp.f32(float %d)
  %qs = getelementptr inbounds nuw float, ptr addrspace(1) %q, i64 0
  store float %e, ptr addrspace(1) %qs, align 4
  %jv.next = add nuw nsw i64 %jv, 32
  %c2 = icmp samesign ult i64 %jv.next, %trip64
  br i1 %c2, label %l2, label %exit

exit:
  ret void
}

define agx_kernel void @next(ptr addrspace(1) %x, i64 %n) {
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
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br i1 %none, label %exit, label %l1

l1:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %l1 ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %l1 ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %l1, label %mid

mid:
  %mr = tail call float @llvm.agx.simd.reduce.f32(float %m.next, i32 7)
  br label %l2

l2:
  %jv = phi i64 [ 0, %mid ], [ %jv.next, %l2 ]
  %jv32 = trunc nuw nsw i64 %jv to i32
  %j = add i32 %start, %jv32
  %j64 = zext i32 %j to i64
  %q = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %j64
  %w = load float, ptr addrspace(1) %q, align 4
  %d = fsub float %w, %mr
  %e = tail call float @llvm.exp.f32(float %d)
  %qs = getelementptr inbounds nuw float, ptr addrspace(1) %q, i64 1
  store float %e, ptr addrspace(1) %qs, align 4
  %jv.next = add nuw nsw i64 %jv, 32
  %c2 = icmp samesign ult i64 %jv.next, %trip64
  br i1 %c2, label %l2, label %exit

exit:
  ret void
}

define agx_kernel void @short(ptr addrspace(1) %x, i64 %n) {
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
  %row = shl i32 %bx, 10
  %start = add i32 %lane, %row
  %trip64 = zext nneg i32 %trip to i64
  br i1 %none, label %exit, label %l1

l1:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %l1 ]
  %m = phi float [ 0xFFF0000000000000, %entry ], [ %m.next, %l1 ]
  %iv32 = trunc nuw nsw i64 %iv to i32
  %i = add i32 %start, %iv32
  %i64 = zext i32 %i to i64
  %p = getelementptr inbounds nuw i16, ptr addrspace(1) %x, i64 %i64
  %v = load float, ptr addrspace(1) %p, align 4
  %m.next = tail call float @llvm.maxnum.f32(float %m, float %v)
  %iv.next = add nuw nsw i64 %iv, 32
  %c = icmp samesign ult i64 %iv.next, %trip64
  br i1 %c, label %l1, label %mid

mid:
  %mr = tail call float @llvm.agx.simd.reduce.f32(float %m.next, i32 7)
  br label %l2

l2:
  %jv = phi i64 [ 0, %mid ], [ %jv.next, %l2 ]
  %jv32 = trunc nuw nsw i64 %jv to i32
  %j = add i32 %start, %jv32
  %j64 = zext i32 %j to i64
  %q = getelementptr inbounds nuw i16, ptr addrspace(1) %x, i64 %j64
  %w = load float, ptr addrspace(1) %q, align 4
  %d = fsub float %w, %mr
  %e = tail call float @llvm.exp.f32(float %d)
  %qs = getelementptr inbounds nuw float, ptr addrspace(1) %q, i64 0
  store float %e, ptr addrspace(1) %qs, align 4
  %jv.next = add nuw nsw i64 %jv, 32
  %c2 = icmp samesign ult i64 %jv.next, %trip64
  br i1 %c2, label %l2, label %exit

exit:
  ret void
}
