; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; MLIR's 64-bit index arithmetic of positions: (zext(by) << 16) + ((zext(sg)
; << 14) & m) | zext(tx). Every node is below 2^32 (by < 64 by the kernel's
; upper bounds, sg < 32, tx < 1024), so the tree is computed in 32 bits and
; zero-extended once (L15-E405). @unbounded's by << 24 may pass 2^32: it
; stays 64-bit.

declare i32 @llvm.agx.threadgroup.position.in.grid.y()
declare i32 @llvm.agx.simdgroup.index.in.threadgroup()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; CHECK-LABEL: @bounded(
; CHECK:       shl nuw i32
; CHECK:       zext i32
; CHECK-NOT:   shl nuw nsw i64
; CHECK:       ret void
define agx_kernel void @bounded(ptr addrspace(1) %o) #0 {
  %by = call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %sg = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %tx = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %byz = zext i32 %by to i64
  %sgz = zext i32 %sg to i64
  %txz = zext i32 %tx to i64
  %a = shl nuw nsw i64 %byz, 16
  %b = shl nuw nsw i64 %sgz, 14
  %c = and i64 %b, 16384
  %d = add nuw nsw i64 %a, %c
  %e = or disjoint i64 %d, %txz
  store i64 %e, ptr addrspace(1) %o, align 8
  ret void
}

; CHECK-LABEL: @unbounded(
; CHECK:       shl nuw nsw i64
define agx_kernel void @unbounded(ptr addrspace(1) %o) {
  %by = call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %byz = zext i32 %by to i64
  %a = shl nuw nsw i64 %byz, 24
  store i64 %a, ptr addrspace(1) %o, align 8
  ret void
}

; A lane's trip umin(1000 - zext(lane), 1000): the subtraction cannot borrow
; (1000 is at least the lane's bound, 31), and the umin of a value never
; above 1000 is the value (L15-E435). @borrow's 1000 - tx, tx < 1024, may
; borrow: it stays 64-bit.
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i64 @llvm.umin.i64(i64, i64)

; CHECK-LABEL: @lane_trip(
; CHECK:       sub nuw i32 1000,
; CHECK-NOT:   umin
; CHECK:       ret void
define agx_kernel void @lane_trip(ptr addrspace(1) %o) {
  %l = call i32 @llvm.agx.thread.index.in.simdgroup()
  %lz = zext i32 %l to i64
  %s = sub nsw i64 1000, %lz
  %u = call i64 @llvm.umin.i64(i64 %s, i64 1000)
  store i64 %u, ptr addrspace(1) %o, align 8
  ret void
}

; CHECK-LABEL: @borrow(
; CHECK:       sub nsw i64 1000,
; CHECK:       ret void
define agx_kernel void @borrow(ptr addrspace(1) %o) #0 {
  %tx = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %tz = zext i32 %tx to i64
  %s = sub nsw i64 1000, %tz
  %u = call i64 @llvm.umin.i64(i64 %s, i64 1000)
  store i64 %u, ptr addrspace(1) %o, align 8
  ret void
}

attributes #0 = { "agx-threads-per-threadgroup"="1024,1,1" "agx-upper-bounds:air.threadgroup_position_in_grid"="64,64,0" }
