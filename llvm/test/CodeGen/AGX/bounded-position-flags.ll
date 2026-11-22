; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-lower-air < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-lower-air -agx-bounded-position-flags=0 < %s | FileCheck %s --check-prefix=OFF

; A row's base, row << 10 with row below 1024 by its upper_bound, read in a
; loop: nuw nsw before the passes that keep the loop's index in 32 bits.
; Not in straight-line code, nor under a zext.

; CHECK-LABEL: @rows(
; CHECK: %base = shl nuw nsw i32 %row, 10
; CHECK: %once = shl i32 %row, 9
; CHECK: %wide = shl i32 %row, 8
; OFF: %base = shl i32 %row, 10
define agx_kernel void @rows(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
entry:
  %row = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %base = shl i32 %row, 10
  %start = add i32 %base, %lane
  %once = shl i32 %row, 9
  %oz = zext i32 %once to i64
  %op = getelementptr float, ptr addrspace(1) %q, i64 %oz
  store float 0.0, ptr addrspace(1) %op
  %wide = shl i32 %row, 8
  %wz = zext i32 %wide to i64
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j.next, %loop ]
  %s = phi float [ 0.0, %entry ], [ %s.next, %loop ]
  %k = add i32 %start, %j
  %kz = zext i32 %k to i64
  %a = getelementptr half, ptr addrspace(1) %p, i64 %kz
  %h = load half, ptr addrspace(1) %a
  %f = fpext half %h to float
  %s.next = fadd float %s, %f
  %j.next = add nuw nsw i32 %j, 32
  %done = icmp eq i32 %j.next, 1024
  br i1 %done, label %exit, label %loop
exit:
  %w = getelementptr float, ptr addrspace(1) %q, i64 %wz
  store float %s.next, ptr addrspace(1) %w
  ret void
}

declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.thread.index.in.simdgroup()

attributes #0 = { "agx-upper-bounds:air.threadgroup_position_in_grid"="1024,0,0" }
