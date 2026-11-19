; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -stop-after=agx-codegenprepare < %s | FileCheck %s

; A 64-bit compare whose operands both lie in [-2^31, 2^31) is the same
; compare of their low words (L15-E427): MLIR's index compares of a padded
; stencil's position, (zext(x) - 1) against sext(n), where x < 1024 by the
; kernel's upper bounds. @wide's sum of two sign extensions may leave the
; range: it stays 64-bit.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: @padded(
; CHECK:       icmp slt i32
; CHECK:       icmp sgt i32
; CHECK-NOT:   icmp {{.*}} i64
; CHECK:       ret void
define agx_kernel void @padded(ptr addrspace(1) %o, i32 %n) #0 {
  %x = call i32 @llvm.agx.thread.position.in.grid.x()
  %xz = zext i32 %x to i64
  %xm = add nsw i64 %xz, -1
  %ns = sext i32 %n to i64
  %lt = icmp slt i64 %xm, %ns
  %gt = icmp sgt i64 %xm, -1
  %in = and i1 %lt, %gt
  %v = zext i1 %in to i32
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: @wide(
; CHECK:       icmp slt i64
; CHECK:       ret void
define agx_kernel void @wide(ptr addrspace(1) %o, i32 %a, i32 %b, i32 %n) #0 {
  %as = sext i32 %a to i64
  %bs = sext i32 %b to i64
  %s = add nsw i64 %as, %bs
  %ns = sext i32 %n to i64
  %lt = icmp slt i64 %s, %ns
  %v = zext i1 %lt to i32
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" }
