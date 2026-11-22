; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
;
; b[x + i] += c[i]: the load and the store of b share one gep, which
; rebaseLoopIndices (L15-E398) rewrites as (b + x)[i] once for both. It
; collected each access's gep, so this one twice, and the second visit read
; the gep the first had erased (L15-E769: a crash, seen first in
; stencil_row under -agx-addressing-modes=0, where another rewrite had
; merged two loads' addresses).

; CHECK-LABEL: inplace_offset:
; CHECK: device_load 0, i32, x, [[V:r[0-9]+]], [[B:r[0-9]+_r[0-9]+]], [[I:r[0-9]+]], unsigned
; CHECK: device_load 0, i32, x, {{r[0-9]+}}, u2_u3, [[I]], unsigned
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, [[B]], [[I]], unsigned

target triple = "agx"
declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @inplace_offset(ptr addrspace(1) noalias %b, ptr addrspace(1) noalias %c) "agx-upper-bounds:air.thread_position_in_grid"="1024,0,0" {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %x = shl nuw nsw i32 %t, 8
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %k = add nuw nsw i32 %x, %i
  %z = zext i32 %k to i64
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %z
  %v = load float, ptr addrspace(1) %pb, align 4
  %zi = zext i32 %i to i64
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i64 %zi
  %w = load float, ptr addrspace(1) %pc, align 4
  %o = fadd float %v, %w
  store float %o, ptr addrspace(1) %pb, align 4
  %i1 = add nuw nsw i32 %i, 1
  %e = icmp ult i32 %i1, 256
  br i1 %e, label %loop, label %exit
exit:
  ret void
}
