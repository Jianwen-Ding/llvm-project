; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-sched-loads-after-stores=0 < %s | FileCheck %s --check-prefix=OFF

; A loop over a thread's own row (its address the thread position times 512)
; of light work: the second half's loads stay after the first half's store
; (L15-E589), which streamed the core's loads and stores in turn with many
; SIMD-groups resident.
; CHECK-LABEL: rows:
; CHECK: device_load
; CHECK: device_store
; CHECK: device_load
; CHECK: device_store
; OFF-LABEL: rows:
; OFF: device_load
; OFF-NEXT: device_load
; OFF: device_store

; A column (the thread position not scaled): both loads first.
; CHECK-LABEL: cols:
; CHECK: device_load
; CHECK-NEXT: device_load
; CHECK: device_store

declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @rows(ptr addrspace(1) noalias %x, ptr addrspace(1) noalias %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = shl i32 %t, 9
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %k = add i32 %row, %j
  %z = zext i32 %k to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %z
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  %v = load <4 x float>, ptr addrspace(1) %p, align 4
  %m = fmul <4 x float> %v, <float 2.0, float 2.0, float 2.0, float 2.0>
  store <4 x float> %m, ptr addrspace(1) %q, align 4
  %p4 = getelementptr float, ptr addrspace(1) %p, i64 4
  %q4 = getelementptr float, ptr addrspace(1) %q, i64 4
  %v4 = load <4 x float>, ptr addrspace(1) %p4, align 4
  %m4 = fmul <4 x float> %v4, <float 2.0, float 2.0, float 2.0, float 2.0>
  store <4 x float> %m4, ptr addrspace(1) %q4, align 4
  %j1 = add i32 %j, 8
  %c = icmp ult i32 %j1, 512
  br i1 %c, label %loop, label %out
out:
  ret void
}

define agx_kernel void @cols(ptr addrspace(1) noalias %x, ptr addrspace(1) noalias %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %r = shl i32 %i, 10
  %k = add i32 %r, %t
  %z = zext i32 %k to i64
  %p = getelementptr float, ptr addrspace(1) %x, i64 %z
  %q = getelementptr float, ptr addrspace(1) %y, i64 %z
  %v = load float, ptr addrspace(1) %p, align 4
  %m = fmul float %v, 2.0
  store float %m, ptr addrspace(1) %q, align 4
  %k2 = add i32 %k, 1024
  %z2 = zext i32 %k2 to i64
  %p2 = getelementptr float, ptr addrspace(1) %x, i64 %z2
  %q2 = getelementptr float, ptr addrspace(1) %y, i64 %z2
  %v2 = load float, ptr addrspace(1) %p2, align 4
  %m2 = fmul float %v2, 2.0
  store float %m2, ptr addrspace(1) %q2, align 4
  %i1 = add i32 %i, 2
  %c = icmp ult i32 %i1, 256
  br i1 %c, label %loop, label %out
out:
  ret void
}
