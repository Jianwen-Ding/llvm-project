; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-rebase-loop-indices=0 < %s | FileCheck %s --check-prefix=OFF

; MLIR's FIR filter (bench fir_taps): x[i + k] and w[k] over the taps k <
; min(K, 64), i a bounded thread position. As (x + 4 i)[k], both loads take
; the counter k as their index: one add an iteration, not two (L15-E398).
; @alone has no access indexed by k besides x[i + k]: it keeps its index.

declare i32 @llvm.agx.thread.position.in.grid.x()
declare i64 @llvm.umin.i64(i64, i64)
declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: fir:
; CHECK:       Loop Header
; CHECK:       iadd
; CHECK-NOT:   iadd
; CHECK:       while_icmp
; OFF-LABEL:   fir:
; OFF:         Loop Header
; OFF:         iadd
; OFF:         iadd
; OFF:         while_icmp
define agx_kernel void @fir(ptr addrspace(1) %x, ptr addrspace(1) %w, ptr addrspace(1) %y, i64 %kk) #0 {
entry:
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %none = icmp slt i64 %kk, 1
  br i1 %none, label %exit, label %pre

pre:
  %n = tail call i64 @llvm.umin.i64(i64 %kk, i64 64)
  br label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.next, %loop ]
  %s = phi float [ 0.0, %pre ], [ %s.next, %loop ]
  %k32 = trunc nuw nsw i64 %k to i32
  %ik = add i32 %i, %k32
  %ikx = zext i32 %ik to i64
  %px = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %ikx
  %vx = load float, ptr addrspace(1) %px, align 4
  %pw = getelementptr inbounds nuw float, ptr addrspace(1) %w, i64 %k
  %vw = load float, ptr addrspace(1) %pw, align 4
  %s.next = tail call float @llvm.fma.f32(float %vx, float %vw, float %s)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi float [ 0.0, %entry ], [ %s.next, %loop ]
  %ix = zext i32 %i to i64
  %py = getelementptr inbounds nuw float, ptr addrspace(1) %y, i64 %ix
  store float %r, ptr addrspace(1) %py, align 4
  ret void
}

; CHECK-LABEL: alone:
; CHECK:       device_load 0, i32, x, r{{[0-9]+}}, u0_u1, r{{[0-9]+}}, unsigned
define agx_kernel void @alone(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %kk) #0 {
entry:
  %i = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %none = icmp slt i64 %kk, 1
  br i1 %none, label %exit, label %pre

pre:
  %n = tail call i64 @llvm.umin.i64(i64 %kk, i64 64)
  br label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.next, %loop ]
  %s = phi float [ 0.0, %pre ], [ %s.next, %loop ]
  %k32 = trunc nuw nsw i64 %k to i32
  %ik = add i32 %i, %k32
  %ikx = zext i32 %ik to i64
  %px = getelementptr inbounds nuw float, ptr addrspace(1) %x, i64 %ikx
  %vx = load float, ptr addrspace(1) %px, align 4
  %s.next = fadd float %vx, %s
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi float [ 0.0, %entry ], [ %s.next, %loop ]
  %ix = zext i32 %i to i64
  %py = getelementptr inbounds nuw float, ptr addrspace(1) %y, i64 %ix
  store float %r, ptr addrspace(1) %py, align 4
  ret void
}

attributes #0 = { "agx-upper-bounds:air.thread_position_in_grid"="65536,0,0" }
